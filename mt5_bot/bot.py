import logging
import time
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import MetaTrader5 as mt5
import pandas as pd

from . import concentration, config, connector, llm_strategy, memory, news, risk, strategy, trader

TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30, "H1": mt5.TIMEFRAME_H1, "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}

TIMEFRAME_MINUTES = {
    "M1": 1, "M5": 5, "M15": 15, "M30": 30, "H1": 60, "H4": 240, "D1": 1440,
}

# How long we tolerate no new closed candle before warning that the feed
# looks stuck (as opposed to just quiet). Weekends/holidays trigger this too
# -- that's expected and the log message says so -- but it also catches a
# genuinely dead data feed that would otherwise fail silently forever.
STALE_RELOG_INTERVAL = timedelta(hours=1)


def _fmt_duration(delta: timedelta) -> str:
    total_minutes = int(delta.total_seconds() // 60)
    hours, minutes = divmod(total_minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    return f"{minutes}m"

log = logging.getLogger("bot")


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config.LOG_FILE, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def discover_symbols():
    if config.SYMBOLS != "auto":
        for s in config.SYMBOLS:
            mt5.symbol_select(s, True)
        return list(config.SYMBOLS)

    filters = config.SYMBOL_GROUP_FILTER
    if isinstance(filters, str):
        filters = [filters]
    filters = [f.lower() for f in filters]

    all_symbols = mt5.symbols_get()
    matched = [
        s.name for s in all_symbols
        if any(f in (s.path or "").lower() for f in filters)
    ]
    for s in matched:
        mt5.symbol_select(s, True)
    log.info("Auto-discovered %d symbols matching %s", len(matched), filters)
    return matched


def fetch_candles(symbol: str, timeframe):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 0, 500)
    if bars is None or len(bars) == 0:
        return None
    df = pd.DataFrame(bars)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def run():
    setup_logging()
    account = connector.connect()
    tf = TIMEFRAME_MAP[config.TIMEFRAME]
    symbols = discover_symbols()
    if not symbols:
        log.error("No symbols to trade, exiting.")
        connector.disconnect()
        return

    log.info("Trading %d symbols on %s: %s", len(symbols), config.TIMEFRAME, ", ".join(symbols))
    log.info("Risk per trade: %.2f%% of balance (RISK_FACTOR=%.2fx of base %.2f%%) | "
              "SL=%sxATR TP=%sxATR | daily loss breaker: %.0f%%",
              config.EFFECTIVE_RISK_PCT_PER_TRADE * 100, config.RISK_FACTOR,
              config.RISK_PCT_PER_TRADE * 100, config.SL_ATR_MULT, config.TP_ATR_MULT,
              config.MAX_DAILY_LOSS_PCT * 100)

    last_seen_candle_time = {s: None for s in symbols}
    last_clock_candle_time = None
    current_day = datetime.now(timezone.utc).date()
    day_start_balance = account.balance
    halted_today = False
    # (timestamp, balance) samples, oldest first -- feeds risk.sizing_balance()
    # so position sizing can't inflate off a fresh, unproven balance spike.
    # Pruned each iteration; resets on restart (falls back to unratcheted
    # sizing until enough history rebuilds, the same as early in any run).
    balance_history = []

    clock_symbol = config.LLM_CLOCK_SYMBOL if config.LLM_CLOCK_SYMBOL in symbols else symbols[0]

    stale_threshold = timedelta(minutes=max(2 * TIMEFRAME_MINUTES[config.TIMEFRAME], 10))
    last_activity_time = datetime.now(timezone.utc)
    is_stale = False
    next_stale_log = None

    try:
        while True:
            now = datetime.now(timezone.utc)
            if now.date() != current_day:
                current_day = now.date()
                acc = mt5.account_info()
                day_start_balance = acc.balance if acc else day_start_balance
                halted_today = False
                log.info("New trading day, balance reset to %.2f", day_start_balance)

            acc = mt5.account_info()
            if acc is None:
                log.error("Lost connection to terminal, retrying...")
                time.sleep(config.POLL_SECONDS)
                continue
            balance = acc.balance

            balance_history.append((now, balance))
            prune_before = now - timedelta(days=config.SIZING_BALANCE_LOOKBACK_DAYS + 1)
            while balance_history and balance_history[0][0] < prune_before:
                balance_history.pop(0)
            sizing_bal = risk.sizing_balance(balance, balance_history, now)

            if not halted_today and day_start_balance > 0:
                pnl_today = balance - day_start_balance
                if -pnl_today / day_start_balance >= config.MAX_DAILY_LOSS_PCT:
                    halted_today = True
                    log.warning("Daily loss circuit breaker hit (%.1f%%). No new entries until tomorrow.",
                                config.MAX_DAILY_LOSS_PCT * 100)

            any_new_candle = False
            calendar_events = news.get_calendar_events()

            if config.USE_LLM_STRATEGY:
                clock_df = fetch_candles(clock_symbol, tf)
                if clock_df is not None and len(clock_df) >= 2:
                    clock_candle_time = clock_df.iloc[-2]["time"]  # last closed candle
                    if last_clock_candle_time != clock_candle_time:
                        last_clock_candle_time = clock_candle_time
                        any_new_candle = True
                        run_llm_decision_cycle(
                            symbols, tf, balance, sizing_bal, acc, halted_today, now, calendar_events
                        )
            else:
                for symbol in symbols:
                    df = fetch_candles(symbol, tf)
                    if df is None or len(df) < 5:
                        continue

                    closed = df.iloc[:-1]  # drop the still-forming candle
                    candle_time = closed.iloc[-1]["time"]
                    is_new_candle = last_seen_candle_time[symbol] != candle_time

                    positions = trader.open_positions(symbol)

                    if is_new_candle:
                        any_new_candle = True
                        last_seen_candle_time[symbol] = candle_time
                        signal = strategy.evaluate(closed)

                        for pos in positions:
                            pos_is_buy = pos.type == mt5.ORDER_TYPE_BUY
                            opposite = (pos_is_buy and signal.direction == "SELL") or \
                                       (not pos_is_buy and signal.direction == "BUY")
                            if opposite:
                                trader.close_position(pos)
                                positions = trader.open_positions(symbol)

                        if (signal.direction in ("BUY", "SELL")
                                and not halted_today
                                and len(positions) < config.MAX_OPEN_POSITIONS_PER_SYMBOL):
                            blocking = news.blocking_event(symbol, now, calendar_events)
                            if blocking:
                                log.info("%s: news blackout (%s), skipping new %s",
                                          symbol, blocking["event"], signal.direction)
                            else:
                                log.info("%s signal=%s reason=%s", symbol, signal.direction, signal.reason)
                                trader.open_trade(symbol, signal.direction, signal.atr, sizing_bal)

            if any_new_candle:
                if is_stale:
                    log.info("Data feed resumed after %s of no new candles.",
                             _fmt_duration(now - last_activity_time))
                is_stale = False
                next_stale_log = None
                last_activity_time = now
            elif now - last_activity_time > stale_threshold:
                if not is_stale or (next_stale_log is not None and now >= next_stale_log):
                    log.warning(
                        "No new %s candle in %s (expected roughly every %d min). "
                        "Market is likely closed (weekend/holiday); if it stays stale "
                        "once the market should be open, the data feed may be stuck.",
                        config.TIMEFRAME, _fmt_duration(now - last_activity_time),
                        TIMEFRAME_MINUTES[config.TIMEFRAME],
                    )
                    is_stale = True
                    next_stale_log = now + STALE_RELOG_INTERVAL

            time.sleep(config.POLL_SECONDS)

    except KeyboardInterrupt:
        log.info("Stopped by user (Ctrl+C).")
    finally:
        connector.disconnect()


def _fallback_exit_management(closed_by_symbol):
    """Used when the LLM call itself fails (network/API error) -- rather than
    leaving open positions completely unmanaged until the next cycle succeeds,
    fall back to the deterministic EMA/MACD/RSI ensemble (strategy.py) for
    EXITS ONLY: close a position if the ensemble signal has flipped against
    it. Does not open any new trades -- entries stay gated on the LLM's
    (higher-bar, memory-informed) judgment, only capital preservation on
    already-open risk happens here."""
    for symbol, (closed, pos) in closed_by_symbol.items():
        if pos is None:
            continue
        signal = strategy.evaluate(closed)
        pos_is_buy = pos.type == mt5.ORDER_TYPE_BUY
        opposite = (pos_is_buy and signal.direction == "SELL") or \
                   (not pos_is_buy and signal.direction == "BUY")
        if opposite:
            log.info("%s: LLM cycle unavailable, closing on deterministic fallback signal=%s",
                      symbol, signal.direction)
            trader.close_position(pos)


def _margin_gate_ok(account) -> bool:
    """True if new entries are allowed on the given account state. margin==0
    means no open positions consume margin at all, so there's nothing to gate.
    Kept as a function (not an inline snapshot) so it can be re-evaluated
    mid-cycle after a CLOSE frees margin -- see run_llm_decision_cycle."""
    return (
        account.margin <= 0 or not account.margin_level
        or account.margin_level >= config.MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES
    )


def run_llm_decision_cycle(symbols, tf, balance, sizing_bal, acc, halted_today, now, calendar_events):
    # Always include symbols the bot currently has an open position on, even
    # if they've since fallen out of the tradable `symbols` universe (e.g. the
    # universe was narrowed while a position from the old universe was still
    # open) -- otherwise that position silently loses LLM-level CLOSE/
    # thesis-break monitoring and is left running on its bare SL/TP alone.
    # Safe to do every cycle: a symbol only gets pulled in while it actually
    # has a position (the `pos is not None` guard below already forbids new
    # BUY/SELL on it), and it drops back out on its own once that position
    # closes -- it never becomes newly entry-eligible.
    extra_symbols = [p.symbol for p in trader.open_positions_all() if p.symbol not in symbols]
    if extra_symbols:
        for s in extra_symbols:
            mt5.symbol_select(s, True)
        log.info("LLM cycle: also monitoring %s (open position outside the tradable symbol list)",
                  ", ".join(extra_symbols))

    symbol_rows = []
    rows_by_symbol = {}
    closed_by_symbol = {}
    for symbol in list(symbols) + extra_symbols:
        df = fetch_candles(symbol, tf)
        if df is None or len(df) < 25:
            continue
        closed = df.iloc[:-1]
        positions = trader.open_positions(symbol)
        pos = positions[0] if positions else None
        closed_by_symbol[symbol] = (closed, pos)
        spread = trader.spread_points(symbol)
        if pos is None and spread > config.MAX_SPREAD_POINTS:
            # structurally too costly to ever trade -- trader.open_trade() would
            # hard-reject it anyway, so don't spend LLM context deciding on it
            continue
        row = llm_strategy.build_symbol_row(symbol, closed, spread, pos)
        if row is None:
            continue
        blocking = news.blocking_event(symbol, now, calendar_events)
        row["news_blackout"] = blocking["event"] if blocking else None
        symbol_rows.append(row)
        rows_by_symbol[symbol] = (row, pos)

    if not symbol_rows:
        log.warning("LLM cycle: no symbols had usable data, skipping")
        return

    open_count = sum(1 for _, pos in rows_by_symbol.values() if pos is not None)
    recent_performance = llm_strategy.get_recent_performance()
    negative_setups = llm_strategy.negative_expectancy_setups(recent_performance.get("by_setup", {}))
    headlines = news.get_headlines()
    if negative_setups:
        log.info("LLM cycle: hard-blocking new entries tagged %s (negative realized expectancy)",
                  ", ".join(sorted(negative_setups)))
    account_summary = {
        "balance": round(balance, 2),
        "equity": round(acc.equity, 2),
        "currency": acc.currency,
        "margin_used": round(acc.margin, 2),
        "margin_free": round(acc.margin_free, 2),
        "margin_level_pct": round(acc.margin_level, 1) if acc.margin_level else None,
        "open_positions": open_count,
        "max_total_open_positions": config.MAX_TOTAL_OPEN_POSITIONS,
        "halted_today": halted_today,
        "risk_pct_per_trade": config.EFFECTIVE_RISK_PCT_PER_TRADE,
        "risk_appetite_factor": config.RISK_FACTOR,
        "sl_atr_mult": config.SL_ATR_MULT,
        "tp_atr_mult": config.TP_ATR_MULT,
        "max_daily_loss_pct": config.MAX_DAILY_LOSS_PCT,
        "recent_performance": recent_performance,
        "trade_memory": memory.refresh_trade_memory(),
    }

    log.info("LLM cycle: requesting decisions for %d symbols (%d with open positions)",
              len(symbol_rows), open_count)
    decisions = llm_strategy.get_decisions(account_summary, symbol_rows, headlines)

    if decisions is None:
        # LLM call itself failed -- don't leave open positions unmanaged until
        # the next cycle happens to succeed; fall back to the deterministic
        # ensemble for exits only (see _fallback_exit_management docstring).
        _fallback_exit_management(closed_by_symbol)
        return

    open_position_count = open_count
    # Hard code-level margin gate: block ALL new entries this cycle if margin
    # is already stressed, independent of what the LLM decides. acc.margin==0
    # means no open positions consume margin at all, so there's nothing to gate.
    margin_gate_ok = _margin_gate_ok(acc)
    if not margin_gate_ok:
        log.warning("margin_level %.1f%% below the %.0f%% floor - blocking new entries "
                     "until this cycle's closes (if any) free it",
                     acc.margin_level, config.MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES)

    # Running snapshot of this bot's open positions, used for the currency
    # concentration check -- updated as trades open within this same cycle so
    # a second candidate correctly sees the first one's exposure.
    open_positions_snapshot = trader.open_positions_all()

    # Aggregate portfolio risk budget: per-trade RISK_PCT and the per-currency
    # concentration cap each look at one trade/theme at a time, so several
    # *uncorrelated* symbols can each look "small" and still sum to an
    # oversized total bet. Track committed $-at-risk across the cycle and stop
    # opening new trades once it would exceed MAX_PORTFOLIO_RISK_PCT of balance.
    committed_risk = risk.open_risk_dollars(open_positions_snapshot)
    portfolio_risk_budget = sizing_bal * config.EFFECTIVE_MAX_PORTFOLIO_RISK_PCT

    # Every CLOSE is applied before any new BUY/SELL is evaluated, regardless
    # of the order the LLM returned them in -- this is what makes the
    # SYSTEM_PROMPT's MAKE_ROOM pairing (close a weak position, open a
    # stronger one, same cycle) actually work: margin_gate_ok/committed_risk
    # are refreshed below right after the closes, so a slot/margin/risk budget
    # freed here is usable by the paired entry immediately, not next cycle.
    close_decisions = [d for d in decisions if d.get("action") == "CLOSE"]
    entry_decisions = [d for d in decisions if d.get("action") in ("BUY", "SELL")]

    for decision in close_decisions:
        symbol = decision.get("symbol")
        reason = decision.get("reason", "")
        if symbol not in rows_by_symbol:
            continue
        _, pos = rows_by_symbol[symbol]
        log.info("%s LLM decision=CLOSE reason=%s", symbol, reason)
        if pos is not None:
            trader.close_position(pos)
            open_position_count -= 1
            open_positions_snapshot = [p for p in open_positions_snapshot if p.symbol != symbol]

    if close_decisions:
        fresh_acc = mt5.account_info()
        if fresh_acc is not None:
            margin_gate_ok = _margin_gate_ok(fresh_acc)
        committed_risk = risk.open_risk_dollars(open_positions_snapshot)

    for decision in entry_decisions:
        symbol = decision.get("symbol")
        action = decision.get("action")
        reason = decision.get("reason", "")
        setup = decision.get("setup")
        if symbol not in rows_by_symbol:
            continue

        row, pos = rows_by_symbol[symbol]
        log.info("%s LLM decision=%s reason=%s", symbol, action, reason)

        if halted_today:
            log.info("%s: daily loss breaker active, skipping new %s", symbol, action)
            continue
        if not margin_gate_ok:
            log.info("%s: margin gate active, skipping new %s", symbol, action)
            continue
        if row.get("news_blackout"):
            log.info("%s: news blackout (%s), skipping new %s", symbol, row["news_blackout"], action)
            continue
        if pos is not None:
            log.info("%s: already has an open position, ignoring %s", symbol, action)
            continue
        if row["atr"] is None:
            continue
        if open_position_count >= config.MAX_TOTAL_OPEN_POSITIONS:
            log.info("%s: global open-position cap (%d) reached, skipping new %s",
                      symbol, config.MAX_TOTAL_OPEN_POSITIONS, action)
            continue
        if concentration.exceeds_concentration_cap(symbol, action, open_positions_snapshot):
            log.info("%s: currency/metal concentration cap reached, skipping new %s", symbol, action)
            continue
        if setup in negative_setups:
            log.info("%s: setup %s has negative realized expectancy, skipping new %s", symbol, setup, action)
            continue
        intended_risk = sizing_bal * config.EFFECTIVE_RISK_PCT_PER_TRADE
        if committed_risk + intended_risk > portfolio_risk_budget:
            log.info("%s: portfolio risk budget (%.2f/%.2f already committed) reached, skipping new %s",
                      symbol, committed_risk, portfolio_risk_budget, action)
            continue

        result = trader.open_trade(symbol, action, row["atr"], sizing_bal, setup_tag=setup)
        if result is not None and result.retcode == mt5.TRADE_RETCODE_DONE:
            open_position_count += 1
            committed_risk += intended_risk
            open_positions_snapshot.append(SimpleNamespace(
                symbol=symbol, type=mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL))


if __name__ == "__main__":
    run()
