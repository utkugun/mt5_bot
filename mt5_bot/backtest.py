"""Offline backtest harness for the LLM strategy against MT5 historical data.

IMPORTANT: this does NOT use the MT5 Strategy Tester. The Strategy Tester only
runs compiled MQL5/MQL4 Expert Advisors inside its own simulated environment;
this bot is a Python script that talks to a live terminal, which the tester
can't load at all. Instead this pulls historical candles via the normal
`MetaTrader5` API (works fine outside the tester) and replays them through the
exact same decision logic bot.py uses live (llm_strategy.build_symbol_row /
get_decisions, risk.calc_lot_size), against a simple simulated broker:
- fills at the close of the candle that triggered the decision, offset by
  half the bar's historical spread to approximate ask/bid
- SL/TP checked against each subsequent bar's high/low (SL assumed to win on
  a bar that touches both, i.e. the conservative/worst-case assumption)
- position sizing, margin cap and account-summary figures reuse risk.py and
  mt5.order_calc_margin/order_calc_profit against live symbol specs (margin
  rates etc. are pulled as of now, not as of the historical date -- a known
  approximation)
- the economic-calendar news gate (news.py) IS replayed against real
  historical events for the test window (fetched once up front, actual dates)
  -- but headlines are NOT: Finnhub's /news endpoint only returns latest
  news, no historical date range, so NEWS_HEADLINES_ENABLED has no effect
  here and the LLM never sees a NEWS section in a backtest, live-only

This calls the real Anthropic API once per simulated decision cycle (one per
newly-closed candle on the clock symbol) -- run with --dry-run first to see
how many calls a given date range/symbol set will cost before spending money.
"""
import argparse
import csv
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import MetaTrader5 as mt5
import pandas as pd

from . import concentration, config, connector, llm_strategy, news, risk, strategy

log = logging.getLogger("backtest")

DEFAULT_SYMBOLS = ["EURUSD", "GBPUSD", "USDJPY", "USDCHF", "AUDUSD", "USDCAD", "NZDUSD", "EURJPY",
                    "XAUUSD", "XAGUSD"]
DEFAULT_CLOCK_SYMBOL = "EURUSD"
WARMUP_DAYS = 20  # extra history fetched before the test window so EMA200 etc. are warmed up
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "backtest_results")

# $ per 1M tokens (input, output), cached 2026-09-04 -- update if pricing changes.
PRICING_PER_MTOK = {
    "claude-sonnet-5": (2.00, 10.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

# Standard Anthropic prompt-cache multipliers on the base input price, for the
# default 5-minute ephemeral cache llm_strategy.py uses: writing to the cache
# costs 1.25x a normal input token, reading from it costs 0.1x.
CACHE_WRITE_MULT = 1.25
CACHE_READ_MULT = 0.1


@dataclass
class SimPosition:
    symbol: str
    direction: str  # "BUY" / "SELL"
    volume: float
    price_open: float
    sl: float
    tp: float
    open_time: pd.Timestamp
    setup: str = None  # the LLM's own setup tag at entry, for sim_recent_performance()

    @property
    def order_type(self) -> int:
        return mt5.ORDER_TYPE_BUY if self.direction == "BUY" else mt5.ORDER_TYPE_SELL


@dataclass
class Trade:
    symbol: str
    direction: str
    open_time: pd.Timestamp
    close_time: pd.Timestamp
    open_price: float
    close_price: float
    volume: float
    pnl: float
    reason: str  # "SL", "TP", "LLM_CLOSE", "END_OF_TEST"
    setup: str = None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler("mt5_bot_backtest.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def fetch_history(symbols, tf, start, end):
    """Returns {symbol: DataFrame} sorted by time ascending, or raises if a
    symbol has no data (bad symbol name / no history for that range)."""
    data = {}
    for symbol in symbols:
        mt5.symbol_select(symbol, True)
        bars = mt5.copy_rates_range(symbol, tf, start, end)
        if bars is None or len(bars) == 0:
            raise RuntimeError(f"No historical data for {symbol} in {start}..{end} "
                                f"(check symbol name / that history is downloaded in MT5)")
        df = pd.DataFrame(bars)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        data[symbol] = df.sort_values("time").reset_index(drop=True)
    return data


def bars_up_to(df: pd.DataFrame, t: pd.Timestamp) -> pd.DataFrame:
    idx = df["time"].searchsorted(t, side="right")
    return df.iloc[:idx]


def check_sl_tp(pos: SimPosition, bar) -> tuple[str, float] | None:
    """Returns (reason, fill_price) if this bar's range touched SL or TP,
    else None. SL checked first when a single bar touches both (worst case)."""
    if pos.direction == "BUY":
        if bar["low"] <= pos.sl:
            return "SL", pos.sl
        if bar["high"] >= pos.tp:
            return "TP", pos.tp
    else:
        if bar["high"] >= pos.sl:
            return "SL", pos.sl
        if bar["low"] <= pos.tp:
            return "TP", pos.tp
    return None


class SimBroker:
    def __init__(self, starting_balance: float):
        self.balance = starting_balance
        self.positions: dict[str, SimPosition] = {}
        self.trades: list[Trade] = []
        self.equity_curve: list[tuple[pd.Timestamp, float]] = []
        # (timestamp, balance) samples, oldest first -- feeds
        # risk.sizing_balance(), same mechanism as bot.py's live loop.
        self.balance_history: list[tuple[pd.Timestamp, float]] = []

    def unrealized_pnl(self, pos: SimPosition, current_price: float) -> float:
        profit = mt5.order_calc_profit(pos.order_type, pos.symbol, pos.volume, pos.price_open, current_price)
        return profit if profit is not None else 0.0

    def equity(self, last_price_by_symbol: dict) -> float:
        total = self.balance
        for symbol, pos in self.positions.items():
            price = last_price_by_symbol.get(symbol)
            if price is not None:
                total += self.unrealized_pnl(pos, price)
        return total

    def margin_used(self) -> float:
        total = 0.0
        for pos in self.positions.values():
            m = mt5.order_calc_margin(pos.order_type, pos.symbol, pos.volume, pos.price_open)
            total += m or 0.0
        return total

    def close(self, symbol: str, price: float, t: pd.Timestamp, reason: str):
        pos = self.positions.pop(symbol, None)
        if pos is None:
            return
        pnl = self.unrealized_pnl(pos, price)
        self.balance += pnl
        self.trades.append(Trade(symbol, pos.direction, pos.open_time, t,
                                  pos.price_open, price, pos.volume, pnl, reason, pos.setup))
        log.info("[%s] CLOSE %s %s @ %.5f pnl=%.2f reason=%s balance=%.2f",
                  t, symbol, pos.direction, price, pnl, reason, self.balance)

    def open(self, symbol: str, direction: str, price: float, sl: float, tp: float,
              volume: float, t: pd.Timestamp, setup: str = None):
        self.positions[symbol] = SimPosition(symbol, direction, volume, price, sl, tp, t, setup)
        log.info("[%s] OPEN %s %s %.2f lots @ %.5f SL=%.5f TP=%.5f",
                  t, symbol, direction, volume, price, sl, tp)


def sim_recent_performance(trades: list, now: pd.Timestamp, days: int = None) -> dict:
    """Backtest equivalent of llm_strategy.get_recent_performance(), sourced
    from the SimBroker's own closed trades instead of real MT5 deal history
    (which is meaningless for simulated trades) -- same shape, so
    llm_strategy.negative_expectancy_setups() works unchanged against either."""
    if days is None:
        days = llm_strategy.PERF_LOOKBACK_DAYS
    cutoff = now - timedelta(days=days)
    recent = [tr for tr in trades if tr.close_time >= cutoff]
    if not recent:
        return {"lookback_days": days, "note": "no closed trade history yet", "by_setup": {}, "by_direction": {}}

    def summarize(rows):
        out = {}
        buckets = defaultdict(list)
        for key, profit in rows:
            buckets[key].append(profit)
        for key, profits in buckets.items():
            n = len(profits)
            if n < llm_strategy.PERF_MIN_SAMPLE:
                continue
            wins = [p for p in profits if p > 0]
            out[key] = {"n": n, "win_rate_pct": round(len(wins) / n * 100, 1), "expectancy": round(sum(profits) / n, 2)}
        return out

    by_setup = summarize([(tr.setup or "UNTAGGED", tr.pnl) for tr in recent])
    by_direction = summarize([(tr.direction, tr.pnl) for tr in recent])
    return {
        "lookback_days": days,
        "closed_trades": len(recent),
        "min_sample_for_stats": llm_strategy.PERF_MIN_SAMPLE,
        "by_setup": by_setup,
        "by_direction": by_direction,
    }


def _fallback_exit_management(rows_by_symbol: dict, broker: "SimBroker", t: pd.Timestamp):
    """Backtest equivalent of bot.py's fallback of the same name: when
    get_decisions() returns None (the LLM call itself failed), don't leave
    open positions unmanaged until the next cycle happens to succeed -- close
    on the deterministic EMA/MACD/RSI ensemble's opposite signal instead.
    Entries stay gated on the LLM; this only closes existing risk."""
    for symbol, (row, pos, bar, point, spread_pts, closed) in rows_by_symbol.items():
        if pos is None:
            continue
        signal = strategy.evaluate(closed)
        pos_is_buy = pos.direction == "BUY"
        opposite = (pos_is_buy and signal.direction == "SELL") or (not pos_is_buy and signal.direction == "BUY")
        if opposite:
            price = bar["close"] - spread_pts * point / 2 if pos_is_buy else bar["close"] + spread_pts * point / 2
            log.info("[%s] %s: LLM cycle unavailable, closing on deterministic fallback signal=%s",
                      t, symbol, signal.direction)
            broker.close(symbol, price, t, "FALLBACK_CLOSE")


def _margin_gate_ok(margin_used_val: float, margin_level_val) -> bool:
    """Mirrors bot.py's _margin_gate_ok: True if new entries are allowed given
    this margin state. margin_used<=0 means no open positions consume margin
    at all, so there's nothing to gate. Kept as a function (not an inline
    snapshot) so it can be re-evaluated mid-cycle after a CLOSE frees margin."""
    return (
        margin_used_val <= 0 or not margin_level_val
        or margin_level_val >= config.MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES
    )


def run_backtest(symbols, clock_symbol, start, end, starting_balance, dry_run=False):
    tf = mt5.TIMEFRAME_M15
    fetch_start = start - timedelta(days=WARMUP_DAYS)
    log.info("Fetching history %s..%s (incl. %d-day warmup) for %s",
              fetch_start, end, WARMUP_DAYS, ", ".join(symbols))
    history = fetch_history(symbols, tf, fetch_start, end)

    calendar_events = news.fetch_calendar_range(fetch_start, end)
    log.info("Economic calendar: %d events fetched for the backtest window (NEWS_CALENDAR_ENABLED=%s)",
              len(calendar_events), config.NEWS_CALENDAR_ENABLED)

    clock_df = history[clock_symbol]
    clock_times = clock_df[(clock_df["time"] >= start) & (clock_df["time"] <= end)]["time"].tolist()
    log.info("%d decision cycles in range (one per closed %s candle on %s)",
              len(clock_times), config.TIMEFRAME, clock_symbol)

    if dry_run:
        log.info("--dry-run: not calling the LLM or simulating trades. "
                  "This range would make %d Anthropic API calls.", len(clock_times))
        return None

    broker = SimBroker(starting_balance)
    day_start_balance = starting_balance
    current_day = None
    halted_today = False

    for t in clock_times:
        day = t.date()
        if day != current_day:
            current_day = day
            day_start_balance = broker.balance
            halted_today = False

        last_price_by_symbol = {}
        rows_by_symbol = {}
        symbol_rows = []

        for symbol in symbols:
            closed = bars_up_to(history[symbol], t)
            if len(closed) < 25:
                continue
            bar = closed.iloc[-1]
            last_price_by_symbol[symbol] = bar["close"]

            pos = broker.positions.get(symbol)
            if pos is not None:
                hit = check_sl_tp(pos, bar)
                if hit is not None:
                    reason, fill_price = hit
                    broker.close(symbol, fill_price, t, reason)
                    pos = None

            spread_pts = float(bar["spread"]) if bar["spread"] == bar["spread"] else 20.0
            info = mt5.symbol_info(symbol)
            point = info.point if info else 0.0001

            sim_pos_obj = None
            if pos is not None:
                sim_pos_obj = type("P", (), {
                    "type": pos.order_type,
                    "volume": pos.volume,
                    "price_open": pos.price_open,
                    "profit": broker.unrealized_pnl(pos, bar["close"]),
                })()

            row = llm_strategy.build_symbol_row(symbol, closed, spread_pts, sim_pos_obj)
            if row is None:
                continue
            blocking = news.blocking_event(symbol, t, calendar_events)
            row["news_blackout"] = blocking["event"] if blocking else None
            symbol_rows.append(row)
            rows_by_symbol[symbol] = (row, pos, bar, point, spread_pts, closed)

        if not symbol_rows:
            continue

        if not halted_today and day_start_balance > 0:
            pnl_today = broker.balance - day_start_balance
            if -pnl_today / day_start_balance >= config.MAX_DAILY_LOSS_PCT:
                halted_today = True
                log.warning("[%s] Daily loss circuit breaker hit (%.1f%%)", t, config.MAX_DAILY_LOSS_PCT * 100)

        equity = broker.equity(last_price_by_symbol)
        margin_used = broker.margin_used()
        margin_free = equity - margin_used
        margin_level_pct = round(equity / margin_used * 100, 1) if margin_used > 0 else None
        sizing_bal = risk.sizing_balance(broker.balance, broker.balance_history, t)
        sim_perf = sim_recent_performance(broker.trades, t)
        negative_setups = llm_strategy.negative_expectancy_setups(sim_perf.get("by_setup", {}))
        if negative_setups:
            log.info("[%s] hard-blocking new entries tagged %s (negative realized expectancy)",
                      t, ", ".join(sorted(negative_setups)))
        account_summary = {
            "balance": round(broker.balance, 2),
            "equity": round(equity, 2),
            "currency": "USD",
            "margin_used": round(margin_used, 2),
            "margin_free": round(margin_free, 2),
            "margin_level_pct": margin_level_pct,
            "open_positions": len(broker.positions),
            "max_total_open_positions": config.MAX_TOTAL_OPEN_POSITIONS,
            "halted_today": halted_today,
            "risk_pct_per_trade": config.RISK_PCT_PER_TRADE,
            "sl_atr_mult": config.SL_ATR_MULT,
            "tp_atr_mult": config.TP_ATR_MULT,
            "max_daily_loss_pct": config.MAX_DAILY_LOSS_PCT,
        }

        decisions = llm_strategy.get_decisions(account_summary, symbol_rows)

        if decisions is None:
            # LLM call itself failed -- don't leave open positions unmanaged
            # until the next cycle happens to succeed (see bot.py's function
            # of the same name; this mirrors it for the sim broker).
            _fallback_exit_management(rows_by_symbol, broker, t)
            broker.balance_history.append((t, broker.balance))
            broker.equity_curve.append((t, broker.equity(last_price_by_symbol)))
            continue

        open_position_count = len(broker.positions)
        # Hard code-level margin gate: block ALL new entries this cycle if
        # margin is already stressed, independent of what the LLM decides.
        # margin_used<=0 means no open positions consume margin at all, so
        # there's nothing to gate (mirrors bot.py's margin_gate_ok).
        margin_gate_ok = _margin_gate_ok(margin_used, margin_level_pct)
        if not margin_gate_ok:
            log.warning("[%s] margin_level %.1f%% below the %.0f%% floor - blocking new entries "
                         "until this cycle's closes (if any) free it",
                         t, margin_level_pct, config.MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES)
        # Running snapshot of sim positions, used for the currency
        # concentration check -- updated as trades open/close within this same
        # cycle so a second candidate correctly sees the first one's exposure
        # (mirrors bot.py's open_positions_snapshot).
        open_positions_snapshot = [
            SimpleNamespace(symbol=p.symbol, type=p.order_type) for p in broker.positions.values()
        ]
        # Aggregate portfolio risk budget -- per-trade RISK_PCT and the
        # per-currency concentration cap each look at one trade/theme at a
        # time, so several *uncorrelated* symbols can each look "small" and
        # still sum to an oversized total bet (mirrors bot.py).
        committed_risk = risk.open_risk_dollars(list(broker.positions.values()))
        portfolio_risk_budget = sizing_bal * config.EFFECTIVE_MAX_PORTFOLIO_RISK_PCT

        # Every CLOSE is applied before any new BUY/SELL is evaluated,
        # regardless of the order the LLM returned them in -- mirrors bot.py's
        # MAKE_ROOM handling: margin_gate_ok/margin_free/committed_risk are
        # refreshed right after the closes, so a slot/margin/risk budget freed
        # here is usable by a paired entry in this same cycle, not the next.
        close_decisions = [d for d in decisions if d.get("action") == "CLOSE"]
        entry_decisions = [d for d in decisions if d.get("action") in ("BUY", "SELL")]

        for decision in close_decisions:
            symbol = decision.get("symbol")
            reason = decision.get("reason", "")
            if symbol not in rows_by_symbol:
                continue
            row, pos, bar, point, spread_pts, closed = rows_by_symbol[symbol]
            log.info("[%s] %s LLM decision=CLOSE reason=%s", t, symbol, reason)
            if pos is not None:
                is_buy = pos.direction == "BUY"
                price = bar["close"] - spread_pts * point / 2 if is_buy else bar["close"] + spread_pts * point / 2
                broker.close(symbol, price, t, "LLM_CLOSE")
                open_position_count -= 1
                open_positions_snapshot = [p for p in open_positions_snapshot if p.symbol != symbol]

        if close_decisions:
            fresh_equity = broker.equity(last_price_by_symbol)
            fresh_margin_used = broker.margin_used()
            margin_free = fresh_equity - fresh_margin_used
            fresh_margin_level = round(fresh_equity / fresh_margin_used * 100, 1) if fresh_margin_used > 0 else None
            margin_gate_ok = _margin_gate_ok(fresh_margin_used, fresh_margin_level)
            committed_risk = risk.open_risk_dollars(list(broker.positions.values()))

        for decision in entry_decisions:
            symbol = decision.get("symbol")
            action = decision.get("action")
            reason = decision.get("reason", "")
            setup = decision.get("setup")
            if symbol not in rows_by_symbol:
                continue
            row, pos, bar, point, spread_pts, closed = rows_by_symbol[symbol]
            log.info("[%s] %s LLM decision=%s reason=%s", t, symbol, action, reason)

            if halted_today or pos is not None or row["atr"] is None:
                continue
            if not margin_gate_ok:
                log.info("[%s] %s: margin gate active, skipping new %s", t, symbol, action)
                continue
            if row.get("news_blackout"):
                log.info("[%s] %s: news blackout (%s), skipping new %s", t, symbol, row["news_blackout"], action)
                continue
            if spread_pts > config.MAX_SPREAD_POINTS:
                log.info("[%s] %s: spread too wide (%.1f pts), skipping entry", t, symbol, spread_pts)
                continue
            if open_position_count >= config.MAX_TOTAL_OPEN_POSITIONS:
                continue
            if concentration.exceeds_concentration_cap(symbol, action, open_positions_snapshot):
                log.info("[%s] %s: currency/metal concentration cap reached, skipping entry", t, symbol)
                continue
            if setup in negative_setups:
                log.info("[%s] %s: setup %s has negative realized expectancy, skipping entry", t, symbol, setup)
                continue
            intended_risk = sizing_bal * config.EFFECTIVE_RISK_PCT_PER_TRADE
            if committed_risk + intended_risk > portfolio_risk_budget:
                log.info("[%s] %s: portfolio risk budget (%.2f/%.2f already committed) reached, skipping entry",
                          t, symbol, committed_risk, portfolio_risk_budget)
                continue

            atr_value = row["atr"]
            sl_dist = atr_value * config.SL_ATR_MULT
            tp_dist = atr_value * config.TP_ATR_MULT
            lot = risk.calc_lot_size(symbol, sizing_bal, sl_dist, margin_free)
            if lot <= 0:
                continue

            if action == "BUY":
                price = bar["close"] + spread_pts * point / 2
                sl = price - sl_dist
                tp = price + tp_dist
            else:
                price = bar["close"] - spread_pts * point / 2
                sl = price + sl_dist
                tp = price - tp_dist

            broker.open(symbol, action, price, sl, tp, lot, t, setup=setup)
            open_position_count += 1
            committed_risk += intended_risk
            open_positions_snapshot.append(SimpleNamespace(
                symbol=symbol, type=mt5.ORDER_TYPE_BUY if action == "BUY" else mt5.ORDER_TYPE_SELL))

        broker.balance_history.append((t, broker.balance))
        broker.equity_curve.append((t, broker.equity(last_price_by_symbol)))

    # close anything still open at the end of the test window at last known price
    final_prices = {s: bars_up_to(history[s], end).iloc[-1]["close"] for s in symbols if len(bars_up_to(history[s], end))}
    for symbol in list(broker.positions.keys()):
        price = final_prices.get(symbol)
        if price is not None:
            broker.close(symbol, price, end, "END_OF_TEST")

    return broker


def log_llm_cost():
    usage = llm_strategy.get_usage_totals()
    if usage["calls"] == 0:
        return
    if config.LLM_PROVIDER == "ollama":
        cost_str = "$0.00 (local)"
        log.info("LLM usage: %d calls, %d input tokens, %d output tokens -> %s",
                  usage["calls"], usage["input_tokens"], usage["output_tokens"], cost_str)
        return
    price_in, price_out = PRICING_PER_MTOK.get(config.LLM_MODEL, (None, None))
    cache_write = usage.get("cache_creation_input_tokens", 0)
    cache_read = usage.get("cache_read_input_tokens", 0)
    cost_str = "unknown model pricing"
    if price_in is not None:
        cost = (
            usage["input_tokens"] / 1e6 * price_in
            + usage["output_tokens"] / 1e6 * price_out
            + cache_write / 1e6 * price_in * CACHE_WRITE_MULT
            + cache_read / 1e6 * price_in * CACHE_READ_MULT
        )
        cost_str = f"~${cost:.2f}"
    log.info(
        "LLM usage: %d calls, %d input tokens, %d output tokens, "
        "%d cache-write tokens, %d cache-read tokens (incl. thinking) -> %s",
        usage["calls"], usage["input_tokens"], usage["output_tokens"], cache_write, cache_read, cost_str,
    )


def save_results(broker: SimBroker, starting_balance: float, output_dir: str = OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)

    trades_path = os.path.join(output_dir, "trades.csv")
    with open(trades_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["symbol", "direction", "open_time", "close_time", "open_price",
                          "close_price", "volume", "pnl", "reason", "setup"])
        for tr in broker.trades:
            writer.writerow([tr.symbol, tr.direction, tr.open_time, tr.close_time,
                              tr.open_price, tr.close_price, tr.volume, round(tr.pnl, 2), tr.reason, tr.setup])

    equity_path = os.path.join(output_dir, "equity_curve.csv")
    with open(equity_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "equity"])
        for t, eq in broker.equity_curve:
            writer.writerow([t, round(eq, 2)])

    wins = [tr for tr in broker.trades if tr.pnl > 0]
    losses = [tr for tr in broker.trades if tr.pnl <= 0]
    total_pnl = broker.balance - starting_balance
    peak = starting_balance
    max_dd = 0.0
    for _, eq in broker.equity_curve:
        peak = max(peak, eq)
        max_dd = max(max_dd, (peak - eq) / peak if peak > 0 else 0)

    log.info("=" * 60)
    log.info("BACKTEST SUMMARY")
    log.info("Trades: %d  (wins=%d losses=%d, win rate=%.1f%%)",
              len(broker.trades), len(wins), len(losses),
              100 * len(wins) / len(broker.trades) if broker.trades else 0.0)
    log.info("Starting balance: %.2f  Final balance: %.2f  PnL: %+.2f (%+.1f%%)",
              starting_balance, broker.balance, total_pnl,
              100 * total_pnl / starting_balance if starting_balance else 0.0)
    log.info("Max drawdown: %.1f%%", max_dd * 100)
    log.info("Trades saved to %s", trades_path)
    log.info("Equity curve saved to %s", equity_path)
    log_llm_cost()


def main():
    parser = argparse.ArgumentParser(description="Backtest the LLM strategy against MT5 historical data.")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                         help=f"comma-separated symbols (default: {','.join(DEFAULT_SYMBOLS)})")
    parser.add_argument("--clock-symbol", default=DEFAULT_CLOCK_SYMBOL)
    parser.add_argument("--days", type=int, default=18, help="how many days back from --end to test")
    parser.add_argument("--end", default=None, help="ISO date, default now (UTC)")
    parser.add_argument("--starting-balance", type=float, default=100000.0)
    parser.add_argument("--dry-run", action="store_true",
                         help="only report how many decision cycles/API calls this range would take, no LLM calls or trading")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                         help="where to write trades.csv/equity_curve.csv (default: ../backtest_results) -- "
                              "override when running more than one backtest at once so they don't clobber each other")
    args = parser.parse_args()

    setup_logging()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc) if args.end else datetime.now(timezone.utc)
    start = end - timedelta(days=args.days)

    account = connector.connect()
    log.info("Connected: login=%s server=%s balance=%.2f %s (backtest uses its own simulated balance, "
              "this is just to read historical data + live symbol specs)",
              account.login, account.server, account.balance, account.currency)
    try:
        broker = run_backtest(symbols, args.clock_symbol, start, end, args.starting_balance, dry_run=args.dry_run)
        if broker is not None:
            save_results(broker, args.starting_balance, args.output_dir)
    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()
