from datetime import timedelta

import MetaTrader5 as mt5

from . import config


def _max_affordable_lot(symbol: str, target_margin: float, info) -> float:
    """Largest lot whose required margin stays <= target_margin, found by binary
    search since brokers commonly tier margin rates up for larger positions
    (margin is not simply linear in volume)."""
    tick = mt5.symbol_info_tick(symbol)
    if tick is None or target_margin <= 0:
        return 0.0

    step = info.volume_step or 0.01
    lo, hi = 0.0, info.volume_max

    margin_at_max = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, hi, tick.ask)
    if margin_at_max is not None and margin_at_max <= target_margin:
        return hi

    for _ in range(25):
        mid = (lo + hi) / 2
        margin = mt5.order_calc_margin(mt5.ORDER_TYPE_BUY, symbol, mid, tick.ask)
        if margin is None:
            hi = mid
            continue
        if margin <= target_margin:
            lo = mid
        else:
            hi = mid

    return max(0.0, (lo // step) * step)


def calc_lot_size(symbol: str, balance: float, sl_distance_price: float, margin_free: float = None) -> float:
    """Position size so that a stop-loss hit costs ~RISK_PCT_PER_TRADE of balance,
    capped so the order never requires more than MARGIN_USAGE_CAP of current free
    margin, AND never more than MAX_MARGIN_PCT_OF_BALANCE of account balance.

    A tight ATR-based stop on a low-volatility major (or an oddly-quoted/thin
    symbol) can mathematically call for a position size far beyond what's sane
    (risking a fixed % of balance over a tiny price move requires huge notional)
    -- this is what previously produced e.g. SEKJPY 88.20 lots. The margin_free
    cap alone doesn't catch this early in a cycle when few positions are open
    (free margin is then ~= balance, so a generous fraction of "free margin" is
    still a huge single-trade size) -- the balance-based cap is the real ceiling,
    independent of how much margin currently happens to be free.
    """
    info = mt5.symbol_info(symbol)
    if info is None or sl_distance_price <= 0:
        return 0.0

    risk_amount = balance * config.EFFECTIVE_RISK_PCT_PER_TRADE
    tick_size = info.trade_tick_size or info.point
    tick_value = info.trade_tick_value
    if not tick_size or not tick_value or tick_size <= 0 or tick_value <= 0:
        return 0.0

    loss_per_lot = (sl_distance_price / tick_size) * tick_value
    if loss_per_lot <= 0:
        return 0.0

    lot = risk_amount / loss_per_lot

    max_lot_by_balance = _max_affordable_lot(symbol, balance * config.EFFECTIVE_MAX_MARGIN_PCT_OF_BALANCE, info)
    lot = min(lot, max_lot_by_balance)

    if margin_free is not None:
        max_lot_by_margin = _max_affordable_lot(symbol, margin_free * config.EFFECTIVE_MARGIN_USAGE_CAP, info)
        lot = min(lot, max_lot_by_margin)

    step = info.volume_step or 0.01
    lot = round(lot / step) * step
    lot = min(info.volume_max, lot)
    if lot < info.volume_min:
        return 0.0  # can't afford even the minimum lot within the margin cap
    return round(lot, 2)


def sizing_balance(current_balance: float, balance_history, now, lookback_days: float = None) -> float:
    """Returns the balance new positions should be SIZED off of: the smaller of
    the live current balance and the balance from `lookback_days` ago.

    A trade only gets bigger once a gain has held for that long -- a fresh,
    unproven equity spike can't inflate size immediately (min() always picks
    the smaller value), while a real drawdown is reflected immediately (same
    reason). This exists because RISK_PCT_PER_TRADE sizes off whatever the
    current balance happens to be: the 2026-08-27 post-mortem found lot sizes
    ~65% bigger than baseline right as the account hit its $258K peak, purely
    because balance had grown -- those oversized trades then absorbed most of
    the losing stretch that gave the peak back. This does NOT touch
    account_summary's "balance" (the LLM always sees the real, true balance)
    or the daily-loss circuit breaker (which must react to a real loss
    immediately, not on a delay).

    balance_history: chronological list of (timestamp, balance) samples, both
    comparable to `now` and to each other via subtraction/`timedelta`. Returns
    current_balance unchanged if no sample is old enough yet (e.g. early in a
    run) -- there's nothing to ratchet against, so sizing behaves as it always
    has until history accumulates.
    """
    if lookback_days is None:
        lookback_days = config.SIZING_BALANCE_LOOKBACK_DAYS
    cutoff = now - timedelta(days=lookback_days)
    aged = None
    for t, bal in balance_history:
        if t <= cutoff:
            aged = bal
        else:
            break
    if aged is None:
        return current_balance
    return min(current_balance, aged)


def trailing_sl_update(direction: str, price_open: float, initial_sl: float,
                        current_sl: float, current_price: float):
    """Mechanical profit-lock: returns a new stop-loss price to ratchet to, or
    None if no update is due right now.

    `initial_sl` MUST be the position's stop-loss as originally set at entry,
    not its current (possibly already-trailed) one -- R is the price distance
    from price_open to that original SL, and it has to stay fixed for the life
    of the position or profit_R inflates itself into a runaway trail every
    time the SL ratchets closer to price.

    Once profit reaches TRAIL_ACTIVATE_R multiples of R, this locks in
    (profit_R - TRAIL_GIVEBACK_R) * R -- i.e. once active it always allows
    giving back at most TRAIL_GIVEBACK_R worth of R from whatever peak profit
    has been reached. Returns None below the activation threshold, or if the
    computed level would loosen (not tighten) `current_sl` -- the SL this
    produces only ever moves in the position's favor.
    """
    r = abs(price_open - initial_sl)
    if r <= 0:
        return None

    is_buy = direction == "BUY"
    profit_r = (current_price - price_open) / r if is_buy else (price_open - current_price) / r
    if profit_r < config.TRAIL_ACTIVATE_R:
        return None

    locked_r = max(0.0, profit_r - config.TRAIL_GIVEBACK_R)
    new_sl = price_open + locked_r * r if is_buy else price_open - locked_r * r

    if is_buy and new_sl <= current_sl:
        return None
    if not is_buy and new_sl >= current_sl:
        return None
    return new_sl


def open_risk_dollars(positions) -> float:
    """Sum of $ (account currency) currently at risk across a list of MT5
    position objects -- i.e. what each position's own stop-loss would cost if
    hit, valued via that symbol's tick_value/tick_size. Positions with no SL
    set (sl == 0) are excluded from the sum since they have no bounded risk to
    add up -- see trader.py's _ensure_stops_attached, which exists precisely
    so this shouldn't happen for bot-opened positions. Used for the aggregate
    portfolio risk-budget gate (config.MAX_PORTFOLIO_RISK_PCT), which the
    per-trade risk % and per-currency concentration cap don't cover on their
    own: many uncorrelated symbols can each be independently "small" risk and
    still sum to an oversized aggregate bet.
    """
    total = 0.0
    for pos in positions:
        if not pos.sl:
            continue
        info = mt5.symbol_info(pos.symbol)
        if info is None:
            continue
        tick_size = info.trade_tick_size or info.point
        tick_value = info.trade_tick_value
        if not tick_size or not tick_value or tick_size <= 0 or tick_value <= 0:
            continue
        sl_distance = abs(pos.price_open - pos.sl)
        total += (sl_distance / tick_size) * tick_value * pos.volume
    return total
