import logging
import time

import MetaTrader5 as mt5

from . import config, risk

log = logging.getLogger(__name__)

# Bitmask values for symbol_info.filling_mode (not exposed as constants by
# the Python wrapper, but stable across the MT5 platform).
_SYMBOL_FILLING_FOK = 1
_SYMBOL_FILLING_IOC = 2


def pick_filling_type(info) -> int:
    mode = info.filling_mode or 0
    if mode & _SYMBOL_FILLING_FOK:
        return mt5.ORDER_FILLING_FOK
    if mode & _SYMBOL_FILLING_IOC:
        return mt5.ORDER_FILLING_IOC
    return mt5.ORDER_FILLING_RETURN


def open_positions(symbol: str):
    positions = mt5.positions_get(symbol=symbol)
    if not positions:
        return []
    return [p for p in positions if p.magic == config.MAGIC_NUMBER]


def open_positions_all():
    """All of this bot's open positions across every symbol -- used for
    portfolio-level checks (currency concentration) that can't be done one
    symbol at a time."""
    positions = mt5.positions_get()
    if not positions:
        return []
    return [p for p in positions if p.magic == config.MAGIC_NUMBER]


def spread_points(symbol: str) -> float:
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None or not info.point:
        return float("inf")
    return (tick.ask - tick.bid) / info.point


def close_position(position):
    info = mt5.symbol_info(position.symbol)
    tick = mt5.symbol_info_tick(position.symbol)
    if info is None or tick is None:
        return None

    is_buy = position.type == mt5.ORDER_TYPE_BUY
    price = tick.bid if is_buy else tick.ask
    order_type = mt5.ORDER_TYPE_SELL if is_buy else mt5.ORDER_TYPE_BUY

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "position": position.ticket,
        "symbol": position.symbol,
        "volume": position.volume,
        "type": order_type,
        "price": price,
        "deviation": config.DEVIATION_POINTS,
        "magic": config.MAGIC_NUMBER,
        "comment": "bot close (opposite signal)",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": pick_filling_type(info),
    }
    result = mt5.order_send(request)
    log.info("Close %s %s ticket=%s -> retcode=%s",
              position.symbol, "BUY" if is_buy else "SELL",
              position.ticket, getattr(result, "retcode", None))
    return result


def open_trade(symbol: str, direction: str, atr_value: float, balance: float, setup_tag: str = None):
    if spread_points(symbol) > config.MAX_SPREAD_POINTS:
        log.info("%s: spread too wide, skipping entry", symbol)
        return None

    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None or atr_value != atr_value:  # NaN check
        return None

    sl_dist = atr_value * config.SL_ATR_MULT
    tp_dist = atr_value * config.TP_ATR_MULT
    acc = mt5.account_info()
    margin_free = acc.margin_free if acc else None
    lot = risk.calc_lot_size(symbol, balance, sl_dist, margin_free)
    if lot <= 0:
        log.warning("%s: computed lot size is 0, skipping", symbol)
        return None

    # Setup tag is embedded in the order comment (not just the log) so realized
    # deal history alone -- via mt5.history_deals_get -- can be grouped by setup
    # type later, without depending on the log file's format or retention.
    comment = f"bot|{setup_tag}" if setup_tag else "ema-rsi-macd bot"
    comment = comment[:31]  # MT5 order comment is truncated by most brokers past this

    digits = info.digits
    if direction == "BUY":
        order_type = mt5.ORDER_TYPE_BUY
        price = tick.ask
        sl = price - sl_dist
        tp = price + tp_dist
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = tick.bid
        sl = price + sl_dist
        tp = price - tp_dist

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot,
        "type": order_type,
        "price": price,
        "sl": round(sl, digits),
        "tp": round(tp, digits),
        "deviation": config.DEVIATION_POINTS,
        "magic": config.MAGIC_NUMBER,
        "comment": comment,
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": pick_filling_type(info),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        log.error("%s: order_send failed retcode=%s comment=%s",
                   symbol, getattr(result, "retcode", None), getattr(result, "comment", None))
        return result

    log.info("%s: opened %s %.2f lots @ %.5f SL=%.5f TP=%.5f",
              symbol, direction, lot, price, sl, tp)
    _ensure_stops_attached(symbol, round(sl, digits), round(tp, digits), info)
    return result


def _ensure_stops_attached(symbol: str, sl: float, tp: float, info, attempts: int = 3):
    """Some brokers/symbols (observed here on metals) silently drop the sl/tp sent
    alongside a TRADE_ACTION_DEAL market order: the deal fills but the resulting
    position comes back with sl=0/tp=0, leaving it running with no stop at all --
    this is how a handful of metals trades lost ~10x their intended risk. Verify
    the position actually has its stops and force them via a follow-up
    TRADE_ACTION_SLTP modify if not, retrying a few times before giving up loudly."""
    tolerance = (info.point or 0.00001) * 2
    for attempt in range(attempts):
        positions = open_positions(symbol)
        if not positions:
            # positions_get() can lag a beat right after order_send returns --
            # not finding it on the first look isn't proof it's missing, only a
            # later attempt that still comes up empty is worth escalating on
            if attempt < attempts - 1:
                time.sleep(0.3)
                continue
            log.error("%s: could not find the position just opened to verify its stops "
                       "after %d attempts", symbol, attempts)
            return
        pos = positions[-1]
        if abs(pos.sl - sl) <= tolerance and abs(pos.tp - tp) <= tolerance:
            if attempt > 0:
                log.info("%s: stop-loss/take-profit confirmed attached after retry", symbol)
            return

        log.warning("%s: position opened WITHOUT its stop-loss/take-profit attached "
                     "(position has sl=%.5f tp=%.5f, requested sl=%.5f tp=%.5f) - forcing it",
                     symbol, pos.sl, pos.tp, sl, tp)
        modify_result = mt5.order_send({
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": pos.ticket,
            "sl": sl,
            "tp": tp,
            "magic": config.MAGIC_NUMBER,
        })
        if modify_result is not None and modify_result.retcode == mt5.TRADE_RETCODE_DONE:
            log.info("%s: stop-loss/take-profit forced onto position via TRADE_ACTION_SLTP", symbol)
            return
        log.error("%s: TRADE_ACTION_SLTP failed retcode=%s comment=%s",
                   symbol, getattr(modify_result, "retcode", None), getattr(modify_result, "comment", None))
        time.sleep(0.3)

    log.error("%s: FAILED to attach stop-loss/take-profit after %d attempts - "
               "position is running with NO STOP. Manual intervention required.", symbol, attempts)
