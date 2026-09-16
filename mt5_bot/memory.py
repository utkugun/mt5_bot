"""Builds a persisted "trade memory" of concrete past decisions paired with
their actual realized outcome, so the LLM can see specific precedent -- not
just aggregate stats (llm_strategy.get_recent_performance) -- before deciding.

The reasoning text for a decision only ever exists in mt5_bot.log (MT5 order
comments are far too short to hold it); the realized P&L only ever exists in
MT5's own deal history (the log never records final outcome). This module
joins the two: parse the log for {symbol, dir, tag, reason}, match each to its
MT5 deal-history round-trip by symbol+direction+volume+time proximity, keep
only closed trades, and persist a small curated set to trade_memory.json.
"""
import json
import logging
import os
import re
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from . import config

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEMORY_PATH = os.path.join(_PROJECT_ROOT, "trade_memory.json")

LOOKBACK_DAYS = 14
MAX_RECENT = 10   # most recent closed trades, for regime-relevant precedent
MAX_EXTREMES = 5  # best/worst by P&L, for the standout lessons recency alone would miss

_DECISION_RE = re.compile(
    r"^(?P<ts>[\d\-]+ [\d:,]+) INFO bot: (?P<symbol>\S+) LLM decision=(?P<action>BUY|SELL|CLOSE) "
    r"reason=(?:(?P<tag>[A-Z_]+) \| )?(?P<reason>.*)$"
)
_OPENED_RE = re.compile(
    r"^(?P<ts>[\d\-]+ [\d:,]+) INFO mt5_bot\.trader: (?P<symbol>\S+): opened (?P<dir>BUY|SELL) "
    r"(?P<vol>[\d.]+) lots @ [\d.]+ SL=[\d.]+ TP=[\d.]+$"
)


def _parse_log_opens(log_path: str, since: datetime) -> list:
    """Ties each executed 'opened' log line back to the LLM decision line (tag +
    reason) that preceded it. Old, pre-setup-catalog log lines have no tag and
    fall back to "UNKNOWN", same convention as get_recent_performance."""
    opens = []
    pending = {}
    with open(log_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m = _DECISION_RE.match(line)
            if m and m.group("action") in ("BUY", "SELL"):
                pending[m.group("symbol")] = (m.group("tag") or "UNKNOWN", m.group("reason").strip())
                continue
            m = _OPENED_RE.match(line)
            if not m:
                continue
            ts_dt = datetime.strptime(m.group("ts"), "%Y-%m-%d %H:%M:%S,%f")
            if ts_dt < since:
                continue
            tag, reason = pending.pop(m.group("symbol"), ("UNKNOWN", ""))
            opens.append({
                "ts": m.group("ts"), "symbol": m.group("symbol"), "dir": m.group("dir"),
                "vol": float(m.group("vol")), "tag": tag, "reason": reason[:140],
            })
    return opens


def _match_to_realized_pnl(opens: list, since: datetime) -> list:
    """Matches each parsed open to its MT5 deal-history round-trip. Only
    positions that have actually closed are returned -- an open position has
    no result yet, so it can't teach anything."""
    now = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(since, now + timedelta(minutes=5))
    if not deals:
        return []

    by_position = defaultdict(list)
    for d in deals:
        if d.magic == config.MAGIC_NUMBER:
            by_position[d.position_id].append(d)

    rounds_by_key = defaultdict(list)
    for pos_deals in by_position.values():
        entries = [d for d in pos_deals if d.entry == mt5.DEAL_ENTRY_IN]
        exits = [d for d in pos_deals if d.entry != mt5.DEAL_ENTRY_IN]
        if not entries or not exits:
            continue  # still open -- excluded, no realized result
        e = entries[0]
        net = sum(d.profit + d.commission + d.swap for d in pos_deals)
        key = (e.symbol, "BUY" if e.type == mt5.ORDER_TYPE_BUY else "SELL", round(e.volume, 2))
        rounds_by_key[key].append({"open_time": e.time, "close_time": exits[-1].time, "profit": net})
    for rounds in rounds_by_key.values():
        rounds.sort(key=lambda r: r["open_time"])

    used = set()
    matched = []
    for o in opens:
        ts = datetime.strptime(o["ts"], "%Y-%m-%d %H:%M:%S,%f")
        key = (o["symbol"], o["dir"], round(o["vol"], 2))
        best_i, best_diff = None, None
        for i, r in enumerate(rounds_by_key.get(key, [])):
            if (key, i) in used:
                continue
            rt = datetime.fromtimestamp(r["open_time"], tz=timezone.utc).replace(tzinfo=None)
            diff = abs((rt - ts).total_seconds())
            if diff < 6 * 3600 and (best_diff is None or diff < best_diff):
                best_diff, best_i = diff, i
        if best_i is None:
            continue
        used.add((key, best_i))
        r = rounds_by_key[key][best_i]
        matched.append({**o, "profit": round(r["profit"], 2)})
    return matched


def _empty_memory() -> dict:
    return {"lookback_days": LOOKBACK_DAYS, "closed_trades_considered": 0, "examples": []}


def _load_existing() -> dict:
    try:
        with open(MEMORY_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return _empty_memory()


def refresh_trade_memory(days: int = LOOKBACK_DAYS) -> dict:
    """Rebuilds trade_memory.json from mt5_bot.log + MT5's realized deal
    history and returns it. On any failure (log missing/rotated, MT5 not
    connected, unexpected format), logs a warning and falls back to whatever
    was last persisted rather than breaking the decision cycle."""
    log_path = os.path.join(_PROJECT_ROOT, config.LOG_FILE)
    since_utc = datetime.now(timezone.utc) - timedelta(days=days)
    # log timestamps are naive local machine time (logging's default), not UTC
    since_local = datetime.now() - timedelta(days=days)

    try:
        opens = _parse_log_opens(log_path, since_local)
        matched = _match_to_realized_pnl(opens, since_utc)
    except Exception:
        log.exception("trade memory: failed to rebuild from %s, keeping previous memory", log_path)
        return _load_existing()

    matched.sort(key=lambda r: r["ts"])
    by_profit = sorted(matched, key=lambda r: r["profit"])

    def to_entry(r, note):
        return {"when": r["ts"][:16], "symbol": r["symbol"], "dir": r["dir"], "tag": r["tag"],
                "reason": r["reason"], "profit": r["profit"], "note": note}

    seen = set()
    examples = []
    for r in reversed(matched[-MAX_RECENT:]):  # most recent first
        key = (r["ts"], r["symbol"])
        if key in seen:
            continue
        seen.add(key)
        examples.append(to_entry(r, "recent"))
    for r in by_profit[:MAX_EXTREMES]:  # biggest losses
        key = (r["ts"], r["symbol"])
        if key in seen or r["profit"] >= 0:
            continue
        seen.add(key)
        examples.append(to_entry(r, "worst"))
    for r in reversed(by_profit[-MAX_EXTREMES:]):  # biggest wins
        key = (r["ts"], r["symbol"])
        if key in seen or r["profit"] <= 0:
            continue
        seen.add(key)
        examples.append(to_entry(r, "best"))

    memory = {"lookback_days": days, "closed_trades_considered": len(matched), "examples": examples}

    try:
        with open(MEMORY_PATH, "w", encoding="utf-8") as f:
            json.dump(memory, f, indent=2)
    except OSError:
        log.warning("trade memory: could not write %s", MEMORY_PATH)

    return memory
