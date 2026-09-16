"""One-off data-prep script: filters finetune_data/claude_decisions.jsonl (every
logged Claude decision cycle) down to only the cycles where every actionable
(BUY/SELL/CLOSE) decision had a good realized outcome, for fine-tuning a local
model on Claude's judgment without also teaching it Claude's mistakes.

A decision's outcome is found by matching it to MT5's own deal history (same
join strategy as memory.py: symbol+direction+time-proximity -> realized P&L of
the round-trip trade), since the JSONL log has no outcome field of its own.
Decisions that were structurally rejected (concentration cap, margin, spread,
etc. -- never actually opened) have no outcome to match and don't count against
the cycle either way. HOLD decisions never count against a cycle.

Not part of the live bot -- run manually: `python -m mt5_bot.build_good_decisions`
from the parent directory.
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import MetaTrader5 as mt5

from . import config, connector

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PATH = os.path.join(_PROJECT_ROOT, config.FINETUNE_LOG_PATH)
OUT_PATH = os.path.join(_PROJECT_ROOT, "finetune_data", "claude_decisions_good.jsonl")

MATCH_WINDOW = timedelta(minutes=5)  # decision -> resulting deal should be near-instant

# MT5 deal.time is reported in local-machine wall-clock terms, not true UTC
# epoch seconds (confirmed empirically: a deal actually placed at 16:15:11
# local machine time comes back from datetime.fromtimestamp(d.time, tz=utc)
# as 16:15:xx too, not 13:15 UTC). claude_decisions.jsonl timestamps are
# genuine UTC (datetime.now(timezone.utc)), so deal times need converting
# before comparison.
_LOCAL_UTC_OFFSET = datetime.now().astimezone().utcoffset()


def _deal_time_utc(raw_ts) -> datetime:
    local_naive = datetime.fromtimestamp(raw_ts, tz=timezone.utc).replace(tzinfo=None)
    return (local_naive - _LOCAL_UTC_OFFSET).replace(tzinfo=timezone.utc)


def _load_examples(path):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return examples


def _load_realized_rounds(since: datetime):
    """All closed round-trip trades (this bot's own, any provider) as a dict
    keyed by (symbol, direction) -> list of {open_time (aware UTC dt), profit},
    sorted by open_time, for nearest-in-time matching."""
    now = datetime.now(timezone.utc)
    # history_deals_get's date_from/date_to are compared against the same
    # local-labeled deal.time values (see _deal_time_utc) -- shift the query
    # bounds by the same offset so a true-UTC range is fully covered.
    deals = mt5.history_deals_get(since + _LOCAL_UTC_OFFSET, now + _LOCAL_UTC_OFFSET + timedelta(minutes=5))
    if not deals:
        return {}

    by_position = defaultdict(list)
    for d in deals:
        if d.magic == config.MAGIC_NUMBER:
            by_position[d.position_id].append(d)

    rounds = defaultdict(list)
    for pos_deals in by_position.values():
        entries = [d for d in pos_deals if d.entry == mt5.DEAL_ENTRY_IN]
        exits = [d for d in pos_deals if d.entry != mt5.DEAL_ENTRY_IN]
        if not entries or not exits:
            continue  # still open -- no realized result yet
        e = entries[0]
        net = sum(d.profit + d.commission + d.swap for d in pos_deals)
        direction = "BUY" if e.type == mt5.ORDER_TYPE_BUY else "SELL"
        rounds[(e.symbol, direction)].append({
            "open_time": _deal_time_utc(e.time),
            "close_time": _deal_time_utc(exits[-1].time),
            "profit": net,
        })
    for r in rounds.values():
        r.sort(key=lambda x: x["open_time"])
    return rounds


def _find_outcome(symbol, direction, decision_ts, rounds, used):
    """Nearest not-yet-used realized round-trip opened within MATCH_WINDOW of
    decision_ts, for BUY/SELL. Returns profit or None if unmatched (rejected
    by a code-level backstop, or still open)."""
    best_i, best_diff = None, None
    for i, r in enumerate(rounds.get((symbol, direction), [])):
        if (symbol, direction, i) in used:
            continue
        diff = abs((r["open_time"] - decision_ts).total_seconds())
        if diff <= MATCH_WINDOW.total_seconds() and (best_diff is None or diff < best_diff):
            best_diff, best_i = diff, i
    if best_i is None:
        return None
    used.add((symbol, direction, best_i))
    return rounds[(symbol, direction)][best_i]["profit"]


def _find_close_outcome(symbol, decision_ts, rounds, used):
    """For a CLOSE decision: the realized profit of whichever direction's
    round-trip actually closed nearest decision_ts (a CLOSE ends a position
    opened earlier, so we search close_time, not open_time, across both
    directions)."""
    best_key, best_i, best_diff = None, None, None
    for direction in ("BUY", "SELL"):
        for i, r in enumerate(rounds.get((symbol, direction), [])):
            if (symbol, direction, i) in used:
                continue
            diff = abs((r["close_time"] - decision_ts).total_seconds())
            if diff <= MATCH_WINDOW.total_seconds() and (best_diff is None or diff < best_diff):
                best_diff, best_i, best_key = diff, i, direction
    if best_i is None:
        return None
    used.add((best_key, best_i))
    return rounds[(symbol, best_key)][best_i]["profit"]


def main():
    connector.connect()
    try:
        examples = _load_examples(SRC_PATH)
        if not examples:
            print(f"No examples found at {SRC_PATH}")
            return

        earliest = min(datetime.fromisoformat(ex["timestamp"]) for ex in examples)
        rounds = _load_realized_rounds(earliest - timedelta(minutes=5))
        used = set()

        kept, dropped, no_actionable = [], [], 0
        evaluable_decisions = 0
        good_decisions = 0
        bad_decisions = 0
        unmatched_decisions = 0

        for ex in examples:
            ts = datetime.fromisoformat(ex["timestamp"])
            decisions = ex.get("decisions", [])
            actionable = [d for d in decisions if d.get("action") in ("BUY", "SELL", "CLOSE")]
            if not actionable:
                no_actionable += 1
                kept.append(ex)  # pure HOLD/empty cycle -- nothing bad happened
                continue

            cycle_ok = True
            for d in actionable:
                symbol = d.get("symbol")
                action = d.get("action")
                if action == "CLOSE":
                    profit = _find_close_outcome(symbol, ts, rounds, used)
                else:
                    profit = _find_outcome(symbol, action, ts, rounds, used)

                if profit is None:
                    unmatched_decisions += 1
                    continue  # rejected by a code-level backstop or still open -- no signal either way

                evaluable_decisions += 1
                if profit >= 0:
                    good_decisions += 1
                else:
                    bad_decisions += 1
                    cycle_ok = False

            (kept if cycle_ok else dropped).append(ex)

        with open(OUT_PATH, "w", encoding="utf-8") as f:
            for ex in kept:
                f.write(json.dumps(ex) + "\n")

        print(f"Source examples:        {len(examples)}")
        print(f"  pure HOLD/no-action:   {no_actionable}")
        print(f"  evaluable decisions:   {evaluable_decisions}  (good={good_decisions}, bad={bad_decisions})")
        print(f"  unmatched decisions:   {unmatched_decisions} (rejected by a backstop or still open -- not counted either way)")
        print(f"Kept (good) cycles:      {len(kept)}")
        print(f"Dropped (bad) cycles:    {len(dropped)}")
        print(f"Written to:              {OUT_PATH}")
    finally:
        connector.disconnect()


if __name__ == "__main__":
    main()
