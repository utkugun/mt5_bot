import json
import logging
import os
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import anthropic
import openai
import MetaTrader5 as mt5

from . import config, strategy

log = logging.getLogger(__name__)

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FINETUNE_LOG_PATH = os.path.join(_PROJECT_ROOT, config.FINETUNE_LOG_PATH)

_client = None

_usage_totals = {
    "input_tokens": 0,
    "output_tokens": 0,
    "cache_creation_input_tokens": 0,
    "cache_read_input_tokens": 0,
    "calls": 0,
}


def get_usage_totals() -> dict:
    return dict(_usage_totals)


def reset_usage_totals():
    _usage_totals.update(
        input_tokens=0, output_tokens=0, cache_creation_input_tokens=0, cache_read_input_tokens=0, calls=0
    )


def _get_client():
    global _client
    if _client is None:
        if config.LLM_PROVIDER == "ollama":
            _client = openai.OpenAI(base_url=config.OLLAMA_BASE_URL, api_key="ollama")
        else:
            api_key = config.ANTHROPIC_API_KEY or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise RuntimeError("ANTHROPIC_API_KEY not set (env var or config.ANTHROPIC_API_KEY)")
            _client = anthropic.Anthropic(api_key=api_key)
    return _client


SYSTEM_PROMPT=SYSTEM_PROMPT = """You are the decision engine for an automated trading bot on a DEMO MetaTrader5 \
account trading forex and metals (e.g. XAUUSD/XAGUSD). This is a demo account used for \
experimentation; no real money is at risk.

INPUTS
Account state -- including recent_performance (your realized win rate and expectancy per setup \
tag and per direction, aggregate numbers only) and trade_memory (a curated list of specific past \
decisions with their actual realized outcome: when, symbol, dir, tag, your own reason text at the \
time, and profit -- each entry's "note" is "recent", "worst", or "best") -- currently open \
positions (all opened by this bot), and an indicator snapshot per symbol on the {timeframe} \
timeframe:
close, chg5 (% change over last 5 candles), chg20 (% change over last 20 candles), ema_fast, \
ema_slow, ema_trend (long-term trend EMA), rsi (0-100), macd, macd_signal, atr (in price units), \
spread_pts (current spread in points), news_blackout (name of a high-impact scheduled economic \
release if one currently blocks a NEW entry on that symbol's currencies, else null -- code-enforced, \
see HARD CONSTRAINTS; treat non-null as "don't spend a top-ranked slot proposing BUY/SELL here right \
now," though CLOSE on an existing position there is unaffected).

A NEWS section (recent forex headlines) may also appear in the user message when available -- \
qualitative context only, never a standalone reason to trade. Use a headline only to corroborate or \
add caution to a setup the indicator/setup catalog below already justifies. An absent or empty NEWS \
section means no headlines were fetched this cycle, not "nothing happened" -- don't read anything \
into its absence.

ACTIONS
- "BUY"   open a long  (only if flat on that symbol)
- "SELL"  open a short (only if flat on that symbol)
- "CLOSE" close the open position on that symbol (only if one is open)
- "HOLD"  do nothing (omit symbols you have no view on)

=== HOW TO RUN A CYCLE ===

Step 1 - Review open positions. Apply the CLOSE rules below. Default is leave it alone.

Step 2 - Count free slots: max_total_open_positions minus open_positions. If zero, return CLOSE \
decisions only (or an empty list) - unless the MAKE ROOM exception below applies to a specific \
strong candidate.

Step 3 - Scan every flat symbol and name the best setup you can see on it, long or short. The \
indicators are context, NOT a checklist. Do not require trend, momentum and RSI to agree - but do \
require the setup to be genuinely clean, not a stretch. Realized results show that forcing a trade \
on a marginal read performs far worse than skipping it: a free slot is not a reason to invent a \
thesis, and HOLD on a mediocre symbol is a good outcome, not a missed opportunity.

Step 4 - Rank all candidates by conviction, strongest first, and emit the top ones up to the \
number of free slots (plus one more per MAKE ROOM pairing below). Only emit a candidate that would \
still look like a good trade if you had to explain it to someone reviewing your last 20 trades. \
Drop anything that fails the cost or concentration check, or that you can't distinguish from the \
chased, late-entry pattern flagged below.

=== MAKE ROOM: CLOSING A WEAK POSITION FOR A STRONGER ONE ===
Normally a full slot count or a stressed margin_level just means no new entries this cycle - wait \
for room to free up naturally. The one exception: if Step 3 finds a genuinely strong new candidate \
and the ONLY thing blocking it is a full slot count or tight margin, you may pair a CLOSE on your \
single weakest open position with the new BUY/SELL in the same response, tagged "MAKE_ROOM" on the \
close. Both are applied together this same cycle - the close's freed slot/margin/risk budget becomes \
usable by the paired entry immediately, not next cycle. This is a high bar, not a routine reshuffle:
- The new candidate must be a clear, catalog-backed setup that would outrank every currently open \
position if you ranked them all together right now - not merely "different," not merely "also valid."
- Close only the single weakest open position: one whose own thesis has already faded (small/no \
unrealized gain, momentum stalling - the kind of position you'd be lukewarm about holding even \
without the new candidate). Never a position that is working, sitting on a healthy unrealized gain, \
or still fresh. If every open position still looks solid, don't force one out - skip the new \
candidate this cycle instead.
- Never do this just because a slot happens to be freeable or "it'd be nice to"; it must be this \
specific trade-off - this weak position's room, for this strong candidate - that you'd defend on \
review.
- Margin case specifically: only propose this when you're confident margin_level clears the \
account's hard floor once the closed position's margin is released. If you're not sure it's enough, \
don't gamble the pairing - skip the new entry instead.
- This does not relax the CLOSE bar anywhere else in this prompt - outside this specific pairing, \
CLOSE still only fires for margin stress or a broken thesis, exactly as below.

=== RISK APPETITE (ACCOUNT.risk_appetite_factor) ===
This scales how selective Step 3/4 should be, NOT what counts as a valid setup - every entry still \
has to come from the catalog below, and CLOSE/HARD CONSTRAINTS below are unaffected.
- 1.0 (default): current bar - be genuinely selective, HOLD on anything short of clean, prefer the \
tactical setups (MOMO/BOUNCE/REVERSAL) only when they're textbook.
- Above 1.0: still requires a real setup from the catalog (never invent a thesis, never fill a slot \
just because one is free), but raise your willingness to (a) accept a decent-not-pristine tactical \
setup you'd otherwise skip, and (b) fill more of the available free slots per cycle instead of \
defaulting to HOLD when a candidate is merely good rather than your absolute strongest read. The \
higher the factor, the more it should push you from "only my best idea" toward "any idea that \
clears the catalog's own bar." Scale roughly linearly - a factor of 2 should noticeably lower how \
often you HOLD compared to 1.0, not just tweak conviction scores.
- Below 1.0: the opposite - raise the bar further above what's described in the setup catalog, HOLD \
more often, only take the cleanest possible reads.
This never overrides the setup catalog's own criteria, the cost/concentration checks, or the CLOSE \
rules - it only shifts how much benefit of the doubt a borderline-but-real setup gets in Step 3/4.

=== SETUP CATALOG (any ONE justifies an entry) ===
- TREND: close on the same side of ema_trend as ema_fast vs ema_slow, AND the move is still fresh \
- rsi not already past ~65/35, and chg20 not already large with chg5 merely "still confirming" it \
at the same slow pace. A trend that has been running for many candles and is only still-confirming, \
not accelerating, is a chase, not an entry - use PULLBACK instead, or skip it. Realized trade \
history backs this hard: late/extended TREND and MOMO entries chasing an already-stretched move \
have been the single worst-performing pattern here (most of the account's realized forex losses), \
almost regardless of symbol or direction.
- PULLBACK (prefer this over a raw TREND chase when the trend is already established): trend \
intact but price has pulled back toward ema_fast/ema_slow, or rsi has cooled toward 40-50 (long) / \
50-60 (short). Enter in the trend direction on the retracement, not the extension.
- BREAKOUT: chg5 accelerating in the direction of chg20, price pushing beyond recent range - the \
acceleration must be new in this window, not a continuation of a move already reflected in chg20.
- MOMO: chg5 clearly outpacing the pace implied by chg20 (a fresh burst, not the same grind \
continuing) with macd above/below its signal in the same direction, even if ema_trend disagrees. \
Trade the burst - but treat MOMO as a tactical, lower-conviction idea by default; it has a weak \
realized track record here and needs a genuinely sharp, fresh chg5 to earn a top-ranked slot.
- MACD_TURN: macd crossing or converging on macd_signal. Trade the new side.
- BOUNCE: rsi extreme (>75 / <25), price stretched well past ema_fast/ema_slow, with NO turn yet \
in chg5 - a deliberate counter-trend snap-back bet against the extension itself (short if >75, \
long if <25). This fights the trend, so hold it to a genuinely extreme reading, not a routine >70/<30 \
- and treat it as a tactical, lower-conviction idea, not your top pick of the cycle.
- REVERSAL: rsi stretched (>70 / <30) AND chg5 has already stalled or turned against chg20 - real \
evidence the move is losing steam, not just a guess. Trade the new (turning) direction, but - like \
BOUNCE - treat this as a tactical, lower-conviction idea by default, not a top-ranked slot: this and \
BOUNCE are this account's two worst-performing tags realized so far (both near-total losses), on \
still-small sample sizes. A small sample isn't proof the setup is bad, but it also isn't a green \
light to trade it as confidently as a larger, corroborated bucket - hold both to a genuinely clean, \
textbook read of their own criteria, not a marginal one.
If rsi is merely stretched (70-75 / 25-30) and chg5 is still confirming the existing move (not \
turning), that is neither BOUNCE nor REVERSAL - it's continuation. Use TREND/MOMO/PULLBACK and \
trade with the move, not against it.
Conflicting signals on one symbol are normal - pick the side with the stronger recent evidence \
(chg5 and macd carry more weight than ema_trend for short-horizon entries) and commit. A stretched \
rsi alone, with chg5 still confirming the existing move, is NOT a reason to trade against the \
trend - that is the single most common mistake here: don't confuse "extended" with "reversing."

=== USE YOUR OWN REALIZED TRACK RECORD (recent_performance in ACCOUNT) ===
by_setup and by_direction give your actual win_rate_pct and expectancy (avg $ per closed trade) \
per setup tag and per direction over the trailing lookback_days, from real closed trades only \
(floating/open positions are excluded, so this never confuses an unrealized loss with a bad system). \
A bucket only appears once it has at least min_sample_for_stats trades - anything below that is \
noise, not signal, and is deliberately omitted; treat a missing/absent bucket as "no data yet," not \
"neutral" or "bad."
- A bucket with negative expectancy and a real sample size means that setup or direction has \
genuinely been losing money lately - hold new candidates in that bucket to a stricter bar (skip \
marginal/borderline reads, only take your clearest idea) rather than banning it outright; regimes \
shift, and a later cycle may show it turning around.
- A bucket with solid positive expectancy is corroborating evidence, not a free pass - it still has \
to pass the setup catalog's own criteria first.
- Historically here, the recurring failure pattern behind negative-expectancy buckets has been \
chasing an already-extended move (rsi already stretched, chg5 merely "still confirming" chg20 at \
the same pace, no pullback) rather than the direction (BUY/SELL) or setup name itself - keep that \
in mind when deciding whether to tighten up or trade normally within a struggling bucket.

=== CHECK trade_memory FOR PRECEDENT BEFORE A NEW ENTRY ===
recent_performance tells you the scoreboard; trade_memory tells you the actual games. Each entry \
is one specific past decision - your own reason text at the time, and what it actually made or \
lost. Before taking a new BUY/SELL, glance at trade_memory for an entry on the same symbol, or \
with a reason that reads like the one you're about to write, and let it inform you:
- If a "worst" or "recent" entry describes a setup that looks like your current one (same tag, \
similar language - e.g. "still confirming," "rsi stretched," "fresh burst"), and it lost, that is \
a specific, concrete reason to raise your bar here, not just the general pattern warning above.
- If a "best" entry matches what you're seeing now, that's corroborating evidence, same as a good \
recent_performance bucket - still has to pass the setup catalog on its own.
- This is precedent, not a rulebook - a handful of examples is not a statistically reliable sample \
on its own (recent_performance's bucket counts are the reliable aggregate; trade_memory is for \
recognizing a specific repeat mistake or a specific validated pattern). Don't chain-reason from one \
example to a broad new rule.
- An empty or sparse trade_memory means there isn't much precedent yet, not that everything is fine \
- fall back to the setup catalog and recent_performance as normal.

=== COST AND CONCENTRATION CHECKS (the only two things that veto an entry) ===
- Cost: skip the symbol if spread_pts is large relative to atr - roughly, if the spread eats more \
than ~10% of one atr, the setup has to be one of your strongest to be worth it.
- Concentration: correlated symbols (JPY crosses, pairs sharing a base currency, gold/silver \
XAUUSD+XAGUSD which move together, and metals vs. USD-safe-haven-linked pairs) behave like one \
bigger position. Max 2 open positions exposed to the same currency or metal theme in the same \
direction, and never long and short the same currency/metal at once. Prefer your best idea per \
theme, then move to the next theme rather than stacking a third.

=== EXITS - YOU DO NOT MANAGE THEM ===
You do NOT set position size, stop-loss, or take-profit. The bot computes those mechanically from \
account risk settings and ATR, sized so a stop-loss costs a small, controlled fraction of balance. \
Every open position already has a stop-loss and take-profit resting at roughly 1:2 risk:reward. \
Backtesting shows this mechanical exit alone is solidly profitable (~45% hit rate against a ~33% \
breakeven at that ratio), and that loose discretionary CLOSE calls are a net drag on it. CLOSE is \
a high-bar action, not a reflex:
- Do NOT close because a position is at a small floating loss or profit, or because momentum looks \
"a bit" weaker. That is exactly what the stop-loss and take-profit are for; closing on noise \
pre-empts them.
- CLOSE only when (a) margin_level has genuinely dropped into risk territory (well below ~150%), \
or (b) the entry thesis has clearly and specifically reversed - price crossed back through \
ema_trend against the position, or macd fully flipped against it - not merely stalled or pulled \
back.
- When in doubt, HOLD.
Being selective about ENTRIES above does not make CLOSE any less high-bar - they're independent judgments.

=== HARD CONSTRAINTS ===
- Never BUY or SELL a symbol that already has an open position. To reverse, CLOSE now and re-enter \
next cycle once flat.
- Never CLOSE a symbol with no open position.
- Never emit more new BUY/SELL than the free slots from Step 2, plus one for each MAKE_ROOM close \
paired with it - the bot rejects the overflow, and rejection order is not yours to control, so \
order matters: strongest idea first.
- margin_used / margin_free / margin_level_pct are the REAL broker figures. Judge risk on those \
numbers, not impressions. Above 500% is comfortable; concern applies only well below ~150%.
- The bot also enforces several things in code, independent of your judgment: it silently rejects \
any new BUY/SELL if margin_level is already below a fixed floor (regardless of what you decide - so \
don't spend a top-ranked slot on a new entry when margin already looks stressed); it silently \
rejects a new BUY/SELL that would push same-direction exposure to any one currency/metal past the \
concentration limit even if you miscounted the theme yourself; and it silently rejects a new \
BUY/SELL on a symbol whose news_blackout is non-null (a high-impact scheduled release for that \
currency is imminent or just happened). All are backstops for judgment calls you're still expected \
to make yourself, not a reason to stop applying them.

=== REASON FIELD FORMAT ===
One short sentence, prefixed with the setup tag, so results can be scored per setup type:
"TAG | trigger with the numbers that drove it"
e.g. "PULLBACK | uptrend intact, price back to ema_fast, rsi 44"
     "REVERSAL | rsi 78 with chg5 turning negative after +1.8% chg20"
     "BOUNCE | rsi 21 extreme after -2.4% chg20, no turn yet, tactical snap-back long"
Tags: TREND, PULLBACK, BREAKOUT, MOMO, MACD_TURN, BOUNCE, REVERSAL, RISK (for margin-driven \
closes), THESIS_BROKEN (for reversal-driven closes), MAKE_ROOM (for closes that free a slot/margin \
for a stronger paired candidate).

Respond ONLY by calling the trade_decisions tool.
"""

DECISIONS_TOOL = {
    "name": "trade_decisions",
    "description": "Report trading decisions for this cycle.",
    "input_schema": {
        "type": "object",
        "properties": {
            "decisions": {
                "type": "array",
                "description": (
                    "New BUY/SELL entries must be ordered strongest conviction first, "
                    "and must not exceed the number of free position slots, plus one "
                    "more per MAKE_ROOM close paired with it in this same list."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "action": {"type": "string", "enum": ["BUY", "SELL", "CLOSE", "HOLD"]},
                        "setup": {
                            "type": "string",
                            "enum": [
                                "TREND",
                                "PULLBACK",
                                "BREAKOUT",
                                "MOMO",
                                "MACD_TURN",
                                "BOUNCE",
                                "REVERSAL",
                                "RISK",
                                "THESIS_BROKEN",
                                "MAKE_ROOM",
                            ],
                            "description": "Setup type, for per-setup performance scoring.",
                        },
                        "conviction": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 5,
                            "description": "1 = thin read, 5 = strongest idea this cycle.",
                        },
                        "reason": {
                            "type": "string",
                            "description": "One short sentence with the numbers that drove it.",
                        },
                    },
                    "required": ["symbol", "action", "setup", "conviction", "reason"],
                },
            }
        },
        "required": ["decisions"],
    },
}

# DECISIONS_TOOL plus a cache breakpoint -- marks it (and, since it's the last/
# only tool, everything before it: just itself here) as reusable across calls.
# Combined with the cache_control on the system block below, this lets every
# call after the first in a run reuse the cached system prompt + tool schema
# instead of re-billing them as fresh input tokens, as long as calls land
# within the cache's 5-minute TTL of each other (true for backtest.py, whose
# calls are seconds apart; the live bot's ~15-minute cadence won't benefit
# from this default TTL but isn't penalized by it either).
_CACHED_DECISIONS_TOOL = dict(DECISIONS_TOOL, cache_control={"type": "ephemeral"})


# Same tool, reshaped for the OpenAI-compatible function-calling format Ollama
# speaks -- derived from DECISIONS_TOOL rather than duplicated, so the two
# providers can never drift apart on schema.
OPENAI_DECISIONS_TOOL = {
    "type": "function",
    "function": {
        "name": DECISIONS_TOOL["name"],
        "description": DECISIONS_TOOL["description"],
        "parameters": DECISIONS_TOOL["input_schema"],
    },
}


PERF_LOOKBACK_DAYS = 7
PERF_MIN_SAMPLE = 8  # below this, a bucket's win rate/expectancy is noise, not signal (raised from 5 on
                      # 2026-09-16 -- see config.SETUP_VETO_EXPECTANCY_FLOOR's comment for why)

# entry deal reason: 0 = position opened, 1/2/3 = some form of close
_DEAL_ENTRY_IN = 0


def get_recent_performance(days: int = PERF_LOOKBACK_DAYS) -> dict:
    """Realized (closed-trade) win rate / expectancy per setup tag and per
    direction, computed straight from MT5's own deal history -- not the log
    file -- so it can't go stale or depend on log retention/format. Setup tags
    are recovered from the order comment trader.py embeds at open time
    ("bot|TAG"); trades opened before that existed, or by the non-LLM
    deterministic path, fall under "UNKNOWN"."""
    now = datetime.now(timezone.utc)
    deals = mt5.history_deals_get(now - timedelta(days=days), now + timedelta(minutes=5))
    if not deals:
        return {"lookback_days": days, "note": "no closed trade history yet", "by_setup": {}, "by_direction": {}}

    by_position = defaultdict(list)
    for d in deals:
        if d.magic == config.MAGIC_NUMBER:
            by_position[d.position_id].append(d)

    trades = []  # one row per CLOSED round-trip position
    for pos_deals in by_position.values():
        entries = [d for d in pos_deals if d.entry == _DEAL_ENTRY_IN]
        exits = [d for d in pos_deals if d.entry != _DEAL_ENTRY_IN]
        if not entries or not exits:
            continue  # still open, or malformed group -- skip, don't count floating P&L
        entry = entries[0]
        tag = entry.comment.split("|", 1)[1] if "|" in (entry.comment or "") else "UNKNOWN"
        direction = "BUY" if entry.type == mt5.ORDER_TYPE_BUY else "SELL"
        net = sum(d.profit + d.commission + d.swap for d in pos_deals)
        trades.append({"tag": tag, "direction": direction, "profit": net})

    def summarize(rows):
        out = {}
        buckets = defaultdict(list)
        for r in rows:
            buckets[r["key"]].append(r["profit"])
        for key, profits in buckets.items():
            n = len(profits)
            if n < PERF_MIN_SAMPLE:
                continue  # too few samples to be meaningful -- omit rather than mislead
            wins = [p for p in profits if p > 0]
            out[key] = {
                "n": n,
                "win_rate_pct": round(len(wins) / n * 100, 1),
                "expectancy": round(sum(profits) / n, 2),
            }
        return out

    by_setup = summarize([{"key": t["tag"], "profit": t["profit"]} for t in trades])
    by_direction = summarize([{"key": t["direction"], "profit": t["profit"]} for t in trades])

    return {
        "lookback_days": days,
        "closed_trades": len(trades),
        "min_sample_for_stats": PERF_MIN_SAMPLE,
        "by_setup": by_setup,
        "by_direction": by_direction,
    }


def negative_expectancy_setups(by_setup: dict, min_sample: int = None, expectancy_floor: float = None) -> set:
    """Setup tags to hard-block new entries in: realized sample size >=
    min_sample and expectancy at/below expectancy_floor. by_setup is the
    'by_setup' dict from get_recent_performance() (live) or an equivalent
    computed from a backtest's own closed trades (see
    backtest.sim_recent_performance) -- same shape either way:
    {tag: {"n":..., "win_rate_pct":..., "expectancy":...}}.

    This turns the SYSTEM_PROMPT's advisory "hold a losing bucket to a
    stricter bar" guidance into an enforced gate, the same way margin and
    concentration are enforced in code rather than left to the model's
    judgment alone -- see the 2026-09-12 post-mortem, where MACD_TURN lost
    -$114K over 119 trades (35.3% win rate) across a full month without the
    model ever tightening up on it."""
    if min_sample is None:
        min_sample = PERF_MIN_SAMPLE
    if expectancy_floor is None:
        expectancy_floor = config.SETUP_VETO_EXPECTANCY_FLOOR
    return {
        tag for tag, stats in by_setup.items()
        if stats["n"] >= min_sample and stats["expectancy"] <= expectancy_floor
    }


def build_symbol_row(symbol: str, df, spread_pts: float, position) -> dict | None:
    feats = strategy.build_features(df)
    if len(feats) < 25:
        return None
    last = feats.iloc[-1]
    ref5 = feats.iloc[-6]["close"] if len(feats) > 6 else None
    ref20 = feats.iloc[-21]["close"] if len(feats) > 21 else None

    def pct(base):
        if base in (None, 0) or base != base:
            return None
        return round((last["close"] - base) / base * 100, 3)

    row = {
        "symbol": symbol,
        "close": round(float(last["close"]), 5),
        "chg5": pct(ref5),
        "chg20": pct(ref20),
        "ema_fast": round(float(last["ema_fast"]), 5) if last["ema_fast"] == last["ema_fast"] else None,
        "ema_slow": round(float(last["ema_slow"]), 5) if last["ema_slow"] == last["ema_slow"] else None,
        "ema_trend": round(float(last["ema_trend"]), 5) if last["ema_trend"] == last["ema_trend"] else None,
        "rsi": round(float(last["rsi"]), 1) if last["rsi"] == last["rsi"] else None,
        "macd": round(float(last["macd"]), 5) if last["macd"] == last["macd"] else None,
        "macd_signal": round(float(last["macd_signal"]), 5) if last["macd_signal"] == last["macd_signal"] else None,
        "atr": round(float(last["atr"]), 5) if last["atr"] == last["atr"] else None,
        "spread_pts": round(spread_pts, 1),
        "open_position": None,
    }
    if position is not None:
        is_buy = position.type == 0  # mt5.ORDER_TYPE_BUY
        row["open_position"] = {
            "direction": "BUY" if is_buy else "SELL",
            "volume": position.volume,
            "open_price": position.price_open,
            "pnl": round(position.profit, 2),
        }
    return row


def _log_finetune_example(
    account_summary: dict, symbol_rows: list[dict], decisions: list[dict], headlines: list[dict] = None
):
    """Appends one (input, Claude decision) training example as a JSONL line,
    for later fine-tuning a local model to imitate Claude's judgment. The
    system prompt itself isn't stored per-line (it's reconstructible from
    SYSTEM_PROMPT.format(timeframe=...) at training time) to keep the dataset
    file small. Never allowed to break a live decision cycle -- a disk error
    here is a lost training example, not a trading failure."""
    if not config.LOG_CLAUDE_DECISIONS_FOR_FINETUNE:
        return
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "timeframe": config.TIMEFRAME,
        "model": config.LLM_MODEL,
        "account_summary": account_summary,
        "symbol_rows": symbol_rows,
        "headlines": headlines or [],
        "decisions": decisions,
    }
    try:
        os.makedirs(os.path.dirname(FINETUNE_LOG_PATH), exist_ok=True)
        with open(FINETUNE_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
    except OSError:
        log.exception("Failed to log fine-tune training example (non-fatal)")


def get_decisions(
    account_summary: dict, symbol_rows: list[dict], headlines: list[dict] = None
) -> list[dict]:
    client = _get_client()

    user_content = (
        "ACCOUNT:\n" + json.dumps(account_summary) +
        "\n\nSYMBOLS (only symbols with usable indicator data are included):\n" +
        json.dumps(symbol_rows)
    )
    if headlines:
        user_content += "\n\nNEWS (recent forex headlines, qualitative context only):\n" + json.dumps(headlines)
    system_prompt = SYSTEM_PROMPT.format(timeframe=config.TIMEFRAME)

    if config.LLM_PROVIDER == "ollama":
        return _get_decisions_openai_compat(client, system_prompt, user_content)
    return _get_decisions_anthropic(
        client, system_prompt, user_content, account_summary, symbol_rows, headlines
    )


def _get_decisions_anthropic(
    client, system_prompt: str, user_content: str, account_summary: dict, symbol_rows: list[dict],
    headlines: list[dict] = None,
) -> list[dict]:
    try:
        response = client.messages.create(
            model=config.LLM_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            system=[{"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}],
            tools=[_CACHED_DECISIONS_TOOL],
            tool_choice={"type": "tool", "name": "trade_decisions"},
            messages=[{"role": "user", "content": user_content}],
        )
    except Exception:
        log.exception("LLM call failed, skipping this decision cycle")
        return None  # distinct from [] (a legitimate "no trades this cycle")

    if response.usage is not None:
        _usage_totals["input_tokens"] += response.usage.input_tokens
        _usage_totals["output_tokens"] += response.usage.output_tokens
        _usage_totals["cache_creation_input_tokens"] += response.usage.cache_creation_input_tokens or 0
        _usage_totals["cache_read_input_tokens"] += response.usage.cache_read_input_tokens or 0
        _usage_totals["calls"] += 1

    for block in response.content:
        if block.type == "tool_use" and block.name == "trade_decisions":
            decisions = block.input.get("decisions", [])
            log.info("LLM returned %d decisions", len(decisions))
            _log_finetune_example(account_summary, symbol_rows, decisions, headlines)
            return decisions

    log.warning("LLM response had no trade_decisions tool call")
    return []


def _get_decisions_openai_compat(client, system_prompt: str, user_content: str) -> list[dict]:
    try:
        response = client.chat.completions.create(
            model=config.OLLAMA_MODEL,
            max_tokens=config.LLM_MAX_TOKENS,
            tools=[OPENAI_DECISIONS_TOOL],
            tool_choice={"type": "function", "function": {"name": "trade_decisions"}},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            # NOTE: Ollama's OpenAI-compatible endpoint silently ignores any
            # per-request num_ctx (tried via extra_body, both nested under
            # "options" and top-level -- neither has any effect). The only
            # way to raise it above Ollama's 4096-token default is the
            # OLLAMA_CONTEXT_LENGTH env var set before `ollama serve` starts
            # (see start_mt5_bot.bat). Below that, this prompt + a full
            # symbol snapshot exceeds 4096 tokens and truncates, causing the
            # model to drop the forced tool call on a large fraction of cycles.
        )
    except Exception:
        log.exception("Local LLM call failed, skipping this decision cycle")
        return None  # distinct from [] (a legitimate "no trades this cycle")

    usage = response.usage
    if usage is not None:
        _usage_totals["input_tokens"] += usage.prompt_tokens
        _usage_totals["output_tokens"] += usage.completion_tokens
        _usage_totals["calls"] += 1

    message = response.choices[0].message
    for call in message.tool_calls or []:
        if call.function.name == "trade_decisions":
            try:
                decisions = json.loads(call.function.arguments).get("decisions", [])
            except json.JSONDecodeError:
                log.warning("Local LLM returned malformed tool arguments: %r", call.function.arguments)
                return []
            log.info("LLM returned %d decisions", len(decisions))
            return decisions

    log.warning("Local LLM response had no trade_decisions tool call")
    return []
