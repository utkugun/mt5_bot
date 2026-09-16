"""Central configuration for the MT5 trading bot. Edit values here, no need
to touch the rest of the code."""

# --- MT5 connection ---
# Leave these None to attach to an already-running, already-logged-in MT5
# terminal (simplest: just open MT5, log into your demo account by hand,
# and leave it running). Fill them in only if you want the bot to log in
# itself instead.
MT5_LOGIN = None          # int, e.g. 12345678
MT5_PASSWORD = None       # str
MT5_SERVER = None         # str, e.g. "MetaQuotes-Demo"
MT5_PATH = None           # str, path to terminal64.exe if it can't auto-detect

# Real credentials live in local_settings.py (git-ignored, see
# local_settings.example.py) so they never end up in version control.
try:
    from .local_settings import MT5_LOGIN, MT5_PASSWORD, MT5_SERVER, MT5_PATH  # noqa: F401,F811
except ImportError:
    pass

# --- Symbols ---
# "auto" (SYMBOL_GROUP_FILTER match) used to pull in all 136 broker symbols,
# including thin EM/exotic crosses (USDTRY, USDBRL, USDARS, USDNGN, USDIDR,
# EURRUB/EURRUR, SEKJPY, MXNJPY, ...). Those showed up in mt5_bot.log as
# "No prices"/"No money"/"Invalid stops" order failures, and their tiny/erratic
# ATR values are exactly what drove calc_lot_size() to oversized lots (e.g.
# SEKJPY 88.20 lots, NZDCAD 18.91 lots on a ~100k account -- see risk.py).
# Pinned down to a curated, liquid basket instead: majors, the most liquid
# JPY/EUR/GBP crosses, and metals. Revert to "auto" (+ SYMBOL_GROUP_FILTER)
# only if you've separately vetted the broker's exotic-symbol liquidity/spread.
SYMBOLS = [
    "EURUSD", "GBPUSD", "USDJPY", "USDCHF", "USDCAD", "AUDUSD", "NZDUSD",
    "EURGBP", "EURJPY", "GBPJPY", "EURCHF", "AUDJPY", "CADJPY", "NZDJPY",
    "XAUUSD", "XAGUSD",
]
SYMBOL_GROUP_FILTER = ["Forex", "Metals"]  # only used if SYMBOLS == "auto"

# --- Timeframe & polling ---
TIMEFRAME = "M15"          # candle timeframe the strategy runs on
POLL_SECONDS = 30          # how often the loop wakes up to check for a newly closed candle

# --- Strategy: ensemble of EMA-crossover trend, long-term trend filter,
# MACD momentum confirmation, and RSI overbought/oversold filter. A trade
# only fires when all of them agree. ---
# EMA_FAST/EMA_SLOW, MACD_SIGNAL and ATR_PERIOD widened from 12/26/9/14 on
# 2026-09-13 -- a 3-way local-LLM (qwen2.5:7b-instruct) backtest sweep over
# the same 7-day window (2026-09-06..13) found the slower/smoother reading
# these produce, paired with the wider SL/TP below, cut that week's loss from
# -26.0% (original values) to -2.8%, raising profit factor 0.53 -> 0.97. See
# backtest_results/sweep/summary.json. Single-week result -- re-validate
# against other weeks before trusting this as a durable edge.
EMA_FAST = 16
EMA_SLOW = 34
EMA_TREND = 200
RSI_PERIOD = 14
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 11
ATR_PERIOD = 20

# --- Risk management ---
RISK_PCT_PER_TRADE = 0.015  # fraction of account BALANCE risked per trade (lowered from 0.10 -- at
                             # 10% risk x up to 5 concurrent positions, margin_level was crashing to
                             # 37-100% in backtesting, forcing the LLM to panic-close positions on
                             # margin stress before SL/TP ever got a chance to work)
SL_ATR_MULT = 2.0           # stop-loss distance = ATR * this (widened from 1.5 -- 2026-09-13 sweep,
                             # see EMA_FAST comment above)
TP_ATR_MULT = 3.5           # take-profit distance = ATR * this (1:1.75 reward:risk; widened from 2.0,
                             # same sweep)
MAX_OPEN_POSITIONS_PER_SYMBOL = 1
MAX_TOTAL_OPEN_POSITIONS = 6     # hard cap on concurrent bot-owned positions across ALL symbols combined
                                 # (lowered from 10 -- same 2026-09-13 sweep: fewer, more selective
                                 # concurrent positions outperformed over the tested week)
                                 # (correlated pairs like JPY-crosses can otherwise all fire in one cycle
                                 # and act as one oversized position instead of several diversified ones)
MAX_SPREAD_POINTS = 40      # skip new entries if current spread is wider than this, in points
MAGIC_NUMBER = 20260904     # tags this bot's own orders/positions
MARGIN_USAGE_CAP = 0.20     # a single new trade may use at most this fraction of CURRENT FREE
                             # margin (lowered from 0.85 -- with few/no positions open, free
                             # margin is ~all of balance, so 0.85 let one trade alone eat 85% of
                             # the account's margin capacity before any other check fired)
MAX_MARGIN_PCT_OF_BALANCE = 0.15  # a single new trade's required margin may also never exceed
                                   # this fraction of account BALANCE, independent of how much
                                   # margin happens to be free right now. This is the fix for the
                                   # actual bug that produced SEKJPY 88.20 lots / NZDCAD 18.91
                                   # lots: a tight/tiny ATR on a low-volatility or oddly-quoted
                                   # symbol makes the risk-% math alone call for a huge lot, and
                                   # early in a cycle (few positions open) MARGIN_USAGE_CAP against
                                   # free margin doesn't catch it because almost all margin is free.
                                   # See risk.py:calc_lot_size.

# Hard, code-level backstop (not LLM judgment) -- block ALL new entries this
# cycle if margin_level is already this stressed, regardless of what the LLM
# decides. The LLM is told to weigh margin_level in the prompt, but that's
# advisory; this is enforced. Doesn't touch existing positions' own SL/TP.
MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES = 150.0  # percent; None/0 margin_used (no open positions) never blocks

# Hard, code-level backstop for currency/metal concentration -- mirrors the
# prompt's own "max 2 positions same currency, same direction" rule, but
# enforced in code instead of trusted to the LLM. A BUY on EURUSD counts as
# one long-EUR + one short-USD exposure; a SELL counts the opposite.
MAX_POSITIONS_PER_CURRENCY_SAME_DIRECTION = 2

# Code-level backstop for aggregate portfolio risk -- the per-trade RISK_PCT
# and per-currency concentration cap above don't stop many *uncorrelated*
# positions from each independently being at risk at once. Blocks a new entry
# if (sum of $ currently at risk across all open bot positions, i.e. distance
# to each position's own SL, valued in account currency) + this trade's own
# intended risk would exceed this fraction of balance. See risk.py:open_risk_dollars.
MAX_PORTFOLIO_RISK_PCT = 0.06

# --- Risk-taking factor ---
# Single dial to scale how much risk the bot takes, both in position SIZE and
# in how selective the LLM is about OPENING a new trade in the first place.
# 1.0 = current behavior (the settings above, unchanged). >1.0 scales:
#   - RISK_PCT_PER_TRADE, MARGIN_USAGE_CAP, MAX_MARGIN_PCT_OF_BALANCE,
#     MAX_PORTFOLIO_RISK_PCT (sizing)
#   - the LLM's entry bar, via risk_appetite_factor in the account summary
#     sent to it each cycle (see SYSTEM_PROMPT in llm_strategy.py) -- higher
#     values make it more willing to take tactical/lower-conviction setups
#     and fill more of the free slots instead of defaulting to HOLD.
# Does NOT scale MAX_DAILY_LOSS_PCT, MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES, the
# concentration caps, or the demo-account check -- those are hard backstops,
# not a risk-appetite dial, and stay fixed regardless of this value (that's
# exactly what caught the -30% day on 2026-09-09; raising RISK_FACTOR makes
# losses bigger *inside* that ceiling, it doesn't move the ceiling).
# Lowered from 1.0 to 0.75 on 2026-09-13 -- see the EMA_FAST comment above for
# the backtest sweep this came from. That sweep only exercised the sizing half
# of this dial (backtest.py's account_summary doesn't send risk_appetite_factor
# to the LLM the way bot.py's live cycle does -- see llm_strategy.SYSTEM_PROMPT's
# RISK APPETITE section) -- live, 0.75 should also make entries more selective,
# consistent with (not fighting) the smaller size tested here.
RISK_FACTOR = 1.0

# Hard ceilings RISK_FACTOR can never push the effective settings past, no
# matter how high it's set -- a second, independent rail so a fat-fingered
# RISK_FACTOR (e.g. 50) can't reintroduce the pre-2026-09-09 blowup risk.
_RISK_PCT_PER_TRADE_CEILING = 0.05   # a single trade can never risk >5% of balance
_MARGIN_USAGE_CAP_CEILING = 0.60     # a single trade can never eat >60% of free margin
_MAX_MARGIN_PCT_OF_BALANCE_CEILING = 0.40  # a single trade's margin can never exceed 40% of balance
_MAX_PORTFOLIO_RISK_PCT_CEILING = 0.25  # all open bot risk combined can never exceed 25% of balance

EFFECTIVE_RISK_PCT_PER_TRADE = min(RISK_PCT_PER_TRADE * RISK_FACTOR, _RISK_PCT_PER_TRADE_CEILING)
EFFECTIVE_MARGIN_USAGE_CAP = min(MARGIN_USAGE_CAP * RISK_FACTOR, _MARGIN_USAGE_CAP_CEILING)
EFFECTIVE_MAX_MARGIN_PCT_OF_BALANCE = min(MAX_MARGIN_PCT_OF_BALANCE * RISK_FACTOR, _MAX_MARGIN_PCT_OF_BALANCE_CEILING)
EFFECTIVE_MAX_PORTFOLIO_RISK_PCT = min(MAX_PORTFOLIO_RISK_PCT * RISK_FACTOR, _MAX_PORTFOLIO_RISK_PCT_CEILING)

# Circuit breaker: once realized loss since midnight (UTC) reaches this
# fraction of the day's starting balance, the bot stops opening NEW trades
# until the next day. It does not touch already-open positions' own SL/TP.
# NOTE: this was 0.30 on 2026-09-09, fired correctly at a -30% day (see
# mt5_bot.log "Daily loss circuit breaker hit (30.0%)"), and was then loosened
# to 0.80 afterwards -- which defeats the point (80% of the account gone is
# not a "circuit breaker", it's already a blown account). Reset to a value
# that actually preserves capital.
MAX_DAILY_LOSS_PCT = 0.15

# How many days a balance gain must "age" before it's allowed to increase
# position sizing (see risk.py:sizing_balance). Position size is always the
# SMALLER of the live balance and the balance from this many days ago, so a
# fresh hot streak can't inflate every subsequent trade's size before it's
# proven durable, but an actual drawdown is still sized down immediately.
# Added after the 2026-08-27 post-mortem: sizing off live balance let a
# short-lived $258K peak (from $100K) balloon lot sizes ~65% right before the
# losing stretch that gave nearly all of it back.
SIZING_BALANCE_LOOKBACK_DAYS = 5

# A setup tag with at least this many realized closed trades (see
# llm_strategy.PERF_MIN_SAMPLE) and an expectancy at or below this $ floor is
# hard-blocked from new entries in code (bot.py / backtest.py), rather than
# left to the LLM's prompt-level judgment alone -- see
# llm_strategy.negative_expectancy_setups(). Raised from 0.0 to -150.0 on
# 2026-09-16 after a local-model backtest (RISK_FACTOR sweep) showed the 0.0
# floor combined with PERF_MIN_SAMPLE=5 spiraling during a rough week: a
# handful of early losses tipped nearly every setup tag negative at once,
# hard-blocking almost all new entries and leaving the bot unable to trade
# its way back out. -150.0 requires a clearer, less noise-driven loss before
# vetoing a setup, while still catching a genuinely broken one (the
# 2026-09-12 MACD_TURN incident this gate was built for lost ~$960/trade on
# average, well past this floor).
SETUP_VETO_EXPECTANCY_FLOOR = -150.0

DEVIATION_POINTS = 20       # max price slippage tolerated on market orders
LOG_FILE = "mt5_bot.log"

# --- LLM-driven decisions ---
# The LLM decides direction (BUY/SELL) and CLOSE timing per symbol each cycle.
# Position size, stop-loss and take-profit stay fully deterministic (computed
# by risk.py from ATR + RISK_PCT_PER_TRADE above) — the model never sets numbers.
USE_LLM_STRATEGY = True

# "anthropic" = Claude API (costs money per call, needs ANTHROPIC_API_KEY).
# "ollama"    = local model via Ollama's OpenAI-compatible server (free, no
#               network calls, needs `ollama serve` running with the model
#               below already pulled). Tested with qwen2.5:7b-instruct, which
#               reliably supports forced tool-calling; not every local model
#               does -- if you switch models, confirm it accepts tool_choice
#               pointing at a specific function before trusting it live.
LLM_PROVIDER = "anthropic"

ANTHROPIC_API_KEY = None    # None = read from ANTHROPIC_API_KEY env var
LLM_MODEL = "claude-opus-5"
LLM_MAX_TOKENS = 4000

OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_MODEL = "qwen2.5:7b-instruct"
# IMPORTANT: Ollama's OpenAI-compatible endpoint (used above) ignores any
# per-request context-size option and always runs at Ollama's 4096-token
# default, regardless of what the model actually supports (this model
# handles up to 32768). The trading system prompt + a full multi-symbol
# snapshot + trade_memory routinely exceeds 4096 tokens, silently truncating
# input and making the model drop the forced tool call on a large fraction
# of cycles. The only fix is raising it server-wide via the
# OLLAMA_CONTEXT_LENGTH env var *before* `ollama serve` starts -- already
# done in start_mt5_bot.bat (set to 12000). If you start `ollama serve`
# manually, set OLLAMA_CONTEXT_LENGTH=12000 first or this silently degrades.
# One symbol whose candle-close timing is used as the "clock" that triggers a
# fresh batched decision cycle (all symbols are sent together in one call,
# not one call per symbol).
LLM_CLOCK_SYMBOL = "EURUSD"

# Log every successful Claude decision (input + output) as a training example
# for later fine-tuning a local model to imitate Claude's judgment. Only
# applies to the "anthropic" provider path -- local-model decisions are never
# logged here, since the whole point is distilling Claude's behavior, not the
# local model's. See llm_strategy._log_finetune_example / FINETUNE_LOG_PATH.
LOG_CLAUDE_DECISIONS_FOR_FINETUNE = True
FINETUNE_LOG_PATH = "finetune_data/claude_decisions.jsonl"  # relative to project root

# --- Financial news ---
# Two independent features, three providers, independently toggle-able (see
# news.py):
#
# Headlines (NEWS_HEADLINES_ENABLED): recent forex/financial headlines handed
# to the LLM as qualitative context (see the NEWS section of SYSTEM_PROMPT in
# llm_strategy.py) -- never a hard filter, the model may ignore them. Merged
# from two sources (deduped, newest first, capped at NEWS_HEADLINES_MAX_ITEMS):
# Finnhub (https://finnhub.io/, free tier signup) -- confirmed free,
# /news?category=forex is not behind Finnhub's paid plan -- and Alpha
# Vantage's NEWS_SENTIMENT endpoint (https://www.alphavantage.co/documentation/#news-sentiment).
# ALPHA_VANTAGE_API_KEY currently set to the public "demo" key in
# local_settings.py, which returns fixed sample data regardless of query
# params -- fine for wiring/testing the code path, but swap in a real free-tier
# key (https://www.alphavantage.co/support/#api-key) before relying on it for
# live headlines. Each source fails open independently -- one being
# down/unkeyed never removes the other's headlines.
#
# Economic calendar (NEWS_CALENDAR_ENABLED): a hard, code-level backstop --
# blocks NEW entries (never touches already-open positions) on symbols
# exposed to a currency with a high-impact scheduled release (NFP, CPI, a
# central bank rate decision, ...) for a window around it, same rationale as
# the margin/concentration backstops above: ATR-based stops are sized off
# pre-event volatility and routinely get skipped straight through on the
# release itself. Source: Financial Modeling Prep (https://financialmodelingprep.com/).
#
# CONFIRMED 2026-09-15 WITH A REAL KEY: FMP's free plan does NOT include
# /economic-calendar -- the current /stable endpoint returns HTTP 402
# ("Restricted Endpoint...upgrade your plan"), and the legacy v3/v4
# equivalents return HTTP 403 ("no longer supported for new users"). The key
# itself is valid (works for FMP's other free endpoints) and is kept in
# local_settings.py -- this gate simply has nothing to enforce with until the
# plan is upgraded (news.fetch_calendar_range() fails open: logs the 402
# once, returns [], never blocks trading). Defaulted OFF below since it's
# currently a no-op; flip to True once/if you upgrade the FMP plan. At that
# point also verify news._CCY_TO_COUNTRY's ISO-style country-code assumption
# ("US"/"EU"/"GB"/"JP"/"CH"/"CA"/"AU"/"NZ") against what FMP actually returns
# -- still unverified since the endpoint has never successfully responded.
NEWS_HEADLINES_ENABLED = True
NEWS_CALENDAR_ENABLED = False

FINNHUB_API_KEY = None       # None = read from FINNHUB_API_KEY env var
FMP_API_KEY = None           # None = read from FMP_API_KEY env var
ALPHA_VANTAGE_API_KEY = None # None = read from ALPHA_VANTAGE_API_KEY env var

# Optionally also settable in local_settings.py (git-ignored), same idea as
# the MT5 credentials above -- checked in news.py before the env var. Kept as
# a separate try/except so an older local_settings.py missing these names
# can't accidentally break the MT5_LOGIN/etc. import above.
try:
    from .local_settings import FINNHUB_API_KEY, FMP_API_KEY, ALPHA_VANTAGE_API_KEY  # noqa: F401,F811
except ImportError:
    pass

NEWS_HEADLINES_CATEGORY = "forex"
# Alpha Vantage has no native "forex" topic (its documented set is
# blockchain/earnings/ipo/mergers_and_acquisitions/financial_markets/
# economy_fiscal/economy_monetary/economy_macro/energy_transportation/
# finance/life_sciences/manufacturing/real_estate/retail_wholesale/
# technology). Tried "financial_markets" first -- confirmed 2026-09-15 it's
# dominated by US equity-specific news (single-stock movers, earnings, etc.)
# with no FX/macro relevance, and its high volume crowded Finnhub's actual
# forex headlines out of the merged top NEWS_HEADLINES_MAX_ITEMS entirely.
# "economy_macro,economy_monetary" (comma-separated = OR'd by Alpha Vantage)
# is central-bank/macro-data focused instead -- the closest real fit for a
# forex/metals bot.
NEWS_HEADLINES_AV_TOPICS = "economy_macro,economy_monetary"
NEWS_HEADLINES_MAX_ITEMS = 8
NEWS_HEADLINES_LOOKBACK_HOURS = 6

NEWS_CALENDAR_MIN_IMPACT = "high"   # "low" / "medium" / "high" -- only events at/above this block entries
NEWS_CALENDAR_BLACKOUT_MIN_BEFORE = 15
NEWS_CALENDAR_BLACKOUT_MIN_AFTER = 15

# Both providers are cached this long between refetches (live bot only --
# backtest.py fetches the whole test range once up front instead). POLL_SECONDS
# is 30s but neither headlines nor the calendar change that fast, and both
# providers' free-tier rate limits are much tighter than MT5's.
NEWS_CACHE_TTL_SECONDS = 900  # 15 min
