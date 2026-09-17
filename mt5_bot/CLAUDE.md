# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A Python trading bot for MetaTrader5 (forex + metals, demo account only). It polls MT5 for newly
closed candles and either:
- evaluates a deterministic EMA/MACD/RSI ensemble signal (`strategy.py`), or
- (default, `config.USE_LLM_STRATEGY = True`) sends a batched snapshot of indicators for all
  symbols to an LLM (`llm_strategy.py`), which returns BUY/SELL/CLOSE/HOLD per symbol.
  `config.LLM_PROVIDER` picks the backend: `"anthropic"` (Claude API, costs money per call) or
  `"ollama"` (default — a local model via Ollama's OpenAI-compatible server, free, no network
  calls, requires `ollama serve` running with `config.OLLAMA_MODEL` pulled). Both paths force the
  same `trade_decisions` tool call and get parsed into the same decision list
  (`_get_decisions_anthropic` / `_get_decisions_openai_compat` in `llm_strategy.py`), so the rest
  of the bot doesn't know or care which one answered.

Position sizing, stop-loss, and take-profit are always computed deterministically from ATR and
account risk settings (`risk.py`) — the LLM only picks direction/timing, never numbers. A
mechanical trailing profit-lock (`risk.trailing_sl_update`, driven by `config.TRAIL_ACTIVATE_R`/
`TRAIL_GIVEBACK_R`) ratchets each open position's stop-loss once it's shown a real gain, so a
winner that reverses without ever touching its original SL/TP still banks most of what it made —
see `risk.py`/`bot.py` below.

This package is run as `python -m mt5_bot.bot` (or `mt5_bot.backtest`) from the **parent**
directory (`utku python`), since `bot.py`/`backtest.py` use relative imports (`from . import
config, ...`). `start_mt5_bot.bat` in the parent dir does this.

## Commands

```bash
# Run the live bot (from the parent directory of mt5_bot/)
python -m mt5_bot.bot

# Backtest against MT5 historical data (also from the parent directory)
python -m mt5_bot.backtest --days 18 --symbols EURUSD,GBPUSD,USDJPY --dry-run   # count API calls first
python -m mt5_bot.backtest --days 18                                            # actually runs it (spends Anthropic API $)
```

There is no test suite, linter, or requirements.txt in this package — dependencies are
`MetaTrader5`, `pandas`, `numpy`, `anthropic`, and `openai` (the last used only for the
OpenAI-compatible call into Ollama), installed into whatever environment runs it. `news.py` adds
no new dependency — it uses stdlib `urllib` rather than `requests`.

Both `bot.py` and `backtest.py` require:
- A running, logged-in MT5 terminal (or credentials in `local_settings.py`) — `connector.py`
  **refuses to proceed if the connected account is not a demo account.**
- When `USE_LLM_STRATEGY` is on (default): either `ollama serve` running locally with
  `config.OLLAMA_MODEL` pulled (default provider, no API key needed), or, if
  `config.LLM_PROVIDER = "anthropic"`, `ANTHROPIC_API_KEY` set in the environment (or
  `config.ANTHROPIC_API_KEY`).
- Optionally, `FINNHUB_API_KEY` / `FMP_API_KEY` for the news features (see `news.py` below) —
  settable via env var or `local_settings.py` (both keys are already there for this account);
  both are entirely optional, missing/invalid keys just disable that feature, never break a run.

## Configuration

Everything tunable lives in `config.py` (single source of truth — comments there explain *why*
values were changed, e.g. risk % and position caps were lowered after backtesting showed margin
stress). Real MT5 credentials go in `local_settings.py` (git-ignored; copy from
`local_settings.example.py`) and are pulled in automatically via a `try/except ImportError` in
`config.py`.

Key knobs: `SYMBOLS`/`SYMBOL_GROUP_FILTER` (which pairs/metals to trade — `SYMBOL_GROUP_FILTER`
takes a string or a list, e.g. `["Forex", "Metals"]`, matched against each symbol's broker
group path; check your broker's actual path naming and adjust if needed), `TIMEFRAME`, risk settings
(`RISK_PCT_PER_TRADE`, `SL_ATR_MULT`/`TP_ATR_MULT`, `TRAIL_ACTIVATE_R`/`TRAIL_GIVEBACK_R` (the
mechanical trailing profit-lock — see `risk.py` below), `MAX_TOTAL_OPEN_POSITIONS`,
`MAX_DAILY_LOSS_PCT`, `MARGIN_USAGE_CAP`), the LLM block (`USE_LLM_STRATEGY`, `LLM_MODEL`,
`LLM_CLOCK_SYMBOL`), and the news block (`NEWS_HEADLINES_ENABLED`, `NEWS_CALENDAR_ENABLED`,
`NEWS_CALENDAR_MIN_IMPACT`, `NEWS_CALENDAR_BLACKOUT_MIN_BEFORE`/`_AFTER` — see `news.py` below).

`RISK_FACTOR` (default `1.0`) is a single dial that scales both position *sizing*
(`EFFECTIVE_RISK_PCT_PER_TRADE`/`EFFECTIVE_MARGIN_USAGE_CAP`/`EFFECTIVE_MAX_MARGIN_PCT_OF_BALANCE`/
`EFFECTIVE_MAX_PORTFOLIO_RISK_PCT`, all computed in `config.py` and consumed by `risk.py`/`bot.py`
instead of the raw base values) and the LLM's bar for *opening* a trade (via `risk_appetite_factor`
in the account summary `bot.py` sends each cycle — see the "RISK APPETITE" section of
`SYSTEM_PROMPT` in `llm_strategy.py`). It does not touch `MAX_DAILY_LOSS_PCT`,
`MIN_MARGIN_LEVEL_FOR_NEW_ENTRIES`, the concentration caps, or the demo-account check — those are
hard backstops, not a risk-appetite dial, and each `EFFECTIVE_*` value is independently clamped to
its own ceiling in `config.py` so `RISK_FACTOR` can't reintroduce the kind of oversized single-trade
risk that the 2026-09-09 -30% day came from.

## Architecture

- **`connector.py`** — MT5 terminal connect/disconnect. Hard safety check: raises if the
  attached account isn't a demo account.
- **`indicators.py`** — pure pandas EMA/RSI/MACD/ATR implementations, no MT5/config dependency.
- **`strategy.py`** — `build_features()` (adds indicator columns) and `evaluate()`, the
  deterministic ensemble signal used only when `USE_LLM_STRATEGY = False`. Also called by
  `llm_strategy.build_symbol_row()` to compute indicators for the LLM path.
- **`llm_strategy.py`** — the default decision path. `build_symbol_row()` turns a symbol's
  candles into a compact indicator dict; `get_decisions()` sends the account summary + all
  symbol rows in one forced tool call (`trade_decisions`, via `tool_choice`) and returns the
  parsed decisions, dispatching to `_get_decisions_anthropic()` or `_get_decisions_openai_compat()`
  per `config.LLM_PROVIDER`. `DECISIONS_TOOL` (Anthropic schema) and `OPENAI_DECISIONS_TOOL`
  (derived from it, OpenAI function-calling shape) must stay in sync — the latter is built
  programmatically from the former so they can't drift. Tracks cumulative token usage/cost via
  `get_usage_totals()`. The `SYSTEM_PROMPT` encodes the actual trading rules (when CLOSE is/isn't
  appropriate, margin thresholds, correlation caution, position caps) — read it before changing
  LLM behavior. Note: not every local model reliably honors forced tool_choice — verify a new
  `OLLAMA_MODEL` actually returns a `trade_decisions` tool call before trusting it live.
- **`news.py`** — two independent, independently-toggleable news features, both fail-open (a
  missing key, plan restriction, or request/parse error just disables that feature, never breaks
  a decision cycle or blocks trading some unrelated way): (1) recent forex headlines from Finnhub
  (`NEWS_HEADLINES_ENABLED`, **live and working** — verified 2026-09-15 with a real key, an
  account exists at `utkugun@gmail.com`, key stored in `local_settings.py`), handed to the LLM as
  pure qualitative context via `get_decisions(..., headlines)`; (2) an economic-calendar entry gate
  from Financial Modeling Prep (`NEWS_CALENDAR_ENABLED`, **defaulted OFF** — see below) — a
  **code-level backstop**, same pattern as the margin/concentration checks:
  `blocking_event(symbol, at, events)` blocks NEW entries (never touches open positions) on
  symbols exposed to a currency with a high-impact scheduled release (NFP/CPI/rate decisions) for
  a window around it. `bot.py`/`backtest.py` both attach a `news_blackout` field to each symbol
  row (event name or `null`) and enforce the gate identically; **headlines are live-only** —
  Finnhub's `/news` has no historical date range, so `backtest.py` never sends a NEWS section to
  the LLM (see its module docstring).
  **FMP calendar status (confirmed 2026-09-15 with a real, valid key):** the free plan does NOT
  include `/economic-calendar` — the current `/stable` endpoint returns HTTP 402 ("upgrade your
  plan"), and the legacy `v3`/`v4` equivalents return HTTP 403 ("no longer supported for new
  users"). This isn't a guess or a docs-page ambiguity anymore — it was tested directly. The key
  is valid and stored in `local_settings.py` for when/if the plan gets upgraded; until then
  `NEWS_CALENDAR_ENABLED = False` because the gate is a confirmed no-op otherwise. If upgraded,
  also verify `news._CCY_TO_COUNTRY`'s ISO-style country-code assumption
  (`"US"`/`"EU"`/`"GB"`/`"JP"`/`"CH"`/`"CA"`/`"AU"`/`"NZ"`) against what FMP actually returns —
  still unverified, since the endpoint has never successfully responded.
- **`risk.py`** — `calc_lot_size()`: sizes a position so a stop-loss hit costs
  `RISK_PCT_PER_TRADE` of balance, capped by `MARGIN_USAGE_CAP` of free margin via a binary
  search over `mt5.order_calc_margin` (margin isn't linear in volume for many brokers).
  `trailing_sl_update()`: the mechanical profit-lock — a pure function of direction, entry
  price, the position's *original* SL (fixed — defines its 1R risk distance), its current SL,
  and the current price. Once profit reaches `TRAIL_ACTIVATE_R` multiples of that R, returns a
  new SL that locks in `(profit_R - TRAIL_GIVEBACK_R) * R`, re-ratcheting tighter as price
  extends further; returns `None` below the activation threshold or whenever the computed level
  would loosen (not tighten) the current SL. It never touches take-profit, so a strong move can
  still run all the way to TP — this only bounds how much of an already-earned gain can be given
  back if price reverses before getting there. Added 2026-09-17 after an XAGUSD trade rode from
  +$15k unrealized down to +$2k because it never touched its SL/TP and the LLM's own
  discretionary CLOSE/PROFIT_LOCK judgment (deliberately conservative, see `llm_strategy.py`'s
  `SYSTEM_PROMPT`) held through the whole round trip — this is a code-level backstop, independent
  of the LLM, same pattern as the margin/concentration gates.
- **`trader.py`** — sends actual `mt5.order_send` requests (open/close), picks a supported
  order-filling mode per symbol, tags all bot orders with `MAGIC_NUMBER` so `open_positions()`
  only ever sees/manages this bot's own trades. `modify_sl()` ratchets an open position's SL via
  `TRADE_ACTION_SLTP` (TP untouched) — used only by the trailing profit-lock.
- **`bot.py`** — the live polling loop. Two decision paths depending on
  `config.USE_LLM_STRATEGY`:
  - **LLM path**: waits for a new closed candle on `LLM_CLOCK_SYMBOL` only, then runs one
    batched `run_llm_decision_cycle()` covering every symbol at once (not one call per symbol).
  - **Non-LLM path**: loops every symbol independently, closes on an opposite deterministic
    signal, then opens on confluence.

  Also owns: daily-loss circuit breaker (tracked from midnight-UTC balance, blocks new entries
  only — never touches existing SL/TP), stale-data-feed detection/logging (distinguishes
  "market closed" from "feed actually stuck"), and `_apply_trailing_stops()` — runs every poll
  tick (every `POLL_SECONDS`, independent of the candle/LLM cadence) against every open
  bot-owned position on every symbol, calling `risk.trailing_sl_update()` / `trader.modify_sl()`.
  It tracks each position's *original* SL in an in-memory `ticket -> sl` dict (seeded from
  whatever SL is on the position the first time its ticket is seen this run, so it resets on
  restart — same known limitation as `balance_history`/`sizing_balance`).
- **`backtest.py`** — replays the **exact same** `llm_strategy`/`risk` decision logic against
  historical MT5 candles via a `SimBroker` (not the MT5 Strategy Tester, which can't run Python
  EAs at all). Fills at candle close ± half historical spread; SL is assumed to win if a single
  bar's range touches both SL and TP (worst-case). The trailing profit-lock is replayed too —
  `apply_trailing_and_check()` ratchets each `SimPosition.sl` off the bar's favorable high/low
  before checking that bar's SL/TP hit, using the same `risk.trailing_sl_update()`; trades closed
  this way show up in `trades.csv` as `reason="TRAIL_SL"` rather than `"SL"`. This makes real
  Anthropic API calls per simulated decision cycle — always run `--dry-run` first to see the call
  count before spending money. Results (trades.csv, equity_curve.csv) are written to
  `../backtest_results/`.

## Working on this code

- Never weaken or remove the demo-account check in `connector.py`.
- Position sizing/SL/TP math must stay deterministic (`risk.py`/`trader.py`) — the LLM must
  never be given control over size, SL, or TP, only direction and CLOSE timing.
- If you change indicator columns sent to the LLM (`build_symbol_row`) or the trading rules,
  update `SYSTEM_PROMPT` in `llm_strategy.py` to match — the prompt is the only place those
  rules are documented for the model.
- `bot.py` and `backtest.py` must stay behaviorally in sync for anything that affects trade
  decisions (position sizing, entry/exit rules, account summary fields) since the backtest's
  entire premise is replaying the live bot's exact logic. The one deliberate exception is
  headlines (live-only, see `news.py` above) — the economic-calendar gate itself IS replayed.
- Don't flip `NEWS_CALENDAR_ENABLED` to `True` expecting it to do anything without an FMP plan
  upgrade first — see the confirmed 402/403 findings under `news.py` above.
