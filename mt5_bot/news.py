"""Financial news integration: two independent features, three providers.

- Headlines (config.NEWS_HEADLINES_ENABLED): recent forex headlines handed to
  the LLM as qualitative context (see the NEWS section of SYSTEM_PROMPT in
  llm_strategy.py) -- never a hard filter, the model may ignore them. Two
  sources merged together (deduped by headline text, newest first, capped at
  NEWS_HEADLINES_MAX_ITEMS): Finnhub (https://finnhub.io/, free tier signup,
  confirmed free -- /news?category=forex is not gated behind Finnhub's paid
  plan) and Alpha Vantage's NEWS_SENTIMENT endpoint
  (https://www.alphavantage.co/documentation/#news-sentiment). Each source
  fails open independently -- one being down/unkeyed never removes the
  other's headlines.

- Economic calendar (config.NEWS_CALENDAR_ENABLED): a hard, code-level
  backstop -- blocks NEW entries (never touches existing open positions) on
  symbols exposed to a currency with a high-impact scheduled release (NFP,
  CPI, a central bank rate decision, ...) for a window around it, the same
  rationale as the margin/concentration backstops in config.py: ATR-based
  stops are sized off pre-event volatility and routinely get skipped straight
  through on the release itself. Source: Financial Modeling Prep
  (https://financialmodelingprep.com/, free tier signup). See config.py's
  "Financial news" section for the two caveats worth verifying with a real
  key before relying on this live (date-range access on the free tier, and
  the exact country codes FMP returns).

Both providers fail open: a missing/invalid key, a plan restriction, a
network error, or a malformed response all just mean no headlines / no
calendar events this cycle -- never an exception that could interrupt a
decision cycle or block trading in some unrelated way.
"""
import json
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from . import config

log = logging.getLogger(__name__)

_FINNHUB_BASE = "https://finnhub.io/api/v1"
_FMP_BASE = "https://financialmodelingprep.com/stable"
_ALPHA_VANTAGE_BASE = "https://www.alphavantage.co/query"

_IMPACT_RANK = {"low": 0, "medium": 1, "high": 2}

# ISO-style country codes assumed for FMP's economic-calendar `country` field.
# Unverified against a real key -- if events aren't matching symbols you'd
# expect, log the raw `country` values fetch_calendar_range() returns and
# adjust this map (e.g. FMP may use "UK" instead of "GB", or "EMU"/"EA"
# instead of "EU" for the Eurozone).
_CCY_TO_COUNTRY = {
    "USD": "US", "EUR": "EU", "GBP": "GB", "JPY": "JP",
    "CHF": "CH", "CAD": "CA", "AUD": "AU", "NZD": "NZ",
}

_headlines_cache = {"until": 0.0, "items": []}
_calendar_cache = {"until": 0.0, "events": []}

_warned_finnhub_key = False
_warned_fmp_key = False
_warned_fmp_fetch = False
_warned_av_key = False


def _finnhub_key():
    return config.FINNHUB_API_KEY or os.environ.get("FINNHUB_API_KEY")


def _fmp_key():
    return config.FMP_API_KEY or os.environ.get("FMP_API_KEY")


def _alpha_vantage_key():
    return config.ALPHA_VANTAGE_API_KEY or os.environ.get("ALPHA_VANTAGE_API_KEY")


def _http_get_json(url: str, params: dict):
    """GET url?params as JSON. Raises on failure -- callers catch broadly,
    since every caller here treats any failure the same way (fail open)."""
    qs = urllib.parse.urlencode(params)
    req = urllib.request.Request(f"{url}?{qs}", headers={"User-Agent": "mt5_bot/news"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read().decode("utf-8"))


def symbol_currencies(symbol: str) -> set:
    """Currency codes this symbol is exposed to, for matching against the
    economic calendar's per-country events. Metals are treated as pure-USD
    exposure -- there's no "XAU country" issuing economic data, and gold/
    silver's dominant short-term driver is USD/Fed-relevant releases, not a
    second currency the way a forex pair has one."""
    if symbol in ("XAUUSD", "XAGUSD"):
        return {"USD"}
    if len(symbol) == 6 and symbol.isalpha():
        return {symbol[:3].upper(), symbol[3:].upper()}
    return set()


def _headline_cutoff():
    return datetime.now(timezone.utc) - timedelta(hours=config.NEWS_HEADLINES_LOOKBACK_HOURS)


def _fetch_finnhub_headlines():
    """Recent forex headlines from Finnhub. Returns None on a request/parse
    failure (caller falls back to cache), or a list (possibly empty) on
    success -- never raises."""
    global _warned_finnhub_key
    api_key = _finnhub_key()
    if not api_key:
        if not _warned_finnhub_key:
            log.warning("FINNHUB_API_KEY not set -- Finnhub headlines disabled (calendar gate unaffected)")
            _warned_finnhub_key = True
        return []

    try:
        raw = _http_get_json(
            f"{_FINNHUB_BASE}/news",
            {"category": config.NEWS_HEADLINES_CATEGORY, "token": api_key},
        )
    except Exception:
        log.exception("Finnhub headlines fetch failed (non-fatal)")
        return None

    cutoff = _headline_cutoff()
    items = []
    for a in raw or []:
        try:
            t = datetime.fromtimestamp(a["datetime"], tz=timezone.utc)
        except (KeyError, TypeError, ValueError, OSError):
            continue
        if t < cutoff:
            continue
        items.append({"time": t.isoformat(), "headline": a.get("headline"), "source": a.get("source")})
    return items


def _fetch_alpha_vantage_headlines():
    """Recent headlines from Alpha Vantage's NEWS_SENTIMENT endpoint. Returns
    None on a request/parse failure (caller falls back to cache), or a list
    (possibly empty) on success -- never raises. Alpha Vantage has no native
    "forex" topic; NEWS_HEADLINES_AV_TOPICS defaults to "financial_markets",
    the closest macro/FX-relevant bucket -- see config.py."""
    global _warned_av_key
    api_key = _alpha_vantage_key()
    if not api_key:
        if not _warned_av_key:
            log.warning("ALPHA_VANTAGE_API_KEY not set -- Alpha Vantage headlines disabled (Finnhub unaffected)")
            _warned_av_key = True
        return []

    try:
        raw = _http_get_json(
            _ALPHA_VANTAGE_BASE,
            {"function": "NEWS_SENTIMENT", "topics": config.NEWS_HEADLINES_AV_TOPICS, "apikey": api_key},
        )
    except Exception:
        log.exception("Alpha Vantage headlines fetch failed (non-fatal)")
        return None

    if not isinstance(raw, dict) or "feed" not in raw:
        # e.g. {"Information": "..."} on a rate-limited/invalid/demo key --
        # not an exception, just nothing usable this cycle.
        return []

    cutoff = _headline_cutoff()
    items = []
    for a in raw.get("feed") or []:
        try:
            t = datetime.strptime(a["time_published"], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        except (KeyError, TypeError, ValueError):
            continue
        if t < cutoff:
            continue
        items.append({"time": t.isoformat(), "headline": a.get("title"), "source": a.get("source")})
    return items


def get_headlines() -> list:
    """Recent forex/financial headlines merged from Finnhub + Alpha Vantage,
    deduped by headline text, newest first, capped at NEWS_HEADLINES_MAX_ITEMS,
    cached for NEWS_CACHE_TTL_SECONDS. Returns [] if disabled. Each source
    fails open independently; if both fail to fetch, falls back to the
    previous cached items rather than going blank. Never raises and never
    blocks a decision cycle."""
    if not config.NEWS_HEADLINES_ENABLED:
        return []

    now = time.monotonic()
    if now < _headlines_cache["until"]:
        return _headlines_cache["items"]

    finnhub_items = _fetch_finnhub_headlines()
    av_items = _fetch_alpha_vantage_headlines()

    if finnhub_items is None and av_items is None:
        _headlines_cache["until"] = now + config.NEWS_CACHE_TTL_SECONDS
        return _headlines_cache["items"]

    merged = (finnhub_items or []) + (av_items or [])
    merged.sort(key=lambda x: x["time"], reverse=True)

    seen = set()
    deduped = []
    for it in merged:
        key = (it["headline"] or "").strip().lower()
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(it)
    deduped = deduped[:config.NEWS_HEADLINES_MAX_ITEMS]

    _headlines_cache.update(until=now + config.NEWS_CACHE_TTL_SECONDS, items=deduped)
    return deduped


_FMP_MAX_SPAN_DAYS = 90  # FMP's documented per-call cap on the from/to range


def _fetch_calendar_chunk(start: datetime, end: datetime) -> list:
    """Single FMP economic-calendar request, no chunking -- caller
    (fetch_calendar_range) is responsible for staying within the 90-day cap."""
    global _warned_fmp_key, _warned_fmp_fetch
    api_key = _fmp_key()
    if not api_key:
        if not _warned_fmp_key:
            log.warning("FMP_API_KEY not set -- economic calendar gate disabled")
            _warned_fmp_key = True
        return []

    try:
        raw = _http_get_json(
            f"{_FMP_BASE}/economic-calendar",
            {"from": start.date().isoformat(), "to": end.date().isoformat(), "apikey": api_key},
        )
    except urllib.error.HTTPError as e:
        if not _warned_fmp_fetch:
            log.error(
                "FMP economic calendar request failed: HTTP %s %s (non-fatal, calendar gate inactive -- "
                "check FMP_API_KEY and that your plan includes date-ranged /economic-calendar access)",
                e.code, e.reason,
            )
            _warned_fmp_fetch = True
        return []
    except Exception:
        if not _warned_fmp_fetch:
            log.exception("FMP economic calendar fetch failed (non-fatal, calendar gate inactive this cycle)")
            _warned_fmp_fetch = True
        return []

    events = []
    for e in raw or []:
        try:
            t = datetime.strptime(e["date"], "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
        except (KeyError, TypeError, ValueError):
            continue
        events.append({
            "country": e.get("country"),
            "currency": e.get("currency"),
            "event": e.get("event"),
            "impact": (e.get("impact") or "").lower(),
            "time": t,
        })
    return events


def fetch_calendar_range(start: datetime, end: datetime) -> list:
    """FMP economic-calendar fetch for [start, end], transparently chunked
    into <=90-day requests (FMP's documented per-call cap) and concatenated.
    Returns parsed event dicts {country, currency, event, impact (lowercased),
    time (aware UTC datetime)}, or [] if disabled, no key, or any request/
    parse failure -- never raises. A failure on one chunk doesn't stop the
    others (partial calendar coverage beats none for the rest of the range)."""
    if not config.NEWS_CALENDAR_ENABLED:
        return []

    events = []
    chunk_start = start
    span = timedelta(days=_FMP_MAX_SPAN_DAYS)
    while chunk_start < end:
        chunk_end = min(chunk_start + span, end)
        events.extend(_fetch_calendar_chunk(chunk_start, chunk_end))
        chunk_start = chunk_end
    return events


def get_calendar_events() -> list:
    """Live rolling window (now +/- a couple hours) of the economic calendar,
    cached for NEWS_CACHE_TTL_SECONDS. For backtesting a specific historical
    window, call fetch_calendar_range(start, end) directly instead."""
    if not config.NEWS_CALENDAR_ENABLED:
        return []

    now = time.monotonic()
    if now < _calendar_cache["until"]:
        return _calendar_cache["events"]

    span = timedelta(minutes=max(config.NEWS_CALENDAR_BLACKOUT_MIN_BEFORE,
                                  config.NEWS_CALENDAR_BLACKOUT_MIN_AFTER) + 120)
    wall_now = datetime.now(timezone.utc)
    events = fetch_calendar_range(wall_now - span, wall_now + span)

    _calendar_cache.update(until=now + config.NEWS_CACHE_TTL_SECONDS, events=events)
    return events


def blocking_event(symbol: str, at: datetime, events: list):
    """The economic-calendar event blocking a NEW entry on `symbol` at time
    `at`, or None. `events` is whatever fetch_calendar_range()/
    get_calendar_events() returned -- caller decides live vs. backtest
    source. Only touches NEW entries; existing positions are never affected."""
    countries = {_CCY_TO_COUNTRY[c] for c in symbol_currencies(symbol) if c in _CCY_TO_COUNTRY}
    if not countries:
        return None
    min_rank = _IMPACT_RANK.get(config.NEWS_CALENDAR_MIN_IMPACT.lower(), 2)
    before = timedelta(minutes=config.NEWS_CALENDAR_BLACKOUT_MIN_BEFORE)
    after = timedelta(minutes=config.NEWS_CALENDAR_BLACKOUT_MIN_AFTER)
    for ev in events:
        if ev["country"] not in countries:
            continue
        if _IMPACT_RANK.get(ev["impact"], -1) < min_rank:
            continue
        if ev["time"] - before <= at <= ev["time"] + after:
            return ev
    return None
