"""Code-level backstop for currency/metal concentration risk -- enforces the
same "max N positions per currency, same direction" rule the SYSTEM_PROMPT
already asks the LLM to self-police, so a misjudgment there (e.g. stacking
five JPY-short positions across unrelated crosses) can't slip through. Mirrors
how MAX_OPEN_POSITIONS_PER_SYMBOL / MAX_TOTAL_OPEN_POSITIONS are already
code-enforced rather than left to the LLM's judgment alone.
"""
from . import config


def _symbol_currencies(symbol: str):
    """Splits a 6-letter symbol into (base, quote), e.g. "EURUSD" -> ("EUR",
    "USD"), "XAUUSD" -> ("XAU", "USD") -- metals behave like a base currency
    here, matching how the prompt already talks about "currency or metal
    theme". Returns None for anything that doesn't fit that shape (unusual
    broker symbol naming) so the caller can skip enforcement rather than
    guess wrong."""
    if len(symbol) != 6 or not symbol.isalpha():
        return None
    return symbol[:3].upper(), symbol[3:].upper()


def _exposures(symbol: str, direction: str):
    """The set of (currency, sign) exposures a position creates: sign +1 for
    a long exposure to that currency, -1 for short. A BUY on EURUSD is long
    EUR / short USD; a SELL is the opposite."""
    ccys = _symbol_currencies(symbol)
    if ccys is None:
        return set()
    base, quote = ccys
    sign = 1 if direction == "BUY" else -1
    return {(base, sign), (quote, -sign)}


def exceeds_concentration_cap(symbol: str, direction: str, open_positions: list,
                               max_per_currency: int = None) -> bool:
    """True if adding this candidate trade would push any currency/metal's
    same-direction exposure count above the cap. open_positions is the bot's
    own currently-open positions (already magic-number filtered), each with
    .symbol and .type (mt5.ORDER_TYPE_BUY/SELL)."""
    max_per_currency = max_per_currency or config.MAX_POSITIONS_PER_CURRENCY_SAME_DIRECTION
    candidate = _exposures(symbol, direction)
    if not candidate:
        return False  # unrecognized symbol shape -- don't block on a guess

    existing_counts = {}
    for pos in open_positions:
        pos_dir = "BUY" if pos.type == 0 else "SELL"  # mt5.ORDER_TYPE_BUY == 0
        for exposure in _exposures(pos.symbol, pos_dir):
            existing_counts[exposure] = existing_counts.get(exposure, 0) + 1

    return any(existing_counts.get(exp, 0) >= max_per_currency for exp in candidate)
