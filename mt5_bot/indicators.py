import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int, slow: int, signal: int):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    hist = macd_line - signal_line
    return macd_line, signal_line, hist


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def pivot_points(df: pd.DataFrame) -> pd.DataFrame:
    """Classic floor-trader daily pivot/R1/S1, computed from the prior
    COMPLETED UTC trading day's high/low/close and held constant through the
    following day. No lookahead -- a day's levels only ever depend on the
    previous trading day's already-closed candles, and weekends/holidays are
    skipped naturally since there's no data for them (Monday's levels use
    Friday's high/low/close). Requires a tz-aware 'time' column."""
    day = df["time"].dt.floor("D")
    by_day = df.groupby(day).agg(high=("high", "max"), low=("low", "min"), close=("close", "last"))
    prev_day = by_day.shift(1)
    pivot = (prev_day["high"] + prev_day["low"] + prev_day["close"]) / 3
    r1 = 2 * pivot - prev_day["low"]
    s1 = 2 * pivot - prev_day["high"]
    return pd.DataFrame({
        "pivot": day.map(pivot),
        "r1": day.map(r1),
        "s1": day.map(s1),
    }, index=df.index)
