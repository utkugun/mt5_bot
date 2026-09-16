from dataclasses import dataclass

import pandas as pd

from . import config, indicators


@dataclass
class Signal:
    direction: str  # "BUY", "SELL", "NONE"
    atr: float
    reason: str


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ema_fast"] = indicators.ema(df["close"], config.EMA_FAST)
    df["ema_slow"] = indicators.ema(df["close"], config.EMA_SLOW)
    df["ema_trend"] = indicators.ema(df["close"], config.EMA_TREND)
    df["rsi"] = indicators.rsi(df["close"], config.RSI_PERIOD)
    df["macd"], df["macd_signal"], df["macd_hist"] = indicators.macd(
        df["close"], config.MACD_FAST, config.MACD_SLOW, config.MACD_SIGNAL
    )
    df["atr"] = indicators.atr(df, config.ATR_PERIOD)
    pivots = indicators.pivot_points(df)
    df["pivot"] = pivots["pivot"]
    df["r1"] = pivots["r1"]
    df["s1"] = pivots["s1"]
    return df


def evaluate(df: pd.DataFrame) -> Signal:
    """Ensemble signal on the last closed candle (df must not include the
    still-forming candle). Requires trend, EMA-crossover, MACD and RSI to
    all agree before firing a BUY/SELL."""
    feats = build_features(df)
    min_history = max(config.EMA_TREND, config.MACD_SLOW, config.RSI_PERIOD) + 5
    if len(feats) < min_history:
        return Signal("NONE", float("nan"), "not enough history")

    last = feats.iloc[-1]
    prev = feats.iloc[-2]

    check_cols = ["ema_fast", "ema_slow", "ema_trend", "rsi", "macd", "macd_signal", "atr"]
    if last[check_cols].isna().any() or prev[check_cols].isna().any():
        return Signal("NONE", float("nan"), "indicator warmup")

    trend_up = last["close"] > last["ema_trend"]
    trend_down = last["close"] < last["ema_trend"]

    ema_bull = last["ema_fast"] > last["ema_slow"]
    ema_bull_cross = ema_bull and prev["ema_fast"] <= prev["ema_slow"]
    ema_bear = last["ema_fast"] < last["ema_slow"]
    ema_bear_cross = ema_bear and prev["ema_fast"] >= prev["ema_slow"]

    macd_bull = last["macd"] > last["macd_signal"]
    macd_bull_cross = macd_bull and prev["macd"] <= prev["macd_signal"]
    macd_bear = last["macd"] < last["macd_signal"]
    macd_bear_cross = macd_bear and prev["macd"] >= prev["macd_signal"]

    rsi_ok_buy = last["rsi"] < config.RSI_OVERBOUGHT
    rsi_ok_sell = last["rsi"] > config.RSI_OVERSOLD

    # Require a fresh cross (EMA or MACD) so the bot doesn't re-fire every
    # candle while conditions merely stay true.
    buy = trend_up and ema_bull and macd_bull and rsi_ok_buy and (ema_bull_cross or macd_bull_cross)
    sell = trend_down and ema_bear and macd_bear and rsi_ok_sell and (ema_bear_cross or macd_bear_cross)

    if buy:
        return Signal("BUY", float(last["atr"]), "trend up + EMA bull + MACD bull + RSI ok")
    if sell:
        return Signal("SELL", float(last["atr"]), "trend down + EMA bear + MACD bear + RSI ok")
    return Signal("NONE", float(last["atr"]), "no confluence")
