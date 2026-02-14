"""Technical indicator computation (27 features).

All indicators are computed from raw OHLCV DataFrames and follow the
equations described in the referenced papers.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-functions grouped by indicator category
# ---------------------------------------------------------------------------

def compute_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute MA5, ΔMA5, Trend, CRP, and RP labels (Eqs 10–14)."""
    df["MA5"] = df["Close"].rolling(5).mean()
    df["ΔMA5"] = df["MA5"].diff()
    df["Trend"] = np.where(
        df["ΔMA5"] > 0, 1, np.where(df["ΔMA5"] < 0, -1, 0)
    )

    # CRP (Eq 13)
    df["CRP"] = 0
    df.loc[
        (df["Trend"] == 1) & (df["Close"] < df["Close"].shift(1)), "CRP"
    ] = 1
    df.loc[
        (df["Trend"] == -1) & (df["Close"] > df["Close"].shift(1)), "CRP"
    ] = -1

    # RP (Eq 14) – 1 = URP, 2 = DRP
    df["RP"] = 0
    for i in range(len(df) - 10):
        c = df["CRP"].iat[i]
        if c != 0:
            window = df["MA5"].iloc[i + 1 : i + 11]
            if (window.max() - window.min()) / df["MA5"].iat[i] > 0.03:
                df.iat[i, df.columns.get_loc("RP")] = 1 if c == 1 else 2
    return df


def compute_candlestick_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute candlestick body / tail indicators (Eqs 15–19)."""
    prev_close = df["Close"].shift(1)
    df["Candle"] = np.sign(df["Close"] - df["Open"])
    df["Body"] = (df["Close"] - df["Open"]).abs() / prev_close
    df["topTail"] = (
        df["High"] - df[["Open", "Close"]].max(axis=1)
    ) / prev_close
    df["bottomTail"] = (
        df[["Open", "Close"]].min(axis=1) - df["Low"]
    ) / prev_close
    df["Whole"] = (df["High"] - df["Low"]) / prev_close
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute volume-based indicators: pctMV20, VR20, PL20."""
    df["pctMV20"] = df["Volume"] / df["Volume"].rolling(20).mean().shift(1)

    up_close = df["Close"] > df["Close"].shift(1)
    down_close = df["Close"] < df["Close"].shift(1)
    eq_close = df["Close"] == df["Close"].shift(1)

    upv = df["Volume"].where(up_close, 0)
    downv = df["Volume"].where(down_close, 0)
    eqv = df["Volume"].where(eq_close, 0)

    df["VR20"] = (
        (upv.rolling(20).sum() + eqv.rolling(20).sum() / 2)
        / (downv.rolling(20).sum() + eqv.rolling(20).sum() / 2)
        * 100
    )
    df["PL20"] = df["Candle"].rolling(20).apply(
        lambda x: (x > 0).sum() / 20 * 100
    )
    return df


def compute_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute CCI14, CCIS14, RSI20, Stochastic, MACD-related, ROC."""
    # CCI14 & CCIS14
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    df["CCI14"] = (tp - tp.rolling(14).mean()) / (0.015 * tp.rolling(14).std())
    df["CCIS14"] = df["CCI14"].rolling(14).mean()

    # RSI20
    d = df["Close"].diff()
    up = d.where(d > 0, 0)
    dn = -d.where(d < 0, 0)
    df["RSI20"] = 100 - 100 / (1 + up.rolling(20).mean() / dn.rolling(20).mean())

    # Stochastic (n=5)
    low5 = df["Low"].rolling(5).min()
    high5 = df["High"].rolling(5).max()
    df["StoK5"] = (df["Close"] - low5) / (high5 - low5) * 100
    df["StoD5"] = df["StoK5"].rolling(3).mean()
    df["StoR5"] = (high5 - df["Close"]) / (high5 - low5) * 100

    # MACD-related
    ema5 = df["Close"].ewm(span=5, adjust=False).mean()
    ema10 = df["Close"].ewm(span=10, adjust=False).mean()
    df["MACDR"] = ema5 - ema10
    df["ROCMA5"] = df["MA5"].pct_change(5) * 100
    df["ROC5"] = df["Close"].pct_change(5) * 100
    return df


def compute_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute A/B ratio indicators (n=26)."""
    n26 = 26
    A = (df["High"] - df["Open"]).rolling(n26).sum()
    B = (df["Open"] - df["Low"]).rolling(n26).sum()
    C = (df["High"] - df["Close"]).rolling(n26).sum()
    D = (df["Close"] - df["Low"]).rolling(n26).sum()
    df["ARatio26"] = A / B * 100
    df["BRatio26"] = C / D * 100
    df["ABRatio26"] = df["ARatio26"] / df["BRatio26"]
    return df


def compute_pct_change(df: pd.DataFrame) -> pd.DataFrame:
    """Compute simple percentage change."""
    df["pctChange"] = df["Close"].pct_change() * 100
    return df


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 27 technical features from a raw OHLCV DataFrame.

    The input DataFrame must have columns: Open, High, Low, Close, Volume
    with a DatetimeIndex.  Rows containing NaN after feature computation
    are dropped.

    Args:
        df: Raw OHLCV DataFrame.

    Returns:
        DataFrame with all original columns plus computed features, NaN rows
        dropped.
    """
    logger.info("Entering compute_all_features")
    df = df.copy()
    df = compute_trend_features(df)
    df = compute_candlestick_features(df)
    df = compute_volume_features(df)
    df = compute_momentum_features(df)
    df = compute_ratio_features(df)
    df = compute_pct_change(df)
    df.dropna(inplace=True)
    logger.info("compute_all_features output shape=%s", df.shape)
    return df
