"""Visualization of turning-point predictions on price charts."""

import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from keras.models import load_model

from src.features.technical import compute_all_features
from src.features.feature_sets import FEATURE_SETS, create_sliding_window_data

logger = logging.getLogger(__name__)


def _load_ticker_data(ticker: str, data_dir: str) -> pd.DataFrame:
    """Load and clean CSV data for a single ticker.

    Args:
        ticker: Ticker symbol.
        data_dir: Directory containing CSV files.

    Returns:
        Cleaned DataFrame with DatetimeIndex.
    """
    fp = os.path.join(data_dir, f"{ticker}.csv")
    df = pd.read_csv(fp, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def _get_predictions(
    df: pd.DataFrame,
    urp_model,
    drp_model,
    scaler,
    feature_set: str,
    window: int,
) -> tuple[pd.DatetimeIndex, np.ndarray, np.ndarray]:
    """Compute URP/DRP probabilities for sliding windows.

    Args:
        df: Raw OHLCV DataFrame.
        urp_model: Trained URP Keras model.
        drp_model: Trained DRP Keras model.
        scaler: Fitted StandardScaler.
        feature_set: Feature set key (e.g. ``"S27"``).
        window: Sliding window size.

    Returns:
        Tuple of (dates, urp_probs, drp_probs).
    """
    df_feat = compute_all_features(df)
    cols = FEATURE_SETS[feature_set] + ["RP"]
    X, _ = create_sliding_window_data([df_feat[cols]], window, "RP")

    mask = np.all(np.isfinite(X), axis=(1, 2))
    X = X[mask]
    dates = pd.to_datetime(df_feat.index[window:][mask])

    flat = X.reshape(-1, X.shape[-1])
    X_sc = scaler.transform(flat).reshape(X.shape).astype("float32")
    urp_probs = urp_model.predict(X_sc, verbose=0)[:, 0]
    drp_probs = drp_model.predict(X_sc, verbose=0)[:, 0]
    return dates, urp_probs, drp_probs


def plot_signals(
    year: int,
    df: pd.DataFrame,
    dates: pd.DatetimeIndex,
    urp_probs: np.ndarray,
    drp_probs: np.ndarray,
    ticker: str,
    threshold: float = 0.5,
) -> None:
    """Plot closing price with URP (buy) and DRP (sell) signals for a year.

    Args:
        year: Calendar year to plot.
        df: Raw OHLCV DataFrame (with Close column).
        dates: Prediction dates array.
        urp_probs: URP probability array.
        drp_probs: DRP probability array.
        ticker: Ticker symbol for title.
        threshold: Probability threshold for signal generation.
    """
    buy_mask = (urp_probs > threshold) & (dates.year == year)
    sell_mask = (drp_probs > threshold) & (dates.year == year)
    buy_dates = dates[buy_mask]
    sell_dates = dates[sell_mask]

    price = df["Close"]
    price.index = pd.to_datetime(price.index)
    year_mask = price.index.year == year
    price_year = price[year_mask]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_year.index, price_year.values, label="Close Price")
    if len(buy_dates):
        ax.scatter(
            buy_dates, price.loc[buy_dates], marker="^", s=100,
            edgecolors="black", label="URP (Buy)", zorder=5, color="green",
        )
    if len(sell_dates):
        ax.scatter(
            sell_dates, price.loc[sell_dates], marker="v", s=100,
            edgecolors="black", label="DRP (Sell)", zorder=5, color="red",
        )
    ax.set_title(f"{ticker} Close Price with URP (Buy) & DRP (Sell) Signals ({year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()


def visualize(cfg: dict, year: int = 2024) -> None:
    """Generate prediction overlay plots for OOS tickers.

    Args:
        cfg: Parsed configuration dictionary.
        year: Calendar year to visualize.
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg.get("evaluation", {})
    paths_cfg = cfg["paths"]

    feature_set = model_cfg["feature_set"]
    hidden = model_cfg["hidden_units"]
    window = model_cfg["window"]
    threshold = eval_cfg.get("threshold", 0.5)
    models_dir = paths_cfg["models"]

    scaler = load(os.path.join(models_dir, "scaler.joblib"))
    urp_model = load_model(os.path.join(models_dir, f"{feature_set}_H{hidden}_URP.h5"))
    drp_model = load_model(os.path.join(models_dir, f"{feature_set}_H{hidden}_DRP.h5"))

    for ticker in data_cfg["oos_tickers"]:
        df = _load_ticker_data(ticker, data_cfg["dir"])
        dates, urp_probs, drp_probs = _get_predictions(
            df, urp_model, drp_model, scaler, feature_set, window,
        )
        plot_signals(year, df, dates, urp_probs, drp_probs, ticker, threshold)
