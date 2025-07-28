#!/usr/bin/env python3

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from keras.models import load_model

from a_utils import compute_all_features, create_sliding_window_data, FEATURE_SETS

# — Config —
DATA_DIR     = "data"
WINDOW       = 10
FEATURE_SET  = "S27"
THRESHOLD    = 0.5
OOS_TICKERS  = ["SPY", "BTC-USD"]


def load_data(ticker: str) -> pd.DataFrame:
    """Load CSV data for a ticker, mirroring training parsing."""
    fp = os.path.join(DATA_DIR, f"{ticker}.csv")
    df = pd.read_csv(fp, parse_dates=[0], index_col=0)
    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df.apply(pd.to_numeric, errors="coerce").dropna()
    return df


def get_predictions(df: pd.DataFrame, urp_model, drp_model, scaler):
    """
    Compute URP and DRP probabilities for sliding windows.
    Returns end-of-window dates and probability arrays.
    """
    # Feature engineering
    df_feat = compute_all_features(df)
    cols = FEATURE_SETS[FEATURE_SET] + ["RP"]
    X, _ = create_sliding_window_data([df_feat[cols]], WINDOW, "RP")

    # Filter finite windows
    mask = np.all(np.isfinite(X), axis=(1, 2))
    X = X[mask]
    dates = pd.to_datetime(df_feat.index[WINDOW:][mask])

    # Scale and predict
    flat = X.reshape(-1, X.shape[-1])
    X_sc = scaler.transform(flat).reshape(X.shape).astype("float32")
    urp_probs = urp_model.predict(X_sc, verbose=0)[:, 0]
    drp_probs = drp_model.predict(X_sc, verbose=0)[:, 0]
    return dates, urp_probs, drp_probs


def plot_signals(year: int, df: pd.DataFrame, dates, urp_probs: np.ndarray, drp_probs: np.ndarray, ticker: str):
    """Plot closing price with URP (buy) and DRP (sell) signals."""
    # Masks
    buy_mask = (urp_probs > THRESHOLD) & (dates.year == year)
    sell_mask = (drp_probs > THRESHOLD) & (dates.year == year)
    buy_dates = dates[buy_mask]
    sell_dates = dates[sell_mask]

    # Price series for year
    price = df['Close']
    price.index = pd.to_datetime(price.index)
    year_mask = price.index.year == year
    price_year = price[year_mask]

    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(price_year.index, price_year.values, label='Close Price')
    if len(buy_dates):
        ax.scatter(buy_dates, price.loc[buy_dates], marker='^', s=100,
                   edgecolors='black', label='URP (Buy)', zorder=5, color='green')
    if len(sell_dates):
        ax.scatter(sell_dates, price.loc[sell_dates], marker='v', s=100,
                   edgecolors='black', label='DRP (Sell)', zorder=5, color='red')

    ax.set_title(f"{ticker} Close Price with URP (Buy) & DRP (Sell) Signals ({year})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main(year: int = 2024):
    scaler = load("saved_models/scaler.joblib")
    urp_model = load_model(f"saved_models/{FEATURE_SET}_H200_URP.h5")
    drp_model = load_model(f"saved_models/{FEATURE_SET}_H200_DRP.h5")
    for ticker in OOS_TICKERS:
        df = load_data(ticker)
        dates, urp_probs, drp_probs = get_predictions(df, urp_model, drp_model, scaler)
        plot_signals(year, df, dates, urp_probs, drp_probs, ticker)

if __name__ == "__main__":
    main()
