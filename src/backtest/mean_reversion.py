"""Mean-reversion backtesting strategy using RSI on Binance 5-minute candles."""

import logging
import os
import time

import numpy as np
import pandas as pd
import requests
from ta.momentum import RSIIndicator

logger = logging.getLogger(__name__)


def fetch_binance_klines(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start_ts: int | None = None,
    end_ts: int | None = None,
    limit: int = 1000,
) -> pd.DataFrame:
    """Fetch one batch of up to ``limit`` candles from Binance.

    Args:
        symbol: Binance trading pair symbol.
        interval: Candle interval (e.g. ``"5m"``).
        start_ts: Start timestamp in milliseconds.
        end_ts: End timestamp in milliseconds.
        limit: Maximum candles per request.

    Returns:
        DataFrame with OHLCV data.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit,
        "startTime": start_ts,
        "endTime": end_ts,
    }
    resp = requests.get(
        url, params={k: v for k, v in params.items() if v is not None}
    )
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(
        data,
        columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades",
            "taker_base_vol", "taker_quote_vol", "ignore",
        ],
    )
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return df


def download_data(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    start: str = "2020-01-01",
    end: str = "2025-07-28",
    cache_file: str = "b_data.pkl",
) -> pd.DataFrame:
    """Download full history from Binance or load from cache.

    Args:
        symbol: Trading pair.
        interval: Candle interval.
        start: Start date ISO string.
        end: End date ISO string.
        cache_file: Path for pickle cache.

    Returns:
        DataFrame with OHLCV data.
    """
    if os.path.exists(cache_file):
        logger.info("Loading cached data from '%s'", cache_file)
        return pd.read_pickle(cache_file)

    start_dt = pd.to_datetime(start)
    end_dt = pd.to_datetime(end)
    all_data: list[pd.DataFrame] = []
    batch_start = int(start_dt.timestamp() * 1000)
    batch_count = 0

    logger.info("Downloading %s %s candles from %s to %s", symbol, interval, start, end)
    while True:
        batch = fetch_binance_klines(
            symbol=symbol,
            interval=interval,
            start_ts=batch_start,
            end_ts=int(end_dt.timestamp() * 1000),
            limit=1000,
        )
        if batch.empty:
            break
        batch_count += 1
        last = batch["close_time"].iloc[-1]
        logger.info("Batch %d: %d candles up to %s", batch_count, len(batch), last)
        all_data.append(batch)
        batch_start = int((last + pd.Timedelta(milliseconds=1)).timestamp() * 1000)
        time.sleep(0.2)

    df = pd.concat(all_data, ignore_index=True)
    logger.info("Total candles: %d", len(df))
    df.to_pickle(cache_file)
    return df


def run_backtest(cfg: dict) -> pd.DataFrame:
    """Execute the mean-reversion RSI backtest.

    Args:
        cfg: Configuration dictionary. Uses keys under ``backtest`` if present,
             otherwise sensible defaults.

    Returns:
        DataFrame of trades with columns: entry_time, exit_time, side,
        entry_price, exit_price, pnl_pct.
    """
    bt_cfg = cfg.get("backtest", {})
    symbol = bt_cfg.get("symbol", "BTCUSDT")
    interval = bt_cfg.get("interval", "5m")
    start = bt_cfg.get("start", "2020-01-01")
    end = bt_cfg.get("end", "2025-07-28")
    cache_file = bt_cfg.get("cache_file", "b_data.pkl")
    rsi_window = bt_cfg.get("rsi_window", 14)

    b_data = download_data(symbol, interval, start, end, cache_file)

    # Compute RSI
    logger.info("Computing %d-period RSI", rsi_window)
    b_data["rsi"] = RSIIndicator(b_data["close"], window=rsi_window).rsi()

    # Generate positions
    logger.info("Generating signals and extracting trades")
    b_data["position"] = 0

    for i in range(1, len(b_data)):
        r = b_data.at[i, "rsi"]
        prev_pos = b_data.at[i - 1, "position"]

        if prev_pos == 0:
            if r < 30:
                b_data.at[i, "position"] = 1
            elif r > 70:
                b_data.at[i, "position"] = -1
        elif prev_pos == 1:
            if b_data.at[i - 1, "rsi"] < 50 <= r:
                b_data.at[i, "position"] = 0
            else:
                b_data.at[i, "position"] = 1
        elif prev_pos == -1:
            if b_data.at[i - 1, "rsi"] > 50 >= r:
                b_data.at[i, "position"] = 0
            else:
                b_data.at[i, "position"] = -1

    # Extract trades
    trades: list[dict] = []
    pos = 0
    entry_idx: int | None = None

    for idx, row in b_data.iterrows():
        if pos == 0 and row["position"] != 0:
            pos = row["position"]
            entry_idx = idx
        elif pos != 0 and row["position"] == 0:
            entry_price = b_data.at[entry_idx, "close"]
            exit_price = row["close"]
            pnl = (exit_price - entry_price) / entry_price * (1 if pos == 1 else -1)
            trades.append(
                {
                    "entry_time": b_data.at[entry_idx, "close_time"],
                    "exit_time": b_data.at[idx, "close_time"],
                    "side": "Long" if pos == 1 else "Short",
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl * 100,
                }
            )
            pos = 0

    trades_df = pd.DataFrame(trades)
    logger.info("Total trades: %d", len(trades_df))

    if not trades_df.empty:
        trades_df["year"] = trades_df["entry_time"].dt.year
        trades_df["month"] = trades_df["entry_time"].dt.month

        monthly = trades_df.groupby(["year", "month"]).agg(
            num_trades=("pnl_pct", "size"),
            win_rate=("pnl_pct", lambda x: (x > 0).mean()),
        ).reset_index()
        logger.info("Monthly summary:\n%s", monthly.to_string(index=False))

        yearly = trades_df.groupby("year").agg(
            num_trades=("pnl_pct", "size"),
            win_rate=("pnl_pct", lambda x: (x > 0).mean()),
        ).reset_index()
        logger.info("Yearly summary:\n%s", yearly.to_string(index=False))

        overall = (trades_df["pnl_pct"] > 0).mean()
        logger.info("Overall win rate: %.2f%%", overall * 100)

    return trades_df
