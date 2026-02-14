"""Data loading utilities for downloading and caching ticker data."""

import glob
import logging
import os

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)


def load_df_list(
    data_dir: str = "data",
    tickers: list[str] | None = None,
    start: str | None = None,
    end: str | None = None,
    reload: bool = False,
) -> list[pd.DataFrame]:
    """Load ticker DataFrames from local CSVs or download via yfinance.

    If CSVs exist in ``data_dir`` and ``reload`` is False, they are loaded
    directly.  Otherwise the given ``tickers`` are downloaded for the
    ``[start, end)`` date range and cached as CSVs.

    Args:
        data_dir: Directory for CSV cache.
        tickers: List of Yahoo Finance ticker symbols.
        start: Start date string (inclusive), e.g. ``"2000-01-01"``.
        end: End date string (exclusive), e.g. ``"2025-07-01"``.
        reload: If True, re-download even when CSVs exist.

    Returns:
        List of DataFrames with DatetimeIndex and numeric OHLCV columns.

    Raises:
        ValueError: When no CSVs exist and tickers/start/end are not provided.
        RuntimeError: When no data could be loaded or downloaded.
    """
    logger.info("Entering load_df_list")
    os.makedirs(data_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    logger.info("Found %d CSV(s) in '%s'", len(paths), data_dir)

    if paths and not reload:
        dfs: list[pd.DataFrame] = []
        for fp in paths:
            logger.info("Loading local CSV: %s", fp)
            df = pd.read_csv(fp, parse_dates=[0], index_col=0)
            df = df.apply(pd.to_numeric, errors="coerce")
            n0 = len(df)
            df.dropna(inplace=True)
            n1 = len(df)
            if n1 < n0:
                logger.warning("Dropped %d NaN rows", n0 - n1)
            logger.info("shape=%s", df.shape)
            dfs.append(df)
        logger.info("Exiting load_df_list with local data")
        return dfs

    if not (tickers and start and end):
        raise ValueError(
            "No CSVs found (or reload=True) and missing tickers/start/end"
        )

    dfs = []
    for t in tickers:
        logger.info("Downloading '%s' from %s to %s", t, start, end)
        df = yf.download(t, start=start, end=end, progress=False)
        if df.empty:
            logger.warning("No data for %s; skipping", t)
            continue
        fp = os.path.join(data_dir, f"{t}.csv")
        df.to_csv(fp)
        logger.info("Saved raw CSV to %s", fp)
        df = pd.read_csv(fp, parse_dates=[0], index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.dropna(inplace=True)
        logger.info("shape=%s", df.shape)
        dfs.append(df)

    if not dfs:
        raise RuntimeError("No dataframes loaded or downloaded")
    logger.info("Exiting load_df_list with downloaded data")
    return dfs
