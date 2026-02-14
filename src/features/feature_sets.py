"""Feature set definitions and sliding-window data creation."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_SETS: dict[str, list[str]] = {
    "S0": ["CRP", "Trend", "Candle", "pctMV20", "VR20", "PL20", "CCI14", "CCIS14",
           "RSI20", "StoK5", "StoD5", "StoR5", "MACDR", "ROCMA5", "ROC5", "pctChange"],
    "S1": ["CRP", "Trend", "Candle", "pctMV20", "VR20", "PL20", "pctChange"],
    "S2": ["CRP", "Trend", "Candle", "CCI14", "CCIS14", "RSI20", "pctChange"],
    "S3": ["CRP", "Trend", "Candle", "StoK5", "StoD5", "StoR5", "pctChange"],
    "S4": ["CRP", "Trend", "Candle", "MACDR", "ROCMA5", "ROC5", "pctChange"],
    "S5": ["CRP", "Trend", "Candle", "ARatio26", "BRatio26", "ABRatio26", "pctChange"],
    "S6": ["CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S7": ["CRP", "Trend", "Candle", "pctMV20", "VR20", "PL20", "Body", "topTail",
           "bottomTail", "Whole"],
    "S8": ["CRP", "Trend", "Candle", "CCI14", "CCIS14", "RSI20", "Body", "topTail",
           "bottomTail", "Whole"],
    "S9": ["CRP", "Trend", "Candle", "StoK5", "StoD5", "StoR5", "Body", "topTail",
           "bottomTail", "Whole"],
    "S10": ["CRP", "Trend", "Candle", "MACDR", "ROCMA5", "ROC5", "Body", "topTail",
            "bottomTail", "Whole"],
    "S11": ["CRP", "Trend", "Candle", "ARatio26", "BRatio26", "ABRatio26", "Body",
            "topTail", "bottomTail", "Whole"],
    "S12": ["pctMV20", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S13": ["VR20", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S14": ["PL20", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S15": ["CCI14", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S16": ["CCIS14", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S17": ["RSI20", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S18": ["StoK5", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S19": ["StoD5", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S20": ["StoR5", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S21": ["MACDR", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S22": ["ROCMA5", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S23": ["ROC5", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S24": ["ARatio26", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S25": ["BRatio26", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S26": ["ABRatio26", "CRP", "Trend", "Candle", "Body", "topTail", "bottomTail", "Whole"],
    "S27": ["CRP", "Trend", "Candle", "pctMV20", "VR20", "PL20", "CCI14", "CCIS14",
            "RSI20", "StoK5", "StoD5", "StoR5", "MACDR", "ROCMA5", "ROC5", "pctChange",
            "ARatio26", "BRatio26", "ABRatio26", "Body", "topTail", "bottomTail", "Whole"],
}


def create_sliding_window_data(
    df_list: list[pd.DataFrame],
    window_size: int,
    target_col: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Create sliding-window arrays for LSTM training.

    For each DataFrame, a window of ``window_size`` feature rows is used as
    input and the value of ``target_col`` at the next row is the label.

    Args:
        df_list: List of feature-engineered DataFrames.
        window_size: Number of timesteps per window.
        target_col: Column name used as the prediction target.

    Returns:
        Tuple ``(X, y)`` where X has shape ``(N, window_size, n_features)``
        and y has shape ``(N,)``.
    """
    logger.info("Entering create_sliding_window_data")
    X: list[np.ndarray] = []
    y: list[float] = []
    for idx, df in enumerate(df_list):
        logger.info("DF#%d shape=%s", idx, df.shape)
        feats = df.drop(columns=[target_col]).values
        tgts = df[target_col].values
        for i in range(len(df) - window_size):
            X.append(feats[i : i + window_size])
            y.append(tgts[i + window_size])
    X_arr = np.array(X)
    y_arr = np.array(y)
    logger.info("windows: X.shape=%s, y.shape=%s", X_arr.shape, y_arr.shape)
    return X_arr, y_arr
