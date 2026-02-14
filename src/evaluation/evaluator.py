"""Out-of-sample evaluation for trained turning-point models."""

import logging
import os

import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score

from src.features.technical import compute_all_features
from src.features.feature_sets import FEATURE_SETS, create_sliding_window_data

logger = logging.getLogger(__name__)


def _evaluate_preds(y_true: np.ndarray, y_pred: np.ndarray, label: str) -> None:
    """Log confusion-matrix metrics and F1 for a single model."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    logger.info(
        "%s → TN=%d  FP=%d  FN=%d  TP=%d  Precision=%.3f  Recall=%.3f  F1=%.3f",
        label, tn, fp, fn, tp, precision, recall, f1,
    )


def evaluate(cfg: dict) -> None:
    """Run OOS evaluation on configured tickers.

    Args:
        cfg: Parsed configuration dictionary.
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    eval_cfg = cfg["evaluation"]
    paths_cfg = cfg["paths"]

    oos_tickers = data_cfg["oos_tickers"]
    feature_set = model_cfg["feature_set"]
    hidden = model_cfg["hidden_units"]
    window = model_cfg["window"]
    threshold = eval_cfg["threshold"]
    models_dir = paths_cfg["models"]

    logger.info("=== OOS Testing on: %s ===", oos_tickers)

    scaler = load(os.path.join(models_dir, "scaler.joblib"))
    urp_model = load_model(os.path.join(models_dir, f"{feature_set}_H{hidden}_URP.h5"))
    drp_model = load_model(os.path.join(models_dir, f"{feature_set}_H{hidden}_DRP.h5"))
    logger.info("Loaded scaler and models")

    # Load OOS data
    dfs_raw: list[pd.DataFrame] = []
    for t in oos_tickers:
        fp = os.path.join(data_cfg["dir"], f"{t}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {fp}; run the train script first")
        df = pd.read_csv(fp, parse_dates=[0], index_col=0)
        df = df.apply(pd.to_numeric, errors="coerce")
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        if dropped:
            logger.warning("Dropped %d non-numeric/NaN rows from %s", dropped, t)
        dfs_raw.append(df)

    cols = FEATURE_SETS[feature_set] + ["RP"]
    dfs_feat = [compute_all_features(df) for df in dfs_raw]

    for ticker, df in zip(oos_tickers, dfs_feat):
        X, y = create_sliding_window_data([df[cols]], window, "RP")

        mask = np.all(np.isfinite(X), axis=(1, 2))
        if mask.sum() < len(mask):
            logger.warning(
                "Dropped %d bad windows for %s", len(mask) - mask.sum(), ticker
            )
        X, y = X[mask], y[mask]

        flat = X.reshape(-1, X.shape[-1])
        X_sc = scaler.transform(flat).reshape(X.shape).astype("float32")

        print(f"\n--- OOS report for {ticker} ---")
        for cls_val, cls_name, model in [
            (1, "URP", urp_model),
            (2, "DRP", drp_model),
        ]:
            probs = model.predict(X_sc, verbose=0)[:, 0]
            preds = (probs > threshold).astype(int)
            _evaluate_preds(
                (y == cls_val).astype(int),
                preds,
                f"{feature_set}-{cls_name}-{ticker}",
            )

    logger.info("=== OOS Testing complete ===")
