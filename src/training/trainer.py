"""Training pipeline for URP/DRP turning-point models."""

import logging
import os

import numpy as np
from joblib import dump
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from src.data.loader import load_df_list
from src.features.technical import compute_all_features
from src.features.feature_sets import FEATURE_SETS, create_sliding_window_data
from src.models.lstm import build_lstm_model

logger = logging.getLogger(__name__)


def _report_balance(name: str, labels: np.ndarray) -> None:
    """Log class balance statistics."""
    pos = int((labels == 1).sum())
    neg = int((labels == 0).sum())
    total = len(labels)
    logger.info(
        "%s: total=%d, +=%d (%.1f%%), -=%d (%.1f%%)",
        name, total, pos, pos / total * 100, neg, neg / total * 100,
    )


def train(cfg: dict) -> None:
    """Run the full training pipeline.

    Args:
        cfg: Parsed configuration dictionary (mirrors ``config/default.yaml``).
    """
    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    paths_cfg = cfg["paths"]

    tickers = data_cfg["train_tickers"]
    logger.info("=== Training on %d tickers ===", len(tickers))

    # 1) Load & feature-engineer
    dfs_raw = load_df_list(
        data_dir=data_cfg["dir"],
        tickers=tickers,
        start=data_cfg["start"],
        end=data_cfg["end"],
        reload=data_cfg["reload"],
    )
    dfs_feat = [compute_all_features(df) for df in dfs_raw]

    # 2) Create sliding windows
    feature_set = model_cfg["feature_set"]
    cols = FEATURE_SETS[feature_set] + ["RP"]
    logger.info("Using feature set %s (%d features)", feature_set, len(cols) - 1)
    X, y = create_sliding_window_data(
        [df[cols] for df in dfs_feat], model_cfg["window"], "RP"
    )
    logger.info("Raw windows: X=%s, y=%s", X.shape, y.shape)

    # 3) Drop non-finite windows
    mask = np.all(np.isfinite(X), axis=(1, 2))
    dropped = X.shape[0] - mask.sum()
    if dropped:
        logger.warning("Dropping %d/%d windows with inf/NaN", dropped, X.shape[0])
    X, y = X[mask], y[mask]

    # 4) Scale features
    flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(flat)
    models_dir = paths_cfg["models"]
    os.makedirs(models_dir, exist_ok=True)
    scaler_path = os.path.join(models_dir, "scaler.joblib")
    dump(scaler, scaler_path)
    logger.info("Saved scaler to %s", scaler_path)
    X_sc = scaler.transform(flat).reshape(X.shape).astype("float32")

    # 5) Train/validation split (chronological)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sc, y, test_size=train_cfg["validation_split"], shuffle=False
    )
    logger.info("Split: X_tr=%s, X_val=%s", X_tr.shape, X_val.shape)

    # 6) Train & save URP/DRP models
    hidden = model_cfg["hidden_units"]
    for cls_val, cls_name in [(1, "URP"), (2, "DRP")]:
        yb_tr = (y_tr == cls_val).astype(int)
        yb_val = (y_val == cls_val).astype(int)
        if len(np.unique(yb_tr)) < 2:
            logger.warning("Skipping %s: only one class in training", cls_name)
            continue

        logger.info("--- Training %s-H%d-%s ---", feature_set, hidden, cls_name)
        _report_balance("train", yb_tr)
        _report_balance("val", yb_val)

        cw = compute_class_weight("balanced", classes=np.array([0, 1]), y=yb_tr)
        class_weight = {0: float(cw[0]), 1: float(cw[1])}

        model = build_lstm_model((model_cfg["window"], X.shape[-1]), hidden)
        es = EarlyStopping(
            monitor="val_loss",
            patience=train_cfg["patience"],
            restore_best_weights=True,
            verbose=1,
        )
        model.fit(
            X_tr,
            yb_tr,
            validation_data=(X_val, yb_val),
            epochs=train_cfg["epochs"],
            batch_size=train_cfg["batch_size"],
            class_weight=class_weight,
            callbacks=[es],
            verbose=1,
        )

        path = os.path.join(
            models_dir, f"{feature_set}_H{hidden}_{cls_name}.h5"
        )
        model.save(path)
        logger.info("Saved %s model to %s", cls_name, path)

    logger.info("=== Training complete ===")
