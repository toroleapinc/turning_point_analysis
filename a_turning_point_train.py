#!/usr/bin/env python3

import logging
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from joblib import dump

from a_utils import (
    load_df_list,
    compute_all_features,
    create_sliding_window_data,
    FEATURE_SETS
)

# — Logging setup —
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# — Config —
TRAIN_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "JPM", "JNJ", "BRK-B",
    "TSLA", "NVDA", "V", "UNH", "PG", "HD", "MA", "BAC", "WMT", "XOM",
    "PFE", "CVX", "NFLX", "ORCL", "NKE", "KO", "T", "CSCO", "MRK", "ABT"
]

START        = "2000-01-01"
END          = "2025-07-01"
DATA_DIR     = "data"
RELOAD       = True

WINDOW       = 10
FEATURE_SET  = "S27"
HIDDEN_UNITS = 200

VALIDATION_SPLIT = 0.1
EPOCHS           = 200
BATCH_SIZE       = 64
PATIENCE         = 20


def report_balance(name, labels):
    pos   = int((labels == 1).sum())
    neg   = int((labels == 0).sum())
    total = len(labels)
    logging.info(f"  {name}: total={total}, +={pos} ({pos/total:.1%}), -={neg} ({neg/total:.1%})")


def build_model(input_shape, hidden_units):
    model = Sequential([
        Input(shape=input_shape),
        LSTM(hidden_units, dropout=0.2, recurrent_dropout=0.2),
        Dropout(0.2),
        Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )
    return model


def main():
    logging.info(f"=== Training on large caps: {TRAIN_TICKERS} ===")

    # 1) Load & feature‐engineer
    dfs_raw = load_df_list(DATA_DIR, TRAIN_TICKERS, START, END, reload=RELOAD)
    dfs_feat = [compute_all_features(df) for df in dfs_raw]

    # 2) Create sliding windows
    cols = FEATURE_SETS[FEATURE_SET] + ["RP"]
    logging.info(f"Using features: {cols}")
    X, y = create_sliding_window_data(
        [df[cols] for df in dfs_feat], WINDOW, "RP"
    )
    logging.info(f"Raw windows: X={X.shape}, y={y.shape}")

    # 3) Drop non‑finite windows
    mask = np.all(np.isfinite(X), axis=(1,2))
    dropped = X.shape[0] - mask.sum()
    if dropped:
        logging.warning(f"Dropping {dropped}/{X.shape[0]} windows with inf/NaN")
    X, y = X[mask], y[mask]
    logging.info(f"After drop: X={X.shape}, y={y.shape}")

    # 4) Scale features
    flat = X.reshape(-1, X.shape[-1])
    scaler = StandardScaler().fit(flat)
    os.makedirs("saved_models", exist_ok=True)
    dump(scaler, "saved_models/scaler.joblib")
    logging.info("Saved feature scaler to saved_models/scaler.joblib")
    X_sc = scaler.transform(flat).reshape(X.shape).astype("float32")

    # 5) Train/validation split (chronological)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_sc, y, test_size=VALIDATION_SPLIT, shuffle=False
    )
    logging.info(f"Split: X_tr={X_tr.shape}, X_val={X_val.shape}")

    # 6) Train & save URP/DRP models
    for cls_val, cls_name in [(1, "URP"), (2, "DRP")]:
        yb_tr  = (y_tr  == cls_val).astype(int)
        yb_val = (y_val == cls_val).astype(int)
        if len(np.unique(yb_tr)) < 2:
            logging.warning(f"Skipping {cls_name}: only one class present in training")
            continue

        logging.info(f"--- Training {FEATURE_SET}-H{HIDDEN_UNITS}-{cls_name} ---")
        report_balance(" train", yb_tr)
        report_balance(" val  ", yb_val)

        cw = compute_class_weight("balanced", classes=np.array([0,1]), y=yb_tr)
        class_weight = {0: float(cw[0]), 1: float(cw[1])}

        model = build_model((WINDOW, X.shape[-1]), HIDDEN_UNITS)
        es = EarlyStopping(monitor="val_loss", patience=PATIENCE, restore_best_weights=True, verbose=1)
        model.fit(
            X_tr, yb_tr,
            validation_data=(X_val, yb_val),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            class_weight=class_weight,
            callbacks=[es],
            verbose=1
        )

        path = f"saved_models/{FEATURE_SET}_H{HIDDEN_UNITS}_{cls_name}.h5"
        model.save(path)
        logging.info(f"Saved {cls_name} model to {path}")

    logging.info("=== Training complete ===")


if __name__ == "__main__":
    main()
