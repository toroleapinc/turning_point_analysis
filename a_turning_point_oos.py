#!/usr/bin/env python3

import logging
import os
import numpy as np
import pandas as pd
from joblib import load
from keras.models import load_model
from sklearn.metrics import confusion_matrix, f1_score

from a_utils import (
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
OOS_TICKERS = ["SPY", "BTC-USD"]
DATA_DIR     = "data"
WINDOW       = 10
FEATURE_SET  = "S27"
THRESHOLD    = 0.7

def evaluate_preds(y_true, y_pred, label):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    precision = tp/(tp+fp) if tp+fp>0 else 0.0
    recall    = tp/(tp+fn) if tp+fn>0 else 0.0
    f1        = f1_score(y_true, y_pred, zero_division=0)
    logging.info(
        f"{label} → TN={tn}  FP={fp}  FN={fn}  TP={tp}  "
        f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}"
    )

def main():
    logging.info(f"=== OOS Testing on: {OOS_TICKERS} ===")

    # 1) Load scaler & models
    scaler    = load("saved_models/scaler.joblib")
    urp_model = load_model("saved_models/S27_H200_URP.h5")
    drp_model = load_model("saved_models/S27_H200_DRP.h5")
    logging.info("Loaded scaler and models")

    # 2) Read only SPY & BTC‑USD CSVs
    dfs_raw = []
    for t in OOS_TICKERS:
        fp = os.path.join(DATA_DIR, f"{t}.csv")
        if not os.path.isfile(fp):
            raise FileNotFoundError(f"Missing {fp}; run the train script first")
        df = pd.read_csv(fp, parse_dates=[0], index_col=0)

        # — **NEW**: force everything numeric & drop any bad rows
        df = df.apply(pd.to_numeric, errors="coerce")
        before = len(df)
        df.dropna(inplace=True)
        dropped = before - len(df)
        if dropped:
            logging.warning(f"Dropped {dropped} non‑numeric/NaN rows from {t}")

        dfs_raw.append(df)

    # 3) Feature‐engineer & window‐ize
    cols = FEATURE_SETS[FEATURE_SET] + ["RP"]
    dfs_feat = [compute_all_features(df) for df in dfs_raw]

    for ticker, df in zip(OOS_TICKERS, dfs_feat):
        X, y = create_sliding_window_data([df[cols]], WINDOW, "RP")

        # drop any inf/NaN windows just in case
        mask = np.all(np.isfinite(X), axis=(1,2))
        if mask.sum() < len(mask):
            logging.warning(f"Dropped {len(mask)-mask.sum()} bad windows for {ticker}")
        X, y = X[mask], y[mask]

        # 4) Scale
        flat = X.reshape(-1, X.shape[-1])
        X_sc = scaler.transform(flat).reshape(X.shape).astype("float32")

        print(f"\n--- OOS report for {ticker} ---")
        for cls_val, cls_name, model in [
            (1, "URP", urp_model),
            (2, "DRP", drp_model)
        ]:
            probs = model.predict(X_sc, verbose=0)[:,0]
            preds = (probs > THRESHOLD).astype(int)
            evaluate_preds((y==cls_val).astype(int),
                           preds,
                           f"{FEATURE_SET}-{cls_name}-{ticker}")

    logging.info("=== OOS Testing complete ===")


if __name__ == "__main__":
    main()
