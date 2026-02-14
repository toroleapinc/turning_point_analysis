"""Tests for feature engineering pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.features.technical import compute_all_features


def _make_synthetic_ohlcv(n: int = 300) -> pd.DataFrame:
    """Create a synthetic OHLCV DataFrame for testing."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=n)
    close = 100 + np.cumsum(rng.normal(0, 1, n))
    high = close + rng.uniform(0.5, 2.0, n)
    low = close - rng.uniform(0.5, 2.0, n)
    opn = close + rng.normal(0, 0.5, n)
    volume = rng.integers(1_000_000, 10_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": opn, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


class TestComputeAllFeatures:
    """Tests for compute_all_features()."""

    def test_output_columns_exist(self) -> None:
        """All expected feature columns should be present."""
        df = _make_synthetic_ohlcv()
        result = compute_all_features(df)
        expected = [
            "MA5", "Trend", "CRP", "RP",
            "Candle", "Body", "topTail", "bottomTail", "Whole",
            "pctMV20", "VR20", "PL20",
            "CCI14", "CCIS14", "RSI20",
            "StoK5", "StoD5", "StoR5",
            "MACDR", "ROCMA5", "ROC5",
            "ARatio26", "BRatio26", "ABRatio26",
            "pctChange",
        ]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"

    def test_no_nan_in_output(self) -> None:
        """Output should have no NaN values after dropna."""
        df = _make_synthetic_ohlcv()
        result = compute_all_features(df)
        assert not result.isna().any().any(), "Found NaN values in output"

    def test_output_not_empty(self) -> None:
        """Output should have rows remaining after feature computation."""
        df = _make_synthetic_ohlcv()
        result = compute_all_features(df)
        assert len(result) > 0, "Output DataFrame is empty"
