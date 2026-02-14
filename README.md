# Turning Point Analysis

LSTM-based detection of stock market reversal points using technical indicators and candlestick pattern features.

## Overview

This project implements a deep learning approach to identify **upward reversal points (URP)** and **downward reversal points (DRP)** in financial time series. Two separate binary LSTM classifiers are trained on 28 large-cap equities and evaluated out-of-sample on SPY and BTC-USD.

A supplementary **mean-reversion backtesting** module uses RSI signals on Binance 5-minute candles for BTC/USDT.

## Project Structure

```
turning_point_analysis/
├── README.md
├── LICENSE
├── requirements.txt
├── pyproject.toml
├── Makefile
├── .gitignore
├── config/
│   └── default.yaml              # All tunable parameters
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py             # Download / cache OHLCV via yfinance
│   ├── features/
│   │   ├── __init__.py
│   │   ├── technical.py          # 27 technical indicators
│   │   └── feature_sets.py       # Feature subsets (S0–S27) & sliding windows
│   ├── models/
│   │   ├── __init__.py
│   │   └── lstm.py               # LSTM model definition
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py            # Full training pipeline
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py          # OOS confusion matrix & F1
│   ├── backtest/
│   │   ├── __init__.py
│   │   └── mean_reversion.py     # RSI mean-reversion strategy
│   └── visualization/
│       ├── __init__.py
│       └── plots.py              # Price charts with signal overlays
├── scripts/
│   ├── train.py                  # CLI: train models
│   ├── evaluate.py               # CLI: OOS evaluation
│   ├── visualize.py              # CLI: generate plots
│   └── backtest.py               # CLI: run backtest
└── tests/
    ├── __init__.py
    └── test_features.py          # Feature engineering tests
```

## Quick Start

```bash
# Install dependencies
make install

# Train URP & DRP models (downloads data on first run)
make train

# Evaluate on out-of-sample tickers
make evaluate

# Generate prediction overlay plots
make visualize

# Run mean-reversion backtest
make backtest

# Run tests
make test
```

## Configuration

All parameters are in `config/default.yaml`:

- **data** — ticker lists, date range, cache directory
- **model** — window size, feature set, LSTM hidden units
- **training** — epochs, batch size, early stopping patience
- **evaluation** — probability threshold for signal classification
- **paths** — model save directory

Override with `--config path/to/custom.yaml` on any script.

## Methodology

### Turning Point Detection

Reversal points are labeled using a rule-based scheme (CRP → RP) applied to moving-average trends. A sliding window of technical features captures temporal dependencies for LSTM classification.

### Feature Engineering (27 Features)

| Category | Features |
|---|---|
| Trend | MA5, ΔMA5, Trend direction |
| Candlestick Reversal | CRP (composite reversal pattern) |
| Candlestick Shape | Candle, Body, topTail, bottomTail, Whole |
| Volume | pctMV20, VR20, PL20 |
| Momentum | CCI14, CCIS14, RSI20, StoK5, StoD5, StoR5 |
| MACD / ROC | MACDR, ROCMA5, ROC5 |
| Price Ratios | ARatio26, BRatio26, ABRatio26 |
| Returns | pctChange |

28 feature subsets (S0–S27) enable systematic ablation analysis.

### Model Architecture

- **Input:** Sliding window of `W` timesteps × `F` features
- **LSTM** with dropout (0.2) and recurrent dropout (0.2)
- **Dropout** (0.2)
- **Dense(1, sigmoid)** — binary output
- **Loss:** Binary cross-entropy with class weighting
- **Early stopping** on validation loss

## References

1. Dong, X., et al. (2020). "A new stock price reversal point prediction method based on a recognition model of candlestick charts." *Chaos, Solitons & Fractals*, 130, 109413. [DOI: 10.1016/j.chaos.2019.109413](https://doi.org/10.1016/j.chaos.2019.109413)

2. Chen, Y., & Hao, Y. (2022). "A novel framework for stock trading using reinforcement learning with candlestick patterns." *Expert Systems with Applications*, 210, 118484. [DOI: 10.1016/j.eswa.2022.118484](https://doi.org/10.1016/j.eswa.2022.118484)

## License

MIT — see [LICENSE](LICENSE) for details.
