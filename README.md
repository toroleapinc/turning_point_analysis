# Turning Point Analysis

LSTM-based detection of stock market reversal points using technical indicators and candlestick pattern features.

## Overview

This project implements a deep learning approach to identify **upward reversal points (URP)** and **downward reversal points (DRP)** in financial time series. The methodology is based on academic research that combines candlestick pattern recognition with technical indicators to train LSTM models for turning point prediction.

The system trains on historical data from 28 large-cap equities and evaluates out-of-sample on SPY (S&P 500 ETF) and BTC-USD (Bitcoin).

## Methodology

### Turning Point Detection

Reversal points are identified using a rule-based labeling scheme applied to historical price data. A sliding window approach captures temporal dependencies in the feature space, allowing the LSTM to learn sequential patterns leading up to market reversals.

Two separate binary classifiers are trained:
- **URP Model** — detects potential upward reversals (buy signals)
- **DRP Model** — detects potential downward reversals (sell signals)

### Feature Engineering (27 Features)

The feature set spans multiple categories of technical analysis:

| Category | Features |
|---|---|
| **Candlestick Reversal Patterns** | CRP (composite reversal pattern score) |
| **Trend Indicators** | Trend direction, moving average crossovers |
| **Momentum Oscillators** | RSI (20-period), Stochastic Oscillator, MACD |
| **Volatility** | CCI (14-period), Bollinger Band signals |
| **Custom Ratios** | A/B Ratios (proprietary price-action metrics) |
| **Ablation Sets** | 28 feature subsets (S0–S27) for systematic feature importance analysis |

All features are computed from OHLCV data retrieved via the `yfinance` API.

### Model Architecture

- **Model:** LSTM (Long Short-Term Memory) neural network
- **Framework:** TensorFlow / Keras
- **Training Data:** 28 large-cap tickers (AAPL, MSFT, AMZN, GOOGL, etc.), 2000–2025
- **Class Balancing:** Weighted loss to handle imbalanced reversal vs. non-reversal samples
- **Regularization:** Early stopping on validation loss

## Project Structure

```
├── a_utils.py                      # Data loading, feature computation, windowing
├── a_turning_point_train.py        # Model training pipeline
├── a_turning_point_oos.py          # Out-of-sample evaluation (SPY, BTC-USD)
├── a_turning_point_visualization.py # Prediction overlay on price charts
├── b_mean_reversion_oos.py         # Mean reversion backtesting strategy
├── requirements.txt
└── README.md
```

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Training

```bash
python a_turning_point_train.py
```

Trains URP and DRP models on the full training universe. Models are saved to `saved_models/`.

### Out-of-Sample Evaluation

```bash
python a_turning_point_oos.py
```

Evaluates trained models on held-out tickers (SPY, BTC-USD). Outputs confusion matrix and F1 score.

### Visualization

```bash
python a_turning_point_visualization.py
```

Generates price charts with predicted reversal points overlaid.

### Mean Reversion Backtest

```bash
python b_mean_reversion_oos.py
```

Runs a mean reversion trading strategy using detected turning points and evaluates performance.

## Results

The evaluation pipeline outputs:
- **Confusion matrices** for URP/DRP classification
- **F1 scores** (macro and per-class)
- **Price charts** with predicted turning points highlighted

## References

1. Chen, Y., & Hao, Y. (2022). "A novel framework for stock trading using reinforcement learning with candlestick patterns." *Expert Systems with Applications*, 210, 118484. [DOI: 10.1016/j.eswa.2022.118484](https://doi.org/10.1016/j.eswa.2022.118484)

2. Dong, X., et al. (2020). "A new stock price reversal point prediction method based on a recognition model of candlestick charts." *Chaos, Solitons & Fractals*, 130, 109413. [DOI: 10.1016/j.chaos.2019.109413](https://doi.org/10.1016/j.chaos.2019.109413)

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
