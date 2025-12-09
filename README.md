# Turning Point Analysis 📈

**LSTM-based stock market turning point detection using technical indicators.**

Predicts regime changes (turning points) in equity price series using a sliding-window LSTM classifier trained on 28 large-cap US stocks.

## Overview

This project implements the approach from recent research on market turning point detection:

1. **Feature Engineering** — 27 technical indicators (momentum, volatility, volume-based) computed from OHLCV data
2. **Sliding Window LSTM** — Binary classifier trained on windowed feature sequences to predict reversal points
3. **Out-of-Sample Testing** — Evaluation on held-out tickers with visualization of predicted vs actual turning points
4. **Mean Reversion Strategy** — Backtesting a mean reversion strategy guided by turning point signals

## Project Structure

| File | Description |
|------|-------------|
| `a_utils.py` | Data loading (via yfinance), feature engineering, sliding window creation |
| `a_turning_point_train.py` | LSTM model training with class balancing and early stopping |
| `a_turning_point_oos.py` | Out-of-sample evaluation and metrics |
| `a_turning_point_visualization.py` | Plotting predicted turning points against price series |
| `b_mean_reversion_oos.py` | Mean reversion strategy backtesting |

## Tech Stack

- **Deep Learning:** Keras/TensorFlow (LSTM)
- **Data:** yfinance, pandas, NumPy
- **ML:** scikit-learn (preprocessing, evaluation)
- **Visualization:** matplotlib

## Quick Start

```bash
pip install -r requirements.txt
python a_turning_point_train.py    # Train the model
python a_turning_point_oos.py      # Evaluate out-of-sample
python a_turning_point_visualization.py  # Visualize results
```

## References

Includes two research papers in the repo that inspired this approach (see PDFs).
