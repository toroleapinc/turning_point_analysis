#!/usr/bin/env python3
"""Run mean-reversion backtest."""

import argparse
import logging
import yaml

from src.backtest.mean_reversion import run_backtest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Mean-reversion backtest")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    trades_df = run_backtest(cfg)
    if not trades_df.empty:
        trades_df.to_csv("backtest_results.csv", index=False)
        print(f"Saved {len(trades_df)} trades to backtest_results.csv")


if __name__ == "__main__":
    main()
