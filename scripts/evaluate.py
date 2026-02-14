#!/usr/bin/env python3
"""Evaluate trained models on out-of-sample tickers."""

import argparse
import logging
import yaml

from src.evaluation.evaluator import evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="OOS evaluation")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg)


if __name__ == "__main__":
    main()
