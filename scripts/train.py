#!/usr/bin/env python3
"""Train URP/DRP turning-point models."""

import argparse
import logging
import yaml

from src.training.trainer import train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train turning-point models")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg)


if __name__ == "__main__":
    main()
