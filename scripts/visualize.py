#!/usr/bin/env python3
"""Generate prediction overlay plots."""

import argparse
import logging
import yaml

from src.visualization.plots import visualize

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize predictions")
    parser.add_argument("--config", default="config/default.yaml", help="Path to config YAML")
    parser.add_argument("--year", type=int, default=2024, help="Year to plot")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    visualize(cfg, year=args.year)


if __name__ == "__main__":
    main()
