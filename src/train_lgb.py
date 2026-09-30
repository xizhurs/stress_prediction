"""Compatibility launcher for the packaged LightGBM training command."""

from __future__ import annotations

import sys

from stress_prediction.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["train-lgb", *sys.argv[1:]]))
