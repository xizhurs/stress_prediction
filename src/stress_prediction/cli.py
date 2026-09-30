"""Command-line interface for stress prediction workflows."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from stress_prediction import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="stress-prediction",
        description="Prepare data and train vegetation stress prediction models.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    commands = parser.add_subparsers(dest="command")
    train_parser = commands.add_parser(
        "train-lgb", help="Train and evaluate the LightGBM classifier."
    )
    train_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/drought_indices.csv"),
        help="Input CSV containing monthly climate and vegetation observations.",
    )
    train_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lightgbm"),
        help="Directory for the model bundle and metrics.",
    )
    train_parser.add_argument("--n-lags", type=int, default=12)
    train_parser.add_argument("--horizon", type=int, default=6)
    train_parser.add_argument("--validation-start", default="2016-01-01")
    train_parser.add_argument("--test-start", default="2019-01-01")
    train_parser.add_argument("--trials", type=int, default=30)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.add_argument("--threads", type=int, default=1)
    predict_parser = commands.add_parser(
        "predict", help="Run batch prediction with a trained model bundle."
    )
    predict_parser.add_argument("--artifact", type=Path, required=True)
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "train-lgb":
        from stress_prediction.training.lightgbm import TrainingConfig, train_lightgbm

        config = TrainingConfig(
            input_path=args.input,
            output_dir=args.output_dir,
            n_lags=args.n_lags,
            horizon=args.horizon,
            validation_start=args.validation_start,
            test_start=args.test_start,
            trials=args.trials,
            seed=args.seed,
            threads=args.threads,
        )
        train_lightgbm(config)
    elif args.command == "predict":
        from stress_prediction.prediction import predict_file

        predict_file(args.artifact, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
