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
    train_parser.add_argument(
        "--recall-target",
        type=float,
        default=0.8,
        help="Decision threshold: the most precise validation threshold whose "
        "recall reaches this value.",
    )
    train_parser.add_argument(
        "--spatial-block-size",
        type=float,
        default=0.5,
        help="Degrees per held-out spatial block (70/15/15 train/validation/test "
        "blocks, on top of the time cut); 0 for a time-only split.",
    )
    for command, model, checkpoint in (
        ("eval-timestral", "Timestral", "mistralai/Timestral-28M"),
        ("eval-chronos2", "Chronos-2", "amazon/chronos-2"),
    ):
        eval_parser = commands.add_parser(
            command,
            help=f"Zero-shot evaluate {model} on the same splits and metrics.",
        )
        eval_parser.add_argument(
            "--input",
            type=Path,
            default=Path("data/drought_indices.csv"),
            help="Input CSV containing monthly climate and vegetation observations.",
        )
        eval_parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path(f"experiments/{command.removeprefix('eval-')}"),
            help="Directory for the metrics and figures.",
        )
        eval_parser.add_argument("--checkpoint", default=checkpoint)
        eval_parser.add_argument("--n-lags", type=int, default=12)
        eval_parser.add_argument("--horizon", type=int, default=6)
        eval_parser.add_argument("--validation-start", default="2016-01-01")
        eval_parser.add_argument("--test-start", default="2019-01-01")
        eval_parser.add_argument("--batch-size", type=int, default=512)
        eval_parser.add_argument("--device", default="cpu")
        eval_parser.add_argument(
            "--recall-target",
            type=float,
            default=0.8,
            help="Decision threshold: the most precise validation threshold whose "
            "recall reaches this value.",
        )
        eval_parser.add_argument(
            "--spatial-block-size",
            type=float,
            default=0.5,
            help="Degrees per held-out spatial block (70/15/15 train/validation/test "
            "blocks, on top of the time cut); 0 for a time-only split.",
        )
        eval_parser.add_argument(
            "--max-samples",
            type=int,
            default=None,
            help="Evaluate a fixed random subset of each split, for quick runs.",
        )
    stacked_parser = commands.add_parser(
        "train-lgb-stacked",
        help="Compare LightGBM with and without zero-shot forecast features.",
    )
    stacked_parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/drought_indices.csv"),
        help="Input CSV containing monthly climate and vegetation observations.",
    )
    stacked_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/lightgbm_stacked"),
        help="Directory for cached forecasts, per-variant results and metrics.",
    )
    stacked_parser.add_argument(
        "--model", choices=("chronos2", "timestral"), default="chronos2"
    )
    stacked_parser.add_argument(
        "--checkpoint",
        default=None,
        help="Defaults to amazon/chronos-2 or mistralai/Timestral-28M per --model.",
    )
    stacked_parser.add_argument("--n-lags", type=int, default=12)
    stacked_parser.add_argument("--horizon", type=int, default=6)
    stacked_parser.add_argument("--validation-start", default="2016-01-01")
    stacked_parser.add_argument("--test-start", default="2019-01-01")
    stacked_parser.add_argument("--batch-size", type=int, default=512)
    stacked_parser.add_argument("--device", default="cpu")
    stacked_parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50_000,
        help="Random training subset to forecast; Chronos-2 on CPU is the slow step.",
    )
    stacked_parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Evaluate a fixed random subset of validation and test, for quick runs.",
    )
    stacked_parser.add_argument("--trials", type=int, default=30)
    stacked_parser.add_argument("--seed", type=int, default=42)
    stacked_parser.add_argument("--threads", type=int, default=1)
    compare_parser = commands.add_parser(
        "compare-models",
        help="Plot validation thresholds and test confusion matrices per model.",
    )
    compare_parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="NAME=DIR",
        help="A run directory containing predictions.csv; repeat for each model.",
    )
    compare_parser.add_argument(
        "--recall-target",
        type=float,
        default=0.8,
        help="Decision threshold: the most precise validation threshold whose "
        "recall reaches this value.",
    )
    compare_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/figures"),
        help="Directory for the comparison figures.",
    )
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
            spatial_block_size=args.spatial_block_size or None,
            recall_target=args.recall_target,
        )
        train_lightgbm(config)
    elif args.command in {"eval-timestral", "eval-chronos2"}:
        options = {
            "input_path": args.input,
            "output_dir": args.output_dir,
            "checkpoint": args.checkpoint,
            "n_lags": args.n_lags,
            "horizon": args.horizon,
            "validation_start": args.validation_start,
            "test_start": args.test_start,
            "batch_size": args.batch_size,
            "device": args.device,
            "max_samples": args.max_samples,
            "spatial_block_size": args.spatial_block_size or None,
            "recall_target": args.recall_target,
        }
        if args.command == "eval-timestral":
            from stress_prediction.training.timestral import (
                TimestralConfig,
                evaluate_timestral,
            )

            evaluate_timestral(TimestralConfig(**options))
        else:
            from stress_prediction.training.chronos2 import (
                Chronos2Config,
                evaluate_chronos2,
            )

            evaluate_chronos2(Chronos2Config(**options))
    elif args.command == "train-lgb-stacked":
        from stress_prediction.training.stacked import (
            StackedConfig,
            train_lightgbm_stacked,
        )

        checkpoint = args.checkpoint or (
            "amazon/chronos-2"
            if args.model == "chronos2"
            else "mistralai/Timestral-28M"
        )
        train_lightgbm_stacked(
            StackedConfig(
                input_path=args.input,
                output_dir=args.output_dir,
                model=args.model,
                checkpoint=checkpoint,
                n_lags=args.n_lags,
                horizon=args.horizon,
                validation_start=args.validation_start,
                test_start=args.test_start,
                batch_size=args.batch_size,
                device=args.device,
                max_samples=args.max_samples,
                max_train_samples=args.max_train_samples,
                trials=args.trials,
                seed=args.seed,
                threads=args.threads,
            )
        )
    elif args.command == "compare-models":
        from stress_prediction.comparison import plot_model_comparison

        runs: dict[str, Path] = {}
        for item in args.run:
            name, separator, directory = item.partition("=")
            if not separator or not name or not directory:
                parser.error(f"--run expects NAME=DIR, got {item!r}")
            runs[name] = Path(directory)
        for path in plot_model_comparison(runs, args.output_dir, args.recall_target):
            print(path)
    elif args.command == "predict":
        from stress_prediction.prediction import predict_file

        predict_file(args.artifact, args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
