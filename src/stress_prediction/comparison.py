"""Side-by-side threshold, confusion-matrix and PR figures for several runs."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from stress_prediction.training.lightgbm import (
    DEFAULT_RECALL_TARGET,
    F_BETA,
    _metrics,
    _select_threshold,
)

THRESHOLD_FIGURE = "threshold_val_models.png"
TEST_FIGURE = "test_results_models.png"
PRECISION_RECALL_FIGURE = "precision_recall_models.png"
METRICS_TABLE = "comparison_metrics.csv"
THRESHOLD_GRID = np.linspace(0.0, 1.0, 101)


def _load_split(run_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray]:
    path = run_dir / "predictions.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing predictions for {run_dir}: {path}")
    predictions = pd.read_csv(path)
    rows = predictions[predictions["split"] == split]
    if rows.empty:
        raise ValueError(f"{path} has no {split!r} rows")
    return rows["label"].to_numpy(), rows["score"].to_numpy()


def _load_runs(
    runs: Mapping[str, Path],
) -> dict[str, dict[str, tuple[np.ndarray, np.ndarray]]]:
    loaded = {
        name: {
            split: _load_split(Path(path), split) for split in ("validation", "test")
        }
        for name, path in runs.items()
    }
    reference_name = next(iter(loaded))
    for name, splits in loaded.items():
        for split, (labels, _) in splits.items():
            if not np.array_equal(labels, loaded[reference_name][split][0]):
                raise ValueError(
                    f"{name} and {reference_name} were not scored on the same "
                    f"{split} samples"
                )
    return loaded


def plot_model_comparison(
    runs: Mapping[str, Path],
    output_dir: Path,
    recall_target: float = DEFAULT_RECALL_TARGET,
) -> list[Path]:
    """Plot validation threshold curves and test confusion matrices per model.

    Each run directory needs a `predictions.csv` written by `train-lgb` or an
    `eval-*` command. The decision threshold is the most precise validation one
    reaching `recall_target`, so changing the target re-selects thresholds from
    the saved scores without re-running any model. Besides the figures it writes
    a table of every model's validation and test metrics at that threshold.
    """
    if not runs:
        raise ValueError("At least one run is required")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    loaded = _load_runs(runs)
    thresholds = {
        name: _select_threshold(
            pd.Series(splits["validation"][0]), splits["validation"][1], recall_target
        )
        for name, splits in loaded.items()
    }
    columns = 2 if len(runs) > 1 else 1
    rows = -(-len(runs) // columns)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / METRICS_TABLE
    pd.DataFrame(
        [
            {
                "model": name,
                "split": split,
                "recall_target": recall_target,
                "decision_threshold": thresholds[name],
                **_metrics(pd.Series(labels), scores, thresholds[name]),
            }
            for name, splits in loaded.items()
            for split, (labels, scores) in splits.items()
        ]
    ).to_csv(metrics_path, index=False)

    figure, axes = plt.subplots(
        rows, columns, figsize=(6.2 * columns, 4.6 * rows), sharex=True, sharey=True
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, (name, splits) in zip(flat_axes, loaded.items(), strict=False):
        labels, scores = splits["validation"]
        precision, recall, f2 = (
            np.array(
                [
                    metric(labels, scores >= threshold, zero_division=0)
                    for threshold in THRESHOLD_GRID
                ]
            )
            for metric in (
                precision_score,
                recall_score,
                partial(fbeta_score, beta=F_BETA),
            )
        )
        axis.plot(THRESHOLD_GRID, precision, label="Precision")
        axis.plot(THRESHOLD_GRID, recall, label="Recall")
        axis.plot(THRESHOLD_GRID, f2, label="F2-score")
        best = thresholds[name]
        axis.scatter(
            [best],
            [fbeta_score(labels, scores >= best, beta=F_BETA, zero_division=0)],
            color="red",
            zorder=3,
            label=f"Selected threshold ({best:.3f})",
        )
        axis.set(title=name, xlabel="Threshold", ylabel="Metric value")
        axis.set_xlim(-0.02, 1.02)
        axis.set_ylim(-0.02, 1.05)
        axis.legend(loc="lower left", fontsize=8)
    for axis in flat_axes[len(loaded) :]:
        axis.set_visible(False)
    figure.suptitle("Validation precision, recall and F2-score by threshold")
    figure.tight_layout()
    threshold_path = output_dir / THRESHOLD_FIGURE
    figure.savefig(threshold_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(
        rows, columns, figsize=(max(4.6 * columns, 7.0), 4.6 * rows)
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, (name, splits) in zip(flat_axes, loaded.items(), strict=False):
        labels, scores = splits["test"]
        predicted = scores >= thresholds[name]
        sns.heatmap(
            confusion_matrix(labels, predicted, labels=[0, 1]),
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["normal", "stress"],
            yticklabels=["normal", "stress"],
            ax=axis,
        )
        axis.set(
            title=(
                f"{name}\n"
                f"P {precision_score(labels, predicted, zero_division=0):.2f} | "
                f"R {recall_score(labels, predicted, zero_division=0):.2f} | "
                f"F2 {fbeta_score(labels, predicted, beta=F_BETA, zero_division=0):.2f}"
            ),
            xlabel="Predicted Labels",
            ylabel="True Labels",
        )
    for axis in flat_axes[len(loaded) :]:
        axis.set_visible(False)
    figure.suptitle("Test confusion matrices at the validation-selected threshold")
    figure.tight_layout()
    test_path = output_dir / TEST_FIGURE
    figure.savefig(test_path, dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12.5, 5.2), sharey=True)
    for axis, split in zip(axes, ("validation", "test"), strict=True):
        for name, splits in loaded.items():
            labels, scores = splits[split]
            precision, recall, _ = precision_recall_curve(labels, scores)
            (line,) = axis.plot(
                recall,
                precision,
                linewidth=2,
                label=f"{name} (AP={average_precision_score(labels, scores):.3f})",
            )
            selected = scores >= thresholds[name]
            axis.scatter(
                [recall_score(labels, selected)],
                [precision_score(labels, selected, zero_division=0)],
                color=line.get_color(),
                edgecolor="black",
                zorder=3,
            )
        base_rate = float(np.mean(splits[split][0]))
        axis.axhline(
            base_rate,
            color="grey",
            linestyle="--",
            linewidth=1,
            label=f"No skill ({base_rate:.2f})",
        )
        axis.set(title=split.capitalize(), xlabel="Recall", ylabel="Precision")
        axis.set_xlim(0, 1.01)
        axis.set_ylim(0, 1.02)
        axis.grid(alpha=0.2)
        axis.legend(loc="lower left")
    figure.suptitle("Precision-recall curves (markers: validation-selected threshold)")
    figure.tight_layout()
    precision_recall_path = output_dir / PRECISION_RECALL_FIGURE
    figure.savefig(precision_recall_path, dpi=160)
    plt.close(figure)
    return [threshold_path, test_path, precision_recall_path, metrics_path]
