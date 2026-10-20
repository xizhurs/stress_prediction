from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_prediction.cli import main
from stress_prediction.comparison import (
    TEST_FIGURE,
    THRESHOLD_FIGURE,
    plot_model_comparison,
)


def _write_run(directory: Path, labels: np.ndarray, noise: float, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    directory.mkdir()
    frames = []
    for split, count in (("validation", 200), ("test", 300)):
        split_labels = labels[:count]
        score = np.clip(split_labels * 0.5 + rng.normal(0.25, noise, count), 0, 1)
        frames.append(
            pd.DataFrame({"split": split, "label": split_labels, "score": score})
        )
    pd.concat(frames).to_csv(directory / "predictions.csv", index=False)
    return directory


def test_plot_model_comparison_writes_both_figures(tmp_path: Path) -> None:
    labels = np.random.default_rng(0).integers(0, 2, 300)
    runs = {
        "Model A": _write_run(tmp_path / "a", labels, 0.1, 1),
        "Model B": _write_run(tmp_path / "b", labels, 0.3, 2),
        "Model C": _write_run(tmp_path / "c", labels, 0.2, 3),
    }

    paths = plot_model_comparison(runs, tmp_path / "figures")

    assert [path.name for path in paths] == [THRESHOLD_FIGURE, TEST_FIGURE]
    assert all(path.stat().st_size > 0 for path in paths)


def test_plot_model_comparison_rejects_different_samples(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    runs = {
        "Model A": _write_run(tmp_path / "a", rng.integers(0, 2, 300), 0.1, 1),
        "Model B": _write_run(tmp_path / "b", rng.integers(0, 2, 300), 0.1, 2),
    }

    with pytest.raises(ValueError, match="same validation samples"):
        plot_model_comparison(runs, tmp_path / "figures")


def test_compare_models_requires_predictions(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()

    with pytest.raises(FileNotFoundError, match="predictions"):
        plot_model_comparison({"Model": tmp_path / "empty"}, tmp_path / "figures")


def test_compare_models_cli_rejects_malformed_run(capsys: object) -> None:
    with pytest.raises(SystemExit) as error:
        main(["compare-models", "--run", "missing-separator"])
    assert error.value.code == 2
    assert "NAME=DIR" in capsys.readouterr().err  # type: ignore[attr-defined]
