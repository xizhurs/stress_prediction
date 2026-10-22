from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stress_prediction.training.lightgbm import (
    TrainingConfig,
    _select_threshold,
    train_lightgbm,
)


def test_train_lightgbm_writes_complete_artifact(tmp_path: Path) -> None:
    periods = 108
    month = np.arange(periods)
    frame = pd.DataFrame(
        {
            "valid_time": pd.date_range("2010-01-01", periods=periods, freq="MS"),
            "latitude": 52.0,
            "longitude": 5.0,
            "vegetation_stress_class": np.where(month % 3 == 0, "severe", "normal"),
            "tp_mm": 50 + np.sin(month / 3),
            "pet_mm": 30 + np.cos(month / 4),
            "T_c": 10 + np.sin(month / 6),
            "ndvi": 0.5 + 0.2 * np.cos(month / 5),
        }
    )
    input_path = tmp_path / "training.csv"
    frame.to_csv(input_path, index=False)
    output_dir = tmp_path / "run"

    artifact_path = train_lightgbm(
        TrainingConfig(
            input_path=input_path,
            output_dir=output_dir,
            n_lags=2,
            horizon=1,
            validation_start="2013-01-01",
            test_start="2016-01-01",
            trials=1,
            seed=7,
            threads=1,
            spatial_block_size=None,
        )
    )

    with artifact_path.open("rb") as source:
        artifact = pickle.load(source)  # noqa: S301
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert artifact["artifact_version"] == 1
    assert artifact["threshold"] >= 0
    assert artifact["feature_columns"]
    assert artifact["data_sha256"]
    assert set(metrics) == {
        "decision_threshold",
        "recall_target",
        "split_samples",
        "validation",
        "test",
    }
    assert metrics["split_samples"] == {
        "training": 33,
        "validation": 36,
        "test": 36,
    }
    for figure_name in (
        "precision_recall.png",
        "confusion_matrix.png",
        "feature_importance.png",
    ):
        assert (output_dir / figure_name).stat().st_size > 0
    predictions = pd.read_csv(output_dir / "predictions.csv")
    assert predictions["split"].value_counts().to_dict() == {
        "validation": 36,
        "test": 36,
    }
    assert predictions["score"].between(0, 1).all()


def test_select_threshold_is_most_precise_threshold_reaching_recall_target() -> None:
    target = pd.Series([1, 1, 0, 1, 0, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2])

    # Recall of 0.5 is first reached at 0.8 (2 of 4), where precision is 1.0.
    assert _select_threshold(target, scores, 0.5) == 0.8
    # Full recall needs every positive, so the lowest positive score is the limit.
    assert _select_threshold(target, scores, 1.0) == 0.3
    # A lower threshold with equal precision recalls more, so ties go lowest.
    tied = pd.Series([1, 1, 1, 0])
    assert _select_threshold(tied, np.array([0.9, 0.8, 0.7, 0.1]), 0.5) == 0.7


def test_select_threshold_rejects_invalid_recall_targets() -> None:
    target = pd.Series([1, 0, 1, 0])
    scores = np.array([0.9, 0.8, 0.7, 0.1])
    for recall_target in (0.0, 1.5):
        with pytest.raises(ValueError, match="recall_target"):
            _select_threshold(target, scores, recall_target)
