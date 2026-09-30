from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from stress_prediction.training.lightgbm import TrainingConfig, train_lightgbm


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
        )
    )

    with artifact_path.open("rb") as source:
        artifact = pickle.load(source)  # noqa: S301
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    assert artifact["artifact_version"] == 1
    assert artifact["threshold"] >= 0
    assert artifact["feature_columns"]
    assert artifact["data_sha256"]
    assert set(metrics) == {"validation", "test"}
