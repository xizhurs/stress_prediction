from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from stress_prediction.features import prediction_feature_extraction
from stress_prediction.prediction import predict_file


def _monthly_data(periods: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2020-01-01", periods=periods, freq="MS"),
            "latitude": 52.0,
            "longitude": 5.0,
            "tp_mm": np.arange(periods, dtype=float),
            "pet_mm": np.arange(periods, dtype=float) + 1,
            "T_c": np.arange(periods, dtype=float) + 2,
            "ndvi": np.arange(periods, dtype=float) + 3,
        }
    )


def test_prediction_artifact_round_trip(tmp_path: Path) -> None:
    observations = _monthly_data(periods=8)
    features = prediction_feature_extraction(observations, n_lags=2, horizon=3)
    model_columns = [
        column for column in features if column not in {"valid_time", "target_time"}
    ]
    classifier = DummyClassifier(strategy="prior").fit(
        features[model_columns], [0, 1, 0, 1, 0, 1]
    )
    artifact = {
        "artifact_version": 1,
        "model": classifier,
        "threshold": 0.4,
        "feature_columns": model_columns,
        "config": {"n_lags": 2, "horizon": 3},
    }
    artifact_path = tmp_path / "model.pkl"
    with artifact_path.open("wb") as destination:
        pickle.dump(artifact, destination)
    input_path = tmp_path / "input.csv"
    observations.to_csv(input_path, index=False)

    output_path = predict_file(artifact_path, input_path, tmp_path / "output.csv")

    output = output_path.read_text(encoding="utf-8")
    assert "stress_probability" in output
    assert "predicted_stress" in output
