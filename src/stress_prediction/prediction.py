"""Batch prediction from a trusted LightGBM model bundle."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stress_prediction.features import prediction_feature_extraction


def load_artifact(path: Path) -> dict[str, Any]:
    """Load and minimally validate a trusted local model bundle."""
    if not path.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {path}")
    with path.open("rb") as source:
        artifact = pickle.load(source)  # noqa: S301
    if not isinstance(artifact, dict) or artifact.get("artifact_version") != 1:
        raise ValueError("Unsupported or invalid model artifact")
    required = {"model", "threshold", "feature_columns", "config"}
    missing = sorted(required.difference(artifact))
    if missing:
        raise ValueError(f"Model artifact is missing: {', '.join(missing)}")
    return artifact


def predict_file(artifact_path: Path, input_path: Path, output_path: Path) -> Path:
    """Generate timestamped stress probabilities from a CSV observation file."""
    if not input_path.is_file():
        raise FileNotFoundError(f"Prediction input does not exist: {input_path}")
    artifact = load_artifact(artifact_path)
    observations = pd.read_csv(input_path, parse_dates=["valid_time"])
    config = artifact["config"]
    features = prediction_feature_extraction(
        observations,
        n_lags=int(config["n_lags"]),
        horizon=int(config["horizon"]),
    )
    expected_columns = list(artifact["feature_columns"])
    missing = sorted(set(expected_columns).difference(features.columns))
    if missing:
        raise ValueError(f"Prediction features are missing: {', '.join(missing)}")

    probabilities = np.asarray(
        artifact["model"].predict_proba(features[expected_columns])
    )[:, 1]
    output = features[["latitude", "longitude", "valid_time", "target_time"]].copy()
    output["stress_probability"] = probabilities
    output["predicted_stress"] = (probabilities >= float(artifact["threshold"])).astype(
        int
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix.lower() == ".parquet":
        output.to_parquet(output_path, index=False)
    elif output_path.suffix.lower() == ".csv":
        output.to_csv(output_path, index=False)
    else:
        raise ValueError("Prediction output must use a .csv or .parquet extension")
    return output_path
