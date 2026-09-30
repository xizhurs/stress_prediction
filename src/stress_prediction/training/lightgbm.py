"""End-to-end LightGBM training workflow."""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import lightgbm
import numpy as np
import optuna
import pandas as pd
import sklearn
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from stress_prediction import __version__
from stress_prediction.data import split_data
from stress_prediction.features import DEFAULT_FEATURES, feature_extraction
from stress_prediction.models import fit_tuned_classifier

LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "vegetation_stress_class"
LABEL_MAPPING = {"normal": 0, "mild": 0, "moderate": 0, "severe": 1}


@dataclass(frozen=True)
class TrainingConfig:
    input_path: Path
    output_dir: Path
    n_lags: int = 12
    horizon: int = 6
    validation_start: str = "2016-01-01"
    test_start: str = "2019-01-01"
    trials: int = 30
    seed: int = 42
    threads: int = 1


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_data(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Training input does not exist: {path}")
    frame = pd.read_csv(path, parse_dates=["valid_time"])
    required = {
        "valid_time",
        "latitude",
        "longitude",
        TARGET_COLUMN,
        *DEFAULT_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    if frame.duplicated(["latitude", "longitude", "valid_time"]).any():
        raise ValueError("Duplicate spatial observations were found")
    unknown_labels = sorted(set(frame[TARGET_COLUMN].dropna()) - LABEL_MAPPING.keys())
    if unknown_labels:
        raise ValueError(f"Unknown stress labels: {', '.join(unknown_labels)}")
    if frame[list(DEFAULT_FEATURES)].isna().any().any():
        raise ValueError("Training features contain missing values")
    return frame


def _select_threshold(target: pd.Series, probabilities: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probabilities)
    if not len(thresholds):
        raise ValueError("Validation data cannot produce a decision threshold")
    scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    return float(thresholds[int(np.argmax(scores))])


def _metrics(
    target: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    return {
        "average_precision": float(average_precision_score(target, probabilities)),
        "roc_auc": float(roc_auc_score(target, probabilities)),
        "precision": float(precision_score(target, predictions, zero_division=0)),
        "recall": float(recall_score(target, predictions, zero_division=0)),
        "f1": float(f1_score(target, predictions, zero_division=0)),
    }


def train_lightgbm(config: TrainingConfig) -> Path:
    """Train, evaluate, and persist a complete LightGBM inference bundle."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    np.random.seed(config.seed)
    frame = _load_data(config.input_path)
    features, labels = feature_extraction(
        frame,
        n_lags=config.n_lags,
        horizon=config.horizon,
        target_column=TARGET_COLUMN,
    )
    binary_target = labels.map(LABEL_MAPPING)
    if binary_target.isna().any():
        raise ValueError("Target mapping produced missing values")
    binary_target = binary_target.astype(int)
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_data(
        features,
        binary_target,
        target_column=TARGET_COLUMN,
        validation_start=config.validation_start,
        test_start=config.test_start,
    )
    for name, values in (
        ("training", train_y),
        ("validation", validation_y),
        ("test", test_y),
    ):
        if values.nunique() < 2:
            raise ValueError(f"The {name} partition must contain both classes")

    LOGGER.info(
        "Training with %d/%d/%d train/validation/test samples",
        len(train_x),
        len(validation_x),
        len(test_x),
    )
    classifier, tuning = fit_tuned_classifier(
        train_x,
        train_y,
        validation_x,
        validation_y,
        trials=config.trials,
        seed=config.seed,
        threads=config.threads,
    )
    validation_probabilities = np.asarray(classifier.predict_proba(validation_x))[:, 1]
    threshold = _select_threshold(validation_y, validation_probabilities)
    results = {
        "validation": _metrics(validation_y, validation_probabilities, threshold),
        "test": _metrics(
            test_y,
            np.asarray(classifier.predict_proba(test_x))[:, 1],
            threshold,
        ),
    }

    config.output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = config.output_dir / "model.pkl"
    serializable_config = {
        **asdict(config),
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
    }
    artifact: dict[str, Any] = {
        "artifact_version": 1,
        "package_version": __version__,
        "created_at": datetime.now(UTC).isoformat(),
        "model": classifier,
        "threshold": threshold,
        "feature_columns": list(train_x.columns),
        "label_mapping": LABEL_MAPPING,
        "config": serializable_config,
        "data_sha256": _file_digest(config.input_path),
        "tuning": tuning,
        "metrics": results,
        "library_versions": {
            "lightgbm": lightgbm.__version__,
            "numpy": np.__version__,
            "optuna": optuna.__version__,
            "pandas": pd.__version__,
            "scikit-learn": sklearn.__version__,
        },
    }
    with artifact_path.open("wb") as destination:
        pickle.dump(artifact, destination)
    (config.output_dir / "metrics.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    LOGGER.info("Saved model bundle to %s", artifact_path)
    return artifact_path
