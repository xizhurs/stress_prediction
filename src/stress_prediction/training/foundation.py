"""Zero-shot evaluation of forecasting foundation models on the stress task.

A model is reduced to one callable that forecasts NDVI quantiles. Everything
else mirrors the LightGBM workflow: the same samples, temporal splits,
validation-selected threshold, metrics and figures.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from stress_prediction.features import DEFAULT_FEATURES, feature_extraction
from stress_prediction.training.lightgbm import (
    LABEL_MAPPING,
    TARGET_COLUMN,
    _file_digest,
    _load_data,
    _metrics,
    _save_classification_figures,
    _select_threshold,
)

LOGGER = logging.getLogger(__name__)
VCI_EPSILON = 1e-6
COVARIATE_COLUMNS = tuple(column for column in DEFAULT_FEATURES if column != "ndvi")

# (ndvi (B, T), past covariates (B, 3, T), known covariates (B, 2, T + P), P)
# -> (quantiles (B, Q) at the final forecast step, quantile levels (Q,))
QuantileForecaster = Callable[
    [np.ndarray, np.ndarray, np.ndarray, int], tuple[np.ndarray, np.ndarray]
]


@dataclass(frozen=True)
class FoundationConfig:
    input_path: Path
    output_dir: Path
    checkpoint: str
    n_lags: int = 12
    horizon: int = 6
    validation_start: str = "2016-01-01"
    test_start: str = "2019-01-01"
    baseline_start_year: int = 2003
    baseline_end_year: int = 2020
    severe_vci: float = 0.25
    batch_size: int = 512
    device: str = "cpu"
    max_samples: int | None = None


def to_numpy(values: Any) -> np.ndarray:
    if hasattr(values, "detach"):
        values = values.detach().cpu().numpy()
    return np.asarray(values, dtype=np.float64)


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _ndvi_climatology(
    frame: pd.DataFrame, start_year: int, end_year: int
) -> pd.DataFrame:
    """Per-pixel monthly NDVI extremes, as used to derive the stress labels."""
    baseline = frame[frame["valid_time"].dt.year.between(start_year, end_year)]
    climatology: pd.DataFrame = (
        baseline.assign(month=baseline["valid_time"].dt.month)
        .groupby(["latitude", "longitude", "month"])["ndvi"]
        .agg(ndvi_min="min", ndvi_max="max")
        .reset_index()
    )
    return climatology


def _climatology_at(
    samples: pd.DataFrame, climatology: pd.DataFrame, month_offset: int
) -> pd.DataFrame:
    """Look up the NDVI extremes for each sample's month shifted by an offset."""
    keys = pd.DataFrame(
        {
            "latitude": samples["latitude"].to_numpy(),
            "longitude": samples["longitude"].to_numpy(),
            "month": ((samples["valid_time"].dt.month - 1 + month_offset) % 12 + 1)
            .astype(int)
            .to_numpy(),
        }
    )
    merged = keys.merge(climatology, on=["latitude", "longitude", "month"], how="left")
    if merged[["ndvi_min", "ndvi_max"]].isna().any().any():
        raise ValueError("NDVI climatology is missing for some samples")
    return merged


def _severe_ndvi_limit(extremes: pd.DataFrame, severe_vci: float) -> np.ndarray:
    span = extremes["ndvi_max"] - extremes["ndvi_min"] + VCI_EPSILON
    return (extremes["ndvi_min"] + severe_vci * span).to_numpy()


def _vci(ndvi: np.ndarray, extremes: pd.DataFrame) -> np.ndarray:
    span = (extremes["ndvi_max"] - extremes["ndvi_min"] + VCI_EPSILON).to_numpy()
    index: np.ndarray = (ndvi - extremes["ndvi_min"].to_numpy()) / span
    clipped: np.ndarray = np.clip(index, 0.0, 1.0)
    return clipped


def _label_rule_agreement(
    frame: pd.DataFrame, climatology: pd.DataFrame, severe_vci: float
) -> float:
    """Share of labels reproduced by thresholding NDVI-derived VCI."""
    extremes = _climatology_at(frame, climatology, 0)
    predicted = _vci(frame["ndvi"].to_numpy(), extremes) <= severe_vci
    actual = frame[TARGET_COLUMN].map(LABEL_MAPPING).astype(int).to_numpy() == 1
    return float(np.mean(predicted == actual))


def _severe_probability(
    quantiles: np.ndarray, levels: np.ndarray, limit: np.ndarray
) -> np.ndarray:
    """P(forecast NDVI <= limit) by interpolating the forecast quantile function."""
    ordered = np.sort(quantiles, axis=1)
    count = ordered.shape[1]
    above = (ordered <= limit[:, None]).sum(axis=1)
    rows = np.arange(len(ordered))
    low = np.clip(above - 1, 0, count - 1)
    high = np.clip(above, 0, count - 1)
    low_value, high_value = ordered[rows, low], ordered[rows, high]
    span = high_value - low_value
    weight = np.divide(limit - low_value, span, out=np.zeros_like(span), where=span > 0)
    probability = levels[low] + weight * (levels[high] - levels[low])
    probability = np.where(above == 0, 0.0, probability)
    return np.where(above == count, 1.0, probability)


def _forecast_ndvi_quantiles(
    forecast: QuantileForecaster,
    samples: pd.DataFrame,
    config: FoundationConfig,
    description: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Forecast NDVI quantiles at the label month for every sample."""
    lags = range(config.n_lags, 0, -1)
    ndvi = samples[[f"ndvi_lag{lag}" for lag in lags]].to_numpy(dtype=np.float32)
    covariates = np.stack(
        [
            samples[[f"{column}_lag{lag}" for lag in lags]].to_numpy(dtype=np.float32)
            for column in COVARIATE_COLUMNS
        ],
        axis=1,
    )
    # The context ends one month before the sample month, as in the lag features.
    steps = np.arange(-config.n_lags, config.horizon + 1)
    month = samples["valid_time"].dt.month.to_numpy()[:, None] - 1 + steps[None, :]
    angle = 2 * np.pi * (month % 12 + 1) / 12
    known = np.stack([np.sin(angle), np.cos(angle)], axis=1).astype(np.float32)

    batches: list[np.ndarray] = []
    levels = np.empty(0)
    for start in tqdm(range(0, len(samples), config.batch_size), desc=description):
        batch = slice(start, start + config.batch_size)
        quantiles, levels = forecast(
            ndvi[batch], covariates[batch], known[batch], config.horizon + 1
        )
        batches.append(quantiles)
    return np.concatenate(batches), levels


def evaluate_foundation(
    config: FoundationConfig,
    forecast: QuantileForecaster,
    *,
    name: str,
    model_info: dict[str, Any],
) -> Path:
    """Zero-shot evaluate a forecaster with the LightGBM splits and metrics."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    frame = _load_data(config.input_path)
    climatology = _ndvi_climatology(
        frame, config.baseline_start_year, config.baseline_end_year
    )
    agreement = _label_rule_agreement(frame, climatology, config.severe_vci)
    LOGGER.info("NDVI/VCI rule reproduces %.2f%% of the labels", 100 * agreement)

    features, labels = feature_extraction(
        frame,
        n_lags=config.n_lags,
        horizon=config.horizon,
        target_column=TARGET_COLUMN,
    )
    binary_target = labels.map(LABEL_MAPPING).astype(int)
    target_time = pd.to_datetime(features["target_time"])
    validation_cutoff = pd.Timestamp(config.validation_start)
    test_cutoff = pd.Timestamp(config.test_start)
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start must be earlier than test_start")
    masks = {
        "validation": (
            (target_time >= validation_cutoff) & (target_time < test_cutoff)
        ),
        "test": target_time >= test_cutoff,
    }

    scores: dict[str, dict[str, np.ndarray]] = {}
    targets: dict[str, pd.Series] = {}
    for split, mask in masks.items():
        samples = features.loc[mask].reset_index(drop=True)
        target = binary_target.loc[mask].reset_index(drop=True)
        if config.max_samples is not None and len(samples) > config.max_samples:
            keep = np.sort(
                np.random.default_rng(0).choice(
                    len(samples), config.max_samples, replace=False
                )
            )
            samples, target = samples.iloc[keep], target.iloc[keep]
            samples, target = (
                samples.reset_index(drop=True),
                target.reset_index(drop=True),
            )
        if target.nunique() < 2:
            raise ValueError(f"The {split} partition must contain both classes")
        quantiles, levels = _forecast_ndvi_quantiles(
            forecast, samples, config, f"{name} {split}"
        )
        limit = _severe_ndvi_limit(
            _climatology_at(samples, climatology, config.horizon), config.severe_vci
        )
        last_extremes = _climatology_at(samples, climatology, -1)
        last_ndvi = samples["ndvi_lag1"].to_numpy()
        targets[split] = target
        scores[split] = {
            "model": _severe_probability(quantiles, levels, limit),
            "persistence": 1.0 - _vci(last_ndvi, last_extremes),
        }

    results: dict[str, Any] = {
        "model": {
            "checkpoint": config.checkpoint,
            **model_info,
            "forecast": "zero-shot NDVI quantiles with tp/pet/T_c past covariates",
        },
        "label_rule_agreement": agreement,
        "split_samples": {split: len(target) for split, target in targets.items()},
    }
    selected_threshold = 0.0
    for method in ("model", "persistence"):
        threshold = _select_threshold(
            targets["validation"], scores["validation"][method]
        )
        section = {
            "decision_threshold": threshold,
            "validation": _metrics(
                targets["validation"], scores["validation"][method], threshold
            ),
            "test": _metrics(targets["test"], scores["test"][method], threshold),
        }
        if method == "model":
            results.update(section)
            selected_threshold = threshold
        else:
            results["persistence_baseline"] = section

    config.output_dir.mkdir(parents=True, exist_ok=True)
    pd.concat(
        [
            pd.DataFrame(
                {
                    "split": split,
                    "label": targets[split].to_numpy(),
                    "score": scores[split]["model"],
                    "persistence_score": scores[split]["persistence"],
                }
            )
            for split in targets
        ],
        ignore_index=True,
    ).to_csv(config.output_dir / "predictions.csv", index=False)
    _save_classification_figures(
        config.output_dir,
        validation_target=targets["validation"],
        validation_probabilities=scores["validation"]["model"],
        test_target=targets["test"],
        test_probabilities=scores["test"]["model"],
        threshold=selected_threshold,
    )
    results["config"] = {
        **asdict(config),
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
    }
    results["data_sha256"] = _file_digest(config.input_path)
    results["created_at"] = datetime.now(UTC).isoformat()
    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved %s evaluation to %s", name, metrics_path)
    return metrics_path
