"""LightGBM stress classifier stacked on zero-shot foundation-model forecasts.

The forecaster (Chronos-2 or Timestral) is never fitted, so its forecasts are
valid features for every split without out-of-fold handling. They are the slow
part, so each split's raw quantiles are cached and the training split can be
randomly subsampled.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stress_prediction.data import split_data
from stress_prediction.features import feature_extraction
from stress_prediction.models import fit_tuned_classifier
from stress_prediction.training.chronos2 import _load_pipeline, chronos2_forecaster
from stress_prediction.training.foundation import (
    FoundationConfig,
    QuantileForecaster,
    _climatology_at,
    _forecast_ndvi_quantiles,
    _ndvi_climatology,
    _severe_ndvi_limit,
    _severe_probability,
)
from stress_prediction.training.lightgbm import (
    LABEL_MAPPING,
    TARGET_COLUMN,
    _file_digest,
    _load_data,
    _metrics,
    _save_evaluation_figures,
    _select_threshold,
    _write_predictions,
)
from stress_prediction.training.timestral import _load_forecaster, timestral_forecaster

LOGGER = logging.getLogger(__name__)
CLIMATOLOGY_COLUMNS = ("ndvi_min_target", "ndvi_max_target")
QUANTILE_LEVELS = {"q10": 0.1, "q50": 0.5, "q90": 0.9}
# Feature, cache and variant names per forecasting model.
PREFIXES = {"chronos2": "chronos", "timestral": "timestral"}


@dataclass(frozen=True)
class StackedConfig(FoundationConfig):
    checkpoint: str = "amazon/chronos-2"
    model: str = "chronos2"
    trials: int = 30
    seed: int = 42
    threads: int = 1
    max_train_samples: int | None = 50_000


def _subsample(index: pd.Index, limit: int | None) -> pd.Index:
    if limit is None or len(index) <= limit:
        return index
    keep = np.sort(np.random.default_rng(0).choice(len(index), limit, replace=False))
    return index[keep]


def _cached_quantiles(
    path: Path,
    samples: pd.DataFrame,
    signature: str,
    compute: Callable[[], tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Reuse saved forecasts only when they were made for exactly these samples."""
    keys = {
        "latitude": samples["latitude"].to_numpy(),
        "longitude": samples["longitude"].to_numpy(),
        "valid_time": samples["valid_time"].to_numpy("datetime64[ns]").astype(np.int64),
    }
    if path.is_file():
        with np.load(path) as cached:
            if str(cached["signature"]) == signature and all(
                np.array_equal(cached[name], values) for name, values in keys.items()
            ):
                LOGGER.info("Reusing cached Chronos-2 forecasts from %s", path)
                return cached["quantiles"], cached["levels"]
    quantiles, levels = compute()
    path.parent.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {
        "signature": signature,
        "quantiles": quantiles,
        "levels": levels,
        **keys,
    }
    np.savez(path, **arrays)
    return quantiles, levels


def _forecast_features(
    samples: pd.DataFrame,
    quantiles: np.ndarray,
    levels: np.ndarray,
    climatology: pd.DataFrame,
    config: StackedConfig,
) -> pd.DataFrame:
    """Summarise the label-month NDVI forecast relative to the severe-stress limit."""
    prefix = PREFIXES[config.model]
    ordered = np.sort(quantiles, axis=1)
    picked = {
        name: ordered[:, int(np.argmin(np.abs(levels - level)))]
        for name, level in QUANTILE_LEVELS.items()
    }
    extremes = _climatology_at(samples, climatology, config.horizon)
    limit = _severe_ndvi_limit(extremes, config.severe_vci)
    median = picked["q50"]
    return pd.DataFrame(
        {
            **{f"{prefix}_{name}": values for name, values in picked.items()},
            f"{prefix}_spread": picked["q90"] - picked["q10"],
            f"{prefix}_delta": median - samples["ndvi_lag1"].to_numpy(),
            f"{prefix}_margin": median - limit,
            f"{prefix}_p_severe": _severe_probability(quantiles, levels, limit),
            "ndvi_min_target": extremes["ndvi_min"].to_numpy(),
            "ndvi_max_target": extremes["ndvi_max"].to_numpy(),
        },
        index=samples.index,
    )


def _load_forecast(config: StackedConfig) -> QuantileForecaster:
    if config.model == "chronos2":
        return chronos2_forecaster(
            _load_pipeline(config.checkpoint, config.device), config.batch_size
        )
    return timestral_forecaster(_load_forecaster(config.checkpoint, config.device))


def train_lightgbm_stacked(
    config: StackedConfig, forecast: QuantileForecaster | None = None
) -> Path:
    """Compare LightGBM with and without foundation-model features."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if config.model not in PREFIXES:
        raise ValueError(f"model must be one of {', '.join(PREFIXES)}")
    prefix = PREFIXES[config.model]
    if forecast is None:
        forecast = _load_forecast(config)
    frame = _load_data(config.input_path)
    digest = _file_digest(config.input_path)
    signature = f"{config.checkpoint}|{config.n_lags}|{config.horizon}|{digest}"
    climatology = _ndvi_climatology(
        frame, config.baseline_start_year, config.baseline_end_year
    )
    features, labels = feature_extraction(
        frame,
        n_lags=config.n_lags,
        horizon=config.horizon,
        target_column=TARGET_COLUMN,
    )
    target = labels.map(LABEL_MAPPING).astype(int)
    target_time = pd.to_datetime(features["target_time"])
    validation_cutoff = pd.Timestamp(config.validation_start)
    test_cutoff = pd.Timestamp(config.test_start)
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start must be earlier than test_start")
    splits = {
        "train": (target_time < validation_cutoff, config.max_train_samples),
        "validation": (
            (target_time >= validation_cutoff) & (target_time < test_cutoff),
            config.max_samples,
        ),
        "test": (target_time >= test_cutoff, config.max_samples),
    }

    parts: list[pd.DataFrame] = []
    for split, (mask, limit) in splits.items():
        samples = features.loc[_subsample(features.index[mask], limit)]
        quantiles, levels = _cached_quantiles(
            config.output_dir / f"{prefix}_{split}.npz",
            samples,
            signature,
            partial(
                _forecast_ndvi_quantiles,
                forecast,
                samples,
                config,
                f"{config.model} {split}",
            ),
        )
        extra = _forecast_features(samples, quantiles, levels, climatology, config)
        parts.append(pd.concat([samples, extra], axis=1))
    selected = pd.concat(parts)
    train_x, train_y, validation_x, validation_y, test_x, test_y = split_data(
        selected,
        target.loc[selected.index],
        target_column=TARGET_COLUMN,
        validation_start=config.validation_start,
        test_start=config.test_start,
        spatial_block_size=config.spatial_block_size,
    )
    for name, values in (
        ("training", train_y),
        ("validation", validation_y),
        ("test", test_y),
    ):
        if values.nunique() < 2:
            raise ValueError(f"The {name} partition must contain both classes")

    model_columns = [
        column
        for column in train_x.columns
        if column.startswith(f"{prefix}_") or column in CLIMATOLOGY_COLUMNS
    ]
    raw_columns = [column for column in train_x.columns if column not in model_columns]
    variants = {
        "baseline": raw_columns,
        "climatology": [*raw_columns, *CLIMATOLOGY_COLUMNS],
        prefix: [*raw_columns, *model_columns],
    }
    results: dict[str, Any] = {
        "split_samples": {
            "training": len(train_x),
            "validation": len(validation_x),
            "test": len(test_x),
        },
        "variants": {},
    }
    for variant, columns in variants.items():
        LOGGER.info("Tuning LightGBM variant %r (%d features)", variant, len(columns))
        classifier, tuning = fit_tuned_classifier(
            train_x[columns],
            train_y,
            validation_x[columns],
            validation_y,
            trials=config.trials,
            seed=config.seed,
            threads=config.threads,
        )
        validation_probabilities = np.asarray(
            classifier.predict_proba(validation_x[columns])
        )[:, 1]
        test_probabilities = np.asarray(classifier.predict_proba(test_x[columns]))[:, 1]
        threshold = _select_threshold(
            validation_y, validation_probabilities, config.recall_target
        )
        results["variants"][variant] = {
            "decision_threshold": threshold,
            "validation": _metrics(validation_y, validation_probabilities, threshold),
            "test": _metrics(test_y, test_probabilities, threshold),
            "tuning": tuning,
        }
        variant_dir = config.output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        _write_predictions(
            variant_dir / "predictions.csv",
            validation_target=validation_y,
            validation_probabilities=validation_probabilities,
            test_target=test_y,
            test_probabilities=test_probabilities,
        )
        _save_evaluation_figures(
            variant_dir,
            validation_target=validation_y,
            validation_probabilities=validation_probabilities,
            test_target=test_y,
            test_probabilities=test_probabilities,
            threshold=threshold,
            classifier=classifier,
            feature_columns=columns,
        )

    results["config"] = {
        **asdict(config),
        "input_path": str(config.input_path),
        "output_dir": str(config.output_dir),
    }
    results["data_sha256"] = digest
    results["created_at"] = datetime.now(UTC).isoformat()
    metrics_path = config.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    LOGGER.info("Saved stacked LightGBM evaluation to %s", metrics_path)
    return metrics_path
