"""Zero-shot Timestral evaluation on the LightGBM vegetation stress task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stress_prediction.training.foundation import (
    FoundationConfig,
    evaluate_foundation,
    package_version,
    to_numpy,
)


@dataclass(frozen=True)
class TimestralConfig(FoundationConfig):
    checkpoint: str = "mistralai/Timestral-28M"


def _load_forecaster(checkpoint: str, device: str) -> Any:
    try:
        from timestral import TimestralForecaster
    except ImportError as error:
        raise ImportError(
            "Timestral evaluation requires the 'foundation' extra: "
            "uv sync --extra foundation"
        ) from error
    return TimestralForecaster.from_pretrained(checkpoint, device=device)


def evaluate_timestral(config: TimestralConfig, forecaster: Any | None = None) -> Path:
    """Zero-shot evaluate Timestral with the LightGBM data splits and metrics."""
    if forecaster is None:
        forecaster = _load_forecaster(config.checkpoint, config.device)

    def forecast(
        ndvi: np.ndarray, covariates: np.ndarray, known: np.ndarray, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        result = forecaster.forecast(
            ndvi[:, None, :],
            prediction_length=length,
            past_covariates=covariates,
            known_covariates=known,
        )
        quantiles = to_numpy(result.quantiles)  # (B, variates, steps, levels)
        return quantiles[:, 0, -1, :], to_numpy(result.quantile_levels)

    return evaluate_foundation(
        config,
        forecast,
        name="Timestral",
        model_info={"timestral_version": package_version("timestral")},
    )
