"""Zero-shot Chronos-2 evaluation on the LightGBM vegetation stress task."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from stress_prediction.training.foundation import (
    COVARIATE_COLUMNS,
    FoundationConfig,
    evaluate_foundation,
    package_version,
    to_numpy,
)

KNOWN_COLUMNS = ("month_sin", "month_cos")


@dataclass(frozen=True)
class Chronos2Config(FoundationConfig):
    checkpoint: str = "amazon/chronos-2"


def _load_pipeline(checkpoint: str, device: str) -> Any:
    try:
        from chronos import Chronos2Pipeline
    except ImportError as error:
        raise ImportError(
            "Chronos-2 evaluation requires the 'chronos' extra: uv sync --extra chronos"
        ) from error
    return Chronos2Pipeline.from_pretrained(checkpoint, device_map=device)


def evaluate_chronos2(config: Chronos2Config, pipeline: Any | None = None) -> Path:
    """Zero-shot evaluate Chronos-2 with the LightGBM data splits and metrics."""
    if pipeline is None:
        pipeline = _load_pipeline(config.checkpoint, config.device)
    levels = np.asarray(pipeline.quantiles, dtype=np.float64)

    def forecast(
        ndvi: np.ndarray, covariates: np.ndarray, known: np.ndarray, length: int
    ) -> tuple[np.ndarray, np.ndarray]:
        history = ndvi.shape[1]
        inputs = []
        for row in range(len(ndvi)):
            past = {
                column: covariates[row, index]
                for index, column in enumerate(COVARIATE_COLUMNS)
            }
            future = {}
            for index, column in enumerate(KNOWN_COLUMNS):
                past[column] = known[row, index, :history]
                future[column] = known[row, index, history:]
            inputs.append(
                {
                    "target": ndvi[row],
                    "past_covariates": past,
                    "future_covariates": future,
                }
            )
        predictions = pipeline.predict(
            inputs, prediction_length=length, batch_size=config.batch_size
        )
        # Each item is (variates, levels, steps); keep NDVI at the final step.
        quantiles = np.stack([to_numpy(item)[0, :, -1] for item in predictions])
        return quantiles, levels

    return evaluate_foundation(
        config,
        forecast,
        name="Chronos-2",
        model_info={
            "chronos_forecasting_version": package_version("chronos-forecasting")
        },
    )
