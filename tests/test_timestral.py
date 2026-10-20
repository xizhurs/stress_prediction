from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

timestral = pytest.importorskip("timestral")
torch = pytest.importorskip("torch")

CHECKPOINT = "mistralai/Timestral-28M"
TARGETS = ("tp_mm", "pet_mm", "T_c", "ndvi")


def _monthly_data(periods: int) -> pd.DataFrame:
    month = np.arange(periods) % 12
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "valid_time": pd.date_range("2000-01-01", periods=periods, freq="MS"),
            "tp_mm": 50
            + 20 * np.sin(2 * np.pi * month / 12)
            + rng.normal(0, 2, periods),
            "pet_mm": 60
            + 30 * np.cos(2 * np.pi * month / 12)
            + rng.normal(0, 2, periods),
            "T_c": 10
            + 8 * np.cos(2 * np.pi * month / 12)
            + rng.normal(0, 0.5, periods),
            "ndvi": 0.5
            + 0.2 * np.sin(2 * np.pi * month / 12)
            + rng.normal(0, 0.02, periods),
        }
    )


def _seasonal_covariates(times: pd.DatetimeIndex) -> np.ndarray:
    month = times.month.to_numpy()
    return np.stack(
        [np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)]
    ).astype(np.float32)


@pytest.fixture(scope="module")
def forecaster():
    return timestral.TimestralForecaster.from_pretrained(CHECKPOINT, device="cpu")


def test_multivariate_forecast_shape_and_finite(forecaster) -> None:
    history, horizon = 36, 6
    frame = _monthly_data(history)
    context = frame[list(TARGETS)].to_numpy(dtype=np.float32).T[None]  # (1, V, T)

    forecast = forecaster.forecast(context, prediction_length=horizon)

    assert forecast.median.shape == (1, len(TARGETS), horizon)
    assert np.isfinite(np.asarray(forecast.median)).all()
    lower, upper = forecast.quantile(0.1), forecast.quantile(0.9)
    assert (np.asarray(lower) <= np.asarray(upper)).all()


def test_forecast_with_known_seasonal_covariates(forecaster) -> None:
    history, horizon = 36, 6
    frame = _monthly_data(history)
    times = pd.date_range(
        frame["valid_time"].iloc[0], periods=history + horizon, freq="MS"
    )
    context = frame[list(TARGETS)].to_numpy(dtype=np.float32).T[None]
    known = _seasonal_covariates(times)[None]  # (1, 2, T + H)

    forecast = forecaster.forecast(
        context, prediction_length=horizon, known_covariates=known
    )

    assert forecast.median.shape == (1, len(TARGETS), horizon)
    assert np.isfinite(np.asarray(forecast.median)).all()
