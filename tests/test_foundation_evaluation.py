from __future__ import annotations

import json
from pathlib import Path
from statistics import NormalDist
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from stress_prediction.cli import main
from stress_prediction.training.chronos2 import Chronos2Config, evaluate_chronos2
from stress_prediction.training.foundation import _severe_probability
from stress_prediction.training.timestral import TimestralConfig, evaluate_timestral

LEVELS = np.linspace(0.01, 0.99, 21)
SPREAD = np.array([NormalDist().inv_cdf(level) for level in LEVELS])
METRIC_NAMES = {"average_precision", "roc_auc", "precision", "recall", "f1"}


class PersistenceForecaster:
    """Stands in for Timestral: repeats the last NDVI with a Gaussian spread."""

    def __init__(self) -> None:
        self.shapes: list[tuple[tuple[int, ...], ...]] = []
        self.prediction_lengths: list[int] = []

    def forecast(
        self,
        context: np.ndarray,
        prediction_length: int,
        past_covariates: np.ndarray,
        known_covariates: np.ndarray,
    ) -> Any:
        self.shapes.append(
            (context.shape, past_covariates.shape, known_covariates.shape)
        )
        self.prediction_lengths.append(prediction_length)
        center = context[:, 0, -1]
        quantiles = center[:, None, None, None] + 0.05 * SPREAD
        quantiles = np.broadcast_to(
            quantiles, (len(context), 1, prediction_length, len(LEVELS))
        )
        return SimpleNamespace(quantiles=quantiles, quantile_levels=LEVELS)


class PersistencePipeline:
    """Stands in for Chronos-2 and enforces its documented input schema."""

    quantiles = list(LEVELS)

    def __init__(self) -> None:
        self.prediction_lengths: list[int] = []
        self.history_lengths: set[int] = set()

    def predict(
        self, inputs: list[dict[str, Any]], prediction_length: int, batch_size: int
    ) -> list[np.ndarray]:
        self.prediction_lengths.append(prediction_length)
        outputs = []
        for item in inputs:
            history = len(item["target"])
            self.history_lengths.add(history)
            assert item["target"].ndim == 1
            assert set(item["future_covariates"]) <= set(item["past_covariates"])
            assert set(item["past_covariates"]) == {
                "tp_mm",
                "pet_mm",
                "T_c",
                "month_sin",
                "month_cos",
            }
            assert all(len(v) == history for v in item["past_covariates"].values())
            assert all(
                len(v) == prediction_length for v in item["future_covariates"].values()
            )
            quantiles = item["target"][-1] + 0.05 * SPREAD
            outputs.append(np.tile(quantiles[None, :, None], (1, 1, prediction_length)))
        return outputs


def _synthetic_observations() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    times = pd.date_range("2000-01-01", "2022-12-01", freq="MS")
    month = times.month.to_numpy()
    frames = []
    for latitude in (51.0, 52.0, 53.0, 54.0):
        frames.append(
            pd.DataFrame(
                {
                    "valid_time": times,
                    "latitude": latitude,
                    "longitude": 5.0,
                    "tp_mm": 50 + 10 * np.sin(2 * np.pi * month / 12),
                    "pet_mm": 40 + 10 * np.cos(2 * np.pi * month / 12),
                    "T_c": 10 + 8 * np.cos(2 * np.pi * month / 12),
                    "ndvi": 0.5
                    + 0.2 * np.sin(2 * np.pi * month / 12)
                    + rng.normal(0, 0.08, len(times)),
                }
            )
        )
    frame = pd.concat(frames, ignore_index=True)
    baseline = frame[frame["valid_time"].dt.year.between(2003, 2020)]
    extremes = (
        baseline.assign(month=baseline["valid_time"].dt.month)
        .groupby(["latitude", "month"])["ndvi"]
        .agg(["min", "max"])
    )
    keyed = frame.join(extremes, on=["latitude", frame["valid_time"].dt.month])
    vci = ((keyed["ndvi"] - keyed["min"]) / (keyed["max"] - keyed["min"] + 1e-6)).clip(
        0, 1
    )
    frame["vegetation_stress_class"] = np.where(vci <= 0.25, "severe", "normal")
    return frame


@pytest.fixture
def observations_path(tmp_path: Path) -> Path:
    path = tmp_path / "observations.csv"
    _synthetic_observations().to_csv(path, index=False)
    return path


def _assert_matches_lightgbm_outputs(metrics_path: Path) -> dict[str, Any]:
    metrics: dict[str, Any] = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["label_rule_agreement"] == 1.0
    assert set(metrics["split_samples"]) == {"validation", "test"}
    for section in (metrics, metrics["persistence_baseline"]):
        assert 0 <= section["decision_threshold"] <= 1
        assert set(section["test"]) == METRIC_NAMES
        assert set(section["validation"]) == METRIC_NAMES
    # A forecaster that carries NDVI forward must beat random ranking.
    assert metrics["test"]["roc_auc"] > 0.5
    for figure_name in ("precision_recall.png", "confusion_matrix.png"):
        assert (metrics_path.parent / figure_name).stat().st_size > 0
    return metrics


def test_severe_probability_interpolates_the_forecast_quantiles() -> None:
    quantiles = np.tile(SPREAD, (4, 1))
    limit = np.array([0.0, -10.0, 10.0, NormalDist().inv_cdf(0.25)])

    probability = _severe_probability(quantiles, LEVELS, limit)

    assert probability[0] == pytest.approx(0.5, abs=1e-6)
    assert probability[1] == 0.0
    assert probability[2] == 1.0
    assert probability[3] == pytest.approx(0.25, abs=0.02)


def test_evaluate_timestral_matches_lightgbm_outputs(
    tmp_path: Path, observations_path: Path
) -> None:
    forecaster = PersistenceForecaster()
    config = TimestralConfig(
        input_path=observations_path,
        output_dir=tmp_path / "timestral",
        batch_size=64,
    )

    metrics_path = evaluate_timestral(config, forecaster=forecaster)

    _assert_matches_lightgbm_outputs(metrics_path)
    # Context ends at the month before the sample, matching the lag features.
    assert forecaster.prediction_lengths == [7] * len(forecaster.prediction_lengths)
    assert forecaster.shapes[0] == ((64, 1, 12), (64, 3, 12), (64, 2, 19))


def test_evaluate_chronos2_matches_lightgbm_outputs(
    tmp_path: Path, observations_path: Path
) -> None:
    pipeline = PersistencePipeline()
    config = Chronos2Config(
        input_path=observations_path,
        output_dir=tmp_path / "chronos2",
        batch_size=64,
    )

    metrics_path = evaluate_chronos2(config, pipeline=pipeline)

    metrics = _assert_matches_lightgbm_outputs(metrics_path)
    assert metrics["model"]["checkpoint"] == "amazon/chronos-2"
    assert set(pipeline.prediction_lengths) == {7}
    assert pipeline.history_lengths == {12}


def test_max_samples_limits_each_split(tmp_path: Path, observations_path: Path) -> None:
    config = TimestralConfig(
        input_path=observations_path,
        output_dir=tmp_path / "subset",
        max_samples=100,
    )

    metrics_path = evaluate_timestral(config, forecaster=PersistenceForecaster())

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["split_samples"] == {"validation": 100, "test": 100}


@pytest.mark.parametrize("command", ["eval-timestral", "eval-chronos2"])
def test_eval_help_is_side_effect_free(command: str, capsys: object) -> None:
    with pytest.raises(SystemExit) as error:
        main([command, "--help"])
    assert error.value.code == 0
    output = capsys.readouterr().out  # type: ignore[attr-defined]
    assert "--checkpoint" in output
    assert "--max-samples" in output
