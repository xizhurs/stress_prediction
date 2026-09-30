from __future__ import annotations

import numpy as np
import pandas as pd

from stress_prediction.data import split_data
from stress_prediction.features import feature_extraction


def _monthly_data(periods: int = 30) -> pd.DataFrame:
    dates = pd.date_range("2019-01-01", periods=periods, freq="MS")
    return pd.DataFrame(
        {
            "valid_time": dates,
            "latitude": 52.0,
            "longitude": 5.0,
            "vegetation_stress_class": np.arange(periods) % 2,
            "tp_mm": np.arange(periods, dtype=float),
            "pet_mm": np.arange(periods, dtype=float) + 1,
            "T_c": np.arange(periods, dtype=float) + 2,
            "ndvi": np.arange(periods, dtype=float) + 3,
        }
    )


def test_feature_extraction_preserves_future_target_time() -> None:
    features, target = feature_extraction(
        _monthly_data(),
        n_lags=2,
        horizon=3,
        target_column="vegetation_stress_class",
    )

    assert (
        features["target_time"] == features["valid_time"] + pd.offsets.MonthBegin(3)
    ).all()
    assert len(features) == len(target) == 25


def test_split_uses_target_time_to_prevent_leakage() -> None:
    features, target = feature_extraction(
        _monthly_data(),
        n_lags=2,
        horizon=3,
        target_column="vegetation_stress_class",
    )
    features["sample_id"] = np.arange(len(features))

    train_x, _, validation_x, _, test_x, _ = split_data(
        features,
        target,
        target_column="vegetation_stress_class",
        validation_start="2020-01-01",
        test_start="2021-01-01",
    )

    sample_target_times = features.set_index("sample_id")["target_time"]
    assert (sample_target_times.loc[train_x["sample_id"]] < "2020-01-01").all()
    assert (
        sample_target_times.loc[validation_x["sample_id"]].between(
            "2020-01-01", "2020-12-01"
        )
    ).all()
    assert (sample_target_times.loc[test_x["sample_id"]] >= "2021-01-01").all()
