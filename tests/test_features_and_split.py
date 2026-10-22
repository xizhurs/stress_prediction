from __future__ import annotations

import numpy as np
import pandas as pd

from stress_prediction.data import assign_spatial_blocks, split_data, split_masks
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


def _grid_data() -> pd.DataFrame:
    frames = []
    for latitude in np.arange(51.0, 53.0, 0.1):
        for longitude in np.arange(4.0, 6.0, 0.1):
            frame = _monthly_data(periods=40)
            frame["latitude"] = latitude
            frame["longitude"] = longitude
            frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def test_assign_spatial_blocks_keeps_blocks_whole_and_near_target_shares() -> None:
    data = _grid_data()
    regions = assign_spatial_blocks(
        data["latitude"].to_numpy(),
        data["longitude"].to_numpy(),
        block_size=0.5,
        seed=0,
    )
    again = assign_spatial_blocks(
        data["latitude"].to_numpy(),
        data["longitude"].to_numpy(),
        block_size=0.5,
        seed=0,
    )

    assert (regions == again).all()
    blocks = list(
        zip(
            np.floor(data["latitude"] / 0.5).astype(int),
            np.floor(data["longitude"] / 0.5).astype(int),
            strict=True,
        )
    )
    per_block = pd.Series(regions).groupby(pd.Series(blocks)).nunique()
    assert (per_block == 1).all()
    shares = pd.Series(regions).value_counts(normalize=True)
    assert set(shares.index) == {"train", "validation", "test"}
    assert shares["train"] > 0.5


def test_spatial_split_holds_out_locations_and_coordinates() -> None:
    features, target = feature_extraction(
        _grid_data(),
        n_lags=2,
        horizon=3,
        target_column="vegetation_stress_class",
    )
    kwargs = {
        "target_column": "vegetation_stress_class",
        "validation_start": "2020-01-01",
        "test_start": "2021-01-01",
    }
    masks = split_masks(
        features,
        validation_start="2020-01-01",
        test_start="2021-01-01",
        spatial_block_size=0.5,
    )
    pixels = {
        name: set(
            zip(
                features.loc[mask, "latitude"].round(3),
                features.loc[mask, "longitude"].round(3),
                strict=True,
            )
        )
        for name, mask in masks.items()
    }
    train_x, _, validation_x, _, test_x, _ = split_data(
        features, target, spatial_block_size=0.5, **kwargs
    )

    assert not pixels["train"] & pixels["validation"]
    assert not pixels["train"] & pixels["test"]
    assert not pixels["validation"] & pixels["test"]
    assert len(train_x) == masks["train"].sum()
    assert len(test_x) == masks["test"].sum()
    for part in (train_x, validation_x, test_x):
        assert "latitude" not in part and "longitude" not in part
