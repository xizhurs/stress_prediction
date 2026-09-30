"""Tabular feature engineering for vegetation stress forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

DEFAULT_FEATURES = ("tp_mm", "pet_mm", "T_c", "ndvi")


def feature_extraction(
    data: pd.DataFrame,
    *,
    n_lags: int = 6,
    horizon: int = 6,
    keep_current: bool = False,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
    target_column: str = "drought_class",
) -> tuple[pd.DataFrame, pd.Series]:
    """Create lagged features and future labels for each spatial location."""
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    required = {
        "latitude",
        "longitude",
        "valid_time",
        target_column,
        *feature_columns,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = data.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="raise")
    frame = frame.sort_values(["latitude", "longitude", "valid_time"])
    groups: list[pd.DataFrame] = []

    for _, group in frame.groupby(["latitude", "longitude"], sort=False):
        group = group.reset_index(drop=True)
        for lag in range(1, n_lags + 1):
            lag_columns = [f"{column}_lag{lag}" for column in feature_columns]
            group[lag_columns] = group[list(feature_columns)].shift(lag)

        group["target_time"] = group["valid_time"].shift(-horizon)
        group["target"] = group[target_column].shift(-horizon)
        if not keep_current:
            group = group.drop(columns=list(feature_columns))

        groups.append(group.iloc[n_lags : len(group) - horizon].copy())

    if not groups:
        raise ValueError("No spatial groups were available for feature extraction")

    supervised = pd.concat(groups, ignore_index=True)
    month = supervised["valid_time"].dt.month
    supervised["month_sin"] = np.sin(2 * np.pi * month / 12)
    supervised["month_cos"] = np.cos(2 * np.pi * month / 12)
    supervised = supervised.dropna(subset=["target", "target_time"])

    features = supervised.drop(columns=["target"])
    target = supervised["target"].rename(target_column)
    return features, target


def prediction_feature_extraction(
    data: pd.DataFrame,
    *,
    n_lags: int,
    horizon: int,
    feature_columns: Sequence[str] = DEFAULT_FEATURES,
) -> pd.DataFrame:
    """Create lagged inference features from observations without future labels."""
    if n_lags < 1:
        raise ValueError("n_lags must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    required = {"latitude", "longitude", "valid_time", *feature_columns}
    missing = sorted(required.difference(data.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    frame = data.copy()
    frame["valid_time"] = pd.to_datetime(frame["valid_time"], errors="raise")
    frame = frame.sort_values(["latitude", "longitude", "valid_time"])
    groups: list[pd.DataFrame] = []
    lag_columns: list[str] = []
    for _, group in frame.groupby(["latitude", "longitude"], sort=False):
        group = group.reset_index(drop=True)
        for lag in range(1, n_lags + 1):
            current_lag_columns = [f"{column}_lag{lag}" for column in feature_columns]
            group[current_lag_columns] = group[list(feature_columns)].shift(lag)
            lag_columns.extend(current_lag_columns)
        groups.append(group.iloc[n_lags:].copy())

    if not groups:
        raise ValueError("No spatial groups were available for prediction")
    prediction_frame = pd.concat(groups, ignore_index=True)
    prediction_frame["target_time"] = prediction_frame["valid_time"] + pd.DateOffset(
        months=horizon
    )
    month = prediction_frame["valid_time"].dt.month
    prediction_frame["month_sin"] = np.sin(2 * np.pi * month / 12)
    prediction_frame["month_cos"] = np.cos(2 * np.pi * month / 12)
    prediction_frame = prediction_frame.drop(columns=list(feature_columns))
    return prediction_frame.dropna(subset=lag_columns).reset_index(drop=True)
