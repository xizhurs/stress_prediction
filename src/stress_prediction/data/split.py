"""Leakage-safe temporal dataset splitting."""

from __future__ import annotations

import pandas as pd


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_column: str,
    validation_start: str = "2016-01-01",
    test_start: str = "2019-01-01",
    target_time_column: str = "target_time",
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Split samples according to label time rather than feature time."""
    if target_time_column not in features:
        raise ValueError(
            f"{target_time_column!r} is required to prevent forecast leakage"
        )
    if len(features) != len(target):
        raise ValueError("features and target must contain the same number of rows")

    validation_cutoff = pd.Timestamp(validation_start)
    test_cutoff = pd.Timestamp(test_start)
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start must be earlier than test_start")

    target_time = pd.to_datetime(features[target_time_column], errors="raise")
    masks = (
        target_time < validation_cutoff,
        (target_time >= validation_cutoff) & (target_time < test_cutoff),
        target_time >= test_cutoff,
    )
    if any(not mask.any() for mask in masks):
        raise ValueError("Temporal cutoffs produced an empty dataset partition")

    metadata_columns = ["valid_time", target_time_column, target_column]
    model_columns = [column for column in features if column not in metadata_columns]

    partitions: list[pd.DataFrame | pd.Series] = []
    for mask in masks:
        partitions.extend(
            [
                features.loc[mask, model_columns].reset_index(drop=True),
                target.loc[mask].reset_index(drop=True),
            ]
        )
    return tuple(partitions)  # type: ignore[return-value]
