"""Leakage-safe temporal and spatial dataset splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd

SPATIAL_FRACTIONS = (0.70, 0.15, 0.15)


def assign_spatial_blocks(
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    block_size: float,
    seed: int = 0,
) -> np.ndarray:
    """Assign each row to the train, validation or test blocks of a degree grid.

    Whole grid blocks are held out, so neighbouring pixels never straddle splits.
    Blocks are shuffled and filled in order until each split holds its share of
    the pixels.
    """
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    pixels = pd.DataFrame({"latitude": latitude, "longitude": longitude})
    pixels = pixels.drop_duplicates().reset_index(drop=True)
    pixels["block"] = list(
        zip(
            np.floor(pixels["latitude"] / block_size).astype(int),
            np.floor(pixels["longitude"] / block_size).astype(int),
            strict=True,
        )
    )
    sizes = pixels["block"].value_counts().sort_index()
    order = np.random.default_rng(seed).permutation(len(sizes))
    shuffled = sizes.iloc[order]
    share = shuffled.cumsum() / shuffled.sum()
    # A block belongs to the split in which its cumulative share starts.
    started = share - shuffled / shuffled.sum()
    split_of_block = np.select(
        [started < SPATIAL_FRACTIONS[0], started < sum(SPATIAL_FRACTIONS[:2])],
        ["train", "validation"],
        default="test",
    )
    lookup = dict(zip(shuffled.index, split_of_block, strict=True))
    blocks = zip(
        np.floor(np.asarray(latitude) / block_size).astype(int),
        np.floor(np.asarray(longitude) / block_size).astype(int),
        strict=True,
    )
    return np.array([lookup[block] for block in blocks])


def split_masks(
    features: pd.DataFrame,
    *,
    validation_start: str,
    test_start: str,
    spatial_block_size: float | None = None,
    spatial_seed: int = 0,
    target_time_column: str = "target_time",
) -> dict[str, np.ndarray]:
    """Boolean row masks for the train, validation and test partitions.

    Partitions are cut by label time. With `spatial_block_size`, each one is also
    restricted to its own held-out spatial blocks, so validation and test
    locations are never seen in training.
    """
    if target_time_column not in features:
        raise ValueError(
            f"{target_time_column!r} is required to prevent forecast leakage"
        )
    validation_cutoff = pd.Timestamp(validation_start)
    test_cutoff = pd.Timestamp(test_start)
    if validation_cutoff >= test_cutoff:
        raise ValueError("validation_start must be earlier than test_start")

    target_time = pd.to_datetime(features[target_time_column], errors="raise")
    masks = {
        "train": (target_time < validation_cutoff).to_numpy(),
        "validation": (
            (target_time >= validation_cutoff) & (target_time < test_cutoff)
        ).to_numpy(),
        "test": (target_time >= test_cutoff).to_numpy(),
    }
    if spatial_block_size is not None:
        region = assign_spatial_blocks(
            features["latitude"].to_numpy(),
            features["longitude"].to_numpy(),
            block_size=spatial_block_size,
            seed=spatial_seed,
        )
        masks = {name: mask & (region == name) for name, mask in masks.items()}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("The cutoffs produced an empty dataset partition")
    return masks


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
    *,
    target_column: str,
    validation_start: str = "2016-01-01",
    test_start: str = "2019-01-01",
    target_time_column: str = "target_time",
    spatial_block_size: float | None = None,
    spatial_seed: int = 0,
) -> tuple[
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
    pd.DataFrame,
    pd.Series,
]:
    """Split samples according to label time, and optionally held-out locations.

    With a spatial split the coordinates are dropped from the model columns, since
    they would let a model look up the neighbouring training pixels.
    """
    if len(features) != len(target):
        raise ValueError("features and target must contain the same number of rows")

    masks = split_masks(
        features,
        validation_start=validation_start,
        test_start=test_start,
        spatial_block_size=spatial_block_size,
        spatial_seed=spatial_seed,
        target_time_column=target_time_column,
    )
    metadata_columns = ["valid_time", target_time_column, target_column]
    if spatial_block_size is not None:
        metadata_columns += ["latitude", "longitude"]
    model_columns = [column for column in features if column not in metadata_columns]

    partitions: list[pd.DataFrame | pd.Series] = []
    for mask in masks.values():
        partitions.extend(
            [
                features.loc[mask, model_columns].reset_index(drop=True),
                target.loc[mask].reset_index(drop=True),
            ]
        )
    return tuple(partitions)  # type: ignore[return-value]
