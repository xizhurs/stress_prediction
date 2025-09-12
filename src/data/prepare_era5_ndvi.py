#!/usr/bin/env python3
"""
Prepare ERA5 monthly NetCDF + monthly NDVI GeoTIFF stacks into a single, aligned dataset.
- Reprojects & resamples NDVI to ERA5 grid
- Aligns monthly timestamps
- (Optional) computes VCI per pixel & calendar month
- Saves merged NetCDF

Example:
    python prepare_era5_ndvi.py \
        --era5 data/data_stream-moda.nc \
        --ndvi_dir data/NDVI \
        --ndvi_pattern "*.tif" \
        --ndvi_scale 0.001 \
        --ndvi_crs EPSG:4326 \
        --out data/netherlands_era5_with_ndvi_vci.nc
"""

from __future__ import annotations
import argparse
import logging
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
import rioxarray  # noqa: F401  # needed for .rio methods


# ----------------------------- Utils ----------------------------- #


def _ensure_datetime64_month_start(
    da: xr.DataArray | xr.Dataset, time_name: str = "time"
) -> xr.DataArray | xr.Dataset:
    """Normalize time coordinate to month start."""
    t = pd.to_datetime(da[time_name].values).astype("datetime64[ns]")
    da = da.assign_coords({time_name: pd.to_datetime(t).to_period("M").to_timestamp()})
    return da


def _sort_lat_ascending(ds: xr.Dataset, lat_name: str = "latitude") -> xr.Dataset:
    """Ensure latitude is ascending (xarray.interp prefers ascending)."""
    if ds[lat_name][0] > ds[lat_name][-1]:
        ds = ds.sortby(lat_name)
    return ds


def _parse_year_from_name(p: Path) -> int:
    m = re.search(r"(19|20)\d{2}", p.stem)
    if not m:
        raise ValueError(f"Cannot parse year from filename: {p.name}")
    return int(m.group(0))


# ----------------------------- Core steps ----------------------------- #


def load_era5_nc(path: Path, time_name: str = "valid_time") -> xr.Dataset:
    """Load ERA5 monthly NC; ensure time is month-start and latitude ascending."""
    ds = xr.open_dataset(path)
    # Some ERA5 files use 'time', others 'valid_time'
    if time_name not in ds.coords and "time" in ds.coords:
        time_name = "time"

    ds = ds.rename({time_name: "time"})
    ds["time"] = pd.to_datetime(ds["time"].values)
    ds = _ensure_datetime64_month_start(ds, time_name="time")
    ds = _sort_lat_ascending(ds, lat_name="latitude")
    return ds


def load_ndvi_tifs(
    ndvi_dir: Path,
    pattern: str = "*.tif",
    ndvi_var_name: Optional[str] = None,
    ndvi_scale: float = 0.001,
    ndvi_crs: str = "EPSG:4326",
    time_name_out: str = "time",
) -> xr.DataArray:
    """
    Load NDVI GeoTIFFs (one file per year, 12 bands = months) and stack into (time, y, x).
    Returns DataArray named 'NDVI' with coords (time, y, x).
    """
    paths = sorted(ndvi_dir.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No GeoTIFFs found in {ndvi_dir} matching {pattern}")

    ndvi_list = []
    for tif in paths:
        year = _parse_year_from_name(tif)

        # Prefer rioxarray.open_rasterio for robust GeoTIFF handling
        da = xr.open_dataset(tif)  # dims: band, y, x ; has CRS/transform

        # Reproject if needed
        if not da.rio.crs or da.rio.crs.to_string() != ndvi_crs:
            da = da.rio.reproject(ndvi_crs)

        # Use first 12 bands as months (Jan..Dec)
        if da.sizes.get("band", 0) < 12:
            raise ValueError(f"{tif} has < 12 bands (months).")
        da12 = da.isel(band=slice(0, 12)).astype("float32")

        # Apply scale if provided (e.g., 0.001)
        if ndvi_scale != 1.0:
            da12 = da12 * ndvi_scale

        # Attach monthly timestamps for that year
        months = pd.date_range(f"{year}-01-01", f"{year}-12-01", freq="MS")
        da12 = da12.assign_coords(band=("band", months)).rename({"band": time_name_out})

        # Name the data var consistently
        # rioxarray returns a DataArray; ensure name exists
        ndvi_list.append(da12)

    ndvi = xr.concat(ndvi_list, dim=time_name_out)  # (time, y, x)
    ndvi = ndvi.sortby(time_name_out)
    ndvi = _ensure_datetime64_month_start(ndvi, time_name=time_name_out)
    return ndvi


def regrid_ndvi_to_era(
    ndvi: xr.DataArray,
    era_grid: xr.Dataset,
    method: str = "linear",
    time_name: str = "time",
) -> xr.DataArray:
    """Interpolate NDVI (time, y, x) to ERA5 grid (latitude, longitude)."""
    # Rename for interp
    ndvi_ll = ndvi.rename({"y": "latitude", "x": "longitude"})
    # Target coords
    lat_t = era_grid["latitude"]
    lon_t = era_grid["longitude"]
    # Interpolate
    ndvi_on_era = ndvi_ll.interp(latitude=lat_t, longitude=lon_t, method=method)
    # Align time exactly to ERA time axis
    ndvi_on_era = _ensure_datetime64_month_start(ndvi_on_era, time_name=time_name)
    ndvi_on_era = ndvi_on_era.reindex({time_name: era_grid[time_name]})
    ndvi_on_era = ndvi_on_era.rename({"band_data": "NDVI"})

    return ndvi_on_era


def merge_and_save(
    ds_era: xr.Dataset,
    ndvi_on_era: xr.DataArray,
    out_path: Path,
) -> None:
    """Merge ERA5 + NDVI (+ VCI) and save to NetCDF."""
    pieces = [ds_era, ndvi_on_era]

    ds_out = xr.merge(pieces)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(out_path)
    logging.info(f"Wrote: {out_path}")


# ----------------------------- CLI ----------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge ERA5 monthly NetCDF with NDVI GeoTIFF stacks; compute VCI."
    )
    p.add_argument(
        "--era5",
        type=Path,
        required=False,
        default=Path("data/data_stream-moda.nc"),
        help="Path to ERA5 monthly NetCDF (time, latitude, longitude).",
    )
    p.add_argument(
        "--ndvi_dir",
        type=Path,
        default=Path("data/NDVI"),
        required=False,
        help="Folder with NDVI GeoTIFFs (one file/year, 12 bands).",
    )
    p.add_argument(
        "--ndvi_pattern",
        type=str,
        default="*.tif",
        help="Glob pattern for NDVI GeoTIFFs.",
    )
    p.add_argument(
        "--ndvi_scale",
        type=float,
        default=0.001,
        help="Multiply NDVI by this factor (e.g., 0.001).",
    )
    p.add_argument(
        "--ndvi_crs",
        type=str,
        default="EPSG:4326",
        help="Expected NDVI CRS; will reproject if different.",
    )
    p.add_argument(
        "--interp",
        type=str,
        default="linear",
        choices=["linear", "nearest"],
        help="Resampling method to ERA5 grid.",
    )

    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/netherlands_era5_with_ndvi.nc"),
        required=False,
        help="Output NetCDF path.",
    )
    p.add_argument(
        "--log", type=str, default="INFO", help="Logging level (DEBUG, INFO, WARNING)."
    )
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )

    logging.info("Loading ERA5...")
    ds_era = load_era5_nc(args.era5)

    logging.info("Loading NDVI GeoTIFF stack...")
    ndvi = load_ndvi_tifs(
        ndvi_dir=args.ndvi_dir,
        pattern=args.ndvi_pattern,
        ndvi_scale=args.ndvi_scale,
        ndvi_crs=args.ndvi_crs,
        time_name_out="time",
    )

    logging.info("Interpolating NDVI to ERA5 grid...")
    ndvi_on_era = regrid_ndvi_to_era(ndvi, ds_era, method=args.interp)

    logging.info("Merging & saving...")
    merge_and_save(ds_era, ndvi_on_era, args.out)

    logging.info("Done.")


if __name__ == "__main__":
    main()
