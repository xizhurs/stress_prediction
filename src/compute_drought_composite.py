#!/usr/bin/env python3
"""
Compute drought indicators and a composite index from a merged ERA5+NDVI NetCDF.

Inputs
------
- A NetCDF that already contains ERA5 monthly variables (e.g., tp, t2m) and NDVI on the same grid.
  Typically produced by your "prepare_era5_ndvi.py" step.

What it does
------------
1) Converts ERA5 precipitation to monthly totals (mm/month) using true days-in-month.
2) Computes SPI-k (default k=3) per pixel (gamma fit -> standard normal).
3) Computes TCI (Temperature Condition Index) per pixel/month using a fixed climatology baseline.
4) Computes VCI (Vegetation Condition Index) per pixel/month using a fixed climatology baseline.
5) Converts SPI/TCI/VCI to 0..1 *stress* scores and blends them into a composite index.
6) Exports a tidy table (lat, lon, time, SPI, TCI, VCI, Composite, class).

Example
-------
python compute_drought_composite.py \
  --in_nc data/netherlands_era5_with_ndvi_vci.nc \
  --tp_var tp \
  --t_var t2m \
  --ndvi_var NDVI \
  --spi_scale 3 \
  --tci_baseline 2003:2020 \
  --vci_baseline 2003:2020 \
  --alpha 0.4 --beta 0.3 --gamma 0.3 \
  --start 1982-01 --end 2022-12 \
  --out data/drought_composite.parquet
"""

from __future__ import annotations
import argparse
import logging
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from climate_indices import compute, indices


# ----------------------------- utils ----------------------------- #


def _parse_period(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s is None:
        return None
    # accept YYYY or YYYY-MM or full date
    try:
        if len(s) == 4:
            return pd.Period(f"{s}-01", freq="M").to_timestamp()
        return pd.Period(s, freq="M").to_timestamp()
    except Exception as e:
        raise ValueError(f"Invalid date/period: {s}. Use 'YYYY' or 'YYYY-MM'.") from e


def _parse_baseline(s: Optional[str]) -> Optional[Tuple[int, int]]:
    if s is None:
        return None
    try:
        a, b = s.split(":")
        return int(a), int(b)
    except Exception as e:
        raise ValueError(f"Invalid baseline '{s}'. Use 'YYYY:YYYY'.") from e


def _days_in_month_index(times: pd.DatetimeIndex) -> xr.DataArray:
    return xr.DataArray(
        pd.Index(times).daysinmonth.values, coords={"time": times}, dims=["time"]
    )


def _spi_gamma(
    ts_mm: np.ndarray, scale: int, start_year: int, end_year: int
) -> np.ndarray:
    """Gamma-fit SPI using climate_indices for a 1D monthly series."""
    return indices.spi(
        values=ts_mm,
        scale=scale,
        distribution=indices.Distribution.gamma,
        data_start_year=start_year,
        calibration_year_initial=start_year,
        calibration_year_final=end_year,
        periodicity=compute.Periodicity.monthly,
    )


def _spi_to_stress(spi: np.ndarray | pd.Series) -> np.ndarray:
    # SPI <= -2 -> 1 (worst), SPI >= +2 -> 0
    return np.clip((-np.asarray(spi) + 2.0) / 4.0, 0.0, 1.0)


# ----------------------------- core ----------------------------- #


def compute_tci(
    df_T: pd.DataFrame, baseline: Optional[Tuple[int, int]]
) -> pd.DataFrame:
    """Compute monthly TCI per pixel using Tmin/Tmax of the baseline climatology."""
    df = df_T.copy()
    df["month"] = df["valid_time"].dt.month
    if baseline is not None:
        y0, y1 = baseline
        base_mask = df["valid_time"].dt.year.between(y0, y1)
    else:
        base_mask = slice(None)

    clim = (
        df[base_mask]
        .groupby(["latitude", "longitude", "month"])["T_c"]
        .agg(Tmin_clim="min", Tmax_clim="max")
        .reset_index()
    )
    out = df.merge(clim, on=["latitude", "longitude", "month"], how="left")
    eps = 1e-6
    out["TCI"] = (
        (out["Tmax_clim"] - out["T_c"]) / (out["Tmax_clim"] - out["Tmin_clim"] + eps)
    ).clip(0, 1)
    return out[["latitude", "longitude", "valid_time", "TCI"]]


def compute_vci(
    df_ndvi: pd.DataFrame, baseline: Optional[Tuple[int, int]]
) -> pd.DataFrame:
    """Compute monthly VCI per pixel using min/max of NDVI climatology."""
    df = df_ndvi.copy()
    df["month"] = df["valid_time"].dt.month
    df["year"] = df["valid_time"].dt.year
    if baseline is not None:
        y0, y1 = baseline
        base_mask = df["year"].between(y0, y1)
    else:
        base_mask = slice(None)

    clim = (
        df[base_mask]
        .groupby(["latitude", "longitude", "month"])["ndvi"]
        .agg(ndvi_min="min", ndvi_max="max")
        .reset_index()
    )
    out = df.merge(clim, on=["latitude", "longitude", "month"], how="left")
    eps = 1e-6
    out["VCI"] = (
        (out["ndvi"] - out["ndvi_min"]) / (out["ndvi_max"] - out["ndvi_min"] + eps)
    ).clip(0, 1)
    return out[["latitude", "longitude", "valid_time", "VCI"]]


def main():
    ap = argparse.ArgumentParser(
        description="Compute SPI, TCI, VCI and a composite drought index."
    )
    ap.add_argument(
        "--in_nc",
        required=False,
        default="data/netherlands_era5_with_ndvi.nc",
        help="Merged ERA5+NDVI NetCDF path.",
    )
    ap.add_argument(
        "--tp_var",
        default="tp",
        help="Precip var name (ERA5 total_precipitation as monthly mean rate m/day).",
    )
    ap.add_argument(
        "--t_var", default="t2m", help="Temperature var name (t2m or mx2t) in Kelvin."
    )
    ap.add_argument(
        "--ndvi_var", default="NDVI", help="NDVI variable name in the NetCDF."
    )
    ap.add_argument(
        "--ndvi_scale",
        type=float,
        default=1.0,
        help="Multiply NDVI by this scale if stored as int.",
    )
    ap.add_argument(
        "--spi_scale",
        type=int,
        default=3,
        help="SPI timescale in months (e.g., 1, 3, 6, 12).",
    )
    ap.add_argument(
        "--tci_baseline",
        type=str,
        default="1991:2020",
        help="TCI baseline 'YYYY:YYYY' or omit for all years.",
    )
    ap.add_argument(
        "--vci_baseline",
        type=str,
        default="2003:2020",
        help="VCI baseline 'YYYY:YYYY' or omit for all years.",
    )
    ap.add_argument(
        "--alpha", type=float, default=0.4, help="Weight for VegStress (1-VCI)."
    )
    ap.add_argument(
        "--beta", type=float, default=0.3, help="Weight for HeatStress (1-TCI)."
    )
    ap.add_argument(
        "--gamma", type=float, default=0.3, help="Weight for ClimStress (SPI/SPEI)."
    )
    ap.add_argument(
        "--start", type=str, default=None, help="First month to keep (YYYY or YYYY-MM)."
    )
    ap.add_argument(
        "--end", type=str, default=None, help="Last month to keep (YYYY or YYYY-MM)."
    )
    ap.add_argument(
        "--out",
        required=False,
        default="data/test.csv",
        help="Output file (.parquet or .csv).",
    )
    ap.add_argument("--log", default="INFO", help="Logging level.")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log.upper(), logging.INFO),
        format="%(levelname)s: %(message)s",
    )
    tci_base = _parse_baseline(args.tci_baseline)
    vci_base = _parse_baseline(args.vci_baseline)
    t0 = _parse_period(args.start)
    t1 = _parse_period(args.end)

    # Load dataset
    logging.info("Loading merged NetCDF...")
    ds = xr.open_dataset(args.in_nc)

    # Normalize/rename time coordinate to 'valid_time' -> 'time'
    tname = (
        "time"
        if "time" in ds.coords
        else ("valid_time" if "valid_time" in ds.coords else None)
    )
    if tname is None:
        raise ValueError("No 'time' or 'valid_time' coordinate found.")
    ds = ds.rename({tname: "time"})
    ds["time"] = pd.to_datetime(ds["time"].values)

    # Optional time filter
    if t0 is not None or t1 is not None:
        start = t0 or ds["time"].min().item()
        end = t1 or ds["time"].max().item()
        ds = ds.sel(time=slice(start, end))

    # --- 1) Precip to mm/month using true days-in-month ---
    if args.tp_var not in ds:
        raise ValueError(f"'{args.tp_var}' not found in dataset.")
    tp = ds[args.tp_var]
    # If value is monthly mean rate (m/day), multiply by days-in-month; if it's already monthly sum (m), change this line.
    days = _days_in_month_index(pd.to_datetime(ds["time"].values))
    tp_mm = (tp * days) * 1000.0  # m/day * days -> m ; then to mm
    df_pr = (
        tp_mm.to_dataframe(name="tp_mm")
        .reset_index()
        .rename(columns={"time": "valid_time"})
        .assign(valid_time=lambda d: pd.to_datetime(d["valid_time"]))
    )

    # --- 2) SPI per pixel ---
    logging.info(f"Computing SPI-{args.spi_scale} ... (gamma fit per pixel)")
    # Determine calibration years from the filtered time range:
    years = pd.to_datetime(df_pr["valid_time"]).dt.year
    start_year, end_year = int(years.min()), int(years.max())

    df_spi = (
        df_pr.groupby(["latitude", "longitude"])
        .apply(
            lambda x: pd.DataFrame(
                {
                    "spi": _spi_gamma(
                        x["tp_mm"].values, args.spi_scale, start_year, end_year
                    ),
                    "valid_time": x["valid_time"].values,
                }
            )
        )
        .reset_index()
        .drop(columns=["level_2"])
    )

    # --- 3) Temperature (K->C) & TCI ---
    if args.t_var not in ds:
        raise ValueError(f"'{args.t_var}' not found in dataset.")
    T_c = (
        (ds[args.t_var] - 273.15)
        .to_dataframe(name="T_c")
        .reset_index()
        .rename(columns={"time": "valid_time"})
    )
    T_c["valid_time"] = pd.to_datetime(T_c["valid_time"])
    df_TCI = compute_tci(T_c, baseline=tci_base)

    # --- 4) NDVI & VCI ---
    if args.ndvi_var not in ds:
        raise ValueError(f"'{args.ndvi_var}' not found in dataset.")
    ndvi = ds[args.ndvi_var]
    if args.ndvi_scale != 1.0:
        ndvi = ndvi * args.ndvi_scale
    df_ndvi = (
        ndvi.to_dataframe(name="ndvi")
        .reset_index()
        .rename(columns={"time": "valid_time"})
    )
    df_ndvi["valid_time"] = pd.to_datetime(df_ndvi["valid_time"])
    df_VCI = compute_vci(df_ndvi, baseline=vci_base)

    # --- 5) Merge all + composite ---
    logging.info("Merging indices and computing composite...")
    df = df_spi.merge(
        df_TCI, on=["latitude", "longitude", "valid_time"], how="left"
    ).merge(df_VCI, on=["latitude", "longitude", "valid_time"], how="left")

    # Stress scores (0..1, higher=worse)
    df["ClimStress"] = _spi_to_stress(df["spi"])
    df["HeatStress"] = 1.0 - df["TCI"]
    df["VegStress"] = 1.0 - df["VCI"]

    # Weighted composite
    a, b, c = float(args.alpha), float(args.beta), float(args.gamma)
    df["DroughtComposite"] = (
        a * df["VegStress"] + b * df["HeatStress"] + c * df["ClimStress"]
    ).clip(0, 1)

    # Classes
    df["drought_class"] = pd.cut(
        df["DroughtComposite"],
        bins=[-1, 0.35, 0.50, 0.70, 1.01],
        labels=["normal", "watch", "moderate", "severe"],
    )

    # --- 6) Save
    out = args.out
    logging.info(f"Writing {out} ...")
    if out.endswith(".parquet"):
        df.to_parquet(out, index=False)
    elif out.endswith(".csv"):
        df.to_csv(out, index=False)
    else:
        raise ValueError("Output must be .parquet or .csv")

    # Quick summary
    logging.info(
        "Class distribution:\n"
        + df["drought_class"].value_counts(dropna=False, normalize=True).to_string()
    )


if __name__ == "__main__":
    main()
