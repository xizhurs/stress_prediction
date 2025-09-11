#!/usr/bin/env python3
"""
Compute drought indicators and a composite index from a merged ERA5+NDVI NetCDF.

Inputs
------
- A NetCDF that already contains ERA5 monthly variables (e.g., tp, t2m), PET (e.g., pev), and NDVI on the same grid.
  Typically produced by your "prepare_era5_ndvi.py" step.

What it does
------------
1) Converts ERA5 precipitation AND PET to monthly totals (mm/month) using true days-in-month.
2) Computes SPEI-k (default k=3) per pixel (distribution fit -> standard normal).
3) Computes TCI (Temperature Condition Index) per pixel/month using a fixed climatology baseline.
4) Computes VCI (Vegetation Condition Index) per pixel/month using a fixed climatology baseline.
5) Converts SPEI/TCI/VCI to 0..1 *stress* scores and blends them into a composite index.
6) Exports a tidy table (lat, lon, time, SPEI, TCI, VCI, Composite, class).

Example
-------
python compute_drought_composite.py \
  --in_nc data/netherlands_era5_with_ndvi.nc \
  --tp_var tp --pet_var pev --pet_is_negative true \
  --t_var t2m --ndvi_var NDVI \
  --spi_scale 3 \
  --tci_baseline 1991:2020 \
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


def _spei_fit(
    precip_mm: np.ndarray,
    pet_mm: np.ndarray,
    scale: int,
    start_year: int,
    end_year: int,
    dist_name: str = "gamma",
) -> np.ndarray:
    """
    SPEI using climate_indices for a 1D monthly series of (P - PET) in mm.

    dist_name: one of {"gamma", "pearson3", "loglogistic"} depending on your
               climate_indices version build. Default "gamma" is safe.
    """
    # Map string to indices.Distribution enum
    dist_map = {
        "gamma": indices.Distribution.gamma,
        "pearson3": getattr(
            indices.Distribution, "pearson3", indices.Distribution.gamma
        ),
        "loglogistic": getattr(
            indices.Distribution, "loglogistic", indices.Distribution.gamma
        ),
    }
    dist = dist_map.get(dist_name.lower(), indices.Distribution.gamma)

    return indices.spei(
        precips_mm=precip_mm,
        pet_mm=pet_mm,
        scale=scale,
        distribution=dist,
        data_start_year=start_year,
        calibration_year_initial=start_year,
        calibration_year_final=end_year,
        periodicity=compute.Periodicity.monthly,
    )


def _index_to_stress(z: np.ndarray | pd.Series) -> np.ndarray:
    # Z <= -2 -> 1 (worst), Z >= +2 -> 0
    return np.clip((-np.asarray(z) + 2.0) / 4.0, 0.0, 1.0)


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
        description="Compute SPEI, TCI, VCI and a composite drought index."
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
        "--pet_var",
        default="pev",
        help="PET var name (ERA5 potential evaporation as monthly mean rate m/day; ERA5 'pev' is negative).",
    )
    ap.add_argument(
        "--pet_is_negative",
        type=lambda s: str(s).lower() in {"1", "true", "yes", "y"},
        default=True,
        help="If PET values are negative (ERA5 'pev'), set true (default).",
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
        help="SPEI timescale in months (e.g., 1, 3, 6, 12).",
    )
    ap.add_argument(
        "--spei_dist",
        type=str,
        default="gamma",
        help="Distribution for SPEI fit: gamma | pearson3 | loglogistic (depends on climate_indices build).",
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
        "--gamma", type=float, default=0.3, help="Weight for ClimStress (SPEI-derived)."
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
        default="data/drought_indices.csv",
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

    # --- 1) Precip & PET to mm/month using true days-in-month ---
    if args.tp_var not in ds:
        raise ValueError(f"'{args.tp_var}' not found in dataset.")
    if args.pet_var not in ds:
        raise ValueError(f"'{args.pet_var}' not found in dataset. Needed for SPEI.")

    days = _days_in_month_index(pd.to_datetime(ds["time"].values))

    # Precipitation: if monthly mean rate (m/day), multiply by days -> m; then to mm
    tp = ds[args.tp_var]
    tp_mm = (tp * days) * 1000.0

    # PET: same conversion; ERA5 'pev' is negative, flip sign if needed
    pet = ds[args.pet_var]
    pet_mm = (pet * days) * 1000.0
    if args.pet_is_negative:
        pet_mm = -pet_mm

    df_pr = (
        tp_mm.to_dataframe(name="tp_mm")
        .reset_index()
        .rename(columns={"time": "valid_time"})
        .assign(valid_time=lambda d: pd.to_datetime(d["valid_time"]))
    )
    df_pet = (
        pet_mm.to_dataframe(name="pet_mm")
        .reset_index()
        .rename(columns={"time": "valid_time"})
        .assign(valid_time=lambda d: pd.to_datetime(d["valid_time"]))
    )

    df_wb = df_pr.merge(df_pet, on=["latitude", "longitude", "valid_time"], how="inner")

    # --- 2) SPEI per pixel ---
    logging.info(f"Computing SPEI-{args.spi_scale} ...")
    years = pd.to_datetime(df_wb["valid_time"]).dt.year
    start_year, end_year = int(years.min()), int(years.max())

    df_spei = (
        df_wb.groupby(["latitude", "longitude"])
        .apply(
            lambda x: pd.DataFrame(
                {
                    "spei": _spei_fit(
                        x["tp_mm"].values,
                        x["pet_mm"].values,
                        args.spi_scale,
                        start_year,
                        end_year,
                        args.spei_dist,
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
    df = df_spei.merge(
        df_TCI, on=["latitude", "longitude", "valid_time"], how="left"
    ).merge(df_VCI, on=["latitude", "longitude", "valid_time"], how="left")

    # Stress scores (0..1, higher=worse)
    df["ClimStress"] = _index_to_stress(df["spei"])
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
