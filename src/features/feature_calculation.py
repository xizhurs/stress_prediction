import pandas as pd
import numpy as np
from tqdm import tqdm


def feature_extraction(
    df,
    # columns: latitude, longitude, valid_time, tp_mm, pet_mm, T_c,
    # ndvi, drought_class (or a numeric target)
    n_lags=6,  # how many past months to use as features
    horizon=6,  # how many months ahead to predict
    keep_current=False,  # keep current-month (t) raw vars as features?
    feat_vars=["tp_mm", "pet_mm", "T_c", "ndvi"],
    target_col="drought_class",  # or "DroughtComposite" for regression
):
    df = df.sort_values(["latitude", "longitude", "valid_time"]).copy()
    out = []
    df = df[feat_vars + ["latitude", "longitude", "valid_time"] + [target_col]]
    for (lat, lon), g in tqdm(df.groupby(["latitude", "longitude"], sort=False)):
        g = g.reset_index(drop=True)

        # 1) Create lag features at t-1..t-n_lags
        for lag in range(1, n_lags + 1):
            g[[f"{v}_lag{lag}" for v in feat_vars]] = g[feat_vars].shift(lag)

        # 2) Create future label at t+horizon (shift negative = align future to current row)
        g["y"] = g[target_col].shift(-horizon)

        # 3) Optionally drop current-month raw vars to avoid leakage in forecasting
        if not keep_current:
            g = g.drop(columns=feat_vars)

        # 4) Drop rows without full lag history or missing future label
        #    (first n_lags rows lack lags; last horizon rows lack future label)
        g = g.iloc[n_lags : len(g) - horizon].copy()

        out.append(g)

    sup = pd.concat(out, ignore_index=True)
    # Seasonality encodings
    m = sup["valid_time"].dt.month
    sup["month_sin"] = np.sin(2 * np.pi * m / 12)
    sup["month_cos"] = np.cos(2 * np.pi * m / 12)
    # If your target is categorical, keep as-is; for regression ensure numeric
    sup = sup.dropna(subset="y")
    X = sup.drop(columns=["y"])
    y = sup["y"]

    return X, y
