import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm


def create_grouped_sequences(
    df: pd.DataFrame,
    seq_len: int = 12,
    horizon: int = 6,
    feature_cols=[
        "tp_mm",
        "pet_mm",
        "T_c",
        "ndvi",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
    ],  # e.g. ["tp_mm","pet_mm","T_c","ndvi","month_sin","month_cos"]
    target_cols="vegetation_stress_class",
    group_cols=("latitude", "longitude"),
    time_col="valid_time",
    enforce_monthly_continuity: bool = True,
    drop_nan_windows: bool = True,
    return_meta: bool = False,
):
    """
    Build (B, Seq, F) sequences per spatial group, predicting labels at t+horizon.

    Assumes df has been pre-imputed (or set drop_nan_windows=False).

    Returns
    -------
    X : np.ndarray  [B, Seq, F]
    y : np.ndarray  [B, Tgt]
    feats_used : list[str]
    meta : list[dict] (optional) with group + reference/label timestamps per sample
    """
    # 1) Sort and basic checks
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([*group_cols, time_col]).reset_index(drop=True)
    if "month_sin" not in df.columns or "month_cos" not in df.columns:
        m = df["valid_time"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    # Feature/target column resolution
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if feature_cols is None:
        # sensible default: all numeric except obvious non-features
        feature_cols = numeric_cols.copy()
    if target_cols is None:
        target_cols = ["drought_class"] if "drought_class" in df.columns else []

    # Ensure features don’t include the target(s)
    feats_used = [c for c in feature_cols if c not in target_cols]
    if len(feats_used) == 0:
        raise ValueError("No feature columns selected after excluding target columns.")

    # 2) Helper to build sequences for a single group
    def _group_windows(g: pd.DataFrame):
        # enforce regular monthly continuity if requested
        months = g[time_col].dt.to_period("M").astype(int).to_numpy()
        vals = g[feats_used].to_numpy()  # [T, F]
        tvals = g[target_cols].to_numpy()

        T = len(g)
        if T < seq_len + horizon:
            return None

        # sliding input windows: (T - seq_len + 1, seq_len, F)
        win = sliding_window_view(vals, (seq_len, vals.shape[1]))[:, 0, :, :]
        # aligned targets at +horizon: (T - seq_len - horizon + 1, Tgt)
        if target_cols:
            y = tvals[seq_len + horizon - 1 :]
        else:
            y = np.empty((T - seq_len - horizon + 1, 0), dtype=np.float32)

        # keep only indices with enough future label
        X = win[:-horizon] if horizon > 0 else win
        # reference time = end of input window (t)
        t_ref = g[time_col].to_numpy()[seq_len - 1 : T - horizon]
        # label time = t + horizon
        t_lab = g[time_col].to_numpy()[seq_len - 1 + horizon :]

        # continuity mask (history must be strictly consecutive months)
        if enforce_monthly_continuity:
            # build ordinals for the end-of-window t and label t+h
            ords = months
            ord_ref = ords[seq_len - 1 : T - horizon]
            ord_lab = ords[seq_len - 1 + horizon :]
            # check label jump
            ok_label = (ord_lab - ord_ref) == horizon

            # check history continuity using windowed diffs == 1
            # (T - 1) diffs; slide over (seq_len-1) consecutive diffs
            diffs = np.diff(ords)  # shape [T-1]
            diffs_win = sliding_window_view(
                diffs, seq_len - 1
            )  # [T-seq_len, seq_len-1]
            # align to X/y (note: X length = T - seq_len - horizon + 1)
            hist_ok = np.all(diffs_win[: len(X)] == 1, axis=1)

            ok = hist_ok & ok_label
            X, y = X[ok], y[ok]
            t_ref, t_lab = t_ref[ok], t_lab[ok]

        if drop_nan_windows:
            nan_mask = ~np.any(np.isnan(X), axis=(1, 2))
            if target_cols:
                nan_mask &= ~np.any(pd.isna(y), axis=1) if y.ndim == 2 else ~pd.isna(y)
            X, y = X[nan_mask], y[nan_mask]
            t_ref, t_lab = t_ref[nan_mask], t_lab[nan_mask]

        if X.size == 0:
            return None

        # collect meta per sample
        meta = [
            {**{gc: g.iloc[0][gc] for gc in group_cols}, "t_ref": tr, "t_label": tl}
            for tr, tl in zip(t_ref, t_lab)
        ]
        return X.astype(np.float32), y, meta

    # 3) Apply per group
    Xs, Ys, Metas = [], [], []
    for _, g in tqdm(df.groupby(list(group_cols), sort=False)):
        out = _group_windows(g)
        if out is None:
            continue
        Xg, yg, mg = out
        Xs.append(Xg)
        Ys.append(yg)
        Metas.extend(mg)

    if not Xs:
        # empty
        X = np.empty((0, seq_len, len(feats_used)), dtype=np.float32)
        y = np.empty((0, len(target_cols))) if target_cols else np.empty((0, 0))
        return (X, y, feats_used) if not return_meta else (X, y, feats_used, Metas)

    X = np.concatenate(Xs, axis=0).astype(np.float32)
    y = np.concatenate(Ys, axis=0).astype(str)

    # Ensure y shape is 2D for consistency
    if y.ndim == 1:
        y = y[:, None]

    return (X, y, feats_used) if not return_meta else (X, y, feats_used, Metas)


def get_split(
    file="data/drought_indices.csv",
    seq_len: int = 12,
    horizon: int = 6,
    feature_cols=(
        "tp_mm",
        "pet_mm",
        "T_c",
        "ndvi",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
    ),
    target_cols: str = "vegetation_stress_class",
    time_col: str = "valid_time",
    enforce_monthly_continuity: bool = True,
    return_meta: bool = False,
):
    df = pd.read_csv(file, parse_dates=[time_col])[
        [
            time_col,
            "latitude",
            "longitude",
            target_cols,
            "tp_mm",
            "pet_mm",
            "T_c",
            "ndvi",
        ]
    ].pipe(
        lambda x: x[x.valid_time.between(datetime(1982, 1, 1), datetime(2022, 12, 31))]
    )
    train_mask = df[time_col] < "2016-01-01"
    val_mask = (df[time_col] >= "2016-01-01") & (df[time_col] < "2019-01-01")
    test_mask = df[time_col] >= "2019-01-01"

    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]
    X_train, y_train, _ = create_grouped_sequences(  # type: ignore
        df=df_train,
        seq_len=seq_len,
        horizon=horizon,
        feature_cols=feature_cols,
        target_cols=target_cols,
        time_col="valid_time",
        enforce_monthly_continuity=enforce_monthly_continuity,
        drop_nan_windows=True,
        return_meta=return_meta,
    )
    X_val, y_val, _ = create_grouped_sequences(  # pyright: ignore[reportAssignmentType]
        df=df_val,
        seq_len=seq_len,
        horizon=horizon,
        feature_cols=feature_cols,
        target_cols=target_cols,
        time_col="valid_time",
        enforce_monthly_continuity=enforce_monthly_continuity,
        drop_nan_windows=True,
        return_meta=return_meta,
    )
    X_test, y_test, _ = (  # pyright: ignore[reportAssignmentType]
        create_grouped_sequences(
            df=df_test,
            seq_len=seq_len,
            horizon=horizon,
            feature_cols=feature_cols,
            target_cols=target_cols,
            time_col="valid_time",
            enforce_monthly_continuity=enforce_monthly_continuity,
            drop_nan_windows=True,
            return_meta=return_meta,
        )
    )
    return (X_train, y_train, X_val, y_val, X_test, y_test)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomJitter:
    """Add small Gaussian noise proportional to feature std."""

    def __init__(self, sigma=0.01, p=0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        if np.random.rand() < self.p:
            noise = np.random.randn(*x.shape).astype(np.float32) * self.sigma
            x = x + noise
        return x


class RandomFeatureScale:
    """Multiply each feature by (1 + eps), eps~N(0, s). Keeps signs/seasonality."""

    def __init__(self, sigma=0.05, p=0.5):
        self.sigma, self.p = sigma, p

    def __call__(self, x):
        if np.random.rand() < self.p:
            f = x.shape[1]
            scale = 1.0 + np.random.randn(f).astype(np.float32) * self.sigma
            x = x * scale[None, :]
        return x


class TimeMask:
    """SpecAugment-style time masking: zero out a short contiguous time span."""

    def __init__(self, max_width=2, p=0.5):
        self.max_width, self.p = max_width, p

    def __call__(self, x):
        if np.random.rand() < self.p and x.shape[0] > 1:
            w = np.random.randint(1, min(self.max_width, x.shape[0]) + 1)
            s = np.random.randint(0, x.shape[0] - w + 1)
            x = x.copy()
            x[s : s + w, :] = 0.0  # if you have mask channels, set mask=0 instead
        return x


class RandomTimeShift:
    """Shift the sequence backward by up to k steps; pad at start by repeating edge.
    NOTE: never shift forward (would leak future info)."""

    def __init__(self, max_shift=1, p=0.5):
        self.max_shift, self.p = max_shift, p

    def __call__(self, x):
        if np.random.rand() < self.p and self.max_shift > 0:
            k = np.random.randint(0, self.max_shift + 1)
            if k > 0:
                pad = np.repeat(x[:1, :], k, axis=0)  # repeat first step
                x = np.concatenate([pad, x[:-k, :]], axis=0)
        return x


class RandomTemporalDropout:
    """Randomly drop (zero) some timesteps independently."""

    def __init__(self, p_drop=0.05, p=0.5):
        self.p_drop, self.p = p_drop, p

    def __call__(self, x):
        if np.random.rand() < self.p:
            mask = (np.random.rand(x.shape[0]) > self.p_drop).astype(np.float32)
            x = x * mask[:, None]
        return x


class SeqDataset(Dataset):
    def __init__(
        self,
        X,
        y,
        means=None,
        stds=None,
        transform=None,
    ):
        self.means = means
        self.stds = stds
        self.X = X  # [B, N, F]
        self.transform = transform
        self.y = y.astype(np.int64)

        self.X = (self.X - self.means) / self.stds

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]  # (Seq, F), float32
        y = self.y[i]
        if self.transform is not None:
            x = self.transform(x)  # must return (Seq, F) float32
        return torch.from_numpy(x), torch.tensor(y)


mapping = {
    "mild": 0,
    "moderate": 0,
    "normal": 0,
    "severe": 1,
}


def create_dataset(
    processed=True,
    seq_len=12,
    horizon=6,
    train_input_dir="data/ts_train/npy",
    ts_data="data/drought_indices.csv",
    scaling_dir="data/ts_train/scaler",
    binary_class=False,
    feature_cols=(
        "tp_mm",
        "pet_mm",
        "T_c",
        "ndvi",
        "month_sin",
        "month_cos",
        "latitude",
        "longitude",
    ),
    target_cols="vegetation_stress_class",
    time_col="valid_time",
):
    # le = LabelEncoder()

    transform = Compose(
        [
            RandomJitter(sigma=0.01, p=0.5),
            RandomFeatureScale(sigma=0.05, p=0.5),
            TimeMask(max_width=2, p=0.5),
            RandomTimeShift(max_shift=1, p=0.5),
            RandomTemporalDropout(p_drop=0.05, p=0.5),
        ]
    )
    if not processed:
        X_train, y_train, X_val, y_val, X_test, y_test = get_split(
            file=ts_data,
            horizon=horizon,
            seq_len=seq_len,
            feature_cols=feature_cols,
            target_cols=target_cols,
            time_col=time_col,
        )
        if binary_class:
            y_train = np.vectorize(mapping.get)(y_train)
            y_val = np.vectorize(mapping.get)(y_val)
            y_test = np.vectorize(mapping.get)(y_test)

        with open(train_input_dir + "/X_train.npy", "wb") as f:
            np.save(f, X_train)
        with open(train_input_dir + "/y_train.npy", "wb") as f:
            np.save(f, y_train)
        with open(train_input_dir + "/X_val.npy", "wb") as f:
            np.save(f, X_val)
        with open(train_input_dir + "/y_val.npy", "wb") as f:
            np.save(f, y_val)
        with open(train_input_dir + "/X_test.npy", "wb") as f:
            np.save(f, X_test)
        with open(train_input_dir + "/y_test.npy", "wb") as f:
            np.save(f, y_test)
        # y_train = le.fit_transform(y_train)
        # y_val = le.transform(y_val)
        # y_test = le.transform(y_test)
        means = np.mean(X_train, keepdims=True, axis=(0, 1))
        stds = np.std(X_train, keepdims=True, axis=(0, 1))

        with open(scaling_dir + "/means.npy", "wb") as f:
            np.save(f, means)
        with open(scaling_dir + "/stds.npy", "wb") as f:
            np.save(f, stds)
    else:
        X_train = np.load(train_input_dir + "/X_train.npy", mmap_mode="r")
        y_train = np.load(train_input_dir + "/y_train.npy", mmap_mode="r")
        X_val = np.load(train_input_dir + "/X_val.npy", mmap_mode="r")
        y_val = np.load(train_input_dir + "/y_val.npy", mmap_mode="r")
        X_test = np.load(train_input_dir + "/X_test.npy", mmap_mode="r")
        y_test = np.load(train_input_dir + "/y_test.npy", mmap_mode="r")
        if binary_class:
            y_train = np.vectorize(mapping.get)(y_train)
            y_val = np.vectorize(mapping.get)(y_val)
            y_test = np.vectorize(mapping.get)(y_test)
        # y_train = le.fit_transform(y_train)
        # y_val = le.transform(y_val)
        # y_test = le.transform(y_test)
        means = np.load(scaling_dir + "/means.npy")
        stds = np.load(scaling_dir + "/stds.npy")

    dataset_train, dataset_val, dataset_test = (
        SeqDataset(
            X_train,
            y_train,
            means,
            stds,
            transform=transform,
        ),
        SeqDataset(X_val, y_val, means, stds),
        SeqDataset(X_test, y_test, means, stds),
    )

    return dataset_train, dataset_val, dataset_test, le


if __name__ == "__main__":
    dataset_train, dataset_val, dataset_test, le = create_dataset(
        processed=False,
        seq_len=36,
        horizon=1,
        train_input_dir="data/ts_train/npy",
        ts_data="data/drought_indices.csv",
        scaling_dir="data/ts_train/scaler",
        binary_class=False,
        feature_cols=(
            "tp_mm",
            "pet_mm",
            "T_c",
            "ndvi",
            "month_sin",
            "month_cos",
            "latitude",
            "longitude",
        ),
        target_cols="vegetation_stress_class",
        time_col="valid_time",
    )
