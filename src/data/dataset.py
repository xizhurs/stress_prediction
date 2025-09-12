import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datetime import datetime
from tqdm import tqdm
from glob import glob


def get_seq_data(
    df: pd.DataFrame,
    seq_len: int = 12,
    horizon: int = 6,
    feats=("tp_mm", "pet_mm", "T_c", "ndvi", "month_sin", "month_cos"),
    label_col: str = "drought_class",
    enforce_monthly_continuity: bool = True,
    return_meta: bool = False,
):
    """
    Build (B, Seq, F) sequences per (lat, lon). Uses sliding windows ending at t,
    predicts label at t + horizon. Ensures monthly continuity if requested.

    df must contain: latitude, longitude, valid_time (datetime64[ns]), label_col,
    and features.
    If month_sin/cos aren't present, they'll be added.
    """
    df = df.copy()

    # Ensure datetime and sort
    df["valid_time"] = pd.to_datetime(df["valid_time"])
    df = df.sort_values(["latitude", "longitude", "valid_time"])

    # Add month features if missing
    if "month_sin" not in df.columns or "month_cos" not in df.columns:
        m = df["valid_time"].dt.month
        df["month_sin"] = np.sin(2 * np.pi * m / 12.0)
        df["month_cos"] = np.cos(2 * np.pi * m / 12.0)

    # Drop non-numeric columns from feats (e.g., if user mistakenly includes
    # 'valid_time')
    feats = [
        f for f in feats if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
    ]

    # Helper to check monthly continuity via period ordinals
    def _is_consecutive_months(periods):
        if len(periods) <= 1:
            return True
        ords = periods.astype("period[M]").astype(int).to_numpy()
        return np.all(np.diff(ords) == 1)

    X, y, meta = [], [], []

    for (_, _), g in tqdm(df.groupby(["latitude", "longitude"], sort=False)):
        g = g.reset_index(drop=True)

        # Precompute monthly period ordinals for continuity checks
        months = g["valid_time"].dt.to_period("M")

        for t in range(seq_len - 1, len(g) - horizon):
            start = t - seq_len + 1
            end = t + 1  # python slice end-exclusive

            if enforce_monthly_continuity:
                # 1) history window continuity
                if not _is_consecutive_months(months.iloc[start : t + 1]):
                    continue
                # 2) label continuity: ensure the label timestamp is exactly horizon
                # months ahead
                #    i.e., period(t+h) = period(t) + horizon
                if (
                    months.iloc[t + horizon].ordinal - months.iloc[t].ordinal
                ) != horizon:
                    continue

            window = g.iloc[start:end]
            x = window[feats].to_numpy(dtype=np.float32)  # (Seq, F)

            # Drop if any NaNs in the window (or implement your imputation before this
            # function)
            if np.isnan(x).any():
                continue

            yi = g.loc[t + horizon, label_col]
            X.append(x)
            y.append(yi)

            if return_meta:
                meta.append(
                    {
                        "latitude": g.loc[t, "latitude"],
                        "longitude": g.loc[t, "longitude"],
                        "t_ref": g.loc[
                            t, "valid_time"
                        ],  # last timestep in the input window
                        "t_label": g.loc[t + horizon, "valid_time"],  # label timestamp
                    }
                )

    X = (
        np.stack(X, axis=0)
        if X
        else np.empty((0, seq_len, len(feats)), dtype=np.float32)
    )
    y = np.array(y)

    if return_meta:
        return X, y, feats, meta
    return X, y, feats


def get_split(
    file="data/drought_indices.csv",
    seq_len: int = 12,
    horizon: int = 6,
    feats=("tp_mm", "pet_mm", "T_c", "ndvi", "month_sin", "month_cos"),
    label_col: str = "drought_class",
    enforce_monthly_continuity: bool = True,
    return_meta: bool = False,
):
    df = pd.read_csv(file, parse_dates=["valid_time"])[
        [
            "valid_time",
            "latitude",
            "longitude",
            "drought_class",
            "tp_mm",
            "pet_mm",
            "T_c",
            "ndvi",
        ]
    ].pipe(
        lambda x: x[x.valid_time.between(datetime(1982, 1, 1), datetime(2022, 12, 31))]
    )
    train_mask = df["valid_time"] < "2016-01-01"
    val_mask = (df["valid_time"] >= "2016-01-01") & (df["valid_time"] < "2019-01-01")
    test_mask = df["valid_time"] >= "2019-01-01"

    df_train = df[train_mask]
    df_val = df[val_mask]
    df_test = df[test_mask]
    X_train, y_train, _ = get_seq_data(
        df_train,
        seq_len,
        horizon,
        feats,
        label_col,
        enforce_monthly_continuity,
        return_meta,
    )
    X_val, y_val, _ = get_seq_data(
        df_val,
        seq_len,
        horizon,
        feats,
        label_col,
        enforce_monthly_continuity,
        return_meta,
    )
    X_test, y_test, _ = get_seq_data(
        df_test,
        seq_len,
        horizon,
        feats,
        label_col,
        enforce_monthly_continuity,
        return_meta,
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
        self.y_str = y
        self.transform = transform
        # encode labels to ints (Transformer/LSTM need ints for CE)
        classes, y_idx = np.unique(self.y_str, return_inverse=True)
        self.classes_ = classes
        self.y = y_idx.astype(np.int64)

        self.X = (self.X - self.means) / self.stds

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        x = self.X[i]  # (Seq, F), float32
        y = self.y[i]
        if self.transform is not None:
            x = self.transform(x)  # must return (Seq, F) float32
        return torch.from_numpy(x), torch.tensor(y)


def create_dataset(
    processed=True,
    train_input_dir="data/ts_train/npy",
    ts_data="data/drought_indices.csv",
    scaling_dir="data/ts_train/scaler",
):
    if not processed:
        X_train, y_train, X_val, y_val, X_test, y_test = get_split(file=ts_data)
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
        means = np.load(scaling_dir + "/means.npy")
        stds = np.load(scaling_dir + "/stds.npy")

        transform = Compose(
            [
                RandomJitter(sigma=0.01, p=0.5),
                RandomFeatureScale(sigma=0.05, p=0.5),
                TimeMask(max_width=2, p=0.5),
                RandomTimeShift(max_shift=1, p=0.5),
                RandomTemporalDropout(p_drop=0.05, p=0.5),
            ]
        )

        dataset_train, dataset_val, dataset_test = (
            SeqDataset(
                X_train,
                y_train,
                means,
                stds,
                transform=transform,
            ),
            SeqDataset(X_val[index_val], y_val[index_val], means, stds),
            SeqDataset(X_test, y_test, means, stds),
        )

    return dataset_train, dataset_val, dataset_test


if __name__ == "__main__":
    dataset_train, dataset_val, dataset_test = create_dataset(batch_size=4)
