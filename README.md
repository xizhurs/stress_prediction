# Drought Stress Prediction

Train a LightGBM classifier to forecast vegetation stress from monthly NDVI and
ERA5 climate variables. The supported workflow is an internal, reproducible CLI
application managed with `uv`.

## Setup

Install `uv`, then create the locked Python 3.11 environment:

```powershell
uv sync --locked --group dev
uv run stress-prediction --help
```

`pyproject.toml` defines compatible dependency ranges. `uv.lock` is committed
and is the exact environment used by development and CI.

Optional dependencies are available for the geospatial pipeline and the
experimental neural models:

```powershell
uv sync --locked --extra geo
uv sync --locked --extra nn
```

## Input Data

LightGBM training expects a CSV with one monthly observation per spatial point:

- `valid_time`
- `latitude`
- `longitude`
- `vegetation_stress_class`
- `tp_mm`
- `pet_mm`
- `T_c`
- `ndvi`

Supported stress labels are `normal`, `mild`, `moderate`, and `severe`.
Duplicate `(latitude, longitude, valid_time)` observations, missing feature
values, and unknown labels are rejected before training.

Raw climate and vegetation datasets are not distributed in the Python package.
Copernicus CDS credentials are required for ERA5 downloads.

## Train

```powershell
uv run stress-prediction train-lgb `
  --input data/drought_indices.csv `
  --output-dir experiments/lightgbm `
  --n-lags 12 `
  --horizon 6 `
  --validation-start 2016-01-01 `
  --test-start 2019-01-01 `
  --trials 30 `
  --seed 42
```

Splits use the future label timestamp, rather than the feature timestamp, to
prevent labels across a temporal cutoff from leaking into an earlier partition.
The decision threshold is selected only from validation data.

Training writes:

- `model.pkl`: trusted internal model bundle containing the estimator,
  threshold, feature order, configuration, versions, metrics, and data hash.
- `metrics.json`: validation and test metrics for inspection and automation.

Only load `model.pkl` files produced by a trusted training run. Python pickle
artifacts are not safe to load from untrusted sources.

## Predict

Prediction input uses the same climate and location columns as training, but it
does not need `vegetation_stress_class`:

```powershell
uv run stress-prediction predict `
  --artifact experiments/lightgbm/model.pkl `
  --input data/current_observations.csv `
  --output experiments/lightgbm/predictions.csv
```

The output contains location, observation time, forecast target time, stress
probability, and binary prediction. CSV and Parquet outputs are supported;
Parquet requires the `geo` extra.

## Development

```powershell
uv run ruff format --check src/stress_prediction tests
uv run ruff check src/stress_prediction tests
uv run mypy src/stress_prediction
uv run pytest --cov=stress_prediction
uv build
```

The PyTorch sequence models and the original geospatial preparation scripts are
still experimental and are not part of the production support contract yet.