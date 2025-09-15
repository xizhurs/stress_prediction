import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.models.lightgbm import LGBMClassifier_tuned
from src.features.feature_calculation import feature_extraction
from src.data.split import split_data
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
)

# to add
# soil moisture, temperature, nutrient, drainage
# landcover, landuse, crop type
# topography
# irrigation data
# lst

df = pd.read_csv("data/drought_indices.csv", parse_dates=["valid_time"])[
    [
        "valid_time",
        "latitude",
        "longitude",
        # "drought_class",
        # "climate_stress_class",
        # "heat_stress_class",
        "vegetation_stress_class",
        "tp_mm",
        "pet_mm",
        "T_c",
        "ndvi",
    ]
]

mapping = {
    "mild": 0,
    "moderate": 0,
    "normal": 0,
    "severe": 1,
}


def train_lgb(
    df,
    n_lags=12,
    horizon=1,
    keep_current=False,
    feat_vars=["tp_mm", "pet_mm", "T_c", "ndvi"],
    target_col="vegetation_stress_class",
    binary=True,
):
    X, y = feature_extraction(
        df,
        n_lags=n_lags,
        horizon=horizon,
        keep_current=keep_current,
        feat_vars=feat_vars,
        target_col=target_col,
    )
    if binary:
        y = pd.Series(np.vectorize(mapping.get)(y))
        X = X.reset_index(drop=True)
    (X_train, y_train, X_val, y_val, X_test, y_test) = split_data(
        X, y, target_col=target_col
    )

    lgbclassifier = LGBMClassifier_tuned(X_train, y_train, X_val, y_val, n_trials=30)
    lgbclassifier()
    lgbclassifier.fit()

    proba_va = lgbclassifier.final_clf.predict_proba(X_val)[:, 1]
    ap_va = average_precision_score(y_val, proba_va)
    auc_va = roc_auc_score(y_val, proba_va)
    print(f"Validation PR-AUC={ap_va:.3f}, AUROC={auc_va:.3f}")

    # Choose an operating threshold for alerts (optimize F1 or recall at fixed precision)
    prec_va, rec_va, thr_va = precision_recall_curve(y_val, proba_va)
    f1_va = 2 * prec_va * rec_va / (prec_va + rec_va + 1e-9)
    best_idx_va = f1_va.argmax()
    print(
        f"Best F1 threshold={thr_va[best_idx_va]:.3f}, P={prec_va[best_idx_va]:.2f}, R={rec_va[best_idx_va]:.2f}"
    )

    proba_te = lgbclassifier.final_clf.predict_proba(X_test)[:, 1]
    ap_te = average_precision_score(y_test, proba_te)
    auc_te = roc_auc_score(y_test, proba_te)

    # Evaluate at different thresholds
    thresholds = [0.25, 0.5, 0.75, thr_va[best_idx_va]]
    for threshold in thresholds:
        yhat = (proba_te >= threshold).astype(int)
        prec = precision_score(y_test, yhat)
        rec = recall_score(y_test, yhat)
        f1 = f1_score(y_test, yhat, average="macro")
        print(f"Threshold={threshold:.2f}, P={prec:.2f}, R={rec:.2f}, F1={f1:.3f}")

    # Use the best threshold found on validation set
    yhat = (proba_te >= thr_va[best_idx_va]).astype(int)
    f1 = f1_score(y_test, yhat, average="macro")
    print(f"Test PR-AUC={ap_te:.3f}, AUROC={auc_te:.3f}, F1={f1:.3f}")
    print("\nClassification report (Test) for lightGBM:")
    print(classification_report(y_test, yhat))
    return y_test, yhat


y_test, yhat = train_lgb(
    df,
    n_lags=12,
    horizon=1,
    keep_current=False,
    feat_vars=["tp_mm", "pet_mm", "T_c", "ndvi"],
    target_col="vegetation_stress_class",
    binary=True,
)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
labels = np.unique(y_test)
cm = confusion_matrix(y_test, yhat, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
sns.heatmap(cm_df, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax[0])
ax[0].set_xlabel("Predicted Labels")
ax[0].set_ylabel("True Labels")
ax[0].set_title("LightGBM")
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
