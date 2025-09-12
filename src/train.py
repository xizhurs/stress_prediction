import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from src.models.lightgbm import LGBMClassifier_tuned
from src.features.feature_calculation import feature_extraction
from src.data.split import split_data

df = pd.read_csv("data/drought_indices.csv", parse_dates=["valid_time"])[
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
]


def train_lgb(df):
    X, y = feature_extraction(
        df, n_lags=12, horizon=6, keep_current=False, target_col="drought_class"
    )
    (X_train, y_train, X_val, y_val, X_test, y_test) = split_data(X, y)

    lgbclassifier = LGBMClassifier_tuned(X_train, y_train, X_val, y_val, n_trials=30)
    lgbclassifier()
    lgbclassifier.fit()
    y_pred_test_lgb = lgbclassifier.final_clf.predict(
        lgbclassifier.scaler.transform(X_test)
    )
    print("\nClassification report (Test) for lightGBM:")
    print(classification_report(y_test, y_pred_test_lgb, digits=3))
    return y_test, y_pred_test_lgb


y_test, y_pred_test_lgb = train_lgb(df)


fig, ax = plt.subplots(1, 3, figsize=(15, 5))
labels = np.unique(y_test)
cm = confusion_matrix(y_test, y_pred_test_lgb, labels=labels)
cm_df = pd.DataFrame(cm, index=labels, columns=labels)
sns.heatmap(cm_df, annot=True, fmt="g", cmap="Blues", cbar=False, ax=ax[0])
ax[0].set_xlabel("Predicted Labels")
ax[0].set_ylabel("True Labels")
ax[0].set_title("LightGBM")
# plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
