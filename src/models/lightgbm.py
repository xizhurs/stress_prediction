import optuna
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, average_precision_score
from optuna.integration import LightGBMPruningCallback
from functools import partial
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class LGBMClassifier_tuned:
    def __init__(self, X_train, y_train, X_val, y_val, n_trials=30, random_state=42):
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

        self.n_trials = n_trials
        self.random_state = random_state
        self.best_params = None
        self.best_iter = None

    def __call__(self):
        scaler = StandardScaler().set_output(transform="pandas")
        self.X_train = scaler.fit_transform(self.X_train)
        self.X_val = scaler.transform(self.X_val)
        self.scaler = scaler

    def _objective(self, trial):
        # class weights: inverse frequency
        vc = self.y_train.value_counts()
        class_weight = (vc.max() / vc).to_dict()  # {0: w0, 1: w1}

        params = {
            "objective": "binary",
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 300),
            "max_depth": trial.suggest_int("max_depth", -1, 16),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 10),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 10.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
            "min_gain_to_split": trial.suggest_float("min_gain_to_split", 0.0, 0.5),
            # Optional but helpful for imbalance:
            # "scale_pos_weight": float((self.y_train == 0).sum() / max(1, (self.y_train == 1).sum())),
        }

        clf = lgb.LGBMClassifier(
            n_estimators=20000,
            class_weight=class_weight,
            random_state=self.random_state,
            **params
        )

        # Use a metric name supported by your LightGBM build
        eval_metric_name = "average_precision"  # or "aucpr" if your build supports it

        clf.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            eval_metric=eval_metric_name,
            callbacks=[
                lgb.early_stopping(stopping_rounds=300, verbose=False),
                LightGBMPruningCallback(trial, eval_metric_name),
            ],
        )

        # Compute AUCPR on validation with sklearn for the objective value
        p_val = clf.predict_proba(self.X_val)[:, 1]
        ap = average_precision_score(self.y_val, p_val)

        # Report to Optuna and allow pruning also here (belt & suspenders)
        trial.report(ap, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # Save best_iteration_
        trial.set_user_attr("best_iteration", getattr(clf, "best_iteration_", None))

        # Optuna *minimizes*, so return negative to maximize AUCPR
        return ap

    def tune_hyperparams(self):
        study = optuna.create_study(direction="maximize")
        study.optimize(self._objective, n_trials=self.n_trials, show_progress_bar=True)
        self.best_params = study.best_params
        self.best_iter = study.best_trial.user_attrs.get("best_iteration", 1000)

    def fit(self):
        self.tune_hyperparams()
        cls_counts = self.y_train.value_counts()
        class_weight = (cls_counts.max() / cls_counts).to_dict()
        self.final_clf = lgb.LGBMClassifier(
            n_estimators=self.best_iter,
            class_weight=class_weight,
            random_state=self.random_state,
            **self.best_params
        )
        self.final_clf.fit(self.X_train, self.y_train)
        return self.final_clf

    # def predict(self, X_test):
    #     return self.final_clf.predict(X_test)
