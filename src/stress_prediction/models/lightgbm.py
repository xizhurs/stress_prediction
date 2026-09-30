"""Deterministic LightGBM hyperparameter tuning and fitting."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna_integration.lightgbm import LightGBMPruningCallback
from sklearn.metrics import log_loss


def fit_tuned_classifier(
    train_features: pd.DataFrame,
    train_target: pd.Series,
    validation_features: pd.DataFrame,
    validation_target: pd.Series,
    *,
    trials: int,
    seed: int,
    threads: int,
) -> tuple[lgb.LGBMClassifier, dict[str, Any]]:
    """Tune on validation log loss and fit a final deterministic classifier."""
    if trials < 1:
        raise ValueError("trials must be at least 1")
    if threads < 1:
        raise ValueError("threads must be at least 1")

    class_counts = train_target.value_counts()
    class_weight = (class_counts.max() / class_counts).to_dict()

    def objective(trial: optuna.Trial) -> float:
        parameters: dict[str, Any] = {
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
            "max_depth": trial.suggest_int("max_depth", 3, 14),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": trial.suggest_int("subsample_freq", 0, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 10.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 10.0),
        }
        classifier = lgb.LGBMClassifier(
            objective="binary",
            n_estimators=5_000,
            class_weight=class_weight,
            random_state=seed,
            n_jobs=threads,
            verbosity=-1,
            **parameters,
        )
        classifier.fit(
            train_features,
            train_target,
            eval_X=validation_features,
            eval_y=validation_target,
            eval_metric="binary_logloss",
            callbacks=[
                lgb.early_stopping(100, verbose=False),
                LightGBMPruningCallback(trial, "binary_logloss"),
            ],
        )
        trial.set_user_attr("best_iteration", classifier.best_iteration_)
        probabilities = np.asarray(classifier.predict_proba(validation_features))[:, 1]
        return float(log_loss(validation_target, probabilities, labels=[0, 1]))

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )
    study.optimize(objective, n_trials=trials, show_progress_bar=False)
    best_iteration = int(study.best_trial.user_attrs.get("best_iteration") or 1_000)
    classifier = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=best_iteration,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=threads,
        verbosity=-1,
        **study.best_params,
    )
    classifier.fit(train_features, train_target)
    tuning = {
        "best_iteration": best_iteration,
        "best_parameters": study.best_params,
        "validation_log_loss": study.best_value,
    }
    return classifier, tuning
