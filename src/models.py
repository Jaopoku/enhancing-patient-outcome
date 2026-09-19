"""Reconstructed baseline model comparison.

Implements the thesis-documented workflow (Chapter 3.4 / 3.5):
  - a 70/30 stratified holdout split
  - within the training set: stratified 5-fold cross-validation used to
    guide Optuna hyperparameter optimization for each of the four model
    families
  - SMOTE applied to the training folds only (never to validation/test data)
  - per-model preprocessing: StandardScaler applied only for SVM and Neural
    Network; Random Forest and XGBoost trained on unscaled (but imputed and
    winsorized) features (thesis Table 4.3)
  - final models (best CV hyperparameters) refit on the full training set
    and evaluated once, on the untouched holdout test set

Note: the thesis references a "Hyperparameter Setting" table with the final
Optuna-tuned values, but those specific values were not recoverable from the
supplied thesis document. This script therefore performs its own fresh
Optuna search rather than reusing the original (unavailable) tuned values.
"""
import json
import warnings
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from preprocess import DROP, TARGET, build_preprocessor, load_data, SCALED_MODELS

RANDOM_STATE = 42
N_FOLDS = 5
N_OPTUNA_TRIALS = 25
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_pipeline(df, model_name: str, model) -> ImbPipeline:
    """Build preprocess -> SMOTE -> model pipeline, scaling only where the
    thesis specifies (SVM, Neural Network)."""
    scale = model_name in SCALED_MODELS
    return ImbPipeline(
        [
            ("preprocess", build_preprocessor(df, scale=scale)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )


def _cv_score(df, model_name, model, X_train, y_train, cv) -> float:
    """Mean 5-fold stratified CV AUC-ROC for one hyperparameter configuration."""
    pipe = _make_pipeline(df, model_name, model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(
            pipe, X_train, y_train, cv=cv, scoring="roc_auc", n_jobs=1
        )
    return float(np.mean(scores))


def _tune_xgboost(df, X_train, y_train, cv):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 10),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model = XGBClassifier(**params)
        return _cv_score(df, "XGBoost", model, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1)
    return best, study.best_value


def _tune_random_forest(df, X_train, y_train, cv):
    def objective(trial):
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 100, 500, step=50),
            max_depth=trial.suggest_int("max_depth", 3, 20),
            min_samples_split=trial.suggest_int("min_samples_split", 2, 10),
            min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 5),
            max_features=trial.suggest_categorical("max_features", ["sqrt", "log2", None]),
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        model = RandomForestClassifier(**params)
        return _cv_score(df, "Random Forest", model, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update(random_state=RANDOM_STATE, n_jobs=-1)
    return best, study.best_value


def _tune_svm(df, X_train, y_train, cv):
    def objective(trial):
        kernel = trial.suggest_categorical("kernel", ["rbf", "linear"])
        params = dict(
            C=trial.suggest_float("C", 1e-2, 100, log=True),
            kernel=kernel,
            probability=True,
            random_state=RANDOM_STATE,
        )
        if kernel == "rbf":
            params["gamma"] = trial.suggest_float("gamma", 1e-4, 1.0, log=True)
        model = SVC(**params)
        return _cv_score(df, "Support Vector Machine", model, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    best = study.best_params
    best.update(probability=True, random_state=RANDOM_STATE)
    return best, study.best_value


def _tune_mlp(df, X_train, y_train, cv):
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 2)
        layer_size = trial.suggest_categorical("layer_size", [32, 64, 100, 128])
        hidden = (layer_size,) * n_layers
        params = dict(
            hidden_layer_sizes=hidden,
            alpha=trial.suggest_float("alpha", 1e-5, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
            max_iter=1000,
            random_state=RANDOM_STATE,
        )
        model = MLPClassifier(**params)
        return _cv_score(df, "Neural Network", model, X_train, y_train, cv)

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE))
    study.optimize(objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=False)
    p = study.best_params
    n_layers = p.pop("n_layers")
    layer_size = p.pop("layer_size")
    best = dict(
        hidden_layer_sizes=(layer_size,) * n_layers,
        alpha=p["alpha"],
        learning_rate_init=p["learning_rate_init"],
        max_iter=1000,
        random_state=RANDOM_STATE,
    )
    return best, study.best_value


TUNERS = {
    "XGBoost": (_tune_xgboost, XGBClassifier),
    "Random Forest": (_tune_random_forest, RandomForestClassifier),
    "Support Vector Machine": (_tune_svm, SVC),
    "Neural Network": (_tune_mlp, MLPClassifier),
}


def evaluate(n_trials: int = N_OPTUNA_TRIALS, verbose: bool = True):
    global N_OPTUNA_TRIALS
    N_OPTUNA_TRIALS = n_trials

    df = load_data()
    X = df.drop(columns=DROP)
    y = df[TARGET]

    # Thesis 4.4: 1,120 train (70%) / 480 test (30%), stratified.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    best_params_all = {}
    cv_scores_all = {}

    for name, (tuner, model_cls) in TUNERS.items():
        if verbose:
            print(f"Tuning {name} ({N_OPTUNA_TRIALS} Optuna trials, {N_FOLDS}-fold CV)...")
        best_params, best_cv_auc = tuner(df, X_train, y_train, cv)
        best_params_all[name] = best_params
        cv_scores_all[name] = best_cv_auc

        final_model = model_cls(**best_params)
        pipe = _make_pipeline(df, name, final_model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        rows.append(
            {
                "Algorithm": name,
                "Accuracy": accuracy_score(y_test, pred),
                "Precision": precision_score(y_test, pred, zero_division=0),
                "Recall": recall_score(y_test, pred, zero_division=0),
                "F1_Score": f1_score(y_test, pred, zero_division=0),
                "AUC_ROC": roc_auc_score(y_test, prob),
                "CV_AUC_mean": best_cv_auc,
            }
        )
        if verbose:
            print(f"  best CV AUC={best_cv_auc:.4f}  holdout AUC={rows[-1]['AUC_ROC']:.4f}")

    result = pd.DataFrame(rows)
    OUT.mkdir(exist_ok=True)
    result.to_csv(OUT / "reconstructed_model_performance.csv", index=False)
    with open(OUT / "reconstructed_best_hyperparameters.json", "w") as f:
        json.dump(best_params_all, f, indent=2, default=str)
    return result


if __name__ == "__main__":
    print(evaluate().to_string(index=False))
