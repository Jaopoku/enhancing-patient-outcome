"""Optional standalone Optuna tuning procedure (XGBoost only, single train/validation split).

NOTE: src/models.py now performs Optuna tuning for all four model families
using proper stratified 5-fold cross-validation on the training set, which
is more faithful to the thesis methodology (Section 4.3.2) than the single
80/20 split used here. This script is kept only as a simpler, standalone
reference for tuning XGBoost in isolation; prefer src/models.py for the
full reconstructed workflow.
"""
from pathlib import Path

import optuna
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocess import DROP, TARGET, build_preprocessor, load_data

RANDOM_STATE = 42


def objective(trial, X_train, y_train, X_valid, y_valid, df):
    model = XGBClassifier(
        n_estimators=trial.suggest_int("n_estimators", 100, 600),
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        subsample=trial.suggest_float("subsample", 0.6, 1.0),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    pipe = ImbPipeline(
        [
            ("preprocess", build_preprocessor(df, scale=False)),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("model", model),
        ]
    )
    pipe.fit(X_train, y_train)
    return roc_auc_score(y_valid, pipe.predict_proba(X_valid)[:, 1])


def main(n_trials: int = 50):
    df = load_data()
    X = df.drop(columns=DROP)
    y = df[TARGET]
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: objective(
            trial, X_train, y_train, X_valid, y_valid, df
        ),
        n_trials=n_trials,
    )

    print("Best validation AUC:", study.best_value)
    print("Best parameters:", study.best_params)
    return study


if __name__ == "__main__":
    main()
