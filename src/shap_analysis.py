"""Reconstructed SHAP interpretability analysis.

Trains the best-performing model family (XGBoost, per the thesis and this
reconstruction's own comparison) using the tuned hyperparameters saved by
models.py, then computes real SHAP values with shap.TreeExplainer on the
holdout test set -- rather than reading back previously reported numbers.

Outputs:
  results/reconstructed_feature_importance.csv
      Mean absolute SHAP value per feature, computed independently here.
  results/reconstructed_shap_summary.png
      SHAP summary (beeswarm) plot, for visual inspection.

This is compared against, but does NOT overwrite, results/reported_feature_importance.csv
(from the workbook) since -- as documented in docs/reconstruction_notes.md --
that file and the thesis's own Table 4.12 disagree with each other.
"""
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from preprocess import DROP, TARGET, build_preprocessor, load_data

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"
BEST_PARAMS_FILE = OUT / "reconstructed_best_hyperparameters.json"

DEFAULT_XGB_PARAMS = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.10,
    subsample=0.80,
    colsample_bytree=0.80,
    eval_metric="logloss",
    random_state=RANDOM_STATE,
    n_jobs=-1,
)


def _load_xgb_params() -> dict:
    """Use the Optuna-tuned XGBoost params from models.py if available,
    otherwise fall back to fixed defaults."""
    if BEST_PARAMS_FILE.exists():
        with open(BEST_PARAMS_FILE) as f:
            saved = json.load(f)
        if "XGBoost" in saved:
            params = dict(saved["XGBoost"])
            params["random_state"] = RANDOM_STATE
            params["n_jobs"] = -1
            return params
    return DEFAULT_XGB_PARAMS


def compute_shap(df=None, *, target=TARGET, output_dir=OUT, params=None) -> pd.DataFrame:
    """Fit on raw records and explain the holdout; optionally select input/target."""
    df = load_data() if df is None else df
    output_dir = Path(output_dir)
    X = df.drop(columns=DROP, errors="ignore")
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )

    # XGBoost is unscaled per Table 4.3.
    preprocessor = build_preprocessor(df, scale=False)
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)
    feature_names = preprocessor.get_feature_names_out()

    X_train_res, y_train_res = SMOTE(random_state=RANDOM_STATE).fit_resample(
        X_train_t, y_train
    )

    params = _load_xgb_params() if params is None else params
    model = XGBClassifier(**params)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X_train_res, y_train_res)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_test_t)

    mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
    importance = (
        pd.DataFrame(
            {
                "Feature": [f.split("__")[-1] for f in feature_names],
                "Mean_Abs_SHAP": mean_abs_shap,
            }
        )
        .sort_values("Mean_Abs_SHAP", ascending=False)
        .reset_index(drop=True)
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    importance.to_csv(output_dir / "reconstructed_feature_importance.csv", index=False)

    plt.figure()
    shap.summary_plot(
        shap_values.values,
        X_test_t,
        feature_names=[f.split("__")[-1] for f in feature_names],
        show=False,
    )
    plt.tight_layout()
    plt.savefig(output_dir / "reconstructed_shap_summary.png", dpi=150)
    plt.close()

    return importance


def load_reported_shap(path: Path = None) -> pd.DataFrame:
    """Read the thesis-reported feature importance CSV for comparison only
    (this is NOT computed here -- see compute_shap() for the real analysis)."""
    path = path or (OUT / "reported_feature_importance.csv")
    if not path.exists():
        raise FileNotFoundError(f"Reported feature importance file not found: {path}")
    return pd.read_csv(path)


if __name__ == "__main__":
    reconstructed = compute_shap()
    print("=== Reconstructed SHAP feature importance (computed here) ===")
    print(reconstructed.to_string(index=False))
    try:
        reported = load_reported_shap()
        print("\n=== Reported feature importance (from workbook, for comparison) ===")
        print(reported.to_string(index=False))
    except FileNotFoundError as e:
        print(e)
