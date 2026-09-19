"""Reconstructed preprocessing utilities for the thesis analysis.

Implements, as closely as the thesis documents (Chapter 4.2, Tables 4.1-4.3):
  - median imputation for missing numeric values (Table 4.1)
  - winsorization of continuous variables at the 1st/99th percentiles,
    fit on the TRAINING data only and applied to both train and test (Table 4.2)
  - z-score standardization, applied ONLY to Support Vector Machine and
    Neural Network features -- Random Forest and XGBoost are tree-based
    and are trained on unscaled (but imputed/winsorized) features (Table 4.3)
  - binary 0/1 encoding for Sex and Smoking_Status (Table 4.3)
  - one-hot encoding for the multi-category Disease_Type variable (Table 4.3)
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "patient_analysis_workbook.xlsx"
TARGET = "Readmission_30D"
OUTCOME_COLUMNS = ["Readmission_30D", "Disease_Progression", "Complication"]
DROP = ["Hospital_ID", "Hospital_Name", *OUTCOME_COLUMNS]

# Column roles, matching thesis Table 4.3 exactly.
CONTINUOUS = [
    "Age",
    "BMI",
    "Systolic_BP",
    "Diastolic_BP",
    "HbA1c",
    "Cholesterol",
    "Creatinine",
    "Medication_Adherence",
    "Prior_Admissions",
    "Comorbidity_Count",
    "Length_of_Stay",
]
BINARY = ["Sex", "Smoking_Status"]
MULTICATEGORY = ["Disease_Type"]

# Models that receive z-score standardized continuous features, per Table 4.3.
SCALED_MODELS = {"Support Vector Machine", "Neural Network"}


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip values to the 1st/99th percentile learned on the fitted (training) data.

    Thesis Table 4.2: 21 outlying observations were retained but winsorized at
    the 1st and 99th percentiles rather than removed.
    """

    def __init__(self, lower_pct: float = 1.0, upper_pct: float = 99.0):
        self.lower_pct = lower_pct
        self.upper_pct = upper_pct

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lower_ = np.nanpercentile(X, self.lower_pct, axis=0)
        self.upper_ = np.nanpercentile(X, self.upper_pct, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        return np.clip(X, self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        # Shape-preserving transformer: output columns == input columns.
        if input_features is not None:
            return np.asarray(input_features, dtype=object)
        return np.asarray([f"x{i}" for i in range(len(self.lower_))], dtype=object)


def load_data(path: Path = DATA) -> pd.DataFrame:
    """Load the patient-level workbook from a local/private path."""
    if not path.exists():
        raise FileNotFoundError(
            f"Private data file not found: {path}\n"
            "Place the workbook at data/patient_analysis_workbook.xlsx."
        )
    return pd.read_excel(path, sheet_name="Patient_Data")


def build_preprocessor(df: pd.DataFrame, scale: bool) -> ColumnTransformer:
    """Build the reconstructed imputation/winsorization/(scaling)/encoding transformer.

    scale: True for Support Vector Machine / Neural Network (StandardScaler applied
           after winsorization); False for Random Forest / XGBoost (no scaling,
           matching Table 4.3).
    """
    features = [c for c in df.columns if c not in DROP]
    continuous = [c for c in CONTINUOUS if c in features]
    binary = [c for c in BINARY if c in features]
    multicategory = [c for c in MULTICATEGORY if c in features]

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("winsorize", Winsorizer(lower_pct=1.0, upper_pct=99.0)),
    ]
    if scale:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    # Sex / Smoking_Status -> single 0/1 column each (Table 4.3: "0/1 Encoding").
    binary_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OrdinalEncoder()),
        ]
    )

    # Disease_Type -> one-hot (Table 4.3: "One-Hot Encoding").
    multicategory_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    transformers = []
    if continuous:
        transformers.append(("continuous", numeric_pipe, continuous))
    if binary:
        transformers.append(("binary", binary_pipe, binary))
    if multicategory:
        transformers.append(("multicategory", multicategory_pipe, multicategory))

    return ColumnTransformer(transformers)
