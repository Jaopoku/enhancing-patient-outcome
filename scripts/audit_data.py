"""Audit the local/private patient workbook.

The workbook itself is intentionally excluded from GitHub.
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "patient_analysis_workbook.xlsx"
OUT = ROOT / "results"


def main():
    if not DATA.exists():
        raise FileNotFoundError(
            f"Private data file not found: {DATA}\n"
            "Place the workbook at data/patient_analysis_workbook.xlsx."
        )

    df = pd.read_excel(DATA, sheet_name="Patient_Data")
    hospital_counts = df.groupby("Hospital_ID").size()

    audit = pd.DataFrame(
        [
            ["Patient_Data rows", len(df)],
            ["Patient_Data columns", len(df.columns)],
            ["Duplicate rows", int(df.duplicated().sum())],
            ["Total missing cells", int(df.isna().sum().sum())],
            ["Hospitals", int(df["Hospital_ID"].nunique())],
            ["Patients per hospital (min)", int(hospital_counts.min())],
            ["Patients per hospital (max)", int(hospital_counts.max())],
            ["Readmission_30D prevalence", df["Readmission_30D"].mean()],
            ["Disease_Progression prevalence", df["Disease_Progression"].mean()],
            ["Complication prevalence", df["Complication"].mean()],
            ["Mean age", df["Age"].mean()],
            ["Mean BMI", df["BMI"].mean()],
            ["Mean systolic BP", df["Systolic_BP"].mean()],
        ],
        columns=["Metric", "Value"],
    )

    OUT.mkdir(exist_ok=True)
    audit.to_csv(OUT / "data_audit.csv", index=False)
    print(audit.to_string(index=False))


if __name__ == "__main__":
    main()
