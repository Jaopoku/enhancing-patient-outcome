"""Read-only investigation of thesis Table 4.12 against the patient workbook.

Writes aggregate findings as Markdown and JSON; never edits patient records.
Scenarios are diagnostic checks, not proposals to select data to fit a result.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "patient_analysis_workbook.xlsx"
OUT = ROOT / "results" / "thesis_consistency"
REFERENCE = {
    "Systolic_BP": (145.3, 19.8, 1),
    "HbA1c": (6.85, 1.60, 2),
    "BMI": (28.1, 4.9, 1),
    "Creatinine": (1.12, 0.39, 2),
    "Age": (57.3, 13.8, 1),
    "Cholesterol": (5.4, 1.2, 1),
    "Comorbidity_Count": (2.1, 1.3, 1),
}


def describe(name, data, ddof=1):
    values = {}
    for col, (ref_mean, ref_sd, digits) in REFERENCE.items():
        mean, sd = float(data[col].mean()), float(data[col].std(ddof=ddof))
        mean_match = f"{mean:.{digits}f}" == f"{ref_mean:.{digits}f}"
        sd_match = f"{sd:.{digits}f}" == f"{ref_sd:.{digits}f}"
        values[col] = {"mean": mean, "sd": sd, "mean_matches": mean_match,
                       "sd_matches": sd_match}
    return {"scenario": name, "n": len(data), "ddof": ddof,
            "matching_mean_sd_pairs": sum(v["mean_matches"] and v["sd_matches"]
                                           for v in values.values()),
            "values": values}


def main():
    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    df = pd.read_excel(SOURCE, sheet_name="Patient_Data")
    cols = list(REFERENCE)
    numeric = df[cols]
    scenarios = [describe("Full cohort: sample SD", df),
                 describe("Full cohort: population SD", df, ddof=0),
                 describe("Complete cases across all columns", df.dropna()),
                 describe("Duplicate rows removed", df.drop_duplicates()),
                 describe("Median imputation", numeric.fillna(numeric.median()))]
    lower, upper = numeric.quantile(.01), numeric.quantile(.99)
    winsorized = numeric.clip(lower=lower, upper=upper, axis=1)
    scenarios.append(describe("Full-cohort 1st/99th percentile winsorization", winsorized))
    # Seed 42 is from the reconstructed code, not an established historical seed.
    for target in ["Readmission_30D", "Disease_Progression", "Complication"]:
        train, test = train_test_split(df, test_size=.30, random_state=42,
                                      stratify=df[target])
        train_lower = train[cols].quantile(.01)
        train_upper = train[cols].quantile(.99)
        for label, subset in [("Train", train), ("Test", test)]:
            scenarios.append(describe(f"{label}, 70/30 split stratified by {target}, seed 42", subset))
            scenarios.append(describe(f"{label}, {target} split, training-fitted winsorization",
                                      subset[cols].clip(train_lower, train_upper, axis=1)))
    for group in ["Disease_Type", "Sex", "Smoking_Status", "Readmission_30D",
                  "Disease_Progression", "Complication", "Hospital_ID"]:
        for value, subset in df.groupby(group):
            scenarios.append(describe(f"Subgroup {group}={value}", subset))
    # Check whether the workbook's stored disease summary describes these records.
    stored = pd.read_excel(SOURCE, sheet_name="Descriptive_Summary").set_index("Disease_Type")
    grouped = df.groupby("Disease_Type").agg(Patients=("Age", "size"), Mean_Age=("Age", "mean"),
             Mean_HbA1c=("HbA1c", "mean"), Readmission_Rate=("Readmission_30D", "mean"),
             Progression_Rate=("Disease_Progression", "mean"), Complication_Rate=("Complication", "mean"))
    summary_agrees = bool(np.allclose(grouped.loc[stored.index].round(3).to_numpy(),
                                      stored[grouped.columns].to_numpy(), atol=1e-10, rtol=0))
    matches = [s["scenario"] for s in scenarios if s["matching_mean_sd_pairs"] == len(cols)]
    assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == source_hash
    metadata = {"source": str(SOURCE), "sha256": source_hash, "rows": len(df),
                "columns": len(df.columns), "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
                "sheets": pd.ExcelFile(SOURCE).sheet_names,
                "stored_disease_summary_matches_to_3_decimals": summary_agrees,
                "matching_scenarios": matches, "scenarios": scenarios}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "checks.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    lines = ["# Thesis Table 4.12 consistency check", "",
             "## Result", "",
             f"No complete match was found in {len(scenarios)} diagnostic scenarios."
             if not matches else f"Matching scenarios: {matches}",
             "This does not identify the historical analysis or prove that no other dataset/version was used.",
             "No patient records were changed. The file fingerprint was checked before and after the audit.", "",
             "## Full patient cohort", "",
             f"Source: `{SOURCE.name}`, `Patient_Data`: {len(df):,} records, {len(df.columns)} columns.",
             f"Missing cells: {metadata['missing_cells']}; duplicate rows: {metadata['duplicate_rows']}.", "",
             "| Predictor | Thesis mean ± SD | Raw mean ± sample SD | Both match at thesis precision? |",
             "| --- | --- | --- | --- |"]
    raw = scenarios[0]["values"]
    for col, (mean, sd, digits) in REFERENCE.items():
        actual = raw[col]
        match = actual['mean_matches'] and actual['sd_matches']
        lines.append(f"| {col} | {mean:.{digits}f} ± {sd:.{digits}f} | "
                     f"{actual['mean']:.{digits}f} ± {actual['sd']:.{digits}f} | {'Yes' if match else 'No'} |")
    smoking = df.Smoking_Status.value_counts()
    lines += ["", f"Smoking: Yes {smoking.get(1, 0)}; No {smoking.get(0, 0)}. "
              "The thesis phrase 'Binary Indicator (Yes/No)' describes coding, not a frequency to reproduce.", "",
              "## Checks and interpretation", "",
              "- Complete-case filtering, duplicate removal and median imputation leave this file unchanged.",
              "- Both sample SD (n−1 denominator) and population SD (n denominator) were checked.",
              "- Winsorization at the 1st/99th percentiles was checked on the full cohort and with training-fitted bounds.",
              "- The reconstructed 70/30 stratified train/test split was checked for each of the three outcomes, with seed 42.",
              "- Individual disease, sex, smoking, outcome and hospital groups were checked diagnostically. "
              "These groups are not replacements for the full cohort described in the thesis.",
              "- No arbitrary row selection, random-seed search, or adjustment of patient values was performed.",
              "- Standardized values would no longer have the table's original measurement units. "
              "SMOTE-generated training examples are not original cohort members and were not used for patient descriptives.", "",
              f"The workbook's existing disease summary agrees with the raw data to its stored precision: {summary_agrees}.",
              "This supports internal consistency of that summary with this file; it does not authenticate the historical dataset.", "",
              "## Thesis observations", "",
              "Source: 20260701_Opoku_Janet_102202985_thesis (2).docx, Chapter 4 preprocessing, "
              "descriptive-statistics section and Table 4.12 (text extracted from the supplied document).", "",
              "- The thesis says zero records were excluded and all 1,600 were retained, so reducing the cohort "
              "is not supported by that description.",
              "- The thesis describes 47 initially missing values, while this workbook contains none. "
              "It may already be processed; the original missing-value locations cannot be recovered from this file.",
              "- The thesis descriptive table gives overall systolic BP as 145.2, while Table 4.12 gives 145.3. "
              "The raw cohort gives 144.8 at one decimal.",
              "- The thesis demographic paragraph lists 798 males and 803 females, totalling 1,601 "
              "despite a stated cohort of 1,600. This is an internal reporting inconsistency.",
              "- The workbook's stored feature importance ranks Age first; Table 4.12 describes systolic BP "
              "and HbA1c as Very High. Original model outputs and label thresholds are needed to reconcile these.", "",
              "## Scenario results", "",
              "Matching pairs means both mean and SD round to the thesis values for that predictor (out of seven).", "",
              "| Scenario | Records | Matching pairs |", "| --- | ---: | ---: |"]
    for s in scenarios:
        lines.append(f"| {s['scenario']} | {s['n']} | {s['matching_mean_sd_pairs']}/7 |")
    lines += ["", "## Next step", "",
              "Retrieve the original pre-cleaning dataset, cleaning log and analysis script/model outputs. "
              "If these are unavailable, use this workbook's computed results and document corrections "
              "with the supervisor. These checks do not support constructing replacement patients to validate the thesis.",
              "", f"Source SHA-256: `{source_hash}`", ""]
    (OUT / "consistency_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Checked {len(scenarios)} scenarios; full-table matches: {len(matches)}")
    print(f"Stored disease summary agrees: {summary_agrees}")
    print(f"Report: {OUT / 'consistency_report.md'}")
    print('Age SD: sample',raw['Age']['sd'],'population',scenarios[1]['values']['Age']['sd'],
          'winsorized',scenarios[5]['values']['Age']['sd'])


if __name__ == "__main__":
    main()
