"""Generate the thesis predictor table from Patient_Data only.

Descriptive values use observed, unmodified records and sample SD (ddof=1).
Importance uses a freshly fitted XGBoost model and holdout mean absolute SHAP.
No reported summary sheet or previously generated results are read.
"""
import argparse
import hashlib
import json
from html import escape
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pandas as pd

from preprocess import DATA, OUTCOME_COLUMNS, TARGET

PREDICTORS = [
    ("Systolic_BP", "Systolic Blood Pressure (mmHg)", 1),
    ("HbA1c", "HbA1c (%)", 2),
    ("BMI", "Body Mass Index (kg/m²)", 1),
    ("Creatinine", "Creatinine (mg/dL)", 2),
    ("Age", "Age (years)", 1),
    ("Cholesterol", "Total Cholesterol (mmol/L)", 1),
    ("Comorbidity_Count", "Comorbidity Count", 1),
    ("Smoking_Status", "Smoking Status", None),
]

# Submitted Table 4.12: reference text only, never used in calculations.
THESIS_REFERENCE = [
    ("145.3 ± 19.8", "Very High"), ("6.85 ± 1.60", "Very High"),
    ("28.1 ± 4.9", "High"), ("1.12 ± 0.39", "High"),
    ("57.3 ± 13.8", "High"), ("5.4 ± 1.2", "Moderate–High"),
    ("2.1 ± 1.3", "High"), ("Binary Indicator (Yes/No)", "Moderate"),
]


def write_correction_record(table, audit, df, target, out, notes):
    """Create a reviewable replacement table and traceable correction record."""
    title = f"Proposed replacement Table 4.12 — {target}"
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<title>{escape(title)}</title><style>'
        'body{font:16px/1.5 Arial,sans-serif;max-width:1100px;margin:40px auto;padding:0 20px;color:#172b3a}'
        'h1{font-size:25px}table{border-collapse:collapse;width:100%;margin:24px 0}'
        'th,td{text-align:left;padding:12px;border-bottom:1px solid #ccd7df}'
        'th{background:#173e57;color:white}tr:nth-child(even){background:#f1f5f8}'
        'p{white-space:pre-line}@media print{body{margin:0;font-size:11pt}}'
        '</style><body>'
        f'<h1>{escape(title)}</h1>'
        '<p>Calculated from the original patient records. Proposed correction for review; '
        'importance comes from a new reconstructed model.</p>'
        + table.to_html(index=False, border=0, escape=True)
        + f'<p>{escape(notes)}</p></body></html>'
    )
    (out / "predictor_table.html").write_text(html, encoding="utf-8")
    lines = ["# Proposed correction record for Table 4.12", "",
             "Status: prepared for review; the submitted thesis has not been edited.", "",
             "The user identifies the supplied workbook as the original dataset. "
             "The cause of the discrepancy with the submitted table remains unresolved.", "",
             "## Descriptive results", "",
             "| Predictor | Submitted mean / frequency | Calculated mean / frequency |",
             "| --- | --- | --- |"]
    for row, reference in zip(table.itertuples(index=False, name=None), THESIS_REFERENCE):
        lines.append(f"| {row[0]} | {reference[0]} | {row[1]} |")
    lines += ["", "## Importance results", "",
              "The proposed replacement uses numeric SHAP scores and ranks. The qualitative "
              "thesis labels have no recovered numerical classification rule. These new "
              "model results do not establish what the historical model produced.", "",
              "| Predictor | Submitted label | Reconstructed importance |",
              "| --- | --- | --- |"]
    for row, reference in zip(table.itertuples(index=False, name=None), THESIS_REFERENCE):
        lines.append(f"| {row[0]} | {reference[1]} | {row[2]} |")
    ranked = audit.sort_values("Mean_Abs_SHAP", ascending=False)
    labels = dict((col, label) for col, label, _ in PREDICTORS)
    top_score = ranked.iloc[0].Mean_Abs_SHAP
    leaders = ", ".join(labels[col] for col in ranked.loc[
        ranked.Mean_Abs_SHAP == top_score, "Feature"])
    lines += ["", "## Suggested replacement description", "",
              f"Table 4.12 presents descriptive statistics for {len(df):,} patient records "
              f"and predictor importance from a reconstructed XGBoost model for {target}. "
              "Continuous variables are summarized using the mean and sample standard "
              "deviation of observed original values. Smoking is summarized using counts "
              "and percentages of nonmissing records. Importance is the mean absolute "
              "SHAP value on the holdout sample; ranks compare the eight listed predictors. "
              f"The highest score among these predictors was obtained by {leaders}. "
              "These results replace unsupported qualitative labels with reproducible scores.", "",
              "## Review before using this correction", "",
              "- Confirm with the supervisor that observed original values, rather than a "
              "historical processed version, are the intended descriptive population.",
              f"- The importance results apply to {target}; they do not establish the same "
              "ranking for other outcomes or reproduce the original tuned model.",
              "- Review Sections 4.3, 4.5 and 4.6.3 and related conclusions for claims that "
              "depend on the old values and ranking. No historical performance metrics have "
              "been revalidated or corrected by this command.",
              "- The documentation review lists additional conflicts in hospital and outcome "
              "counts that need separate reconciliation.", "", "## Calculation details", "", notes]
    (out / "correction_record.md").write_text("\n".join(lines), encoding="utf-8")


def validate_data(df, target):
    required = [p[0] for p in PREDICTORS] + [target]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")
    df = df.copy()
    for column in required:
        df[column] = pd.to_numeric(df[column], errors="raise")
        values = df[column].dropna()
        if not np.isfinite(values).all():
            raise ValueError(f"{column} contains infinite values.")
        if values.empty:
            raise ValueError(f"{column} has no observed values.")
    if not set(df.Smoking_Status.dropna().unique()) <= {0, 1}:
        raise ValueError("Smoking_Status must be coded 0=No, 1=Yes (or missing).")
    if df[target].isna().any() or set(df[target].unique()) != {0, 1}:
        raise ValueError(f"{target} must contain both 0 and 1, with no missing outcomes.")
    return df


def summarize(df, importance):
    """Return presentation and audit tables; ranks cover the eight listed predictors."""
    scores = importance.set_index("Feature")["Mean_Abs_SHAP"]
    columns = [p[0] for p in PREDICTORS]
    if scores.index.has_duplicates or not set(columns) <= set(scores.index):
        raise ValueError("SHAP output must contain each requested predictor exactly once.")
    selected = scores.loc[columns]
    if not np.isfinite(selected).all() or (selected < 0).any():
        raise ValueError("SHAP importance must be finite and nonnegative.")
    ranks = selected.rank(method="min", ascending=False).astype(int)
    display, audit = [], []
    for column, label, decimals in PREDICTORS:
        values = df[column].dropna()
        n = len(values)
        mean = float(values.mean()) if n else None
        sd = float(values.std(ddof=1)) if n > 1 else None
        if decimals is None:
            yes = int((values == 1).sum())
            no = int((values == 0).sum())
            description = (f"Yes: {yes} ({yes / n:.1%}); No: {no} ({no / n:.1%})"
                           if n else "No observed values")
        else:
            description = (f"{mean:.{decimals}f} ± {sd:.{decimals}f}"
                           if n > 1 else "Insufficient observations")
        score = float(scores[column])
        rank = int(ranks[column])
        display.append({"Predictor": label, "Mean / Frequency": description,
                        "Relative Importance": f"Rank {rank}; SHAP {score:.4f}"})
        audit.append({"Feature": column, "N_Observed": n,
                      "N_Missing": len(df) - n, "Mean": mean, "Sample_SD": sd,
                      "Mean_Abs_SHAP": score, "Rank_Among_Listed_Predictors": rank})
    return pd.DataFrame(display), pd.DataFrame(audit)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--sheet", default="Patient_Data")
    parser.add_argument("--target", choices=OUTCOME_COLUMNS, default=TARGET)
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1]
                        / "results" / "raw_patient_summary")
    args = parser.parse_args()
    # Hash before reading so the run records exactly which source was used.
    source_hash = hashlib.sha256(args.data.read_bytes()).hexdigest()
    df = validate_data(pd.read_excel(args.data, sheet_name=args.sheet), args.target)
    from shap_analysis import DEFAULT_XGB_PARAMS, RANDOM_STATE, compute_shap
    params = DEFAULT_XGB_PARAMS.copy()
    importance = compute_shap(df, target=args.target, output_dir=args.out, params=params)
    table, audit = summarize(df, importance)
    table.to_csv(args.out / "predictor_table.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(args.out / "predictor_statistics.csv", index=False)
    notes = (
        f"Source: {args.data.name}, sheet {args.sheet}; {len(df):,} records. "
        "Means and sample SDs use observed raw values, before imputation or winsorization. "
        "Smoking percentages use nonmissing smoking records (0=No, 1=Yes).\n\n"
        f"Importance: fresh XGBoost model predicting {args.target}; stratified 70/30 "
        f"train/test split, seed {RANDOM_STATE}; imputation and winsorization fitted "
        "on training data only, SMOTE on training data only. SHAP scores are mean "
        "absolute contributions on the holdout in raw model-output (log-odds) units. "
        "Ranks compare the eight listed predictors; all model features appear in "
        "reconstructed_feature_importance.csv. Fixed reconstruction parameters are "
        "recorded in run_metadata.json; historical fitted models are not available.\n\n"
        "The thesis does not specify numeric cutoffs for Very High/High/Moderate–High/"
        "Moderate. This table therefore reports calculated scores and ranks. "
        "Reported thesis numbers and summary sheets are not inputs to this calculation.\n"
    )
    markdown = "# Predictor table calculated from raw patients\n\n"
    markdown += "| Predictor | Mean / Frequency | Relative Importance |\n"
    markdown += "| --- | --- | --- |\n"
    markdown += "\n".join("| " + " | ".join(row) + " |"
                          for row in table.itertuples(index=False, name=None))
    (args.out / "predictor_table.md").write_text(markdown + "\n\n" + notes, encoding="utf-8")
    write_correction_record(table, audit, df, args.target, args.out, notes)
    metadata = {"source": str(args.data.resolve()), "source_sha256": source_hash,
                "sheet": args.sheet, "records": len(df), "target": args.target,
                "model_parameters": params, "test_fraction": 0.30,
                "sample_sd_ddof": 1, "notes": notes,
                "versions": {p: version(p) for p in
                             ["pandas", "numpy", "scikit-learn", "imbalanced-learn",
                              "xgboost", "shap", "openpyxl"]}}
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(table.to_string(index=False))
    print(f"\nSaved results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
