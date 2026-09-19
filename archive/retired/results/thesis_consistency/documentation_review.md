# Documentation review

Source: `20260701_Opoku_Janet_102202985_thesis (2).docx` in the project root.
Scope: extracted document text and tables in Chapters 3–4, compared with the
patient-workbook audit and `docs/reconstruction_notes.md`. This is a content
review, not a visual-layout review. Original files were not edited.

## What the thesis documents

- Section 3.3 and Section 3.7: retrospective cohort, a 70/30 training/holdout
  split, five-fold cross-validation within training, Optuna tuning and SHAP.
- Table 4.1: median imputation for 47 missing cells (BMI 12, HbA1c 18,
  cholesterol 9, creatinine 8).
- Table 4.2: no duplicates, 21 outlying observations, winsorization at the
  1st/99th percentiles, zero excluded records.
- Table 4.3: standardization for SVM and neural networks, binary encoding for
  sex and smoking, and one-hot encoding for disease type.
- Tables 4.4–4.5: 1,120 training records and 480 test records; training-only
  SMOTE increases 425 readmitted records to 695, balancing 695 non-readmitted
  records, for 1,390 training examples after resampling.
- Section 4.5 and Table 4.12: systolic blood pressure and HbA1c are described
  as the strongest predictors, with qualitative importance labels.

## Findings requiring reconciliation

1. **Hospital count:** the sampling/sample-size sections and Chapter 4 describe
   sixteen hospitals, while Section 3.6 says five participating hospitals.
   The supplied workbook contains sixteen hospital IDs.
2. **Readmission count:** Section 4.2.3 gives 607 readmissions (37.9%), agreeing
   with the workbook. Table 4.7 instead gives 542 (33.9%). Its progression and
   complication counts also differ: 845 and 715, compared with 812 and 705 in
   the workbook.
3. **Blood-pressure mean:** Table 4.6 gives an overall systolic mean of 145.2;
   Table 4.12 gives 145.3. The supplied records give 144.8 at one decimal.
4. **Demographic totals:** Section 4.3 lists 798 males plus 803 females,
   totalling 1,601, despite a cohort of 1,600.
5. **Predictor descriptives:** none of the seven continuous-variable
   mean/SD pairs in Table 4.12 matches the supplied full cohort at the table's
   displayed precision. The separate consistency report documents the values
   and 47 diagnostic checks. No complete table match was found in those checks.
6. **Importance provenance:** the workbook's stored importance sheet ranks
   Age first, while the thesis describes systolic blood pressure first. The
   thesis text reviewed does not provide numerical thresholds for Very High,
   High, Moderate–High or Moderate, nor an unambiguous outcome-specific model
   mapping for every Table 4.12 entry.

## Reproduction details not established by the reviewed documentation

The exact historical random seed, split membership, fitted model, numeric
importance-label thresholds and final tuned parameter values were not found
in the reviewed text/tables. The list of tables mentions a hyperparameter
setting table, but that mention is not a recoverable parameter specification.
Details embedded only in images have not been established by this text review.

The documentation does not establish whether the Table 4.12 descriptives were
computed before or after each individual cleaning step, nor supply the original
missing-cell locations and replacement values. The current workbook has no
missing cells and may already be processed. It is therefore insufficient to
reconstruct the complete history of the data.

## Project documentation issues

`docs/reconstruction_notes.md` says: “The thesis material provides enough
historical implementation detail to determine the precise cause.” That claim
is unsupported by the available evidence and contradicts the same document's
discussion of missing settings. It should instead say that the available
material does **not** establish the precise cause of the discrepancies.

The same paragraph describes possible discrepancies as “slight” without
establishing their size. The notes also call Optuna optional, while the current
`src/models.py` implements tuning within its main model workflow. The new
`src/predictor_summary.py` uses explicitly documented fixed reconstruction
parameters, so its scores are not the historical tuned-model scores.

## Recommended resolution

Keep the submitted document and patient file as preserved references. Retrieve
the original data export, cleaning history and analysis outputs, then reconcile
the contradictory counts and tables with the supervisor. If those materials
cannot be recovered, identify which current results can be reproduced and
document any approved corrections. The documentation does not justify altering
patient measurements or generating replacement records to validate the thesis.
