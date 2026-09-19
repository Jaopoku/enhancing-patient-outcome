# Reported vs reconstructed analysis

## What this reconstruction now implements

Following a review of the thesis document (Chapter 3 and Chapter 4, including
Tables 4.1-4.3, 4.5, and 4.9-4.12), the reconstruction was corrected to match
the documented methodology as closely as possible:

- median imputation for missing values (Table 4.1)
- **winsorization at the 1st/99th percentiles**, fit on the training data only (Table 4.2)
- **StandardScaler applied only to Support Vector Machine and Neural Network**
  features; Random Forest and XGBoost trained on unscaled features (Table 4.3)
- 0/1 encoding for Sex and Smoking_Status; one-hot encoding for Disease_Type (Table 4.3)
- a 70/30 stratified holdout split (1,120 / 480 records, matching Table 4.4)
- **stratified 5-fold cross-validation within the training set**, used to
  guide **Optuna hyperparameter optimization** for each of the four model
  families (matching Section 4.3.2 / Table 4.9)
- SMOTE applied to training folds only, never to validation or test data
- final models refit on the full training set with the best CV
  hyperparameters, evaluated once on the untouched holdout test set
- SHAP values computed directly from the fitted XGBoost model with
  `shap.TreeExplainer`, rather than read back from a stored file

These corrections address every methodology gap identified in the earlier
review of this repository (missing winsorization, missing cross-validation,
missing hyperparameter tuning, and a non-functional SHAP script).

## Result: the gap did not close

After implementing all of the above, the reconstructed metrics remain far
below the thesis-reported values for every model, including XGBoost:

| Algorithm | Accuracy (reconstructed) | Accuracy (reported) | AUC-ROC (reconstructed) | AUC-ROC (reported) |
|---|---|---|---|---|
| XGBoost | 64.0% | 93.9% | 0.648 | 0.985 |
| Random Forest | 64.0% | 90.2% | 0.662 | 0.963 |
| Support Vector Machine | 68.1% | 89.9% | 0.714 | 0.961 |
| Neural Network | 65.0% | 89.6% | 0.697 | 0.959 |

(See `reported_vs_reconstructed.csv` for full precision/recall/F1 figures.)

## Interpretation

Because implementing every documented preprocessing and validation step did
**not** close the gap, the remaining difference is unlikely to be explained
solely by the earlier missing-methodology issues (winsorization, CV,
hyperparameter search). Plausible remaining explanations include:

1. **The specific Optuna-tuned hyperparameter values used in the original
   analysis were not recoverable from the supplied thesis document** (a
   "Hyperparameter Setting" table is referenced but its contents were not
   present in extractable form). This reconstruction performs its own fresh,
   time-limited hyperparameter search (10 Optuna trials per model, due to
   compute constraints) rather than reusing the original values.
2. **Feature engineering differences.** The thesis narrative references
   derived or additional clinical detail that may not be fully captured by
   the 14 raw feature columns available in the supplied workbook.
3. **A difference between the dataset used for the thesis analysis and the
   patient-level workbook supplied for this reconstruction.** The consistent,
   large gap across all four independently-implemented model families
   (rather than one specific model performing poorly) is more typical of a
   data-level difference than a methodology-level one.

This reconstruction does not have enough information to determine which of
these explanations (or what combination) accounts for the remaining gap.
No data or code was altered to force the reconstructed results toward the
reported values.

## Status

The reported values in `results/reported_model_performance.csv` are
preserved unchanged. The reconstructed values in
`results/reconstructed_model_performance.csv` reflect a methodologically
faithful, independently-run implementation of the documented workflow, and
should be read as an empirical reproducibility check rather than a
replacement for the thesis-reported results.
