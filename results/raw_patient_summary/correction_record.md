# Proposed correction record for Table 4.12

Status: prepared for review; the submitted thesis has not been edited.

The user identifies the supplied workbook as the original dataset. The cause of the discrepancy with the submitted table remains unresolved.

## Descriptive results

| Predictor | Submitted mean / frequency | Calculated mean / frequency |
| --- | --- | --- |
| Systolic Blood Pressure (mmHg) | 145.3 ± 19.8 | 145.3 ± 19.8 |
| HbA1c (%) | 6.85 ± 1.60 | 6.85 ± 1.60 |
| Body Mass Index (kg/m²) | 28.1 ± 4.9 | 28.1 ± 4.9 |
| Creatinine (mg/dL) | 1.12 ± 0.39 | 1.12 ± 0.39 |
| Age (years) | 57.3 ± 13.8 | 57.3 ± 13.8 |
| Total Cholesterol (mmol/L) | 5.4 ± 1.2 | 5.4 ± 1.2 |
| Comorbidity Count | 2.1 ± 1.3 | 2.1 ± 1.3 |
| Smoking Status | Binary Indicator (Yes/No) | Yes: 773 (48.3%); No: 827 (51.7%) |

## Importance results

The proposed replacement uses numeric SHAP scores and ranks. The qualitative thesis labels have no recovered numerical classification rule. These new model results do not establish what the historical model produced.

| Predictor | Submitted label | Reconstructed importance |
| --- | --- | --- |
| Systolic Blood Pressure (mmHg) | Very High | Rank 1; SHAP 3.3163 |
| HbA1c (%) | Very High | Rank 2; SHAP 2.1514 |
| Body Mass Index (kg/m²) | High | Rank 3; SHAP 1.5329 |
| Creatinine (mg/dL) | High | Rank 4; SHAP 1.1420 |
| Age (years) | High | Rank 6; SHAP 0.5391 |
| Total Cholesterol (mmol/L) | Moderate–High | Rank 7; SHAP 0.4885 |
| Comorbidity Count | High | Rank 5; SHAP 0.9259 |
| Smoking Status | Moderate | Rank 8; SHAP 0.0756 |

## Suggested replacement description

Table 4.12 presents descriptive statistics for 1,600 patient records and predictor importance from a reconstructed XGBoost model for Readmission_30D. Continuous variables are summarized using the mean and sample standard deviation of observed original values. Smoking is summarized using counts and percentages of nonmissing records. Importance is the mean absolute SHAP value on the holdout sample; ranks compare the eight listed predictors. The highest score among these predictors was obtained by Systolic Blood Pressure (mmHg). These results replace unsupported qualitative labels with reproducible scores.

## Review before using this correction

- Confirm with the supervisor that observed original values, rather than a historical processed version, are the intended descriptive population.
- The importance results apply to Readmission_30D; they do not establish the same ranking for other outcomes or reproduce the original tuned model.
- Review Sections 4.3, 4.5 and 4.6.3 and related conclusions for claims that depend on the old values and ranking. No historical performance metrics have been revalidated or corrected by this command.
- The documentation review lists additional conflicts in hospital and outcome counts that need separate reconciliation.

## Calculation details

Source: patient_analysis_workbook.xlsx, sheet Patient_Data; 1,600 records. Means and sample SDs use observed raw values, before imputation or winsorization. Smoking percentages use nonmissing smoking records (0=No, 1=Yes).

Importance: fresh XGBoost model predicting Readmission_30D; stratified 70/30 train/test split, seed 42; imputation and winsorization fitted on training data only, SMOTE on training data only. SHAP scores are mean absolute contributions on the holdout in raw model-output (log-odds) units. Ranks compare the eight listed predictors; all model features appear in reconstructed_feature_importance.csv. Fixed reconstruction parameters are recorded in run_metadata.json; historical fitted models are not available.

The thesis does not specify numeric cutoffs for Very High/High/Moderate–High/Moderate. This table therefore reports calculated scores and ranks. Reported thesis numbers and summary sheets are not inputs to this calculation.
