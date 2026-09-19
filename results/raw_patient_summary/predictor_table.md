# Predictor table calculated from raw patients

| Predictor | Mean / Frequency | Relative Importance |
| --- | --- | --- |
| Systolic Blood Pressure (mmHg) | 145.3 ± 19.8 | Rank 1; SHAP 3.3163 |
| HbA1c (%) | 6.85 ± 1.60 | Rank 2; SHAP 2.1514 |
| Body Mass Index (kg/m²) | 28.1 ± 4.9 | Rank 3; SHAP 1.5329 |
| Creatinine (mg/dL) | 1.12 ± 0.39 | Rank 4; SHAP 1.1420 |
| Age (years) | 57.3 ± 13.8 | Rank 6; SHAP 0.5391 |
| Total Cholesterol (mmol/L) | 5.4 ± 1.2 | Rank 7; SHAP 0.4885 |
| Comorbidity Count | 2.1 ± 1.3 | Rank 5; SHAP 0.9259 |
| Smoking Status | Yes: 773 (48.3%); No: 827 (51.7%) | Rank 8; SHAP 0.0756 |

Source: patient_analysis_workbook.xlsx, sheet Patient_Data; 1,600 records. Means and sample SDs use observed raw values, before imputation or winsorization. Smoking percentages use nonmissing smoking records (0=No, 1=Yes).

Importance: fresh XGBoost model predicting Readmission_30D; stratified 70/30 train/test split, seed 42; imputation and winsorization fitted on training data only, SMOTE on training data only. SHAP scores are mean absolute contributions on the holdout in raw model-output (log-odds) units. Ranks compare the eight listed predictors; all model features appear in reconstructed_feature_importance.csv. Fixed reconstruction parameters are recorded in run_metadata.json; historical fitted models are not available.

The thesis does not specify numeric cutoffs for Very High/High/Moderate–High/Moderate. This table therefore reports calculated scores and ranks. Reported thesis numbers and summary sheets are not inputs to this calculation.
