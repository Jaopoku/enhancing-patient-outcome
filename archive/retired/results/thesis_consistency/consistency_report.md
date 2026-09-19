# Thesis Table 4.12 consistency check

## Result

No complete match was found in 47 diagnostic scenarios.
This does not identify the historical analysis or prove that no other dataset/version was used.
No patient records were changed. The file fingerprint was checked before and after the audit.

## Full patient cohort

Source: `patient_analysis_workbook.xlsx`, `Patient_Data`: 1,600 records, 19 columns.
Missing cells: 0; duplicate rows: 0.

| Predictor | Thesis mean ± SD | Raw mean ± sample SD | Both match at thesis precision? |
| --- | --- | --- | --- |
| Systolic_BP | 145.3 ± 19.8 | 144.8 ± 19.2 | No |
| HbA1c | 6.85 ± 1.60 | 6.82 ± 1.53 | No |
| BMI | 28.1 ± 4.9 | 28.1 ± 5.0 | No |
| Creatinine | 1.12 ± 0.39 | 1.09 ± 0.38 | No |
| Age | 57.3 ± 13.8 | 57.4 ± 12.1 | No |
| Cholesterol | 5.4 ± 1.2 | 5.6 ± 1.2 | No |
| Comorbidity_Count | 2.1 ± 1.3 | 2.0 ± 1.4 | No |

Smoking: Yes 334; No 1266. The thesis phrase 'Binary Indicator (Yes/No)' describes coding, not a frequency to reproduce.

## Checks and interpretation

- Complete-case filtering, duplicate removal and median imputation leave this file unchanged.
- Both sample SD (n−1 denominator) and population SD (n denominator) were checked.
- Winsorization at the 1st/99th percentiles was checked on the full cohort and with training-fitted bounds.
- The reconstructed 70/30 stratified train/test split was checked for each of the three outcomes, with seed 42.
- Individual disease, sex, smoking, outcome and hospital groups were checked diagnostically. These groups are not replacements for the full cohort described in the thesis.
- No arbitrary row selection, random-seed search, or adjustment of patient values was performed.
- Standardized values would no longer have the table's original measurement units. SMOTE-generated training examples are not original cohort members and were not used for patient descriptives.

The workbook's existing disease summary agrees with the raw data to its stored precision: True.
This supports internal consistency of that summary with this file; it does not authenticate the historical dataset.

## Thesis observations

Source: 20260701_Opoku_Janet_102202985_thesis (2).docx, Chapter 4 preprocessing, descriptive-statistics section and Table 4.12 (text extracted from the supplied document).

- The thesis says zero records were excluded and all 1,600 were retained, so reducing the cohort is not supported by that description.
- The thesis describes 47 initially missing values, while this workbook contains none. It may already be processed; the original missing-value locations cannot be recovered from this file.
- The thesis descriptive table gives overall systolic BP as 145.2, while Table 4.12 gives 145.3. The raw cohort gives 144.8 at one decimal.
- The thesis demographic paragraph lists 798 males and 803 females, totalling 1,601 despite a stated cohort of 1,600. This is an internal reporting inconsistency.
- The workbook's stored feature importance ranks Age first; Table 4.12 describes systolic BP and HbA1c as Very High. Original model outputs and label thresholds are needed to reconcile these.

## Scenario results

Matching pairs means both mean and SD round to the thesis values for that predictor (out of seven).

| Scenario | Records | Matching pairs |
| --- | ---: | ---: |
| Full cohort: sample SD | 1600 | 0/7 |
| Full cohort: population SD | 1600 | 0/7 |
| Complete cases across all columns | 1600 | 0/7 |
| Duplicate rows removed | 1600 | 0/7 |
| Median imputation | 1600 | 0/7 |
| Full-cohort 1st/99th percentile winsorization | 1600 | 0/7 |
| Train, 70/30 split stratified by Readmission_30D, seed 42 | 1120 | 0/7 |
| Train, Readmission_30D split, training-fitted winsorization | 1120 | 0/7 |
| Test, 70/30 split stratified by Readmission_30D, seed 42 | 480 | 0/7 |
| Test, Readmission_30D split, training-fitted winsorization | 480 | 0/7 |
| Train, 70/30 split stratified by Disease_Progression, seed 42 | 1120 | 0/7 |
| Train, Disease_Progression split, training-fitted winsorization | 1120 | 0/7 |
| Test, 70/30 split stratified by Disease_Progression, seed 42 | 480 | 0/7 |
| Test, Disease_Progression split, training-fitted winsorization | 480 | 0/7 |
| Train, 70/30 split stratified by Complication, seed 42 | 1120 | 0/7 |
| Train, Complication split, training-fitted winsorization | 1120 | 0/7 |
| Test, 70/30 split stratified by Complication, seed 42 | 480 | 0/7 |
| Test, Complication split, training-fitted winsorization | 480 | 1/7 |
| Subgroup Disease_Type=Cardiovascular Disease | 394 | 0/7 |
| Subgroup Disease_Type=Diabetes | 586 | 0/7 |
| Subgroup Disease_Type=Hypertension | 620 | 0/7 |
| Subgroup Sex=Female | 766 | 0/7 |
| Subgroup Sex=Male | 834 | 0/7 |
| Subgroup Smoking_Status=0 | 1266 | 0/7 |
| Subgroup Smoking_Status=1 | 334 | 0/7 |
| Subgroup Readmission_30D=0 | 993 | 0/7 |
| Subgroup Readmission_30D=1 | 607 | 0/7 |
| Subgroup Disease_Progression=0 | 788 | 0/7 |
| Subgroup Disease_Progression=1 | 812 | 0/7 |
| Subgroup Complication=0 | 895 | 0/7 |
| Subgroup Complication=1 | 705 | 0/7 |
| Subgroup Hospital_ID=1 | 100 | 0/7 |
| Subgroup Hospital_ID=2 | 100 | 0/7 |
| Subgroup Hospital_ID=3 | 100 | 0/7 |
| Subgroup Hospital_ID=4 | 100 | 0/7 |
| Subgroup Hospital_ID=5 | 100 | 0/7 |
| Subgroup Hospital_ID=6 | 100 | 0/7 |
| Subgroup Hospital_ID=7 | 100 | 0/7 |
| Subgroup Hospital_ID=8 | 100 | 0/7 |
| Subgroup Hospital_ID=9 | 100 | 0/7 |
| Subgroup Hospital_ID=10 | 100 | 0/7 |
| Subgroup Hospital_ID=11 | 100 | 0/7 |
| Subgroup Hospital_ID=12 | 100 | 0/7 |
| Subgroup Hospital_ID=13 | 100 | 0/7 |
| Subgroup Hospital_ID=14 | 100 | 0/7 |
| Subgroup Hospital_ID=15 | 100 | 0/7 |
| Subgroup Hospital_ID=16 | 100 | 0/7 |

## Next step

Retrieve the original pre-cleaning dataset, cleaning log and analysis script/model outputs. If these are unavailable, use this workbook's computed results and document corrections with the supervisor. These checks do not support constructing replacement patients to validate the thesis.

Source SHA-256: `8cc4ed880f1f92b583ff7b814512db80460a779179c1eeb53ade2d2541a2d9c8`
