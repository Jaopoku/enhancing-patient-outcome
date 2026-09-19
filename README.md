# Patient-data analysis

Reconstructed Python analysis of the workbook data.


- Added `src/predictor_summary.py` to calculate the eight-predictor table directly
  from `Patient_Data`, without reading the workbook's summary sheets.
- Extended the shared SHAP function to accept a supplied dataset, outcome,
  output directory and model parameters.
- Added input validation for required columns, numeric values, smoking codes
  and binary outcomes, plus tests for statistics, missing values and tied ranks.
- Added HTML, Markdown and CSV results, a thesis comparison record, and run
  metadata recording the input fingerprint and software versions.
- Fixed the workbook filename from `patient_analysis_workbook.xlsx.xlsx` to
  `patient_analysis_workbook.xlsx`. This rename did not change its contents.
- Moved superseded scripts, earlier reports and saved tuning parameters into
  `archive/` and updated the project documentation. Active model code, tests,
  patient data and thesis documents were retained.


## Run the predictor table

```powershell
python -m pip install -r requirements.txt
python src/predictor_summary.py
Start-Process ./results/raw_patient_summary/predictor_table.html
```

Input: `data/patient_analysis_workbook.xlsx`, sheet `Patient_Data`. Use only one `.xlsx` extension.

Results in `results/raw_patient_summary/` include HTML/Markdown/CSV tables, unrounded statistics, SHAP scores and plot, a comparison record, and `run_metadata.json` with the source fingerprint and parameters.

## How the results are generated

### 1. Load and validate the patient records

The command reads `Patient_Data` using pandas and openpyxl. It checks that the
requested predictors and outcome exist, rejects invalid or infinite numeric
values, and requires smoking codes of 0/1 and a nonmissing outcome containing
both classes. It calculates a SHA-256 fingerprint of the workbook before reading
it so the output can be associated with a particular input version.

### 2. Calculate descriptive statistics

For each continuous predictor, the program calculates the arithmetic mean and
sample standard deviation (`pandas.std(ddof=1)`) from nonmissing observed values
across the full cohort. These statistics are calculated before model imputation,
winsorization or resampling. Display precision follows the thesis table; the
statistics CSV retains unrounded values and observed/missing counts.

Smoking is reported as counts and percentages of nonmissing smoking records,
with 1 meaning Yes and 0 meaning No.

#### Predictor columns and display precision

| Table predictor | Patient_Data column | Decimal places for mean and SD |
| --- | --- | ---: |
| Systolic blood pressure (mmHg) | `Systolic_BP` | 1 |
| HbA1c (%) | `HbA1c` | 2 |
| Body mass index (kg/m²) | `BMI` | 1 |
| Creatinine (mg/dL) | `Creatinine` | 2 |
| Age (years) | `Age` | 1 |
| Total cholesterol (mmol/L) | `Cholesterol` | 1 |
| Comorbidity count | `Comorbidity_Count` | 1 |
| Smoking status | `Smoking_Status` | Counts and percentages to 1 decimal |

Comorbidity count is a discrete numeric variable summarized using mean and SD.
The labels assume that source measurements already use the stated units; the
script does not perform unit conversion.

#### Mean

For a predictor with `n` observed values `x₁, …, xₙ`:

```text
mean = (x₁ + x₂ + ... + xₙ) / n
```

Each observed record has equal weight. Missing values are omitted separately
for each predictor, rather than dropping a patient from every calculation.

#### Sample standard deviation

```text
sample variance = Σ(xᵢ − mean)² / (n − 1)
sample SD       = sqrt(sample variance)
display         = mean ± sample SD
```

The denominator is `n − 1`, implemented with `ddof=1`. The displayed SD measures
variation among patient values; it is not a standard error or confidence interval.
At least two observed values are required for a sample SD. The summary displays
`Insufficient observations` when a numeric predictor has only one observed value;
the input validator rejects predictors with no observed values.

The corresponding calculation in `summarize()` is:

```python
values = df[column].dropna()
n = len(values)
mean = float(values.mean())
sd = float(values.std(ddof=1)) if n > 1 else None
```

Worked arithmetic example (illustrative, not patient records): for values
`120, 140, 160`, the mean is `420 / 3 = 140`. The sum of squared deviations is
`400 + 0 + 400 = 800`; sample SD is `sqrt(800 / 2) = 20`. The result is
`140.0 ± 20.0` at one decimal place.

For the currently recorded workbook, the full-cohort calculation gives
`145.3 ± 19.8` for systolic blood pressure after display rounding. The script
computes this from source values, not from the thesis reference text.

#### Smoking counts and percentages

```text
n_valid = number of nonmissing smoking records
n_yes   = number with Smoking_Status == 1
n_no    = number with Smoking_Status == 0
yes_pct = 100 × n_yes / n_valid
no_pct  = 100 × n_no / n_valid
```

In the recorded run, `n_valid = 1,600`, `n_yes = 773` and `n_no = 827`:

```text
Yes: 100 × 773 / 1600 = 48.3125% → 48.3%
No:  100 × 827 / 1600 = 51.6875% → 51.7%
```

Python's percentage formatter operates on fractions, so the code uses
`f"{n_yes / n_valid:.1%}"` without multiplying by 100 a second time.

#### Missing values and rounding

Missing numeric predictor values are excluded from descriptive calculations,
but imputed later for model fitting. Observed and missing counts are exported as
`N_Observed` and `N_Missing`. A missing outcome is rejected rather than silently
excluding that patient from the analysis.

Calculations use unrounded source values. Python's fixed-point formatting rounds
only the displayed mean and SD to the precision listed above. Ranks use full
SHAP scores, not their four-decimal display values. Consequently, scores that
look equal after rounding can have different ranks. Recalculating a statistic
from already-rounded table entries need not reproduce the original calculation.

### 3. Fit the reconstruction model

The default outcome is `Readmission_30D`. Hospital identifiers/names and all
three outcome columns are excluded from the model predictors. The configured
predictor columns in `preprocess.py` are used for model fitting, including
predictors beyond the eight shown in the summary table.

Records are split into 70% training and 30% holdout data, stratified by the
selected outcome with random seed 42. For 1,600 records this gives 1,120 training
records and 480 holdout records before resampling.

Preprocessing is fitted on training data only:

- Numeric values: median imputation followed by clipping to the training
  1st/99th percentile limits.
- Binary variables: most-frequent imputation and ordinal encoding.
- Disease type: most-frequent imputation and one-hot encoding.
- XGBoost receives unscaled features. SMOTE balances the transformed training
  data only; the holdout data is not resampled.

The main table command fits XGBoost with these fixed reconstruction parameters:

| Parameter | Value |
| --- | ---: |
| `n_estimators` | 300 |
| `max_depth` | 6 |
| `learning_rate` | 0.10 |
| `subsample` | 0.80 |
| `colsample_bytree` | 0.80 |
| `random_state` | 42 |
| `eval_metric` | `logloss` |
| `n_jobs` | -1 |

Other model settings use the installed library's defaults. This command does
not run Optuna or load previously tuned parameters.

### 4. Calculate SHAP importance

`shap.TreeExplainer` explains the fitted model on the transformed holdout data.
For each feature, the program averages the absolute SHAP values across holdout
records. These scores are in raw model-output (log-odds) units for this binary
XGBoost model; they are not percentages or causal effects.

The summary table ranks the eight requested predictors by descending score.
Ties receive the same minimum rank. The full feature-importance CSV and plot
also include the model's other features. The historical labels such as Very
High and High are not recalculated because their numerical thresholds have
not been recovered.

#### Mean absolute SHAP formula

Let `φᵢⱼ` be the SHAP contribution of feature `j` for holdout record `i`, and
let `m` be the number of holdout records:

```text
importance(j) = (|φ₁ⱼ| + |φ₂ⱼ| + ... + |φₘⱼ|) / m
```

For this 1,600-record dataset, `m = 480`. The implementation is:

```python
mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
```

Taking absolute values prevents positive and negative contributions from
cancelling. An illustrative set of SHAP contributions `−0.4, 0.2, 0.6` has mean
absolute importance `(0.4 + 0.2 + 0.6) / 3 = 0.4`. This example is arithmetic
only; actual contributions are calculated by TreeExplainer from the fitted model.

The global magnitude score does not specify whether higher predictor values
increase or decrease predictions. Use the SHAP plot and individual contributions
to inspect direction. These are model explanations, not clinical effect estimates.

#### Ranking and interpretation

The summary selects the eight predictor scores and calculates:

```python
ranks = selected.rank(method="min", ascending=False).astype(int)
```

Rank 1 has the largest mean absolute SHAP score. For example, scores
`0.8, 0.8, 0.3` receive ranks `1, 1, 3`. Table rows remain in the thesis predictor
order; they are not sorted by rank. A rank in the summary applies only to these
eight predictors, not to all model features.

Descriptive statistics answer “What values occur in the patient cohort?”
SHAP importance answers “How strongly does this fitted model use a feature on
the holdout sample?” Neither calculation is derived from the other. A higher
mean does not imply higher predictive importance, and a SHAP score does not
measure an increase in accuracy or a percentage contribution to risk.

### 5. Save outputs and the calculation record

| Output | Contents |
| --- | --- |
| `predictor_table.html` | Formatted table to view, print or copy |
| `predictor_table.md` | Table with calculation notes |
| `predictor_table.csv` | Displayed table values |
| `predictor_statistics.csv` | Unrounded statistics, counts, SHAP scores and ranks |
| `reconstructed_feature_importance.csv` | Scores for all model features |
| `reconstructed_shap_summary.png` | Holdout SHAP plot |
| `run_metadata.json` | Input fingerprint, sheet, record count, target, parameters and package versions |
| `correction_record.md` | Submitted values compared with calculated results, with proposed wording for review |

Submitted thesis values are stored as reference text for the comparison record
only. They do not feed into the calculated descriptive statistics or model.
Rerunning the command replaces outputs in the selected output directory; use
`--out` to retain separate runs.

## Latest recorded result

The recorded run used 1,600 records and the workbook fingerprint
`6342d65f371db3a544c677950a590d9fe21f85118aa7fc1951fc23eff6fb4523`.
Its displayed results are shown below. This is a snapshot; the generated files
and metadata are authoritative for later runs.

| Predictor | Mean / frequency | Mean absolute SHAP | Rank among listed predictors |
| --- | --- | ---: | ---: |
| Systolic blood pressure (mmHg) | 145.3 ± 19.8 | 3.3163 | 1 |
| HbA1c (%) | 6.85 ± 1.60 | 2.1514 | 2 |
| BMI (kg/m²) | 28.1 ± 4.9 | 1.5329 | 3 |
| Creatinine (mg/dL) | 1.12 ± 0.39 | 1.1420 | 4 |
| Age (years) | 57.3 ± 13.8 | 0.5391 | 6 |
| Total cholesterol (mmol/L) | 5.4 ± 1.2 | 0.4885 | 7 |
| Comorbidity count | 2.1 ± 1.3 | 0.9259 | 5 |
| Smoking status | Yes: 773 (48.3%); No: 827 (51.7%) | 0.0756 | 8 |

The seven continuous-variable summaries match the thesis at its displayed
precision for this workbook version. The numeric importance scores are freshly
reconstructed results, not recovered historical SHAP values. Generating this
table does not validate the thesis's model-performance metrics.

## Verification

Four tests cover sample SD, missing-value denominators, score-based ranking
including ties, invalid smoking/outcome data, and missing predictors. The latest
cleanup check passed all four tests and confirmed that the active modules import
and the workbook loads 1,600 records.

The input fingerprint and versions in `run_metadata.json` help reproduce a run.
Dependency ranges in `requirements.txt` are not an exact environment lock;
different library versions can change model results. Verify the input fingerprint
and use the recorded versions when comparing runs.

## Active files

- `src/predictor_summary.py`: main table command.
- `src/preprocess.py`: shared loading and preprocessing.
- `src/shap_analysis.py`: shared and standalone SHAP analysis.
- `src/models.py`: four-model comparison with Optuna tuning.
- `scripts/audit_data.py`: patient-data audit.
- `tests/test_predictor_summary.py`: numerical and validation tests.
- `docs/` and the root thesis DOCX: preserved documentation.
- `results/reported_*.csv`: historical references, not current calculations.
- `archive/`: superseded scripts and earlier outputs, preserved for recovery.

## Other commands

```powershell
python scripts/audit_data.py
python -m unittest discover -s tests -v
python src/models.py
python src/shap_analysis.py
```

`models.py` runs 25 Optuna trials per model with five-fold cross-validation. It writes performance and tuned parameters under `results/`. Standalone SHAP uses saved parameters when present, otherwise fixed defaults. Retune after changing the workbook. The predictor-table command always uses fixed parameters.

For another source or outcome:

```powershell
python src/predictor_summary.py --data "C:/path/patients.xlsx" --sheet Patient_Data
python src/predictor_summary.py --target Disease_Progression --out results/progression_summary
python src/predictor_summary.py --target Complication --out results/complication_summary
```

## Calculation and provenance

Descriptive values use observed means and sample SDs (`ddof=1`). Smoking is 0=No, 1=Yes; percentages exclude missing values. Importance uses fresh XGBoost, a stratified 70/30 split, seed 42, training-fitted preprocessing and training-only SMOTE. The default outcome is `Readmission_30D`. Ranks compare the eight listed predictors; scores are not causal effects.

The current workbook differs from the earlier audited version. Earlier reports and tuning parameters were archived to avoid presenting them as current findings or silently reusing them. Matching thesis statistics does not establish a dataset's origin or reproduce historical model performance. See `docs/reconstruction_notes.md` for fingerprints. Patient files remain excluded by `.gitignore`.
