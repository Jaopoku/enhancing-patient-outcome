# Reconstruction and dataset versions

The implementation is reconstructed. Original historical fitted models and complete settings have not been recovered.

## Workbook versions

- Earlier audited SHA-256: `8cc4ed880f1f92b583ff7b814512db80460a779179c1eeb53ade2d2541a2d9c8`.
- Current SHA-256 at cleanup: `6342d65f371db3a544c677950a590d9fe21f85118aa7fc1951fc23eff6fb4523`.

The current workbook was found with a doubled `.xlsx` extension and renamed without changing its contents. Its displayed means and SDs match the seven continuous-variable thesis entries. This does not authenticate its origin or reproduce the historical model.

The earlier audit concerned the earlier workbook. Findings and the documentation review remain in `archive/retired/`. Previous notes are preserved in `archive/previous_analysis.zip`. These are historical evidence, not current-workbook findings.

## Workflows

`predictor_summary.py` computes observed patient statistics and fresh holdout SHAP with fixed parameters, recording the input fingerprint per run. `models.py` performs fresh Optuna tuning. Standalone `shap_analysis.py` uses saved parameters if present, otherwise defaults.

Earlier tuning parameters were archived so they are not silently reused. Archived standalone SHAP output used those previously saved parameters. `results/reported_*.csv` files remain historical reference values. Patient records and source thesis documents were preserved during cleanup.
