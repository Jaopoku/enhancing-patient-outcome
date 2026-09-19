"""Numerical checks for observed-data statistics and predictor ranking."""
import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from predictor_summary import PREDICTORS, summarize, validate_data


class PredictorSummaryTests(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({p[0]: [1.0, 3.0, None] for p in PREDICTORS})
        self.df["Smoking_Status"] = [1, 0, None]
        self.df["Readmission_30D"] = [1, 0, 1]
        self.scores = pd.DataFrame({"Feature": [p[0] for p in PREDICTORS],
                                    "Mean_Abs_SHAP": [8, 7, 6, 5, 4, 3, 2, 1]})

    def test_raw_sample_sd_and_nonmissing_denominator(self):
        table, audit = summarize(self.df, self.scores)
        self.assertEqual(table.iloc[0]["Mean / Frequency"], "2.0 ± 1.4")
        self.assertAlmostEqual(audit.iloc[0].Sample_SD, 2 ** 0.5)
        self.assertEqual(audit.iloc[0].N_Missing, 1)
        self.assertEqual(table.iloc[-1]["Mean / Frequency"],
                         "Yes: 1 (50.0%); No: 1 (50.0%)")

    def test_rank_uses_scores_not_row_order_and_preserves_ties(self):
        self.scores["Mean_Abs_SHAP"] = [1, 2, 3, 4, 5, 6, 8, 8]
        _, audit = summarize(self.df, self.scores.iloc[::-1])
        self.assertEqual(audit.Rank_Among_Listed_Predictors.tolist(),
                         [8, 7, 6, 5, 4, 3, 1, 1])

    def test_rejects_invalid_smoking_or_missing_outcome(self):
        self.df.loc[0, "Smoking_Status"] = 2
        with self.assertRaisesRegex(ValueError, "Smoking_Status"):
            validate_data(self.df, "Readmission_30D")
        self.df.loc[0, "Smoking_Status"] = 1
        self.df.loc[0, "Readmission_30D"] = None
        with self.assertRaisesRegex(ValueError, "missing outcomes"):
            validate_data(self.df, "Readmission_30D")

    def test_missing_predictor_does_not_silently_disappear(self):
        with self.assertRaisesRegex(ValueError, "Missing required columns"):
            validate_data(self.df.drop(columns="Age"), "Readmission_30D")
        with self.assertRaisesRegex(ValueError, "each requested predictor"):
            summarize(self.df, self.scores.iloc[1:])


if __name__ == "__main__":
    unittest.main()
