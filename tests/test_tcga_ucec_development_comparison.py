import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "practicum"))
from compare_tcga_ucec_development_settings import class_weights, standardize
from train_tcga_ucec_stage import CLASSES


class DevelopmentComparisonTest(unittest.TestCase):
    def test_class_weights_use_training_frequency_only(self):
        labels = pd.DataFrame({"Cancer_type": ["Stage I"] * 8 + ["Stage II"] * 4 + ["Stage III"] * 2 + ["Stage IV"]})
        weights, counts = class_weights(labels)
        self.assertEqual(counts, {"Stage I": 8, "Stage II": 4, "Stage III": 2, "Stage IV": 1})
        self.assertEqual(weights.tolist(), [0.46875, 0.9375, 1.875, 3.75])

    def test_standardization_fits_only_training_rows(self):
        train = pd.DataFrame({"A_RNA": [1.0, 3.0], "B_RNA": [2.0, 6.0]}, index=["t1", "t2"])
        validation = pd.DataFrame({"A_RNA": [101.0], "B_RNA": [102.0]}, index=["v1"])
        train_z, validation_z, scaler = standardize(train, validation)
        self.assertAlmostEqual(scaler.mean_[0], 2.0)
        self.assertAlmostEqual(scaler.mean_[1], 4.0)
        self.assertAlmostEqual(train_z["A_RNA"].mean(), 0.0)
        self.assertGreater(validation_z.loc["v1", "A_RNA"], 90)
        self.assertEqual(list(train_z.columns), list(validation_z.columns))


if __name__ == "__main__":
    unittest.main()
