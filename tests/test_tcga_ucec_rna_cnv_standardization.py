import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_tcga_ucec_development_settings import standardize  # noqa: E402


class RnaCnvStandardizationTest(unittest.TestCase):
    def test_scaler_is_fit_on_train_and_applied_to_validation(self):
        train = pd.DataFrame({"G_RNA": [1.0, 3.0], "G_cnv": [-1.0, 1.0]})
        validation = pd.DataFrame({"G_RNA": [5.0], "G_cnv": [3.0]})
        train_z, validation_z, scaler = standardize(train, validation)
        self.assertTrue(np.allclose(train_z.mean().to_numpy(), [0.0, 0.0]))
        self.assertTrue(np.allclose(scaler.mean_, [2.0, 0.0]))
        self.assertTrue(np.allclose(validation_z.iloc[0].to_numpy(), [3.0, 3.0]))


if __name__ == "__main__":
    unittest.main()
