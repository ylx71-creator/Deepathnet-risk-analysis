import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_tcga_ucec_rna_cnv_development import condition_frame  # noqa: E402


class RnaCnvDevelopmentComparisonTest(unittest.TestCase):
    def test_conditions_keep_expected_modalities(self):
        matrix = pd.DataFrame({"A_RNA": [1.0], "B_RNA": [2.0], "A_cnv": [0.0], "B_cnv": [1.0]})
        self.assertEqual(condition_frame(matrix, ("RNA",)).columns.tolist(), ["A_RNA", "B_RNA"])
        self.assertEqual(condition_frame(matrix, ("RNA", "cnv")).shape[1], 4)


if __name__ == "__main__":
    unittest.main()
