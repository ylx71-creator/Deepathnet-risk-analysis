import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_tcga_ucec_deepathnet_inputs import normalize_stage, parse_args  # noqa: E402


class PrepareTcgaUcecTest(unittest.TestCase):
    def test_main_stages_and_observed_substages(self):
        groups = {
            "Stage I": ["Stage I", "Stage IA", "Stage IB", "Stage IB1", "Stage IC"],
            "Stage II": ["Stage II", "Stage IIA", "Stage IIB"],
            "Stage III": [
                "Stage III", "Stage IIIA", "Stage IIIB", "Stage IIIC",
                "Stage IIIC1", "Stage IIIC2",
            ],
            "Stage IV": ["Stage IV", "Stage IVA", "Stage IVB"],
        }
        for expected, values in groups.items():
            for value in values:
                with self.subTest(value=value):
                    self.assertEqual(normalize_stage(value), expected)
        self.assertEqual(normalize_stage("  stage iiic2  "), "Stage III")

    def test_missing_stage_is_not_assigned_a_class(self):
        for value in [None, float("nan"), pd.NA, "", "Unknown", "Not Reported"]:
            with self.subTest(value=value):
                self.assertIsNone(normalize_stage(value))

    def test_unrecognized_stage_cannot_silently_match_a_prefix(self):
        for value in ["Stage III unknown", "Stage V", "Stage III/IV", "Stage IIII"]:
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    normalize_stage(value)

    def test_default_target_uses_main_stage(self):
        with patch.object(sys, "argv", ["prepare_tcga_ucec_deepathnet_inputs.py"]):
            self.assertEqual(parse_args().label_column, "stage_main")


if __name__ == "__main__":
    unittest.main()
