import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "practicum"))
from screen_tcga_ucec_compact_depth import DEPTH_ONE_SPECIFICATION, paired_depth_comparison


class CompactDepthScreenTest(unittest.TestCase):
    def test_depth_one_setting_keeps_the_compact_width_and_heads(self):
        self.assertEqual(DEPTH_ONE_SPECIFICATION["config_overrides"], {
            "dim": 256,
            "heads": 8,
            "depth": 1,
            "dropout": 0.0,
            "emb_dropout": 0.0,
        })

    def test_pairing_has_unambiguous_depth_columns(self):
        paired = paired_depth_comparison(
            [{"seed": 2, "best_validation_macro_f1": 0.30, "validation_accuracy": 0.65}],
            [{"seed": 2, "best_validation_macro_f1": 0.28, "validation_accuracy": 0.60}],
        )
        self.assertEqual(paired.loc[0, "seed"], 2)
        self.assertAlmostEqual(
            paired.loc[0, "macro_f1_difference_depth_one_minus_depth_two"], 0.02
        )
        self.assertIn("depth_one_accuracy", paired.columns)


if __name__ == "__main__":
    unittest.main()
