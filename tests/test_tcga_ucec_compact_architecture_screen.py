import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "practicum"))
from screen_tcga_ucec_compact_architecture import COMPACT_SPECIFICATION, paired_comparison


class CompactArchitectureScreenTest(unittest.TestCase):
    def test_compact_setting_changes_only_width_and_heads(self):
        self.assertEqual(COMPACT_SPECIFICATION["config_overrides"], {
            "dim": 256, "heads": 8, "dropout": 0.0, "emb_dropout": 0.0,
        })

    def test_paired_comparison_matches_seed_before_subtracting(self):
        compact = [
            {"seed": 2, "best_validation_macro_f1": 0.31, "validation_accuracy": 0.7},
            {"seed": 1, "best_validation_macro_f1": 0.29, "validation_accuracy": 0.6},
        ]
        baseline = [
            {"seed": 1, "best_validation_macro_f1": 0.30, "validation_accuracy": 0.61},
            {"seed": 2, "best_validation_macro_f1": 0.28, "validation_accuracy": 0.62},
        ]
        paired = paired_comparison(compact, baseline)
        self.assertEqual(paired.seed.tolist(), [1, 2])
        self.assertAlmostEqual(paired.loc[0, "macro_f1_difference_compact_minus_baseline"], -0.01)
        self.assertAlmostEqual(paired.loc[1, "macro_f1_difference_compact_minus_baseline"], 0.03)


if __name__ == "__main__":
    unittest.main()
