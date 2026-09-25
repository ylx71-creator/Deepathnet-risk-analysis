import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "practicum"))
from compare_tcga_ucec_transformer_dropout import CONDITIONS, aggregate_summaries


class TransformerDropoutComparisonTest(unittest.TestCase):
    def test_conditions_change_only_transformer_dropout(self):
        self.assertEqual(set(CONDITIONS), {"no_transformer_dropout", "transformer_dropout_0p1"})
        self.assertEqual(CONDITIONS["no_transformer_dropout"]["config_overrides"],
                         {"dropout": 0.0, "emb_dropout": 0.0})
        self.assertEqual(CONDITIONS["transformer_dropout_0p1"]["config_overrides"],
                         {"dropout": 0.1, "emb_dropout": 0.1})

    def test_aggregate_reports_condition_mean_and_variation(self):
        summaries = [
            {"condition": "no_transformer_dropout", "best_validation_macro_f1": 0.2,
             "validation_accuracy": 0.6, "validation_roc_auc_ovo": 0.5},
            {"condition": "no_transformer_dropout", "best_validation_macro_f1": 0.4,
             "validation_accuracy": 0.8, "validation_roc_auc_ovo": 0.7},
        ]
        aggregate = aggregate_summaries(summaries).set_index("condition")
        self.assertAlmostEqual(aggregate.loc["no_transformer_dropout", "best_validation_macro_f1_mean"], 0.3)
        self.assertAlmostEqual(aggregate.loc["no_transformer_dropout", "validation_accuracy_mean"], 0.7)
        self.assertGreater(aggregate.loc["no_transformer_dropout", "best_validation_macro_f1_std"], 0)


if __name__ == "__main__":
    unittest.main()
