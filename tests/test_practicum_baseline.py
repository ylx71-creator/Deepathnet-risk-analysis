import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from run_baseline_independent import (  # noqa: E402
    compute_classification_metrics,
    select_model_score,
    select_requested_features,
)


class PracticumBaselineTest(unittest.TestCase):
    def test_select_requested_features_keeps_train_order_and_rna_only(self):
        train = pd.DataFrame(
            {
                "GENE3_CNV": [1, 2],
                "GENE1_RNA": [3, 4],
                "GENE2_RNA": [5, 6],
            },
            index=["s1", "s2"],
        )
        test = pd.DataFrame(
            {
                "GENE2_RNA": [7],
                "GENE1_RNA": [8],
                "GENE4_RNA": [9],
            },
            index=["s3"],
        )

        selected_train, selected_test = select_requested_features(train, test, ["RNA"])

        self.assertEqual(selected_train.columns.tolist(), ["GENE1_RNA", "GENE2_RNA"])
        self.assertEqual(selected_test.columns.tolist(), ["GENE1_RNA", "GENE2_RNA"])

    def test_select_model_score_uses_high_probability_column(self):
        classes = ["Average", "High"]
        probabilities = [[0.8, 0.2], [0.3, 0.7]]

        high_scores = select_model_score(classes, probabilities, "High")

        self.assertEqual(high_scores, [0.2, 0.7])

    def test_compute_classification_metrics_reports_high_class_behavior(self):
        metrics = compute_classification_metrics(
            y_true=["Average", "Average", "High", "High"],
            y_pred=["Average", "High", "Average", "High"],
            y_score=[0.1, 0.8, 0.4, 0.9],
            positive_label="High",
        )

        self.assertEqual(metrics["acc"], 0.5)
        self.assertEqual(metrics["f1"], 0.5)
        self.assertEqual(metrics["precision_high"], 0.5)
        self.assertEqual(metrics["recall_high"], 0.5)
        self.assertEqual(metrics["tn"], 1)
        self.assertEqual(metrics["fp"], 1)
        self.assertEqual(metrics["fn"], 1)
        self.assertEqual(metrics["tp"], 1)
        self.assertEqual(metrics["auc"], 0.75)


if __name__ == "__main__":
    unittest.main()
