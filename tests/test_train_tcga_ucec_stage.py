import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))
import train_tcga_ucec_stage as training  # noqa: E402


class StageTrainingTest(unittest.TestCase):
    def setUp(self):
        self.ids = [f"sample{i}" for i in range(48)]
        self.labels = pd.DataFrame({"Cancer_type": training.CLASSES * 12},
                                   index=pd.Index(self.ids, name="Cell_line"))
        self.sheet = pd.DataFrame({"sample_submitter_id": self.ids,
                                   "case_submitter_id": [f"patient{i}" for i in range(48)]})

    def test_split_is_reproducible_and_keeps_test_fixed(self):
        dev, test = self.labels.iloc[:40], self.labels.iloc[40:]
        first, assignments = training.make_splits(dev, test, self.sheet, .2, 1)
        second, _ = training.make_splits(dev, test, self.sheet, .2, 1)
        self.assertEqual(first, second)
        self.assertEqual(first["test"], list(test.index))
        self.assertEqual({k: len(v) for k, v in first.items()}, {"train": 32, "validation": 8, "test": 8})
        self.assertTrue(assignments.case_submitter_id.is_unique)
        # Changing test labels must not influence the train/validation selection.
        changed = test.copy()
        changed.Cancer_type = list(reversed(changed.Cancer_type))
        third, _ = training.make_splits(dev, changed, self.sheet, .2, 1)
        self.assertEqual(first, third)

    def test_patient_overlap_is_rejected_even_with_distinct_sample_ids(self):
        self.sheet.loc[47, "case_submitter_id"] = self.sheet.loc[0, "case_submitter_id"]
        with self.assertRaisesRegex(ValueError, "one selected sample per patient"):
            training.make_splits(self.labels.iloc[:40], self.labels.iloc[40:], self.sheet, .2, 1)

    def test_predictions_keep_ids_and_macro_f1_includes_all_stages(self):
        probabilities = np.array([[.7, .1, .1, .1]] * 4)
        metrics, predictions, report, matrix = training.summarize_predictions(
            ["d", "a", "c", "b"], np.arange(4), probabilities,
        )
        self.assertEqual(predictions.Cell_line.tolist(), ["d", "a", "c", "b"])
        self.assertAlmostEqual(metrics["accuracy"], .25)
        self.assertAlmostEqual(metrics["macro_f1"], .1)
        self.assertEqual(matrix.values.sum(), 4)
        self.assertEqual(report.loc["Stage IV", "recall"], 0)

    def test_configured_transformer_dropout_reaches_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pathways = root / "pathways.csv"
            pd.DataFrame({"name": ["path1"], "genes": ["A|B"]}).to_csv(pathways, index=False)
            x = pd.DataFrame([[1.0, 2.0], [3.0, 4.0]], index=["a", "b"],
                             columns=["A_RNA", "B_RNA"])
            y = pd.DataFrame({"Cancer_type": ["Stage I", "Stage II"]}, index=x.index)
            dataset = training.MultiOmicMulticlassDataset(
                x, y, "train", ["RNA"], {name: i for i, name in enumerate(training.CLASSES)},
            )
            config = {
                "pathway_file": str(pathways), "cancer_only": True,
                "dim": 8, "depth": 1, "heads": 2, "mlp_ratio": 2, "out_mlp_ratio": 2,
                "pathway_dropout": 0, "dropout": 0.125, "emb_dropout": 0.25,
            }
            model, _ = training.make_model(dataset, config)
            block = model.blocks[0]
            self.assertEqual(block.mlp.drop.p, 0.125)
            self.assertEqual(block.attn.proj_drop.p, 0.125)
            self.assertEqual(block.attn.attn_drop.p, 0.25)

            zero_config = {**config, "dropout": 0.0, "emb_dropout": 0.0}
            legacy_config = {key: value for key, value in zero_config.items()
                             if key not in {"dropout", "emb_dropout"}}
            torch.manual_seed(7)
            zero_model, _ = training.make_model(dataset, zero_config)
            torch.manual_seed(7)
            legacy_model, _ = training.make_model(dataset, legacy_config)
            for zero, legacy in zip(zero_model.parameters(), legacy_model.parameters()):
                self.assertTrue(torch.equal(zero, legacy))

    def run_synthetic_pipeline(self, smoke):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            x = pd.DataFrame(np.random.RandomState(1).uniform(0, 2, (48, 3)),
                             index=self.labels.index, columns=["A_RNA", "B_RNA", "C_RNA"])
            for role, rows in [("train", slice(0, 40)), ("test", slice(40, 48))]:
                x.iloc[rows].to_csv(root / f"{role}_data.csv")
                self.labels.iloc[rows].to_csv(root / f"{role}_labels.csv")
            self.sheet.to_csv(root / "selected_sample_sheet.tsv", sep="\t", index=False)
            pd.DataFrame({"name": ["path1", "path2"], "genes": ["A|B", "B|C"]}).to_csv(root / "pathways.csv", index=False)
            config = {
                "task": "multiclass", "data_type": ["RNA"], "seed": 1,
                "validation_size": .2, "validation_seed": 1, "batch_size": 4,
                "num_of_epochs": 1, "work_dir": str(root / "new_output"),
                "pathway_file": str(root / "pathways.csv"), "cancer_only": True,
                "dim": 8, "depth": 1, "heads": 2, "mlp_ratio": 2, "out_mlp_ratio": 2,
                "pathway_dropout": 0, "lr": .001, "weight_decay": 0,
            }
            for role in ["train", "test"]:
                config[f"data_file_{role}"] = str(root / f"{role}_data.csv")
                config[f"target_file_{role}"] = str(root / f"{role}_labels.csv")
            config_path = root / "config.json"
            config_path.write_text(json.dumps(config))
            argv = ["train_tcga_ucec_stage.py", str(config_path), "--cpu-threads", "1"]
            if smoke:
                argv.append("--smoke-test")
            evaluated_roles = []
            original_predict = training.predict

            def record_predict(model, loader, device):
                evaluated_roles.append(loader.dataset.mode)
                return original_predict(model, loader, device)

            with patch.object(sys, "argv", argv), patch.object(training, "predict", side_effect=record_predict):
                training.main()
            output = next((root / "new_output").iterdir())
            summary = json.loads((output / "run_summary.json").read_text())
            self.assertTrue((output / "best_model.pth").is_file())
            self.assertEqual(summary["best_epoch"], 1)
            self.assertEqual(summary["test_evaluated"], not smoke)
            self.assertEqual(evaluated_roles, ["val"] if smoke else ["val", "test"])
            self.assertEqual((output / "test_predictions.csv").exists(), not smoke)
            predictions = pd.read_csv(output / "validation_predictions.csv")
            self.assertEqual(len(predictions), 8)
            self.assertTrue(predictions.Cell_line.is_unique)
            if not smoke:
                predictions = pd.read_csv(output / "test_predictions.csv")
                self.assertEqual(predictions.Cell_line.tolist(), self.ids[40:])

    def test_smoke_pipeline_saves_weights_without_evaluating_test(self):
        self.run_synthetic_pipeline(smoke=True)

    def test_full_pipeline_evaluates_test_once_after_selection(self):
        self.run_synthetic_pipeline(smoke=False)


if __name__ == "__main__":
    unittest.main()
