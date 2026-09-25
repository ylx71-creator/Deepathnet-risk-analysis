"""Regression checks for patient isolation and development-only file access."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/practicum"))
import screen_tcga_ucec_subtypes as screen


class SubtypeLeakageTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data = self.root / "data/processed/tcga_ucec_subtypes"
        rna = self.root / "data/processed/tcga_ucec_rna"
        self.data.mkdir(parents=True)
        rna.mkdir(parents=True)

        classes = screen.CLASSES
        rows = []
        labels = []
        for class_index, subtype in enumerate(classes):
            for patient_index in range(7):
                sample = f"TCGA-{class_index:02d}-{patient_index:04d}-01A"
                rows.append({"Cell_line": sample, "G1_RNA": class_index,
                             "G2_RNA": patient_index})
                labels.append({"Cell_line": sample, "Cancer_type": subtype})
        expression = pd.DataFrame(rows).set_index("Cell_line")
        target = pd.DataFrame(labels).set_index("Cell_line")
        expression.to_csv(self.data / "data_file.csv")
        target.to_csv(self.data / "target_file.csv")
        pd.DataFrame({
            "Cell_line": expression.index,
            "case_submitter_id": [sample[:12] for sample in expression.index],
        }).to_csv(self.data / "patient_label_audit.tsv", sep="\t", index=False)

        self.old_test = [expression.index[index * 7] for index in range(len(classes))]
        pd.DataFrame({"Cell_line": self.old_test, "Cancer_type": classes}).to_csv(
            rna / "test_label_file.csv", index=False
        )
        with patch.object(screen, "ROOT", self.root), patch.object(screen, "DATA", self.data):
            screen.prepare()

    def tearDown(self):
        self.temporary.cleanup()

    def test_patients_disjoint_and_historical_test_retained(self):
        table = pd.read_csv(self.data / "splits/assignments.csv")
        self.assertTrue(table.case_submitter_id.is_unique)
        self.assertTrue(table.Cell_line.is_unique)
        held = set(table.loc[table.split.eq("test"), "Cell_line"])
        self.assertEqual(set(self.old_test), held)
        for _, part in table.groupby("split"):
            self.assertEqual(set(part.subtype), set(screen.CLASSES))

    def test_training_loader_never_reads_test_or_full_cohort(self):
        actual = pd.read_csv
        opened = []

        def guarded(path, *args, **kwargs):
            name = Path(path).name
            self.assertIn(name, {"train_data.csv", "train_labels.csv",
                                 "validation_data.csv", "validation_labels.csv"})
            opened.append(name)
            return actual(path, *args, **kwargs)

        with patch.object(screen, "DATA", self.data):
            with patch.object(screen.pd, "read_csv", side_effect=guarded):
                tx, ty, vx, vy = screen.load_development()
        self.assertEqual(len(opened), 4)
        self.assertEqual(len(tx) + len(vx), 24)
        self.assertFalse(set(tx.index) & set(vx.index))

    def test_modified_split_is_rejected(self):
        with patch.object(screen, "DATA", self.data):
            with patch.object(screen, "digest", return_value="changed"):
                with self.assertRaisesRegex(ValueError, "Fixed development file changed"):
                    screen.load_development()


if __name__ == "__main__":
    unittest.main()
