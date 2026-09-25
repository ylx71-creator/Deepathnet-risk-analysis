import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "practicum"))
from diagnose_tcga_ucec_overfit import select_samples
from train_tcga_ucec_stage import CLASSES


class OverfitSelectionTest(unittest.TestCase):
    def test_balanced_selection_excludes_held_out_patients(self):
        rows = []
        for split in ["train", "validation", "test"]:
            for label in CLASSES:
                for i in range(5):
                    sample = f"{split}-{label}-{i}"
                    rows.append({"Cell_line": sample, "case_submitter_id": sample,
                                 "split": split, "Cancer_type": label})
        frame = pd.DataFrame(rows)
        chosen = select_samples(frame, 4, 1)
        self.assertTrue(chosen.split.eq("train").all())
        self.assertEqual(chosen.Cancer_type.value_counts().to_dict(), dict.fromkeys(CLASSES, 4))
        pd.testing.assert_frame_equal(chosen, select_samples(frame, 4, 1))
        with self.assertRaises(ValueError):
            select_samples(frame, 6, 1)
        frame.loc[1, "case_submitter_id"] = frame.loc[0, "case_submitter_id"]
        with self.assertRaises(ValueError):
            select_samples(frame, 4, 1)


if __name__ == "__main__":
    unittest.main()
