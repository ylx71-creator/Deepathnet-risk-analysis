import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from download_tcga_ucec_cnv import download_one, select_patient_matched_files  # noqa: E402


class DownloadTcgaUcecCnvTest(unittest.TestCase):
    def test_selects_exact_rna_sample_and_reports_missing_patient(self):
        label_file = Path(self._testMethodName + ".csv")
        try:
            pd.DataFrame({"Cell_line": ["TCGA-AA-0001-01A", "TCGA-BB-0002-01B"]}).to_csv(
                label_file, index=False
            )
            manifest = [
                self._file("a", "TCGA-AA-0001", ["TCGA-AA-0001-01B"]),
                self._file("b", "TCGA-AA-0001", ["TCGA-AA-0001-01A"]),
            ]
            selected, missing = select_patient_matched_files(manifest, label_file)
        finally:
            label_file.unlink(missing_ok=True)

        self.assertEqual(selected.loc[0, "file_id"], "b")
        self.assertTrue(selected.loc[0, "exact_rna_sample_match"])
        self.assertEqual(missing, ["TCGA-BB-0002"])

    def test_connection_reset_is_retried(self):
        with patch("download_tcga_ucec_cnv.urlopen", side_effect=ConnectionResetError):
            with self.assertRaisesRegex(RuntimeError, "failed after 2 attempts"):
                download_one(
                    {"file_id": "example", "md5sum": "0" * 32},
                    Path("."),
                    retries=2,
                )

    @staticmethod
    def _file(file_id, patient_id, sample_ids):
        return {
            "file_id": file_id,
            "file_name": f"{file_id}.tsv",
            "md5sum": "0" * 32,
            "file_size": 1,
            "analysis": {"workflow_type": "ASCAT2"},
            "cases": [{"submitter_id": patient_id, "samples": [{"submitter_id": x} for x in sample_ids]}],
        }


if __name__ == "__main__":
    unittest.main()
