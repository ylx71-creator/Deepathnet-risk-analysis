import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from prepare_tcga_ucec_rna_cnv_inputs import combine_modalities  # noqa: E402


class PrepareTcgaUcecRnaCnvTest(unittest.TestCase):
    def test_combination_keeps_only_genes_with_complete_centered_cnv(self):
        rna = pd.DataFrame(
            {"GENE1_RNA": [1.0, 2.0], "GENE2_RNA": [3.0, 4.0]},
            index=pd.Index(["sample1", "sample2"], name="Cell_line"),
        )
        cnv = pd.DataFrame({"GENE1": [-1.0, 0.0], "GENE2": [np.nan, 1.0]}, index=rna.index)

        combined, genes = combine_modalities(rna, cnv)

        self.assertEqual(genes, ["GENE1"])
        self.assertEqual(combined.columns.tolist(), ["GENE1_RNA", "GENE1_cnv"])
        self.assertEqual(combined.loc["sample1", "GENE1_cnv"], -1.0)


if __name__ == "__main__":
    unittest.main()
