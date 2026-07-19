import sys
import unittest
from pathlib import Path

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "practicum"
sys.path.insert(0, str(SCRIPT_DIR))

from compare_pathway_gprofiler import (  # noqa: E402
    build_comparison_table,
    normalize_name,
    parse_gene_list,
    rank_deepathnet_explanation,
    rank_gprofiler_terms,
)


class ComparePathwayGProfilerTest(unittest.TestCase):
    def test_parse_gene_list_handles_gprofiler_list_strings(self):
        self.assertEqual(parse_gene_list("['A', 'B', 'C']"), ["A", "B", "C"])
        self.assertEqual(parse_gene_list("A,B"), ["A", "B"])
        self.assertEqual(parse_gene_list(None), [])

    def test_normalize_name_removes_case_and_punctuation(self):
        self.assertEqual(
            normalize_name("Class A/1 (Rhodopsin-like receptors)"),
            "class a 1 rhodopsin like receptors",
        )

    def test_rank_deepathnet_explanation_pivots_high_and_delta(self):
        explanation = pd.DataFrame(
            {
                "cancer_type": ["Average", "High", "Average", "High"],
                "pathway": ["Pathway A", "Pathway A", "Pathway B", "Pathway B"],
                "importance": [0.1, 0.4, 0.3, 0.2],
            }
        )

        ranked = rank_deepathnet_explanation(explanation)
        row_a = ranked.set_index("pathway").loc["Pathway A"]

        self.assertEqual(row_a["importance_High"], 0.4)
        self.assertAlmostEqual(row_a["delta_High_minus_Average"], 0.3)
        self.assertEqual(row_a["rank_High"], 1)
        self.assertEqual(row_a["rank_delta"], 1)

    def test_build_comparison_table_matches_by_gene_overlap(self):
        deepathnet_ranked = pd.DataFrame(
            {
                "pathway": ["Pathway A", "Pathway B"],
                "importance_Average": [0.1, 0.2],
                "importance_High": [0.4, 0.1],
                "delta_High_minus_Average": [0.3, -0.1],
                "rank_High": [1, 2],
                "rank_delta": [1, 2],
            }
        )
        deepathnet_genes = {
            "Pathway A": {"G1", "G2"},
            "Pathway B": {"G3"},
        }
        gprofiler_ranked = rank_gprofiler_terms(
            pd.DataFrame(
                {
                    "source": ["REAC", "GO:BP"],
                    "native": ["REAC:1", "GO:1"],
                    "name": ["Receptor signaling", "Unrelated process"],
                    "p_value": [0.01, 0.02],
                    "intersection_size": [2, 1],
                    "term_size": [10, 20],
                    "intersection": ["['G2', 'G4']", "['G5']"],
                }
            )
        )

        comparison, all_matches, gprofiler_only = build_comparison_table(
            deepathnet_ranked, deepathnet_genes, gprofiler_ranked
        )
        row_a = comparison.set_index("deeppathnet_pathway").loc["Pathway A"]

        self.assertEqual(row_a["interpretation_group"], "concordant")
        self.assertEqual(row_a["best_match_source"], "REAC")
        self.assertEqual(row_a["best_match_overlap_genes"], "G2")
        self.assertEqual(len(all_matches), 1)
        self.assertEqual(gprofiler_only.iloc[0]["gprofiler_name"], "Unrelated process")


if __name__ == "__main__":
    unittest.main()
