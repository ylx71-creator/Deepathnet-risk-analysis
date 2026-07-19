"""
Compare DeePathNet pathway attribution with g:Profiler enrichment results.

Default inputs are the practicum risk RNA explanation and g:Profiler all-sources
outputs generated from DESeq2 significant genes.
"""

import argparse
import ast
import re
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_EXPLANATION = (
    Path(__file__).resolve().parents[2]
    / "work_dirs/practicum/DeePathNet/risk_rna/explanation_202606291649_Cancer_type.csv"
)
DEFAULT_DEEPPATHNET_PATHWAY_FILE = (
    Path(__file__).resolve().parents[2]
    / "data/graph_predefined/LCPathways/41568_2020_240_MOESM4_ESM.csv"
)
DEFAULT_GPROFILER_RESULTS = (
    Path(__file__).resolve().parents[3]
    / "practicum/output/gprofiler/gprofiler_all_sources_results.csv"
)
DEFAULT_OUTPUT_DIR = (
    Path(__file__).resolve().parents[2]
    / "work_dirs/practicum/pathway_gprofiler_comparison"
)


def parse_gene_list(value):
    if isinstance(value, list):
        genes = value
    elif value is None or (isinstance(value, float) and pd.isna(value)):
        genes = []
    else:
        text = str(value).strip()
        if not text:
            genes = []
        elif text.startswith("[") and text.endswith("]"):
            try:
                parsed = ast.literal_eval(text)
                genes = parsed if isinstance(parsed, list) else [parsed]
            except (SyntaxError, ValueError):
                genes = text.split(",")
        else:
            genes = text.split(",")
    return [str(gene).strip().strip("'\"") for gene in genes if str(gene).strip().strip("'\"")]


def normalize_name(value):
    text = str(value).lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def rank_deepathnet_explanation(explanation_df):
    required = {"cancer_type", "pathway", "importance"}
    missing = required - set(explanation_df.columns)
    if missing:
        raise ValueError(f"Missing DeePathNet explanation columns: {sorted(missing)}")

    ranked = (
        explanation_df.pivot_table(
            index="pathway",
            columns="cancer_type",
            values="importance",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename_axis(None, axis=1)
    )
    for label in ["Average", "High"]:
        if label not in ranked.columns:
            ranked[label] = 0.0
    ranked = ranked.rename(
        columns={"Average": "importance_Average", "High": "importance_High"}
    )
    ranked["delta_High_minus_Average"] = (
        ranked["importance_High"] - ranked["importance_Average"]
    )
    ranked["rank_High"] = (
        ranked["importance_High"].rank(ascending=False, method="min").astype(int)
    )
    ranked["rank_delta"] = (
        ranked["delta_High_minus_Average"].rank(ascending=False, method="min").astype(int)
    )
    return ranked.sort_values(["rank_High", "rank_delta", "pathway"]).reset_index(drop=True)


def load_deepathnet_pathway_genes(pathway_file):
    pathway_df = pd.read_csv(pathway_file)
    required = {"name", "genes"}
    missing = required - set(pathway_df.columns)
    if missing:
        raise ValueError(f"Missing DeePathNet pathway columns: {sorted(missing)}")
    return {
        row["name"]: set(parse_gene_list(str(row["genes"]).replace("|", ",")))
        for _, row in pathway_df.iterrows()
    }


def rank_gprofiler_terms(gprofiler_df):
    required = {"source", "native", "name", "p_value", "intersection"}
    missing = required - set(gprofiler_df.columns)
    if missing:
        raise ValueError(f"Missing g:Profiler columns: {sorted(missing)}")

    ranked = gprofiler_df.copy()
    ranked["intersection_genes"] = ranked["intersection"].map(parse_gene_list)
    ranked["intersection_gene_set"] = ranked["intersection_genes"].map(set)
    ranked["intersection_genes_text"] = ranked["intersection_genes"].map(
        lambda genes: ";".join(genes)
    )
    ranked["rank_gprofiler"] = ranked["p_value"].rank(ascending=True, method="min").astype(int)
    if "intersection_size" not in ranked.columns:
        ranked["intersection_size"] = ranked["intersection_genes"].map(len)
    if "term_size" not in ranked.columns:
        ranked["term_size"] = np.nan
    return ranked.sort_values(["rank_gprofiler", "source", "name"]).reset_index(drop=True)


def _name_match_type(deepathnet_name, gprofiler_name):
    deepathnet_norm = normalize_name(deepathnet_name)
    gprofiler_norm = normalize_name(gprofiler_name)
    if deepathnet_norm and deepathnet_norm == gprofiler_norm:
        return "exact_name"
    if (
        deepathnet_norm
        and gprofiler_norm
        and (deepathnet_norm in gprofiler_norm or gprofiler_norm in deepathnet_norm)
    ):
        return "name_contains"
    return ""


def _source_priority(source):
    if source in {"REAC", "KEGG"}:
        return 3
    if source in {"GO:BP", "GO:MF"}:
        return 2
    if source == "GO:CC":
        return 1
    return 0


def build_comparison_table(deepathnet_ranked, deepathnet_genes, gprofiler_ranked):
    all_match_rows = []
    comparison_rows = []
    gprofiler_match_tracker = {idx: 0 for idx in gprofiler_ranked.index}

    for _, dpn_row in deepathnet_ranked.iterrows():
        pathway = dpn_row["pathway"]
        pathway_genes = set(deepathnet_genes.get(pathway, set()))
        matches = []

        for g_idx, gp_row in gprofiler_ranked.iterrows():
            gp_genes = set(gp_row["intersection_gene_set"])
            overlap_genes = sorted(pathway_genes & gp_genes)
            name_match = _name_match_type(pathway, gp_row["name"])
            if not overlap_genes and not name_match:
                continue

            union_size = len(pathway_genes | gp_genes)
            overlap_jaccard = len(overlap_genes) / union_size if union_size else 0
            match_basis = name_match or "gene_overlap"
            match_row = {
                "deeppathnet_pathway": pathway,
                "deeppathnet_gene_count": len(pathway_genes),
                "gprofiler_index": g_idx,
                "gprofiler_source": gp_row["source"],
                "gprofiler_native": gp_row["native"],
                "gprofiler_name": gp_row["name"],
                "gprofiler_p_value": gp_row["p_value"],
                "gprofiler_rank": gp_row["rank_gprofiler"],
                "gprofiler_intersection_size": gp_row["intersection_size"],
                "overlap_gene_count": len(overlap_genes),
                "overlap_jaccard": overlap_jaccard,
                "overlap_genes": ";".join(overlap_genes),
                "match_basis": match_basis,
                "source_priority": _source_priority(gp_row["source"]),
            }
            matches.append(match_row)
            all_match_rows.append(match_row)
            gprofiler_match_tracker[g_idx] = max(
                gprofiler_match_tracker[g_idx], len(overlap_genes)
            )

        if matches:
            match_df = pd.DataFrame(matches)
            match_df["basis_priority"] = match_df["match_basis"].map(
                {"exact_name": 4, "name_contains": 3, "gene_overlap": 2}
            ).fillna(1)
            best = match_df.sort_values(
                [
                    "basis_priority",
                    "overlap_gene_count",
                    "source_priority",
                    "gprofiler_p_value",
                ],
                ascending=[False, False, False, True],
            ).iloc[0]
            interpretation_group = "concordant"
        else:
            best = None
            interpretation_group = "deeppathnet_only"

        comparison_rows.append(
            {
                "deeppathnet_pathway": pathway,
                "importance_High": dpn_row["importance_High"],
                "importance_Average": dpn_row["importance_Average"],
                "delta_High_minus_Average": dpn_row["delta_High_minus_Average"],
                "rank_High": dpn_row["rank_High"],
                "rank_delta": dpn_row["rank_delta"],
                "deeppathnet_gene_count": len(pathway_genes),
                "best_match_source": "" if best is None else best["gprofiler_source"],
                "best_match_native": "" if best is None else best["gprofiler_native"],
                "best_match_name": "" if best is None else best["gprofiler_name"],
                "best_match_p_value": np.nan if best is None else best["gprofiler_p_value"],
                "best_match_gprofiler_rank": np.nan if best is None else best["gprofiler_rank"],
                "best_match_overlap_gene_count": 0 if best is None else best["overlap_gene_count"],
                "best_match_overlap_jaccard": 0 if best is None else best["overlap_jaccard"],
                "best_match_overlap_genes": "" if best is None else best["overlap_genes"],
                "match_basis": "" if best is None else best["match_basis"],
                "interpretation_group": interpretation_group,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    all_matches = pd.DataFrame(all_match_rows)
    if not all_matches.empty:
        all_matches = all_matches.drop(columns=["source_priority"]).sort_values(
            [
                "deeppathnet_pathway",
                "overlap_gene_count",
                "gprofiler_p_value",
            ],
            ascending=[True, False, True],
        )

    gprofiler_only_rows = []
    for g_idx, gp_row in gprofiler_ranked.iterrows():
        if gprofiler_match_tracker[g_idx] == 0:
            gprofiler_only_rows.append(
                {
                    "gprofiler_source": gp_row["source"],
                    "gprofiler_native": gp_row["native"],
                    "gprofiler_name": gp_row["name"],
                    "gprofiler_p_value": gp_row["p_value"],
                    "gprofiler_rank": gp_row["rank_gprofiler"],
                    "gprofiler_intersection_size": gp_row["intersection_size"],
                    "gprofiler_intersection_genes": gp_row["intersection_genes_text"],
                    "interpretation_group": "gprofiler_only",
                }
            )
    gprofiler_only = pd.DataFrame(gprofiler_only_rows)
    return comparison, all_matches, gprofiler_only


def write_outputs(
    explanation_file,
    deepathnet_pathway_file,
    gprofiler_results_file,
    output_dir,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    explanation = pd.read_csv(explanation_file)
    gprofiler_results = pd.read_csv(gprofiler_results_file)

    deepathnet_ranked = rank_deepathnet_explanation(explanation)
    deepathnet_genes = load_deepathnet_pathway_genes(deepathnet_pathway_file)
    gprofiler_ranked = rank_gprofiler_terms(gprofiler_results)
    comparison, all_matches, gprofiler_only = build_comparison_table(
        deepathnet_ranked,
        deepathnet_genes,
        gprofiler_ranked,
    )

    output_paths = {
        "deeppathnet_ranked": output_dir / "deeppathnet_pathway_ranked.csv",
        "gprofiler_ranked": output_dir / "gprofiler_ranked.csv",
        "comparison": output_dir / "pathway_gprofiler_comparison.csv",
        "pairwise_matches": output_dir / "pathway_gprofiler_pairwise_matches.csv",
        "gprofiler_only": output_dir / "gprofiler_only_terms.csv",
    }

    deepathnet_ranked.to_csv(output_paths["deeppathnet_ranked"], index=False)
    gprofiler_ranked.drop(columns=["intersection_gene_set"]).to_csv(
        output_paths["gprofiler_ranked"], index=False
    )
    comparison.to_csv(output_paths["comparison"], index=False)
    all_matches.to_csv(output_paths["pairwise_matches"], index=False)
    gprofiler_only.to_csv(output_paths["gprofiler_only"], index=False)

    return output_paths, comparison, all_matches, gprofiler_only


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare DeePathNet pathway explanation with g:Profiler enrichment."
    )
    parser.add_argument("--explanation", type=Path, default=DEFAULT_EXPLANATION)
    parser.add_argument(
        "--deepathnet-pathway-file",
        type=Path,
        default=DEFAULT_DEEPPATHNET_PATHWAY_FILE,
    )
    parser.add_argument(
        "--gprofiler-results",
        type=Path,
        default=DEFAULT_GPROFILER_RESULTS,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main():
    args = parse_args()
    output_paths, comparison, all_matches, gprofiler_only = write_outputs(
        explanation_file=args.explanation,
        deepathnet_pathway_file=args.deepathnet_pathway_file,
        gprofiler_results_file=args.gprofiler_results,
        output_dir=args.output_dir,
    )

    print("Wrote pathway-gProfiler comparison outputs:")
    for name, path in output_paths.items():
        print(f"- {name}: {path}")
    print()
    print("Comparison summary:")
    print(comparison["interpretation_group"].value_counts().to_string())
    print(f"Pairwise overlap matches: {len(all_matches)}")
    print(f"gProfiler-only terms: {len(gprofiler_only)}")
    print()
    print("Top DeePathNet High pathways with best g:Profiler match:")
    preview_columns = [
        "deeppathnet_pathway",
        "importance_High",
        "delta_High_minus_Average",
        "rank_High",
        "best_match_source",
        "best_match_name",
        "best_match_p_value",
        "best_match_overlap_gene_count",
        "best_match_overlap_genes",
        "interpretation_group",
    ]
    print(comparison.sort_values("rank_High")[preview_columns].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
