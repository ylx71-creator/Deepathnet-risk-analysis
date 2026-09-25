#!/usr/bin/env python3
"""Build patient-matched TCGA-UCEC RNA+CNV matrices for DeePathNet.

CNV values are open GDC ASCAT2 gene-level integer copy numbers.  They are
centered at diploid copy number two, so zero means diploid, not missing.  Genes
with any missing CNV value among the matched patients are excluded instead of
silently converting missingness to diploid CNV.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECTS_ROOT = REPO_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rna-dir", type=Path, default=REPO_ROOT / "data/processed/tcga_ucec_rna"
    )
    parser.add_argument(
        "--cnv-dir",
        type=Path,
        default=PROJECTS_ROOT / "practicum/data/tcga_ucec_gdc/cnv_ascat2",
    )
    parser.add_argument(
        "--split-file",
        type=Path,
        default=(
            REPO_ROOT
            / "work_dirs/practicum/DeePathNet/tcga_ucec_stage_main_rna"
            / "train_20260909_181424_960563/split_assignments.csv"
        ),
        help="Original fixed 300/76/94 RNA split assignments.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/processed/tcga_ucec_rna_cnv",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def read_centered_cnv(path: Path, rna_genes: set[str]) -> pd.Series:
    frame = pd.read_csv(path, sep="\t", usecols=["gene_name", "copy_number"])
    if frame.gene_name.isna().any():
        frame = frame.dropna(subset=["gene_name"])
    frame = frame[frame.gene_name.isin(rna_genes)].copy()
    values = pd.to_numeric(frame.copy_number, errors="coerce")
    series = pd.Series(values.to_numpy(dtype=float) - 2.0, index=frame.gene_name.astype(str))
    return series.groupby(level=0).mean()


def combine_modalities(rna: pd.DataFrame, cnv: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    rna_genes = {column[:-4] for column in rna.columns}
    if set(cnv.columns).difference(rna_genes):
        raise ValueError("CNV contains genes outside the RNA feature set")
    complete_genes = sorted(gene for gene in cnv.columns if cnv[gene].notna().all())
    if not complete_genes:
        raise ValueError("No RNA pathway genes have complete CNV values")
    rna_columns = [f"{gene}_RNA" for gene in complete_genes]
    cnv_columns = [f"{gene}_cnv" for gene in complete_genes]
    combined = pd.concat([rna.loc[:, rna_columns], cnv.loc[:, complete_genes].set_axis(cnv_columns, axis=1)], axis=1)
    if not np.isfinite(combined.to_numpy(dtype=float)).all():
        raise ValueError("Combined RNA+CNV matrix contains non-finite values")
    return combined, complete_genes


def load_matching_inputs(rna_dir: Path, cnv_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rna_path = rna_dir / "data_file.csv"
    labels_path = rna_dir / "target_file.csv"
    manifest_path = cnv_dir / "selected_manifest.tsv"
    for path, label in ((rna_path, "RNA matrix"), (labels_path, "RNA labels"), (manifest_path, "CNV manifest")):
        require_file(path, label)
    rna = pd.read_csv(rna_path, index_col="Cell_line")
    labels = pd.read_csv(labels_path, index_col="Cell_line")
    manifest = pd.read_csv(manifest_path, sep="\t")
    if list(labels.columns) != ["Cancer_type"] or not rna.index.equals(labels.index):
        raise ValueError("RNA matrix and labels must have the same Cell_line index")
    if manifest.patient_id.duplicated().any() or manifest.rna_sample_id.duplicated().any():
        raise ValueError("CNV manifest must select one file per patient and RNA sample")
    manifest = manifest.set_index("rna_sample_id", drop=False)
    matched_ids = [sample_id for sample_id in rna.index if sample_id in manifest.index]
    if not matched_ids:
        raise ValueError("No CNV manifest samples matched the RNA matrix")
    manifest = manifest.loc[matched_ids]
    return rna.loc[matched_ids], labels.loc[matched_ids], manifest


def subset_original_split(
    combined: pd.DataFrame, labels: pd.DataFrame, split_file: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    require_file(split_file, "original split assignments")
    assignments = pd.read_csv(split_file).set_index("Cell_line")
    required = {"case_submitter_id", "split", "Cancer_type"}
    if not required.issubset(assignments.columns):
        raise ValueError("Original split assignments have unexpected columns")
    matched = assignments.loc[combined.index].copy()
    if not matched.Cancer_type.equals(labels.Cancer_type):
        raise ValueError("Original split labels do not match RNA labels")
    if set(matched["split"]) != {"train", "validation", "test"}:
        raise ValueError("Each original split must retain at least one matched sample")
    return combined.loc[matched.index], matched.reset_index()


def main() -> None:
    args = parse_args()
    rna, labels, manifest = load_matching_inputs(args.rna_dir, args.cnv_dir)
    rna_genes = {column[:-4] for column in rna.columns}
    cnv_rows = []
    files_dir = args.cnv_dir / "files"
    total = len(manifest)
    for index, row in enumerate(manifest.itertuples(index=False), start=1):
        cnv_path = files_dir / f"{row.file_id}.tsv"
        require_file(cnv_path, "downloaded CNV TSV")
        cnv_rows.append(read_centered_cnv(cnv_path, rna_genes))
        if index == 1 or index % 50 == 0 or index == total:
            print(f"Loaded {index}/{total}: {row.rna_sample_id}", flush=True)
    cnv = pd.DataFrame(cnv_rows, index=manifest.rna_sample_id).reindex(columns=sorted(rna_genes))
    cnv.index.name = "Cell_line"
    combined, complete_genes = combine_modalities(rna, cnv)
    combined, assignments = subset_original_split(combined, labels, args.split_file)
    labels = labels.loc[combined.index]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output_dir / "data_file_all_matched_samples.csv")
    labels.to_csv(args.output_dir / "target_file.csv")
    assignments.to_csv(args.output_dir / "fixed_split_assignments.csv", index=False)
    dropped = sorted(rna_genes.difference(complete_genes))
    pd.DataFrame({"gene": dropped, "reason": "missing_cnv_in_at_least_one_matched_patient"}).to_csv(
        args.output_dir / "dropped_rna_pathway_genes_missing_cnv.csv", index=False
    )
    for split in ("train", "validation", "test"):
        sample_ids = assignments.loc[assignments["split"].eq(split), "Cell_line"].tolist()
        combined.loc[sample_ids].to_csv(args.output_dir / f"{split}_data_file.csv")
        labels.loc[sample_ids].to_csv(args.output_dir / f"{split}_label_file.csv")
    development_ids = assignments.loc[assignments["split"].ne("test"), "Cell_line"].tolist()
    combined.loc[development_ids].to_csv(args.output_dir / "development_data_file.csv")
    labels.loc[development_ids].to_csv(args.output_dir / "development_label_file.csv")

    summary = {
        "source_rna_dir": str(args.rna_dir.resolve()),
        "source_cnv_dir": str(args.cnv_dir.resolve()),
        "cnv_workflow": "ASCAT2",
        "cnv_transform": "copy_number_minus_2",
        "matched_patients": int(len(combined)),
        "rna_pathway_genes_before_cnv_filter": int(len(rna_genes)),
        "genes_retained_with_complete_cnv": int(len(complete_genes)),
        "genes_dropped_for_missing_cnv": int(len(dropped)),
        "combined_features": int(combined.shape[1]),
        "split_counts": assignments.groupby(["split", "Cancer_type"]).size().unstack(fill_value=0).to_dict("index"),
        "test_evaluated": False,
    }
    (args.output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
