#!/usr/bin/env python3
"""Prepare TCGA-UCEC GDC RNA files for DeePathNet.

This script converts the GDC STAR-count per-sample TSV files into the
sample-by-feature CSV layout expected by DeePathNet's practicum configs.
It also flattens clinical labels into a DeePathNet-compatible target file.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
PROJECTS_ROOT = REPO_ROOT.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build TCGA-UCEC RNA DeePathNet input files."
    )
    parser.add_argument(
        "--gdc-root",
        type=Path,
        default=PROJECTS_ROOT / "practicum/data/tcga_ucec_gdc",
        help="Root directory containing downloaded TCGA-UCEC GDC files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "data/processed/tcga_ucec_rna",
        help="Directory for processed DeePathNet-ready CSV files.",
    )
    parser.add_argument(
        "--config-dir",
        type=Path,
        default=REPO_ROOT / "configs/practicum/tcga_ucec_rna",
        help="Directory for the generated DeePathNet config.",
    )
    parser.add_argument(
        "--pathway-file",
        type=Path,
        default=REPO_ROOT
        / "data/graph_predefined/LCPathways/41568_2020_240_MOESM4_ESM.csv",
        help="Pathway file used to restrict features and configure DeePathNet.",
    )
    parser.add_argument(
        "--value-column",
        default="tpm_unstranded",
        help="GDC STAR-count column to use as expression value.",
    )
    parser.add_argument(
        "--transform",
        choices=["none", "log2p"],
        default="log2p",
        help="Expression transform to apply after reading each sample.",
    )
    parser.add_argument(
        "--gene-type",
        default="protein_coding",
        help="Gene type to keep. Use 'all' to disable this filter.",
    )
    parser.add_argument(
        "--label-column",
        default="stage_main",
        choices=["stage_main", "grade_binary", "histology_binary"],
        help="Clinical label to export as DeePathNet's Cancer_type target.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test fraction for the train/test split.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--keep-multiple-samples-per-case",
        action="store_true",
        help="Keep all primary-tumor samples. Default keeps one sample per case.",
    )
    parser.add_argument(
        "--use-all-genes-for-model",
        action="store_true",
        help="Use all kept genes in train/test files instead of pathway-overlap genes.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")


def normalize_stage(figo_stage: object) -> str | None:
    if pd.isna(figo_stage):
        return None
    value = str(figo_stage).strip().upper()
    if not value or value in {"NOT REPORTED", "UNKNOWN", "NAN"}:
        return None
    # Match the complete stage so III cannot be mistaken for I or II.
    match = re.fullmatch(r"STAGE\s+(IV|III|II|I)(?:[ABC][12]?)?", value)
    if match is None:
        raise ValueError(f"Unrecognized FIGO stage: {figo_stage!r}")
    return f"Stage {match.group(1)}"


def normalize_grade(tumor_grade: object) -> str | None:
    if pd.isna(tumor_grade):
        return None
    value = str(tumor_grade).strip().upper()
    if value in {"G1", "G2"}:
        return "Low_Grade"
    if value in {"G3", "G4", "HIGH GRADE"}:
        return "High_Grade"
    return None


def normalize_histology(primary_diagnosis: object) -> str | None:
    if pd.isna(primary_diagnosis):
        return None
    value = str(primary_diagnosis).strip().lower()
    if not value or value in {"not reported", "unknown", "nan"}:
        return None
    if "endometrioid" in value:
        return "Endometrioid"
    return "Non_Endometrioid"


def read_star_count_file(path: Path, value_column: str, gene_type: str, transform: str) -> pd.Series:
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    required = {"gene_id", "gene_name", "gene_type", value_column}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df = df[~df["gene_id"].astype(str).str.startswith("N_")].copy()
    df = df[df["gene_name"].notna() & (df["gene_name"].astype(str) != "")]
    if gene_type != "all":
        df = df[df["gene_type"] == gene_type]

    values = pd.to_numeric(df[value_column], errors="coerce").fillna(0.0)
    if transform == "log2p":
        values = np.log2(values + 1.0)

    series = pd.Series(values.to_numpy(), index=df["gene_name"].astype(str))
    series = series.groupby(level=0).mean()
    series.index = [f"{gene}_RNA" for gene in series.index]
    return series


def load_pathway_genes(pathway_file: Path) -> set[str]:
    pathway_df = pd.read_csv(pathway_file)
    genes: set[str] = set()
    for value in pathway_df["genes"].dropna():
        genes.update(gene.strip() for gene in str(value).split("|") if gene.strip())
    return {f"{gene}_RNA" for gene in genes}


def select_one_sample_per_case(sample_sheet: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sample_sheet.sort_values(["case_submitter_id", "sample_submitter_id", "file_name"])
    selected = ordered.drop_duplicates("case_submitter_id", keep="first").copy()
    dropped = ordered[~ordered["file_id"].isin(selected["file_id"])].copy()
    return selected, dropped


def build_expression_matrix(
    sample_sheet: pd.DataFrame,
    files_dir: Path,
    value_column: str,
    gene_type: str,
    transform: str,
) -> pd.DataFrame:
    rows: list[pd.Series] = []
    sample_ids: list[str] = []
    total = len(sample_sheet)
    for idx, row in enumerate(sample_sheet.itertuples(index=False), start=1):
        file_path = files_dir / row.file_name
        require_file(file_path, "RNA STAR-count TSV")
        rows.append(read_star_count_file(file_path, value_column, gene_type, transform))
        sample_ids.append(row.sample_submitter_id)
        if idx == 1 or idx % 50 == 0 or idx == total:
            print(f"Loaded {idx}/{total}: {row.sample_submitter_id}", flush=True)

    matrix = pd.DataFrame(rows, index=sample_ids).fillna(0.0)
    matrix.index.name = "Cell_line"
    matrix = matrix.reindex(sorted(matrix.columns), axis=1)
    return matrix


def build_label_table(sample_sheet: pd.DataFrame, clinical_file: Path) -> pd.DataFrame:
    clinical = pd.read_csv(clinical_file, sep="\t", low_memory=False)
    clinical["stage_main"] = clinical["figo_stage"].map(normalize_stage)
    clinical["grade_binary"] = clinical["tumor_grade"].map(normalize_grade)
    clinical["histology_binary"] = clinical["primary_diagnosis"].map(normalize_histology)

    keep_cols = [
        "case_submitter_id",
        "vital_status",
        "age_at_index",
        "primary_diagnosis",
        "figo_stage",
        "tumor_grade",
        "stage_main",
        "grade_binary",
        "histology_binary",
    ]
    labels = sample_sheet[
        ["sample_submitter_id", "case_submitter_id", "sample_type", "file_name"]
    ].merge(clinical[keep_cols], on="case_submitter_id", how="left")
    labels = labels.rename(columns={"sample_submitter_id": "Cell_line"})
    return labels


def write_config(
    config_dir: Path,
    output_dir: Path,
    pathway_file: Path,
    label_column: str,
    seed: int,
    cancer_only: bool = True,
) -> Path:
    config_dir.mkdir(parents=True, exist_ok=True)
    rel = lambda path: str(path.relative_to(REPO_ROOT))
    config = {
        "data_file_train": rel(output_dir / "train_data_file.csv"),
        "target_file_train": rel(output_dir / "train_label_file.csv"),
        "data_file_test": rel(output_dir / "test_data_file.csv"),
        "target_file_test": rel(output_dir / "test_label_file.csv"),
        "pathway_file": rel(pathway_file),
        "model": "DeePathNet",
        "do_cv": False,
        "work_dir": f"work_dirs/practicum/DeePathNet/tcga_ucec_{label_column}_rna",
        "data_type": ["RNA"],
        "task": "multiclass",
        "validation_size": 0.2,
        "validation_seed": seed,
        "seed": seed,
        "num_repeat": 1,
        "batch_size": 64,
        "num_workers": 1,
        "log_freq": 20,
        "num_of_epochs": 100,
        "dim": 512,
        "mlp_ratio": 2,
        "out_mlp_ratio": 8,
        "heads": 16,
        "depth": 2,
        "dropout": 0,
        "emb_dropout": 0,
        "pathway_dropout": 0.5,
        "weight_decay": 1e-5,
        "cancer_only": cancer_only,
        "lr": 1e-5,
        "save_checkpoints": True,
        "save_scores": True,
        "drop_last": False,
        "suffix": f"_tcga_ucec_{label_column}_rna",
        "saved_model": "",
    }
    config_path = config_dir / f"deepathnet_tcga_ucec_{label_column}_rna.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    return config_path


def main() -> None:
    args = parse_args()
    sample_sheet_file = (
        args.gdc_root
        / "metadata/tcga_ucec_primary_tumor_rna_star_counts_sample_sheet.tsv"
    )
    clinical_file = args.gdc_root / "clinical/tcga_ucec_clinical_flat.tsv"
    files_dir = args.gdc_root / "rna_star_counts/files"
    require_file(sample_sheet_file, "sample sheet")
    require_file(clinical_file, "clinical flat table")
    require_file(args.pathway_file, "pathway file")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_sheet = pd.read_csv(sample_sheet_file, sep="\t", low_memory=False)
    if args.keep_multiple_samples_per_case:
        selected_sample_sheet = sample_sheet.copy()
        dropped_sample_sheet = sample_sheet.iloc[0:0].copy()
    else:
        selected_sample_sheet, dropped_sample_sheet = select_one_sample_per_case(
            sample_sheet
        )

    selected_sample_sheet.to_csv(args.output_dir / "selected_sample_sheet.tsv", sep="\t", index=False)
    dropped_sample_sheet.to_csv(
        args.output_dir / "dropped_duplicate_primary_tumor_samples.tsv",
        sep="\t",
        index=False,
    )

    expression_all = build_expression_matrix(
        selected_sample_sheet,
        files_dir,
        args.value_column,
        args.gene_type,
        args.transform,
    )
    all_matrix_file = (
        args.output_dir
        / f"tcga_ucec_expression_{args.value_column}_{args.transform}_{args.gene_type}_all_genes.csv"
    )
    expression_all.to_csv(all_matrix_file)

    pathway_features = sorted(set(expression_all.columns).intersection(load_pathway_genes(args.pathway_file)))
    expression_pathway = expression_all.loc[:, pathway_features]
    pathway_matrix_file = (
        args.output_dir
        / f"tcga_ucec_expression_{args.value_column}_{args.transform}_{args.gene_type}_pathway_genes.csv"
    )
    expression_pathway.to_csv(pathway_matrix_file)

    labels = build_label_table(selected_sample_sheet, clinical_file)
    labels.to_csv(args.output_dir / "tcga_ucec_labels_all_candidates.tsv", sep="\t", index=False)

    model_matrix = expression_all if args.use_all_genes_for_model else expression_pathway
    model_matrix.to_csv(args.output_dir / "data_file_all_selected_samples.csv")

    model_labels = labels[["Cell_line", args.label_column]].dropna().copy()
    model_labels = model_labels.rename(columns={args.label_column: "Cancer_type"})
    model_labels = model_labels[model_labels["Cell_line"].isin(model_matrix.index)]
    model_labels = model_labels.drop_duplicates("Cell_line").set_index("Cell_line")
    model_matrix = model_matrix.loc[model_labels.index]
    model_matrix.to_csv(args.output_dir / "data_file.csv")

    class_counts = model_labels["Cancer_type"].value_counts().to_dict()
    if len(class_counts) < 2:
        raise ValueError(f"{args.label_column} has fewer than two classes: {class_counts}")
    if min(class_counts.values()) < 2:
        raise ValueError(
            f"{args.label_column} has a class with fewer than two samples: {class_counts}"
        )

    train_ids, test_ids = train_test_split(
        model_labels.index,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=model_labels["Cancer_type"],
    )
    train_ids = sorted(train_ids)
    test_ids = sorted(test_ids)
    model_matrix.loc[train_ids].to_csv(args.output_dir / "train_data_file.csv")
    model_matrix.loc[test_ids].to_csv(args.output_dir / "test_data_file.csv")
    model_labels.loc[train_ids].to_csv(args.output_dir / "train_label_file.csv")
    model_labels.loc[test_ids].to_csv(args.output_dir / "test_label_file.csv")
    model_labels.to_csv(args.output_dir / "target_file.csv")

    config_path = write_config(
        args.config_dir,
        args.output_dir,
        args.pathway_file,
        args.label_column,
        args.seed,
        cancer_only=set(model_matrix.columns).issubset(pathway_features),
    )

    summary = {
        "gdc_root": str(args.gdc_root),
        "output_dir": str(args.output_dir),
        "selected_samples": int(selected_sample_sheet.shape[0]),
        "dropped_duplicate_primary_tumor_samples": int(dropped_sample_sheet.shape[0]),
        "all_gene_features": int(expression_all.shape[1]),
        "pathway_gene_features": int(expression_pathway.shape[1]),
        "model_features": int(model_matrix.shape[1]),
        "label_column": args.label_column,
        "class_counts": {k: int(v) for k, v in class_counts.items()},
        "missing_label_samples": int(labels[args.label_column].isna().sum()),
        "train_class_counts": model_labels.loc[train_ids, "Cancer_type"].value_counts().to_dict(),
        "test_class_counts": model_labels.loc[test_ids, "Cancer_type"].value_counts().to_dict(),
        "train_samples": int(len(train_ids)),
        "test_samples": int(len(test_ids)),
        "value_column": args.value_column,
        "transform": args.transform,
        "gene_type": args.gene_type,
        "config_path": str(config_path),
    }
    (args.output_dir / "prepare_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
