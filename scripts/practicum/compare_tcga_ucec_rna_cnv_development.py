#!/usr/bin/env python3
"""Compare matched RNA-only and RNA+CNV DeePathNet on UCEC development data.

The script reads only the 292 fixed training patients and 75 fixed validation
patients created after CNV matching.  It never opens the retained 94-patient
test files or labels.
"""

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import compare_tcga_ucec_development_settings as development
from compare_tcga_ucec_transformer_dropout import aggregate_summaries, run_one
import train_tcga_ucec_stage as training


CONDITIONS = {
    "rna_only_matched": {
        "description": "RNA only, restricted to the CNV-matched patients and genes.",
        "omics_types": ("RNA",),
        "config_overrides": {},
    },
    "rna_cnv_matched": {
        "description": "RNA plus centered ASCAT2 gene-level CNV for the same patients and genes.",
        "omics_types": ("RNA", "cnv"),
        "config_overrides": {},
    },
}


def load_development_data(source_dir):
    x = pd.read_csv(source_dir / "development_data_file.csv", index_col="Cell_line")
    y = pd.read_csv(source_dir / "development_label_file.csv", index_col="Cell_line")
    assignments = pd.read_csv(source_dir / "fixed_split_assignments.csv")
    assignments = assignments[assignments["split"].isin(["train", "validation"])].copy()
    if not x.index.equals(y.index) or set(x.index) != set(assignments.Cell_line):
        raise ValueError("Development matrix, labels, and fixed assignments must match")
    if not pd.api.types.is_numeric_dtype(x.dtypes.iloc[0]) or not x.notna().all().all():
        raise ValueError("Development features must be numeric and complete")
    train_ids = assignments.loc[assignments["split"].eq("train"), "Cell_line"].tolist()
    validation_ids = assignments.loc[assignments["split"].eq("validation"), "Cell_line"].tolist()
    if len(train_ids) != 292 or len(validation_ids) != 75:
        raise ValueError("Expected the fixed CNV-matched 292/75 development split")
    if any(set(y.loc[ids, "Cancer_type"]) != set(training.CLASSES) for ids in (train_ids, validation_ids)):
        raise ValueError("Each development split must include all four stages")
    return x, y, assignments, train_ids, validation_ids


def condition_frame(x, omics_types):
    columns = [column for column in x.columns if column.rsplit("_", 1)[-1] in omics_types]
    frame = x.loc[:, columns]
    genes = {column.rsplit("_", 1)[0] for column in frame.columns}
    if set(omics_types) == {"RNA", "cnv"} and len(frame.columns) != len(genes) * 2:
        raise ValueError("RNA+CNV comparison requires both modalities for every retained gene")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=training.REPO_ROOT / "data/processed/tcga_ucec_rna_cnv")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Positive settings and unique seeds are required")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source = args.source_dir.resolve()
    x, y, assignments, train_ids, validation_ids = load_development_data(source)
    base_config = json.loads((training.REPO_ROOT / "configs/practicum/tcga_ucec_rna/deepathnet_tcga_ucec_stage_main_rna.json").read_text())
    base_config.update({"dim": 256, "heads": 8, "depth": 2, "lr": 1e-4, "dropout": 0.0, "emb_dropout": 0.0})
    output = training.REPO_ROOT / "work_dirs/practicum/DeePathNet/tcga_ucec_stage_main_rna" / f"rna_cnv_development_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    assignments.to_csv(output / "development_split_assignments.csv", index=False)
    (output / "comparison_metadata.json").write_text(json.dumps({"source_dir": str(source), "train_patients": len(train_ids), "validation_patients": len(validation_ids), "conditions": CONDITIONS, "test_evaluated": False}, indent=2) + "\n")
    summaries = []
    for name, spec in CONDITIONS.items():
        frame = condition_frame(x, spec["omics_types"])
        for seed in args.seeds:
            summaries.append(run_one(name, spec, seed, copy.deepcopy(base_config), frame.loc[train_ids], y.loc[train_ids], frame.loc[validation_ids], y.loc[validation_ids], train_ids, validation_ids, output, args.epochs, args.batch_size, device, spec["omics_types"]))
            pd.DataFrame(summaries).to_csv(output / "run_summary.csv", index=False)
            aggregate_summaries(summaries).to_csv(output / "condition_summary.csv", index=False)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
