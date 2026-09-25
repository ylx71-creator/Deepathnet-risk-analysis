#!/usr/bin/env python3
"""Screen train-only Z-score standardization for compact RNA+CNV UCEC training.

The script reads only the fixed 292/75 development split and never opens test
data or test labels. It changes input scaling only, relative to the completed
raw RNA+CNV condition.
"""

import argparse
import copy
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from compare_tcga_ucec_rna_cnv_development import condition_frame, load_development_data
from compare_tcga_ucec_transformer_dropout import aggregate_summaries, run_one
import train_tcga_ucec_stage as training


SPECIFICATION = {
    "description": "Train-only Z-score for every RNA and CNV feature; compact architecture unchanged.",
    "config_overrides": {},
}


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
    frame = condition_frame(x, ("RNA", "cnv"))
    base_config = json.loads((training.REPO_ROOT / "configs/practicum/tcga_ucec_rna/deepathnet_tcga_ucec_stage_main_rna.json").read_text())
    base_config.update({"dim": 256, "heads": 8, "depth": 2, "lr": 1e-4, "dropout": 0.0, "emb_dropout": 0.0})
    output = training.REPO_ROOT / "work_dirs/practicum/DeePathNet/tcga_ucec_stage_main_rna" / f"rna_cnv_standardization_screen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    assignments.to_csv(output / "development_split_assignments.csv", index=False)
    (output / "screen_metadata.json").write_text(json.dumps({"source_dir": str(source), "train_patients": len(train_ids), "validation_patients": len(validation_ids), "seeds": args.seeds, "epochs": args.epochs, "condition": SPECIFICATION, "test_evaluated": False}, indent=2) + "\n")
    summaries = []
    for seed in args.seeds:
        summaries.append(run_one("rna_cnv_train_zscore", SPECIFICATION, seed, copy.deepcopy(base_config), frame.loc[train_ids], y.loc[train_ids], frame.loc[validation_ids], y.loc[validation_ids], train_ids, validation_ids, output, args.epochs, args.batch_size, device, ("RNA", "cnv"), True))
        pd.DataFrame(summaries).to_csv(output / "run_summary.csv", index=False)
        aggregate_summaries(summaries).to_csv(output / "condition_summary.csv", index=False)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
