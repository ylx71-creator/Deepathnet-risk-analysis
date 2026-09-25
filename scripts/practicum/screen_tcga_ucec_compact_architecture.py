"""Screen a smaller DeePathNet architecture on the fixed TCGA-UCEC development split.

The screen never reads test data or test labels. It compares a 256-dimensional,
eight-head model against completed 512-dimensional baseline runs using matching
seeds and the same fixed 300/76 train/validation patients.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import compare_tcga_ucec_development_settings as development
from compare_tcga_ucec_transformer_dropout import aggregate_summaries, run_one


COMPACT_SPECIFICATION = {
    "description": "256-dimensional, eight-head Transformer; all other selected settings unchanged.",
    "config_overrides": {"dim": 256, "heads": 8, "dropout": 0.0, "emb_dropout": 0.0},
}


def paired_comparison(compact_summaries, baseline_summaries):
    compact = pd.DataFrame(compact_summaries).set_index("seed")
    baseline = pd.DataFrame(baseline_summaries).set_index("seed")
    if set(compact.index) != set(baseline.index):
        raise ValueError("Compact and baseline runs must use identical seeds")
    paired = pd.DataFrame(index=sorted(compact.index))
    paired.index.name = "seed"
    paired["baseline_macro_f1"] = baseline.loc[paired.index, "best_validation_macro_f1"]
    paired["compact_macro_f1"] = compact.loc[paired.index, "best_validation_macro_f1"]
    paired["macro_f1_difference_compact_minus_baseline"] = (
        paired["compact_macro_f1"] - paired["baseline_macro_f1"]
    )
    paired["baseline_accuracy"] = baseline.loc[paired.index, "validation_accuracy"]
    paired["compact_accuracy"] = compact.loc[paired.index, "validation_accuracy"]
    return paired.reset_index()


def read_baseline_summaries(path, seeds):
    frame = pd.read_csv(path / "run_summary.csv")
    baseline = frame[frame.condition.eq("no_transformer_dropout") & frame.seed.isin(seeds)].copy()
    if set(baseline.seed) != set(seeds) or baseline.test_evaluated.any():
        raise ValueError("Baseline comparison does not contain the requested validation-only seeds")
    return baseline.to_dict("records")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("baseline_comparison", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("epochs, batch-size, cpu-threads, and unique seeds are required")
    source = args.source_run.resolve()
    baseline_path = args.baseline_comparison.resolve()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_metadata, base_config, paths, x, y, assignments, train_ids, validation_ids = (
        development.load_development_data(source)
    )
    baseline_summaries = read_baseline_summaries(baseline_path, args.seeds)
    output = source.parent / f"compact_architecture_screen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    assignments.to_csv(output / "development_split_assignments.csv", index=False)
    metadata = {
        "source_run": str(source),
        "baseline_comparison": str(baseline_path),
        "source_input_hashes": {key: source_metadata["input_sha256"][key] for key in paths},
        "training_patients": len(train_ids),
        "validation_patients": len(validation_ids),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "condition": COMPACT_SPECIFICATION,
        "fixed_changes": {"lr": 1e-4},
        "test_evaluated": False,
    }
    (output / "screen_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Output: {output}; train=300 validation=76 seeds={args.seeds} test_evaluated=false", flush=True)
    compact_summaries = []
    for seed in args.seeds:
        compact_summaries.append(run_one(
            "compact_dim256_heads8", COMPACT_SPECIFICATION, seed, base_config,
            x.loc[train_ids], y.loc[train_ids], x.loc[validation_ids], y.loc[validation_ids],
            train_ids, validation_ids, output, args.epochs, args.batch_size, device,
        ))
        pd.DataFrame(compact_summaries).to_csv(output / "compact_run_summary.csv", index=False)
    aggregate_summaries(compact_summaries).to_csv(output / "compact_condition_summary.csv", index=False)
    paired_comparison(compact_summaries, baseline_summaries).to_csv(
        output / "paired_baseline_comparison.csv", index=False
    )
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
