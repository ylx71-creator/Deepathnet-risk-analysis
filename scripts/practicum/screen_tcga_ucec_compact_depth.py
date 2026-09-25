"""Screen one Transformer block against the completed two-block compact model.

This script never reads test data or test labels. It keeps the fixed 300/76
TCGA-UCEC development patients and changes only Transformer depth from two to one.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

import compare_tcga_ucec_development_settings as development
from compare_tcga_ucec_transformer_dropout import aggregate_summaries, run_one


DEPTH_ONE_SPECIFICATION = {
    "description": "256-dimensional, eight-head, one-block Transformer; all other selected settings unchanged.",
    "config_overrides": {
        "dim": 256,
        "heads": 8,
        "depth": 1,
        "dropout": 0.0,
        "emb_dropout": 0.0,
    },
}


def read_depth_two_summaries(path, seeds):
    frame = pd.read_csv(path / "compact_run_summary.csv")
    baseline = frame[frame.seed.isin(seeds)].copy()
    if set(baseline.seed) != set(seeds) or baseline.test_evaluated.any():
        raise ValueError("Depth-two comparison does not contain the requested validation-only seeds")
    return baseline.to_dict("records")


def paired_depth_comparison(depth_one_summaries, depth_two_summaries):
    """Pair matching seeds and make the depth-only contrast explicit in output."""
    depth_two_by_seed = {int(summary["seed"]): summary for summary in depth_two_summaries}
    paired = []
    for depth_one in depth_one_summaries:
        seed = int(depth_one["seed"])
        depth_two = depth_two_by_seed[seed]
        paired.append({
            "seed": seed,
            "depth_two_macro_f1": float(depth_two["best_validation_macro_f1"]),
            "depth_one_macro_f1": float(depth_one["best_validation_macro_f1"]),
            "macro_f1_difference_depth_one_minus_depth_two": (
                float(depth_one["best_validation_macro_f1"])
                - float(depth_two["best_validation_macro_f1"])
            ),
            "depth_two_accuracy": float(depth_two["validation_accuracy"]),
            "depth_one_accuracy": float(depth_one["validation_accuracy"]),
            "accuracy_difference_depth_one_minus_depth_two": (
                float(depth_one["validation_accuracy"]) - float(depth_two["validation_accuracy"])
            ),
        })
    return pd.DataFrame(paired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("depth_two_comparison", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("epochs, batch-size, cpu-threads, and unique seeds are required")
    source = args.source_run.resolve()
    baseline_path = args.depth_two_comparison.resolve()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_metadata, base_config, paths, x, y, assignments, train_ids, validation_ids = (
        development.load_development_data(source)
    )
    depth_two_summaries = read_depth_two_summaries(baseline_path, args.seeds)
    output = source.parent / f"compact_depth_screen_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    assignments.to_csv(output / "development_split_assignments.csv", index=False)
    metadata = {
        "source_run": str(source),
        "depth_two_comparison": str(baseline_path),
        "source_input_hashes": {key: source_metadata["input_sha256"][key] for key in paths},
        "training_patients": len(train_ids),
        "validation_patients": len(validation_ids),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "condition": DEPTH_ONE_SPECIFICATION,
        "fixed_changes": {"lr": 1e-4},
        "test_evaluated": False,
    }
    (output / "screen_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Output: {output}; train=300 validation=76 seeds={args.seeds} test_evaluated=false", flush=True)
    depth_one_summaries = []
    for seed in args.seeds:
        depth_one_summaries.append(run_one(
            "compact_depth1", DEPTH_ONE_SPECIFICATION, seed, base_config,
            x.loc[train_ids], y.loc[train_ids], x.loc[validation_ids], y.loc[validation_ids],
            train_ids, validation_ids, output, args.epochs, args.batch_size, device,
        ))
        pd.DataFrame(depth_one_summaries).to_csv(output / "depth_one_run_summary.csv", index=False)
    aggregate_summaries(depth_one_summaries).to_csv(output / "depth_one_condition_summary.csv", index=False)
    paired_depth_comparison(depth_one_summaries, depth_two_summaries).to_csv(
        output / "paired_depth_comparison.csv", index=False
    )
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
