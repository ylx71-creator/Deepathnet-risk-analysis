"""Compare Transformer dropout settings on the fixed TCGA-UCEC development split.

This script never reads test data or test labels. It reuses the fixed 300/76
train/validation assignment from the original UCEC training run and evaluates
each dropout condition from a fresh initialization for every requested seed.
"""

import argparse
import copy
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import compare_tcga_ucec_development_settings as development
import train_tcga_ucec_stage as training


CONDITIONS = {
    "no_transformer_dropout": {
        "description": "Transformer MLP/projection and attention dropout are both 0.0.",
        "config_overrides": {"dropout": 0.0, "emb_dropout": 0.0},
    },
    "transformer_dropout_0p1": {
        "description": "Transformer MLP/projection and attention dropout are both 0.1.",
        "config_overrides": {"dropout": 0.1, "emb_dropout": 0.1},
    },
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_one(condition_name, specification, seed, base_config, train_x, train_y,
            validation_x, validation_y, train_ids, validation_ids, output, epochs,
            batch_size, device, omics_types=("RNA",), standardize_inputs=False):
    set_seed(seed)
    config = copy.deepcopy(base_config)
    config.update({"seed": seed, "lr": 1e-4, **specification["config_overrides"]})
    run_dir = output / condition_name / f"seed_{seed}"
    run_dir.mkdir(parents=True)
    scaler = None
    if standardize_inputs:
        train_x, validation_x, scaler = development.standardize(train_x, validation_x)
        pd.DataFrame({
            "feature": train_x.columns,
            "training_mean": scaler.mean_,
            "training_standard_deviation": scaler.scale_,
        }).to_csv(run_dir / "training_scaler.csv", index=False)
    class_map = {name: index for index, name in enumerate(training.CLASSES)}
    train_loader = development.make_loader(
        train_x, train_y, "train", class_map, batch_size, True, omics_types
    )
    validation_loader = development.make_loader(
        validation_x, validation_y, "val", class_map, batch_size, False, omics_types
    )
    model, pathways = training.make_model(train_loader.dataset, config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    metadata = {
        "condition": condition_name,
        "description": specification["description"],
        "seed": seed,
        "config": config,
        "train_patients": len(train_ids),
        "validation_patients": len(validation_ids),
        "pathway_count": len(pathways),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "omics_types": list(omics_types),
        "standardize_inputs": standardize_inputs,
        "scaler_fit_patients": len(train_ids) if scaler is not None else 0,
        "test_evaluated": False,
    }
    (run_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_score, best_epoch, history = -float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        loss, seen = training.train_epoch(model, train_loader, optimizer, device)
        targets, probabilities = training.predict(model, validation_loader, device)
        metrics, _, _, _ = training.summarize_predictions(validation_ids, targets, probabilities)
        history.append({
            "epoch": epoch,
            "training_loss": loss,
            "training_samples_seen": seen,
            **{f"validation_{name}": value for name, value in metrics.items()},
            "seconds": time.perf_counter() - epoch_started,
        })
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch = metrics["macro_f1"], epoch
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            training.save_evaluation(run_dir, "validation", validation_ids, targets, probabilities)
        pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)
        print(
            f"{condition_name} seed={seed} epoch={epoch}/{epochs} "
            f"loss={loss:.4f} validation_macro_f1={metrics['macro_f1']:.4f}",
            flush=True,
        )
    model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device, weights_only=True))
    targets, probabilities = training.predict(model, validation_loader, device)
    selected_metrics = training.save_evaluation(
        run_dir, "selected_validation", validation_ids, targets, probabilities
    )
    summary = {
        "condition": condition_name,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "validation_accuracy": selected_metrics["accuracy"],
        "validation_roc_auc_ovo": selected_metrics["roc_auc_ovo"],
        "elapsed_seconds": time.perf_counter() - started,
        "test_evaluated": False,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def aggregate_summaries(summaries):
    frame = pd.DataFrame(summaries)
    return frame.groupby("condition", as_index=False).agg(
        best_validation_macro_f1_mean=("best_validation_macro_f1", "mean"),
        best_validation_macro_f1_std=("best_validation_macro_f1", "std"),
        validation_accuracy_mean=("validation_accuracy", "mean"),
        validation_accuracy_std=("validation_accuracy", "std"),
        validation_roc_auc_ovo_mean=("validation_roc_auc_ovo", "mean"),
        validation_roc_auc_ovo_std=("validation_roc_auc_ovo", "std"),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1 or any(seed < 0 for seed in args.seeds):
        raise ValueError("epochs, batch-size, cpu-threads, and seeds must be non-negative")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Seeds must be unique")
    source = args.source_run.resolve()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    source_metadata, base_config, paths, x, y, assignments, train_ids, validation_ids = (
        development.load_development_data(source)
    )
    output = source.parent / f"transformer_dropout_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    assignments.to_csv(output / "development_split_assignments.csv", index=False)
    metadata = {
        "source_run": str(source),
        "source_input_hashes": {key: source_metadata["input_sha256"][key] for key in paths},
        "training_patients": len(train_ids),
        "validation_patients": len(validation_ids),
        "seeds": args.seeds,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads,
        "conditions": CONDITIONS,
        "fixed_changes": {"lr": 1e-4},
        "test_evaluated": False,
    }
    (output / "comparison_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"Output: {output}; train=300 validation=76 seeds={args.seeds} test_evaluated=false",
        flush=True,
    )
    summaries = []
    for condition_name, specification in CONDITIONS.items():
        for seed in args.seeds:
            summaries.append(run_one(
                condition_name, specification, seed, base_config, x.loc[train_ids], y.loc[train_ids],
                x.loc[validation_ids], y.loc[validation_ids], train_ids, validation_ids, output,
                args.epochs, args.batch_size, device,
            ))
            pd.DataFrame(summaries).to_csv(output / "run_summary.csv", index=False)
            aggregate_summaries(summaries).to_csv(output / "condition_summary.csv", index=False)
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
