"""Balanced training-only memorization checks; never evaluate held-out patients."""

import argparse
import copy
import hashlib
import json
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import train_tcga_ucec_stage as training


def select_samples(assignments, per_class, seed):
    if per_class < 1:
        raise ValueError("per-class must be positive")
    if not assignments.Cell_line.is_unique or not assignments.case_submitter_id.is_unique:
        raise ValueError("Repeated samples or patients in source splits")
    eligible = assignments.loc[assignments.split == "train"].sort_values("Cell_line")
    chosen = []
    for label in training.CLASSES:
        group = eligible[eligible.Cancer_type == label]
        if len(group) < per_class:
            raise ValueError(f"Not enough training patients for {label}")
        chosen.append(group.sample(n=per_class, random_state=seed))
    return pd.concat(chosen).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--per-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")
    source = args.source_run.resolve()
    original = json.loads((source / "run_metadata.json").read_text())
    config = original["config"]
    paths = {key: training.resolve_path(config[key]) for key in ["data_file_train", "target_file_train", "pathway_file"]}
    for key, path in paths.items():
        if hashlib.sha256(path.read_bytes()).hexdigest() != original["input_sha256"][key]:
            raise ValueError(f"Source input changed since the original run: {key}")
    assignments = pd.read_csv(source / "split_assignments.csv")
    selected = select_samples(assignments, args.per_class, args.seed)
    x, y = training.read_pair(paths["data_file_train"], paths["target_file_train"])
    ids = selected.Cell_line.tolist()
    x, y = x.loc[ids], y.loc[ids]
    if selected.Cancer_type.tolist() != y.Cancer_type.tolist():
        raise ValueError("Selected labels disagree with source labels")
    output = source.parent / f"overfit_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    selected.to_csv(output / "selected_training_samples.csv", index=False)
    torch.set_num_threads(4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    conditions = [
        ("original_settings", config["lr"], config["pathway_dropout"], config["weight_decay"], False),
        ("memorization_settings", 1e-3, 0.0, 0.0, False),
        ("standardized_memorization", 1e-3, 0.0, 0.0, True),
    ]
    metadata = {
        "source_run": str(source), "seed": args.seed, "samples_per_stage": args.per_class,
        "max_steps": args.steps, "device": str(device), "original_config": config,
        "conditions": conditions, "held_out_evaluated": False,
        "success_rule": "5 consecutive evaluations with training accuracy 1.0 and cross-entropy < 0.05",
        "warning": "Same samples used for optimization and evaluation; these are not generalization scores.",
        "input_sha256": {key: original["input_sha256"][key] for key in paths},
    }
    (output / "diagnostic_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Output: {output}; balanced training-only samples: {len(ids)}", flush=True)
    results = []
    for name, lr, dropout, decay, standardize in conditions:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        settings = copy.deepcopy(config)
        settings.update(lr=lr, pathway_dropout=dropout, weight_decay=decay)
        values = x.copy()
        condition_dir = output / name
        condition_dir.mkdir()
        if standardize:
            scaler = StandardScaler().fit(values)
            values = pd.DataFrame(scaler.transform(values), index=x.index, columns=x.columns)
            pd.DataFrame({"feature": x.columns, "mean": scaler.mean_, "scale": scaler.scale_}).to_csv(
                condition_dir / "scaler.csv", index=False)
        dataset = training.MultiOmicMulticlassDataset(values, y, mode="train", omics_types=["RNA"],
                    class_name_to_id={label: i for i, label in enumerate(training.CLASSES)})
        loader = DataLoader(dataset, batch_size=len(ids), shuffle=False, num_workers=0)
        model, _ = training.make_model(dataset, settings)
        model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=decay)
        history, consecutive, best_loss = [], 0, float("inf")
        started = time.perf_counter()
        for step in range(args.steps + 1):
            optimization_loss = None
            if step:
                optimization_loss, _ = training.train_epoch(model, loader, optimizer, device)
            targets, probabilities = training.predict(model, loader, device)
            metrics, _, _, _ = training.summarize_predictions(ids, targets, probabilities)
            loss = float(-np.log(probabilities[np.arange(len(ids)), targets].clip(1e-12)).mean())
            history.append({"step": step, "optimization_loss": optimization_loss,
                            "memorization_loss": loss, **metrics})
            if loss < best_loss:
                best_loss = loss
                best = {"step": step, "loss": loss, **metrics}
                best_targets, best_probabilities = targets.copy(), probabilities.copy()
            successful = metrics["accuracy"] == 1.0 and loss < .05
            consecutive = consecutive + 1 if successful else 0
            if step % 25 == 0 or consecutive >= 5 or step == args.steps:
                print(f"{name} step={step} loss={loss:.5f} acc={metrics['accuracy']:.3f} macro_f1={metrics['macro_f1']:.3f}", flush=True)
                pd.DataFrame(history).to_csv(condition_dir / "history.csv", index=False)
            if consecutive >= 5:
                break
        training.save_evaluation(condition_dir, "best_training", ids, best_targets, best_probabilities)
        training.save_evaluation(condition_dir, "final_training", ids, targets, probabilities)
        torch.save(model.state_dict(), condition_dir / "final_model.pth")
        result = {"condition": name, "steps_completed": step, "memorization_passed": consecutive >= 5,
                  "best": best, "final": history[-1], "elapsed_seconds": time.perf_counter() - started}
        results.append(result)
        (output / "summary.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        del model, optimizer, loader, dataset
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
