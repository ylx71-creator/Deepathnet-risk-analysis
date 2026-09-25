"""Compare one preprocessing or optimization change at a time on development data.

The script never reads test data or test labels. It reuses the fixed 300/76
train/validation assignment recorded in a prior TCGA-UCEC training run.
"""

import argparse
import copy
import hashlib
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

import train_tcga_ucec_stage as training


CONDITIONS = {
    "standardization": {
        "description": "Gene-wise Z-score fitted on the 300 training samples only.",
        "standardize": True,
    },
    "learning_rate": {
        "description": "Learning rate 1e-4; all other settings stay at the original values.",
        "lr": 1e-4,
    },
    "class_weighting": {
        "description": "Inverse-frequency weighted cross-entropy from the 300 training labels only.",
        "class_weighting": True,
    },
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_development_data(source):
    metadata = json.loads((source / "run_metadata.json").read_text())
    config = metadata["config"]
    if metadata["mode"] != "train" or metadata["epochs"] != 100:
        raise ValueError("Source must be a completed 100-epoch training run")
    paths = {
        key: training.resolve_path(config[key])
        for key in ["data_file_train", "target_file_train", "pathway_file"]
    }
    for key, path in paths.items():
        if sha256(path) != metadata["input_sha256"][key]:
            raise ValueError(f"Source input changed since the original run: {key}")
    x, y = training.read_pair(paths["data_file_train"], paths["target_file_train"])
    assignments = pd.read_csv(source / "split_assignments.csv")
    assignments = assignments[assignments["split"].isin(["train", "validation"])].copy()
    if len(assignments) != len(x) or assignments.Cell_line.nunique() != len(x):
        raise ValueError("Source development split does not match the development input")
    if assignments.case_submitter_id.duplicated().any():
        raise ValueError("Development split contains duplicate patients")
    if set(assignments.Cell_line) != set(x.index):
        raise ValueError("Source split and development input have different sample IDs")
    train_ids = assignments.loc[assignments["split"] == "train", "Cell_line"].tolist()
    validation_ids = assignments.loc[assignments["split"] == "validation", "Cell_line"].tolist()
    expected_counts = {"train": 300, "validation": 76}
    if {"train": len(train_ids), "validation": len(validation_ids)} != expected_counts:
        raise ValueError("Expected the fixed 300/76 development split")
    for ids in [train_ids, validation_ids]:
        if set(y.loc[ids, "Cancer_type"]) != set(training.CLASSES):
            raise ValueError("Every stage must occur in both development splits")
    return metadata, config, paths, x, y, assignments, train_ids, validation_ids


def class_weights(labels):
    counts = labels.Cancer_type.value_counts()
    if set(counts.index) != set(training.CLASSES):
        raise ValueError("All stage labels are required to calculate class weights")
    weights = [len(labels) / (len(training.CLASSES) * counts[label]) for label in training.CLASSES]
    return torch.tensor(weights, dtype=torch.float32), counts.to_dict()


def standardize(train_x, validation_x):
    scaler = StandardScaler().fit(train_x)
    train_z = pd.DataFrame(scaler.transform(train_x), index=train_x.index, columns=train_x.columns)
    validation_z = pd.DataFrame(
        scaler.transform(validation_x), index=validation_x.index, columns=validation_x.columns
    )
    return train_z, validation_z, scaler


def make_loader(x, y, role, class_map, batch_size, shuffle, omics_types=("RNA",)):
    dataset = training.MultiOmicMulticlassDataset(
        x, y, mode=role, omics_types=list(omics_types), class_name_to_id=class_map
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)


def run_condition(name, specification, config, train_x, train_y, validation_x, validation_y,
                  train_ids, validation_ids, output, epochs, batch_size, device):
    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    condition_config = copy.deepcopy(config)
    condition_config.update({key: value for key, value in specification.items() if key in {"lr"}})
    condition_dir = output / name
    condition_dir.mkdir()
    scaler = None
    if specification.get("standardize"):
        train_x, validation_x, scaler = standardize(train_x, validation_x)
        pd.DataFrame({
            "feature": train_x.columns,
            "training_mean": scaler.mean_,
            "training_standard_deviation": scaler.scale_,
        }).to_csv(condition_dir / "training_scaler.csv", index=False)
    weights = None
    counts = train_y.Cancer_type.value_counts().reindex(training.CLASSES).to_dict()
    if specification.get("class_weighting"):
        weights, counts = class_weights(train_y)
        weights = weights.to(device)
    class_map = {name: index for index, name in enumerate(training.CLASSES)}
    train_loader = make_loader(train_x, train_y, "train", class_map, batch_size, True)
    validation_loader = make_loader(validation_x, validation_y, "val", class_map, batch_size, False)
    model, pathways = training.make_model(train_loader.dataset, condition_config)
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=condition_config["lr"], weight_decay=condition_config["weight_decay"]
    )
    metadata = {
        "name": name,
        "description": specification["description"],
        "config": condition_config,
        "standardize": bool(specification.get("standardize")),
        "class_weighting": bool(specification.get("class_weighting")),
        "training_class_counts": counts,
        "class_weights": weights.cpu().tolist() if weights is not None else None,
        "epochs": epochs,
        "batch_size": batch_size,
        "device": str(device),
        "features": list(train_x.columns),
        "pathway_count": len(pathways),
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_evaluated": False,
    }
    (condition_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    best_score, best_epoch, history = -float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        loss, seen = training.train_epoch(model, train_loader, optimizer, device, class_weights=weights)
        targets, probabilities = training.predict(model, validation_loader, device)
        metrics, _, _, _ = training.summarize_predictions(validation_ids, targets, probabilities)
        row = {"epoch": epoch, "training_loss": loss, "training_samples_seen": seen,
               **{f"validation_{key}": value for key, value in metrics.items()},
               "seconds": time.perf_counter() - epoch_started}
        history.append(row)
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch = metrics["macro_f1"], epoch
            torch.save(model.state_dict(), condition_dir / "best_model.pth")
            training.save_evaluation(condition_dir, "validation", validation_ids, targets, probabilities)
        pd.DataFrame(history).to_csv(condition_dir / "history.csv", index=False)
        print(f"{name} epoch={epoch}/{epochs} loss={loss:.4f} validation_macro_f1={metrics['macro_f1']:.4f}", flush=True)
    model.load_state_dict(torch.load(condition_dir / "best_model.pth", map_location=device, weights_only=True))
    targets, probabilities = training.predict(model, validation_loader, device)
    final_metrics = training.save_evaluation(condition_dir, "selected_validation", validation_ids, targets, probabilities)
    summary = {
        "name": name,
        "best_epoch": best_epoch,
        "best_validation_macro_f1": best_score,
        "selected_validation_metrics": final_metrics,
        "elapsed_seconds": time.perf_counter() - started,
        "test_evaluated": False,
    }
    (condition_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def original_reference(source, metadata):
    history = pd.read_csv(source / "history.csv")
    validation = json.loads((source / "validation_metrics.json").read_text())
    selected = json.loads((source / "run_summary.json").read_text())
    return {
        "name": "original_reference",
        "description": "Existing verified 100-epoch run; no retraining performed.",
        "best_epoch": selected["best_epoch"],
        "best_validation_macro_f1": selected["best_validation_macro_f1"],
        "selected_validation_metrics": validation,
        "elapsed_seconds": selected["elapsed_seconds"],
        "test_evaluated_during_comparison": False,
        "source_run": str(source),
        "training_loss_epoch_100": float(history.iloc[-1]["train_loss"]),
        "configuration": metadata["config"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_run", type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.cpu_threads < 1:
        raise ValueError("epochs, batch-size and cpu-threads must be positive")
    source = args.source_run.resolve()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata, config, paths, x, y, assignments, train_ids, validation_ids = load_development_data(source)
    output = source.parent / f"development_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir()
    assignments.to_csv(output / "development_split_assignments.csv", index=False)
    run_metadata = {
        "source_run": str(source), "source_input_hashes": {key: metadata["input_sha256"][key] for key in paths},
        "training_patients": len(train_ids), "validation_patients": len(validation_ids),
        "test_evaluated": False, "epochs": args.epochs, "batch_size": args.batch_size,
        "cpu_threads": args.cpu_threads, "device": str(device), "conditions": CONDITIONS,
    }
    (output / "comparison_metadata.json").write_text(json.dumps(run_metadata, indent=2) + "\n")
    print(f"Output: {output}; train=300 validation=76 test_evaluated=false", flush=True)
    summaries = [original_reference(source, metadata)]
    for name, specification in CONDITIONS.items():
        summaries.append(run_condition(
            name, specification, config, x.loc[train_ids], y.loc[train_ids],
            x.loc[validation_ids], y.loc[validation_ids], train_ids, validation_ids,
            output, args.epochs, args.batch_size, device,
        ))
        pd.DataFrame(summaries).to_csv(output / "comparison_summary.csv", index=False)
        (output / "comparison_summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"Complete: {output}", flush=True)


if __name__ == "__main__":
    main()
