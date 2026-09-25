"""Train main-stage classification with a patient-separated validation set.

Run with --smoke-test for two small training batches and validation only.
Normal runs select weights on validation Macro-F1, then evaluate test once.
"""

import argparse
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from models import MultiOmicMulticlassDataset  # noqa: E402
from model_transformer_lrp import DeePathNet  # noqa: E402

CLASSES = ["Stage I", "Stage II", "Stage III", "Stage IV"]


def resolve_path(value):
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def read_pair(data_path, label_path):
    x = pd.read_csv(data_path, index_col=0)
    y = pd.read_csv(label_path, index_col=0)
    if not x.index.is_unique or not y.index.is_unique or not x.columns.is_unique:
        raise ValueError("Duplicate sample IDs or feature names")
    if set(x.index) != set(y.index) or list(y.columns) != ["Cancer_type"]:
        raise ValueError("Expression and Cancer_type labels must match exactly")
    if x.index.name != "Cell_line" or y.index.name != "Cell_line":
        raise ValueError("Expected Cell_line sample IDs")
    if not set(y.Cancer_type).issubset(CLASSES):
        raise ValueError("Expected Stage I, II, III, IV labels only")
    if not np.isfinite(x.to_numpy(dtype=float)).all():
        raise ValueError("Expression contains missing or non-finite values")
    if any(not c.endswith("_RNA") or "_" in c[:-4] for c in x.columns):
        raise ValueError("Expected GENE_RNA features compatible with the model dataset")
    return x, y.loc[x.index]


def make_splits(development_labels, test_labels, sample_sheet, fraction, seed):
    if not sample_sheet.sample_submitter_id.is_unique:
        raise ValueError("Sample sheet must have unique sample IDs")
    cases = sample_sheet.set_index("sample_submitter_id").case_submitter_id
    all_ids = list(development_labels.index) + list(test_labels.index)
    if len(set(all_ids)) != len(all_ids):
        raise ValueError("Development and test samples overlap")
    if not set(all_ids).issubset(cases.index):
        raise ValueError("Missing patient metadata")
    if cases.loc[all_ids].isna().any() or cases.loc[all_ids].duplicated().any():
        raise ValueError("This workflow requires exactly one selected sample per patient")
    train_ids, val_ids = train_test_split(
        development_labels.index, test_size=fraction, random_state=seed,
        stratify=development_labels.Cancer_type,
    )
    splits = {"train": sorted(train_ids), "validation": sorted(val_ids), "test": list(test_labels.index)}
    rows = []
    for role, ids in splits.items():
        labels = test_labels if role == "test" else development_labels
        if set(labels.loc[ids, "Cancer_type"]) != set(CLASSES):
            raise ValueError(f"All four stages must be present in {role}")
        for sample in ids:
            rows.append({"Cell_line": sample, "case_submitter_id": cases[sample],
                         "split": role, "Cancer_type": labels.loc[sample, "Cancer_type"]})
    return splits, pd.DataFrame(rows)


def load_pathways(path, genes):
    pathways = {}
    frame = pd.read_csv(path)
    for row in frame.itertuples(index=False):
        members = [gene for gene in row.genes.split("|") if gene in genes]
        if members:
            pathways[row.name] = members
    if not pathways:
        raise ValueError("No pathways overlap the input genes")
    covered = {gene for members in pathways.values() for gene in members}
    return pathways, set(genes) - covered


def make_model(dataset, config):
    pathways, non_pathway = load_pathways(resolve_path(config["pathway_file"]), dataset.genes_to_id)
    if not non_pathway and not config["cancer_only"]:
        raise ValueError("Set cancer_only=true when there are no non-pathway genes")
    model = DeePathNet(
        len(dataset.omics_types), len(CLASSES), dataset.genes_to_id, dataset.id_to_genes, pathways, non_pathway,
        embed_dim=config["dim"], depth=config["depth"], num_heads=config["heads"],
        mlp_ratio=config["mlp_ratio"], out_mlp_ratio=config["out_mlp_ratio"],
        drop_rate=config.get("dropout", 0.0),
        attn_drop_rate=config.get("emb_dropout", 0.0),
        pathway_drop_rate=config["pathway_dropout"], only_cancer_genes=config["cancer_only"],
    )
    return model, pathways


def train_epoch(model, loader, optimizer, device, max_batches=None, class_weights=None):
    model.train()
    total_loss, seen = 0.0, 0
    for batch_index, (x, y) in enumerate(loader):
        if max_batches is not None and batch_index >= max_batches:
            break
        optimizer.zero_grad(set_to_none=True)
        logits = model(x.float().to(device))
        loss = torch.nn.functional.cross_entropy(
            logits,
            y.flatten().long().to(device),
            weight=class_weights,
        )
        if not torch.isfinite(loss):
            raise ValueError("Non-finite training loss")
        loss.backward()
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Non-finite model gradients")
        optimizer.step()
        seen += len(x)
        total_loss += loss.item() * len(x)
    return total_loss / seen, seen


def predict(model, loader, device):
    model.eval()
    probabilities, targets = [], []
    with torch.no_grad():
        for x, y in loader:
            logits = model(x.float().to(device))
            probabilities.append(torch.softmax(logits, dim=-1).cpu().numpy())
            targets.append(y.numpy().reshape(-1))
    return np.concatenate(targets), np.concatenate(probabilities)


def summarize_predictions(sample_ids, targets, probabilities):
    if probabilities.shape != (len(sample_ids), len(CLASSES)) or len(targets) != len(sample_ids):
        raise ValueError("Prediction shape does not match sample IDs and classes")
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid class probabilities")
    predicted = probabilities.argmax(axis=1)
    labels = list(range(len(CLASSES)))
    metrics = {
        "accuracy": float(accuracy_score(targets, predicted)),
        "macro_f1": float(f1_score(targets, predicted, labels=labels, average="macro", zero_division=0)),
        "roc_auc_ovo": float(roc_auc_score(targets, probabilities, labels=labels, multi_class="ovo"))
        if set(targets) == set(labels) else None,
    }
    frame = pd.DataFrame({"Cell_line": list(sample_ids), "y_true": [CLASSES[i] for i in targets],
                          "y_pred": [CLASSES[i] for i in predicted]})
    for i, name in enumerate(CLASSES):
        frame[f"prob_{name.replace(' ', '_')}"] = probabilities[:, i]
    report = pd.DataFrame(classification_report(
        targets, predicted, labels=labels, target_names=CLASSES, output_dict=True, zero_division=0,
    )).T
    matrix = pd.DataFrame(confusion_matrix(targets, predicted, labels=labels), index=CLASSES, columns=CLASSES)
    matrix.index.name = "true_stage"
    return metrics, frame, report, matrix


def save_evaluation(output, name, sample_ids, targets, probabilities):
    metrics, frame, report, matrix = summarize_predictions(sample_ids, targets, probabilities)
    frame.to_csv(output / f"{name}_predictions.csv", index=False)
    report.to_csv(output / f"{name}_classification_report.csv")
    matrix.to_csv(output / f"{name}_confusion_matrix.csv")
    (output / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads(resolve_path(args.config).read_text())
    if config["task"] != "multiclass" or config["data_type"] != ["RNA"]:
        raise ValueError("This entrypoint expects RNA main-stage classification")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cpu_threads < 1:
        raise ValueError("cpu-threads must be positive")
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = args.batch_size or (4 if args.smoke_test else config["batch_size"])
    epochs = 1 if args.smoke_test else (args.epochs or config["num_of_epochs"])
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch-size must be positive")
    mode = "smoke" if args.smoke_test else "train"
    output = resolve_path(config["work_dir"]) / f"{mode}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, force=True, format="%(asctime)s %(message)s", handlers=[
        logging.FileHandler(output / "training.log"), logging.StreamHandler(),
    ])
    logger = logging.getLogger(__name__)
    input_paths = {key: resolve_path(config[key]) for key in (
        "data_file_train", "target_file_train", "data_file_test", "target_file_test", "pathway_file",
    )}
    x, y = read_pair(input_paths["data_file_train"], input_paths["target_file_train"])
    test_x, test_y = read_pair(input_paths["data_file_test"], input_paths["target_file_test"])
    if set(x.columns) != set(test_x.columns):
        raise ValueError("Train and test features differ")
    test_x = test_x.loc[:, x.columns]
    sheet_path = input_paths["data_file_train"].parent / "selected_sample_sheet.tsv"
    splits, assignments = make_splits(y, test_y, pd.read_csv(sheet_path, sep="\t"),
                                     config["validation_size"], config["validation_seed"])
    assignments.to_csv(output / "split_assignments.csv", index=False)
    class_map = {name: i for i, name in enumerate(CLASSES)}

    def loader(frame, labels, role, shuffle=False):
        dataset = MultiOmicMulticlassDataset(frame, labels, mode=role, omics_types=["RNA"], class_name_to_id=class_map)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0, drop_last=False)

    train_loader = loader(x.loc[splits["train"]], y.loc[splits["train"]], "train", True)
    val_loader = loader(x.loc[splits["validation"]], y.loc[splits["validation"]], "val")
    model, pathways = make_model(train_loader.dataset, config)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    metadata = {
        "mode": mode, "config": config, "epochs": epochs, "batch_size": batch_size,
        "cpu_threads": args.cpu_threads, "device": str(device), "python": sys.version,
        "torch": torch.__version__, "selection_metric": "validation_macro_f1",
        "max_training_batches": 2 if args.smoke_test else None,
        "class_name_to_id": class_map, "features": list(x.columns), "pathways": pathways,
        "input_sha256": {key: hashlib.sha256(path.read_bytes()).hexdigest() for key, path in input_paths.items()},
        "split_counts": assignments.groupby(["split", "Cancer_type"]).size().unstack(fill_value=0).to_dict("index"),
        "parameters": sum(p.numel() for p in model.parameters()),
    }
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    logger.info("Output: %s; %s; train=%d validation=%d test=%d; pathways=%d parameters=%d",
                output, device, len(splits["train"]), len(splits["validation"]), len(splits["test"]),
                len(pathways), metadata["parameters"])
    best_score, best_epoch, history = -float("inf"), None, []
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        loss, seen = train_epoch(model, train_loader, optimizer, device, 2 if args.smoke_test else None)
        targets, probabilities = predict(model, val_loader, device)
        metrics, _, _, _ = summarize_predictions(splits["validation"], targets, probabilities)
        history.append({"epoch": epoch, "train_loss": loss, "training_samples_seen": seen,
                        **{f"validation_{k}": v for k, v in metrics.items()},
                        "seconds": time.perf_counter() - epoch_started})
        if metrics["macro_f1"] > best_score:
            best_score, best_epoch = metrics["macro_f1"], epoch
            torch.save(model.state_dict(), output / "best_model.pth")
            save_evaluation(output, "validation", splits["validation"], targets, probabilities)
        pd.DataFrame(history).to_csv(output / "history.csv", index=False)
        logger.info("Epoch %d/%d loss=%.4f validation_macro_f1=%.4f seconds=%.1f",
                    epoch, epochs, loss, metrics["macro_f1"], history[-1]["seconds"])
    model.load_state_dict(torch.load(output / "best_model.pth", map_location=device, weights_only=True))
    # The held-out test loader is used only after validation selects the weights.
    test_metrics = None
    if not args.smoke_test:
        test_loader = loader(test_x, test_y, "test")
        targets, probabilities = predict(model, test_loader, device)
        test_metrics = save_evaluation(output, "test", test_x.index, targets, probabilities)
    summary = {"mode": mode, "best_epoch": best_epoch, "best_validation_macro_f1": best_score,
               "test_evaluated": not args.smoke_test, "test_metrics": test_metrics,
               "elapsed_seconds": time.perf_counter() - started, "output_dir": str(output)}
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    logger.info("Completed: %s", json.dumps(summary))


if __name__ == "__main__":
    main()
