"""Paired nested development CV for TCGA-UCEC RNA subtype prediction.

The 91-patient holdout is never opened. Each outer training fold contains an
inner split for neural-network epoch selection. The selected model is then
reinitialized and fit on the full outer training fold for that fixed number of
epochs before the outer validation fold is evaluated.
"""
import argparse
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import screen_tcga_ucec_subtypes as screen
import train_tcga_ucec_stage as shared
from model_transformer_lrp import DeePathNet
from models import MultiOmicMulticlassDataset

ROOT = screen.ROOT
CLASSES = screen.CLASSES
MAPPING = {name: index for index, name in enumerate(CLASSES)}
PATHWAY_FILE = ROOT / "data/graph_predefined/LCPathways/41568_2020_240_MOESM4_ESM.csv"


def combine_development():
    train_x, train_y, validation_x, validation_y = screen.load_development()
    x = pd.concat([train_x, validation_x])
    y = pd.concat([train_y, validation_y]).loc[x.index]
    if not x.index.is_unique or not x.columns.is_unique or not x.index.equals(y.index):
        raise ValueError("Development identifiers must be unique and aligned")
    return x, y


def scale(train_x, validation_x):
    scaler = StandardScaler().fit(train_x)
    if scaler.n_samples_seen_ != len(train_x):
        raise ValueError("Scaler was not fit on the expected training patients")
    scaled_train = pd.DataFrame(
        scaler.transform(train_x).astype("float32"), index=train_x.index, columns=train_x.columns
    )
    scaled_validation = pd.DataFrame(
        scaler.transform(validation_x).astype("float32"),
        index=validation_x.index,
        columns=validation_x.columns,
    )
    return scaled_train, scaled_validation, scaler


def make_model_and_sets(name, train_x, train_labels, validation_x, validation_labels, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    train_y = train_labels.Cancer_type.map(MAPPING).to_numpy(dtype=np.int64)
    validation_y = validation_labels.Cancer_type.map(MAPPING).to_numpy(dtype=np.int64)
    if name == "mlp":
        model = torch.nn.Sequential(
            torch.nn.Linear(train_x.shape[1], 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, len(CLASSES)),
        )
        train_set = TensorDataset(torch.from_numpy(train_x.values), torch.from_numpy(train_y))
        validation_set = TensorDataset(
            torch.from_numpy(validation_x.values), torch.from_numpy(validation_y)
        )
    elif name == "deepathnet":
        train_set = MultiOmicMulticlassDataset(
            train_x, train_labels, "train", ["RNA"], MAPPING
        )
        validation_set = MultiOmicMulticlassDataset(
            validation_x, validation_labels, "val", ["RNA"], MAPPING
        )
        pathways, unused = shared.load_pathways(PATHWAY_FILE, train_set.genes_to_id)
        if unused:
            raise ValueError("Expected every retained RNA gene to occur in LCPathways")
        model = DeePathNet(
            1, len(CLASSES), train_set.genes_to_id, train_set.id_to_genes,
            pathways, unused, embed_dim=256, depth=2, num_heads=8,
            mlp_ratio=2, out_mlp_ratio=8, drop_rate=0,
            attn_drop_rate=0, pathway_drop_rate=0.5,
            only_cancer_genes=True,
        )
    else:
        raise ValueError("Unknown neural model: " + name)
    return model, train_set, validation_set


def choose_epoch(name, train_x, train_y, inner_x, inner_y, seed, max_epochs, patience):
    model, train_set, inner_set = make_model_and_sets(
        name, train_x, train_y, inner_x, inner_y, seed
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    inner_loader = DataLoader(inner_set, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    best_score, best_epoch, history = -1.0, 0, []
    for epoch in range(1, max_epochs + 1):
        loss, _ = shared.train_epoch(model, train_loader, optimizer, torch.device("cpu"))
        truth, probabilities = shared.predict(model, inner_loader, torch.device("cpu"))
        score = f1_score(
            truth, probabilities.argmax(axis=1), labels=range(len(CLASSES)),
            average="macro", zero_division=0,
        )
        history.append({"epoch": epoch, "train_loss": loss, "inner_macro_f1": score})
        if score > best_score:
            best_score, best_epoch = score, epoch
        if epoch - best_epoch >= patience:
            break
    return best_epoch, best_score, pd.DataFrame(history)


def refit_and_predict(name, train_x, train_y, validation_x, validation_y, seed, epochs):
    model, train_set, validation_set = make_model_and_sets(
        name, train_x, train_y, validation_x, validation_y, seed
    )
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    validation_loader = DataLoader(validation_set, batch_size=64, shuffle=False)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
    losses = []
    for epoch in range(1, epochs + 1):
        loss, _ = shared.train_epoch(model, train_loader, optimizer, torch.device("cpu"))
        losses.append({"epoch": epoch, "train_loss": loss})
    truth, probabilities = shared.predict(model, validation_loader, torch.device("cpu"))
    return truth, probabilities, pd.DataFrame(losses), sum(p.numel() for p in model.parameters())


def score(model, fold, ids, truth, probabilities, selected_epoch=None, parameters=None):
    prediction = probabilities.argmax(axis=1)
    row = {
        "model": model,
        "fold": fold,
        "macro_f1": f1_score(
            truth, prediction, labels=range(len(CLASSES)), average="macro", zero_division=0
        ),
        "accuracy": accuracy_score(truth, prediction),
        "balanced_accuracy": balanced_accuracy_score(truth, prediction),
        "selected_epoch": selected_epoch,
        "parameters": parameters,
    }
    predictions = pd.DataFrame(
        probabilities, index=ids, columns=["prob_" + name for name in CLASSES]
    )
    predictions.index.name = "Cell_line"
    predictions.insert(0, "fold", fold)
    predictions.insert(1, "model", model)
    predictions["truth"] = np.array(CLASSES)[truth]
    predictions["prediction"] = np.array(CLASSES)[prediction]
    return row, predictions


def save_pooled_reports(results, predictions, output):
    summary = pd.DataFrame(results)
    summary.to_csv(output / "fold_metrics.csv", index=False)
    aggregate = summary.groupby("model")[["macro_f1", "accuracy", "balanced_accuracy"]].agg(
        ["mean", "std"]
    )
    aggregate.columns = ["_".join(column) for column in aggregate.columns]
    aggregate.to_csv(output / "cv_summary.csv")
    all_predictions = pd.concat(predictions)
    all_predictions.to_csv(output / "out_of_fold_predictions.csv")
    for model, frame in all_predictions.groupby("model"):
        truth = frame.truth.map(MAPPING).to_numpy()
        predicted = frame.prediction.map(MAPPING).to_numpy()
        pd.DataFrame(
            classification_report(
                truth, predicted, labels=range(len(CLASSES)), target_names=CLASSES,
                output_dict=True, zero_division=0,
            )
        ).T.to_csv(output / f"{model}_pooled_report.csv")
        pd.DataFrame(
            confusion_matrix(truth, predicted, labels=range(len(CLASSES))),
            index=CLASSES, columns=CLASSES,
        ).to_csv(output / f"{model}_pooled_confusion.csv")
    pivot = summary.pivot(index="fold", columns="model", values="macro_f1")
    differences = pd.DataFrame({
        "deepathnet_minus_logistic": pivot.deepathnet - pivot.logistic,
        "deepathnet_minus_mlp": pivot.deepathnet - pivot.mlp,
    })
    differences.to_csv(output / "paired_macro_f1_differences.csv")
    return summary, aggregate, differences, all_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cpu-threads", type=int, default=4)
    args = parser.parse_args()
    if args.folds < 2 or args.epochs < 1 or args.patience < 1 or args.cpu_threads < 1:
        raise ValueError("Positive settings and at least two folds are required")
    torch.set_num_threads(args.cpu_threads)
    x, y = combine_development()
    encoded = y.Cancer_type.map(MAPPING).to_numpy(dtype=np.int64)
    output = ROOT / "work_dirs/practicum/DeePathNet/tcga_ucec_subtypes" / datetime.now().strftime(
        "nested_cv_%Y%m%d_%H%M%S"
    )
    output.mkdir(parents=True)
    print("OUTPUT " + str(output), flush=True)
    metadata = {
        "patients": len(x), "features": x.shape[1], "outer_folds": args.folds,
        "outer_seed": args.seed, "inner_validation_fraction": 0.2,
        "max_epochs": args.epochs, "patience": args.patience,
        "models": ["majority", "logistic", "mlp", "deepathnet"],
        "deepathnet": {"dim": 256, "heads": 8, "depth": 2, "pathway_dropout": 0.5},
        "preprocessing": "Z-score fitted independently on inner training for epoch selection and full outer training for refit.",
        "test_read": False,
        "test_policy": "The 91-patient holdout is not opened or evaluated.",
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    outer = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    results, predictions, assignments = [], [], []
    for fold, (outer_train_pos, outer_validation_pos) in enumerate(outer.split(x, encoded), 1):
        outer_train_ids = x.index[outer_train_pos]
        outer_validation_ids = x.index[outer_validation_pos]
        outer_train_y = y.loc[outer_train_ids]
        outer_validation_y = y.loc[outer_validation_ids]
        inner_train_ids, inner_validation_ids = train_test_split(
            outer_train_ids, test_size=0.2, random_state=args.seed + fold,
            stratify=outer_train_y.Cancer_type,
        )
        for sid in outer_train_ids:
            assignments.append({
                "fold": fold, "Cell_line": sid, "outer_role": "train",
                "inner_role": "validation" if sid in set(inner_validation_ids) else "train",
                "subtype": y.loc[sid, "Cancer_type"],
            })
        for sid in outer_validation_ids:
            assignments.append({
                "fold": fold, "Cell_line": sid, "outer_role": "validation",
                "inner_role": "not_used", "subtype": y.loc[sid, "Cancer_type"],
            })
        fold_dir = output / f"fold_{fold}"
        fold_dir.mkdir()
        inner_train_x, inner_validation_x, inner_scaler = scale(
            x.loc[inner_train_ids], x.loc[inner_validation_ids]
        )
        pd.DataFrame(
            {"mean": inner_scaler.mean_, "scale": inner_scaler.scale_}, index=x.columns
        ).to_csv(fold_dir / "inner_training_scaler.csv")
        outer_train_x, outer_validation_x, outer_scaler = scale(
            x.loc[outer_train_ids], x.loc[outer_validation_ids]
        )
        pd.DataFrame(
            {"mean": outer_scaler.mean_, "scale": outer_scaler.scale_}, index=x.columns
        ).to_csv(fold_dir / "outer_training_scaler.csv")
        outer_truth = outer_validation_y.Cancer_type.map(MAPPING).to_numpy(dtype=np.int64)
        majority_probabilities = np.zeros((len(outer_validation_ids), len(CLASSES)))
        majority_class = np.bincount(
            outer_train_y.Cancer_type.map(MAPPING).to_numpy(dtype=np.int64)
        ).argmax()
        majority_probabilities[:, majority_class] = 1
        row, frame = score(
            "majority", fold, outer_validation_ids, outer_truth, majority_probabilities
        )
        results.append(row)
        predictions.append(frame)
        logistic = LogisticRegression(C=1.0, max_iter=2000, random_state=args.seed).fit(
            outer_train_x, outer_train_y.Cancer_type.map(MAPPING)
        )
        row, frame = score(
            "logistic", fold, outer_validation_ids, outer_truth,
            logistic.predict_proba(outer_validation_x),
        )
        results.append(row)
        predictions.append(frame)
        for name in ("mlp", "deepathnet"):
            model_seed = args.seed * 1000 + fold
            selected_epoch, inner_score, history = choose_epoch(
                name, inner_train_x, y.loc[inner_train_ids], inner_validation_x,
                y.loc[inner_validation_ids], model_seed, args.epochs, args.patience,
            )
            history.to_csv(fold_dir / f"{name}_inner_history.csv", index=False)
            truth, probabilities, refit_history, parameters = refit_and_predict(
                name, outer_train_x, outer_train_y, outer_validation_x,
                outer_validation_y, model_seed, selected_epoch,
            )
            refit_history.to_csv(fold_dir / f"{name}_refit_history.csv", index=False)
            row, frame = score(
                name, fold, outer_validation_ids, truth, probabilities,
                selected_epoch=selected_epoch, parameters=parameters,
            )
            row["inner_best_macro_f1"] = inner_score
            results.append(row)
            predictions.append(frame)
            print(
                f"fold={fold} model={name} selected_epoch={selected_epoch} "
                f"outer_macro_f1={row['macro_f1']:.4f}", flush=True
            )
        pd.DataFrame(results).to_csv(output / "fold_metrics.partial.csv", index=False)
    assignments = pd.DataFrame(assignments)
    assignments.to_csv(output / "fold_assignments.csv", index=False)
    summary, aggregate, differences, all_predictions = save_pooled_reports(
        results, predictions, output
    )
    expected = set(x.index)
    for model, frame in all_predictions.groupby("model"):
        if set(frame.index) != expected or not frame.index.is_unique:
            raise ValueError(f"{model} does not have exactly one out-of-fold prediction per patient")
    print("\nCV SUMMARY\n" + aggregate.to_string(), flush=True)
    print("\nPAIRED MACRO-F1 DIFFERENCES\n" + differences.to_string(), flush=True)


if __name__ == "__main__":
    main()
