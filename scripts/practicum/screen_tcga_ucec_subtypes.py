"""Prepare patient-disjoint splits, or run a development-only subtype screen."""
import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

import train_tcga_ucec_stage as shared
from models import MultiOmicMulticlassDataset
from model_transformer_lrp import DeePathNet

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/processed/tcga_ucec_subtypes"
CLASSES = ["CN_HIGH", "CN_LOW", "MSI", "POLE"]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    dest = DATA / "splits"
    if dest.exists():
        raise ValueError("Fixed split directory exists; refusing to overwrite")
    x = pd.read_csv(DATA / "data_file.csv", index_col=0)
    y = pd.read_csv(DATA / "target_file.csv", index_col=0)
    audit = pd.read_csv(DATA / "patient_label_audit.tsv", sep="\t").set_index("Cell_line")
    assert x.index.equals(y.index) and x.index.is_unique
    assert audit.loc[x.index, "case_submitter_id"].is_unique
    old_test = pd.read_csv(ROOT / "data/processed/tcga_ucec_rna/test_label_file.csv", usecols=[0]).iloc[:, 0]
    test_ids = sorted(set(old_test) & set(x.index))
    dev_ids = sorted(set(x.index) - set(test_ids))
    train_ids, val_ids = train_test_split(dev_ids, test_size=0.2, random_state=1,
                                        stratify=y.loc[dev_ids, "Cancer_type"])
    splits = {"train": sorted(train_ids), "validation": sorted(val_ids), "test": test_ids}
    dest.mkdir()
    rows = []
    for name, ids in splits.items():
        assert set(y.loc[ids, "Cancer_type"]) == set(CLASSES)
        x.loc[ids].to_csv(dest / (name + "_data.csv"))
        y.loc[ids].to_csv(dest / (name + "_labels.csv"))
        for sid in ids:
            rows.append({"Cell_line": sid, "case_submitter_id": audit.loc[sid, "case_submitter_id"],
                         "split": name, "subtype": y.loc[sid, "Cancer_type"]})
    assignments = pd.DataFrame(rows)
    assert assignments.case_submitter_id.is_unique
    assignments.to_csv(dest / "assignments.csv", index=False)
    counts = pd.crosstab(assignments.split, assignments.subtype)
    counts.to_csv(dest / "class_counts.csv")
    manifest = {"split_seed": 1, "counts": counts.to_dict("index"),
                "policy": "Preserve all 91 previously held-out staging patients with subtype labels; stratify remaining patients 80/20.",
                "test_limitation": "Internal historical holdout; initial staging results were previously inspected. Not external validation.",
                "files": {p.name: digest(p) for p in dest.glob("*.csv")}}
    (dest / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(counts.to_string(), flush=True)


def load_development():
    source = DATA / "splits"
    manifest = json.loads((source / "manifest.json").read_text())
    frames = []
    for role in ("train", "validation"):
        pair = []
        for kind in ("data", "labels"):
            path = source / (role + "_" + kind + ".csv")
            if digest(path) != manifest["files"][path.name]:
                raise ValueError("Fixed development file changed: " + str(path))
            pair.append(pd.read_csv(path, index_col=0))
        assert pair[0].index.equals(pair[1].index)
        frames.extend(pair)
    tx, ty, vx, vy = frames
    assert not set(tx.index) & set(vx.index)
    assert tx.columns.equals(vx.columns)
    assert np.isfinite(tx.values).all() and np.isfinite(vx.values).all()
    return tx, ty, vx, vy


def evaluate(name, ids, truth, prob, output):
    pred = prob.argmax(axis=1)
    assert prob.shape == (len(truth), 4) and np.isfinite(prob).all()
    assert np.allclose(prob.sum(axis=1), 1, atol=1e-5)
    metrics = {"model": name, "macro_f1": f1_score(truth, pred, labels=range(4), average="macro", zero_division=0),
               "accuracy": accuracy_score(truth, pred), "balanced_accuracy": balanced_accuracy_score(truth, pred)}
    pd.DataFrame(classification_report(truth, pred, labels=range(4), target_names=CLASSES,
                                      output_dict=True, zero_division=0)).T.to_csv(output / (name + "_report.csv"))
    pd.DataFrame(confusion_matrix(truth, pred, labels=range(4)), index=CLASSES, columns=CLASSES).to_csv(output / (name + "_confusion.csv"))
    table = pd.DataFrame(prob, index=ids, columns=["prob_" + c for c in CLASSES])
    table["truth"] = np.array(CLASSES)[truth]
    table["prediction"] = np.array(CLASSES)[pred]
    table.to_csv(output / (name + "_validation_predictions.csv"))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    if args.prepare:
        prepare()
        return
    if args.epochs < 1 or args.patience < 1:
        raise ValueError("Positive epochs/patience required")
    torch.set_num_threads(4)
    tx, ty, vx, vy = load_development()
    output = ROOT / "work_dirs/practicum/DeePathNet/tcga_ucec_subtypes" / datetime.now().strftime("screen_%Y%m%d_%H%M%S")
    output.mkdir(parents=True)
    print("OUTPUT " + str(output), flush=True)
    mapping = {c: i for i, c in enumerate(CLASSES)}
    train_y = ty.Cancer_type.map(mapping).to_numpy(dtype=np.int64)
    val_y = vy.Cancer_type.map(mapping).to_numpy(dtype=np.int64)
    scaler = StandardScaler().fit(tx)
    assert scaler.n_samples_seen_ == len(tx)
    pd.DataFrame({"mean": scaler.mean_, "scale": scaler.scale_}, index=tx.columns).to_csv(output / "training_scaler.csv")
    train_x = pd.DataFrame(scaler.transform(tx).astype("float32"), index=tx.index, columns=tx.columns)
    val_x = pd.DataFrame(scaler.transform(vx).astype("float32"), index=vx.index, columns=vx.columns)
    metadata = {"train": len(tx), "validation": len(vx), "features": tx.shape[1], "classes": CLASSES,
                "seed": 1, "scaler_fit_samples": int(scaler.n_samples_seen_), "test_read": False,
                "selection": "validation macro-F1", "epochs_max": args.epochs, "patience": args.patience,
                "nn_settings": {"optimizer": "Adam", "lr": 1e-4, "weight_decay": 1e-5, "batch_size": 64},
                "deepathnet": {"dim": 256, "heads": 8, "depth": 2, "pathway_dropout": 0.5, "transformer_dropout": 0},
                "mlp": [tx.shape[1], 256, 4], "logistic": {"C": 1.0, "max_iter": 2000},
                "caveat": "Single-seed screen, not equal-budget tuning or final generalization evidence."}
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    results = []
    majority = np.zeros((len(vx), 4))
    majority[:, np.bincount(train_y).argmax()] = 1
    results.append(evaluate("majority", vx.index, val_y, majority, output))
    lr = LogisticRegression(C=1.0, max_iter=2000, random_state=1).fit(train_x, train_y)
    results.append(evaluate("logistic", vx.index, val_y, lr.predict_proba(val_x), output))
    pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
    import joblib
    joblib.dump({"model": lr, "scaler": scaler, "features": list(tx.columns), "classes": CLASSES}, output / "logistic.joblib")
    for name in ("mlp", "deepathnet"):
        torch.manual_seed(1)
        np.random.seed(1)
        if name == "mlp":
            model = torch.nn.Sequential(torch.nn.Linear(tx.shape[1], 256), torch.nn.ReLU(), torch.nn.Linear(256, 4))
            train_set = TensorDataset(torch.from_numpy(train_x.values), torch.from_numpy(train_y))
            val_set = TensorDataset(torch.from_numpy(val_x.values), torch.from_numpy(val_y))
        else:
            train_set = MultiOmicMulticlassDataset(train_x, ty, "train", ["RNA"], mapping)
            val_set = MultiOmicMulticlassDataset(val_x, vy, "val", ["RNA"], mapping)
            pathways, unused = shared.load_pathways(ROOT / "data/graph_predefined/LCPathways/41568_2020_240_MOESM4_ESM.csv", train_set.genes_to_id)
            assert not unused
            model = DeePathNet(1, 4, train_set.genes_to_id, train_set.id_to_genes, pathways, unused,
                               embed_dim=256, depth=2, num_heads=8, mlp_ratio=2, out_mlp_ratio=8,
                               drop_rate=0, attn_drop_rate=0, pathway_drop_rate=0.5, only_cancer_genes=True)
            (output / "pathways.json").write_text(json.dumps(pathways, indent=2))
        train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
        history, best, best_epoch = [], -1, 0
        for epoch in range(1, args.epochs + 1):
            loss, seen = shared.train_epoch(model, train_loader, optimizer, torch.device("cpu"))
            truth, prob = shared.predict(model, val_loader, torch.device("cpu"))
            score = f1_score(truth, prob.argmax(axis=1), labels=range(4), average="macro", zero_division=0)
            history.append({"epoch": epoch, "train_loss": loss, "validation_macro_f1": score})
            if score > best:
                best, best_epoch = score, epoch
                torch.save(model.state_dict(), output / (name + "_best.pth"))
            pd.DataFrame(history).to_csv(output / (name + "_history.csv"), index=False)
            print(f"{name} epoch={epoch} loss={loss:.4f} val_macro_f1={score:.4f} best={best:.4f}", flush=True)
            if epoch - best_epoch >= args.patience:
                break
        model.load_state_dict(torch.load(output / (name + "_best.pth"), map_location="cpu", weights_only=True))
        truth, prob = shared.predict(model, val_loader, torch.device("cpu"))
        result = evaluate(name, vx.index, truth, prob, output)
        result.update({"selected_epoch": best_epoch, "epochs_run": epoch, "parameters": sum(p.numel() for p in model.parameters())})
        results.append(result)
        pd.DataFrame(results).to_csv(output / "summary.csv", index=False)
    print(pd.DataFrame(results).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
