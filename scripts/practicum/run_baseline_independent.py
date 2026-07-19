"""
Run practicum independent-test baselines without modifying DeePathNet's
original baseline scripts.

Example:
    python scripts/practicum/run_baseline_independent.py \
        configs/practicum/risk_rna/baseline_independent_rf.json
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


STAMP = datetime.today().strftime("%Y%m%d%H%M")
OUTPUT_NA_NUM = -100


def resolve_path(path_value, base_dir):
    path = Path(path_value)
    if path.is_absolute():
        return path
    return base_dir / path


def select_requested_features(data_input_train, data_input_test, data_type):
    common_features = [
        column for column in data_input_train.columns if column in data_input_test.columns
    ]
    data_input_train = data_input_train.loc[:, common_features]
    data_input_test = data_input_test.loc[:, common_features]

    if data_type and data_type[0] != "DR":
        selected_features = []
        for column in data_input_train.columns:
            parts = column.split("_")
            first_token = parts[0]
            second_token = parts[1] if len(parts) > 1 else ""
            if second_token in data_type or first_token in data_type:
                selected_features.append(column)
        data_input_train = data_input_train.loc[:, selected_features]
        data_input_test = data_input_test.loc[:, selected_features]

    return data_input_train, data_input_test


def filter_to_pathway_genes(data_input_train, data_input_test, pathway_file):
    genes = np.unique([column.split("_")[0] for column in data_input_train.columns])
    pathway_df = pd.read_csv(pathway_file)
    pathway_df["genes"] = pathway_df["genes"].map(
        lambda value: "|".join([gene for gene in value.split("|") if gene in genes])
    )
    cancer_genes = set(
        gene for genes_string in pathway_df["genes"].values for gene in genes_string.split("|") if gene
    )
    selected_features = [
        column
        for column in data_input_train.columns
        if column.split("_")[0] in cancer_genes or column.split("_")[0] == "tissue"
    ]
    return data_input_train.loc[:, selected_features], data_input_test.loc[:, selected_features]


def get_model(model_name, seed):
    if model_name == "majority":
        return None
    if model_name == "lr":
        return LogisticRegression(max_iter=5000, random_state=seed)
    if model_name == "rf":
        return RandomForestClassifier(n_jobs=-1, random_state=seed)
    if model_name == "svm":
        return SVC(probability=True, random_state=seed)
    if model_name == "svm-linear":
        return SVC(kernel="linear", probability=True, random_state=seed)
    if model_name == "mlp":
        return MLPClassifier(max_iter=1000, random_state=seed)
    raise ValueError(f"Unsupported baseline model: {model_name}")


def select_model_score(classes, probabilities, positive_label):
    classes = list(classes)
    probabilities = np.asarray(probabilities)
    if positive_label not in classes:
        raise ValueError(f"Positive label {positive_label!r} is not in model classes: {classes}")
    positive_index = classes.index(positive_label)
    return probabilities[:, positive_index].tolist()


def compute_classification_metrics(y_true, y_pred, y_score, positive_label):
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    y_true_positive = (y_true == positive_label).astype(int)

    metrics = {
        "acc": accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0),
        "precision_high": precision_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
        "recall_high": recall_score(
            y_true, y_pred, pos_label=positive_label, zero_division=0
        ),
    }
    try:
        metrics["auc"] = roc_auc_score(y_true_positive, y_score)
    except ValueError:
        metrics["auc"] = np.nan

    labels = ["Average", positive_label] if positive_label != "Average" else [positive_label]
    if len(labels) == 2:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=labels).ravel()
    else:
        tn, fp, fn, tp = np.nan, np.nan, np.nan, np.nan
    metrics.update({"tn": tn, "fp": fp, "fn": fn, "tp": tp})
    return metrics


def align_target_and_input(data_target, data_input):
    common_index = data_target.index.intersection(data_input.index)
    return data_target.loc[common_index], data_input.loc[common_index]


def run_baseline(config_file):
    repo_root = Path(__file__).resolve().parents[2]
    config_path = resolve_path(config_file, repo_root)
    configs = json.loads(config_path.read_text())
    seed = configs.get("seed", 12345)
    np.random.seed(seed)

    work_dir = resolve_path(configs["work_dir"], repo_root)
    work_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("practicum_baseline_independent")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    log_file = work_dir / f"{STAMP}_{config_path.stem}.log"
    handler = logging.FileHandler(log_file)
    handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.info(config_path.read_text())

    data_target_train = pd.read_csv(
        resolve_path(configs["target_file_train"], repo_root), index_col=0
    )
    data_input_train = pd.read_csv(
        resolve_path(configs["data_file_train"], repo_root), index_col=0
    )
    data_target_test = pd.read_csv(
        resolve_path(configs["target_file_test"], repo_root), index_col=0
    )
    data_input_test = pd.read_csv(
        resolve_path(configs["data_file_test"], repo_root), index_col=0
    )

    if "pathway_file" in configs:
        data_input_train, data_input_test = filter_to_pathway_genes(
            data_input_train,
            data_input_test,
            resolve_path(configs["pathway_file"], repo_root),
        )

    data_input_train, data_input_test = select_requested_features(
        data_input_train, data_input_test, configs["data_type"]
    )
    data_input_train = data_input_train.fillna(0)
    data_input_test = data_input_test.fillna(0)
    data_target_train = data_target_train.fillna(OUTPUT_NA_NUM)
    data_target_test = data_target_test.fillna(OUTPUT_NA_NUM)

    data_target_train, data_input_train = align_target_and_input(
        data_target_train, data_input_train
    )
    data_target_test, data_input_test = align_target_and_input(
        data_target_test, data_input_test
    )

    logger.info(f"Input training data shape: {data_input_train.shape}")
    logger.info(f"Input training target shape: {data_target_train.shape}")
    logger.info(f"Input test data shape: {data_input_test.shape}")
    logger.info(f"Input test target shape: {data_target_test.shape}")

    model_name = configs["model"]
    positive_label = configs.get("positive_label", "High")
    score_rows = []
    prediction_rows = []

    for target_name in data_target_train.columns:
        y_train = data_target_train[target_name]
        y_test = data_target_test[target_name]
        train_mask = y_train != OUTPUT_NA_NUM
        test_mask = y_test != OUTPUT_NA_NUM

        X_train = data_input_train.loc[train_mask]
        y_train = y_train.loc[train_mask]
        X_test = data_input_test.loc[test_mask]
        y_test = y_test.loc[test_mask]

        if model_name == "majority":
            majority_label = y_train.value_counts().idxmax()
            y_pred = pd.Series(majority_label, index=y_test.index)
            y_score = [float(majority_label == positive_label)] * len(y_test)
        else:
            model = get_model(model_name, seed)
            model.fit(X_train, y_train)
            y_pred = pd.Series(model.predict(X_test), index=y_test.index)
            if hasattr(model, "predict_proba"):
                y_score = select_model_score(
                    model.classes_, model.predict_proba(X_test), positive_label
                )
            else:
                y_score = (y_pred == positive_label).astype(float).tolist()

        metrics = compute_classification_metrics(
            y_true=y_test,
            y_pred=y_pred,
            y_score=y_score,
            positive_label=positive_label,
        )
        score_rows.append({"target": target_name, "model": model_name, **metrics})

        target_predictions = pd.DataFrame(
            {
                "target": target_name,
                "sample_id": y_test.index,
                "y_true": y_test.values,
                "y_pred": y_pred.values,
                f"score_{positive_label}": y_score,
            }
        )
        prediction_rows.append(target_predictions)

    score_df = pd.DataFrame(score_rows)
    predictions_df = pd.concat(prediction_rows, ignore_index=True)
    score_file = work_dir / f"scores_{STAMP}_{config_path.stem}.csv"
    predictions_file = work_dir / f"predictions_{STAMP}_{config_path.stem}.csv"
    score_df.to_csv(score_file, index=False)
    predictions_df.to_csv(predictions_file, index=False)

    print(score_df.to_string(index=False))
    print(f"Saved scores: {score_file}")
    print(f"Saved predictions: {predictions_file}")
    return score_df, predictions_df


def main():
    if len(sys.argv) != 2:
        raise SystemExit(
            "Usage: python scripts/practicum/run_baseline_independent.py CONFIG.json"
        )
    run_baseline(sys.argv[1])


if __name__ == "__main__":
    main()
