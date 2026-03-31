# -*- coding: utf-8 -*-
"""Evaluate PYOD baselines with optional multi-dataset benchmarks."""

import argparse
import glob
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from pyod.models.copod import COPOD
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.ocsvm import OCSVM
from pyod.models.pca import PCA

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from basic_data_anomaly_detection import get_ts_eval


CLASSIFIER_FACTORIES = {
    "KNN": lambda: KNN(),
    "PCA": lambda: PCA(n_components=3),
    "COPOD": lambda: COPOD(),
    "IForest": lambda: IForest(),
    "LOF": lambda: LOF(),
    "OCSVM": lambda: OCSVM(),
}

# One benchmark can contain one or multiple independent datasets.
def _build_smd_datasets():
    """Build dataset specs for all SMD machines (machine-x-n)."""
    smd_base_dir = ROOT_DIR / "data_dir" / "SMD" / "raw"
    smd_datasets = []
    
    # Discover all machine-x-n files and create specs for each
    train_dir = smd_base_dir / "train"
    import glob
    train_files = sorted(glob.glob(str(train_dir / "machine-*.txt")))
    
    for train_file in train_files:
        machine_id = Path(train_file).stem  # e.g., "machine-1-1"
        smd_datasets.append({
            "dataset_id": machine_id,
            "data_dir": smd_base_dir,
            "train_file": f"train/{machine_id}.txt",
            "test_file": f"test/{machine_id}.txt",
            "label_file": f"test_label/{machine_id}.txt",
            "feature_index_col": None,
            "label_column": None,
            "header": None,
        })
    
    return smd_datasets


def _build_qad_datasets():
    """Build dataset specs for all QAD 100Hz datasets (qad_*_1 to qad_*_16)."""
    qad_base_dir = ROOT_DIR / "data_dir" / "QAD" / "raw" / "qad_clean_pkl_100Hz"
    qad_datasets = []
    
    # Discover all train_*.pkl files and create specs for each
    train_files = sorted(glob.glob(str(qad_base_dir / "train_*.pkl")))
    
    for train_file in train_files:
        dataset_num = Path(train_file).stem.replace("train_", "")  # e.g., "1", "10", etc.
        qad_datasets.append({
            "dataset_id": f"qad_{dataset_num}",
            "data_dir": qad_base_dir,
            "train_file": f"train_{dataset_num}.pkl",
            "test_file": f"test_{dataset_num}.pkl",
            "label_file": f"test_label_{dataset_num}.pkl",
            "file_format": "pickle",
        })
    
    return qad_datasets


BENCHMARK_DATASETS = {
    "SWaT": [
        {
            "dataset_id": "SWaT",
            "data_dir": ROOT_DIR / "data_dir" / "SWaT" / "rawraw",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "label_file": "labels.csv",
            "feature_index_col": 0,
            "label_column": None,
        }
    ],
    "PSM": [
        {
            "dataset_id": "PSM",
            "data_dir": ROOT_DIR / "data_dir" / "PSM" / "raw",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "label_file": "test_label.csv",
            "feature_index_col": None,
            "label_column": "label",
            "drop_feature_columns": ["timestamp_(min)"],
        },
    ],
    "SMD": _build_smd_datasets(),
    "QAD": _build_qad_datasets(),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Run PYOD baselines and macro-average metrics across datasets.")
    parser.add_argument(
        "--benchmarks",
        type=str,
        default="all",
        help="Comma-separated benchmark names, or 'all'.",
    )
    parser.add_argument(
        "--classifiers",
        type=str,
        default="all",
        help="Comma-separated classifier names, or 'all'.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap on training rows for quick checks.",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Optional cap on test rows (and labels) for quick checks.",
    )
    return parser.parse_args()


def _select_keys(available, requested_csv):
    if requested_csv.strip().lower() == "all":
        return list(available.keys())
    requested = [item.strip() for item in requested_csv.split(",") if item.strip()]
    invalid = [item for item in requested if item not in available]
    if invalid:
        raise ValueError(f"Unknown names: {invalid}. Available: {list(available.keys())}")
    return requested


def load_dataset(spec, max_train_samples=None, max_test_samples=None):
    data_dir = spec["data_dir"]
    
    # Handle pickle files (QAD)
    if spec.get("file_format") == "pickle":
        with open(data_dir / spec["train_file"], "rb") as f:
            x_train_df = pickle.load(f)
        with open(data_dir / spec["test_file"], "rb") as f:
            x_test_df = pickle.load(f)
        with open(data_dir / spec["label_file"], "rb") as f:
            y_test = pickle.load(f)
        
        if isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy().ravel()
        elif isinstance(y_test, pd.DataFrame):
            y_test = y_test.to_numpy().ravel()
        
        x_train = x_train_df.to_numpy() if isinstance(x_train_df, pd.DataFrame) else x_train_df
        x_test = x_test_df.to_numpy() if isinstance(x_test_df, pd.DataFrame) else x_test_df
    else:
        # Handle CSV files (SWaT, PSM, SMD)
        header = spec.get("header", "infer")  # Default to "infer", can be None for no header
        
        x_train_df = pd.read_csv(
            data_dir / spec["train_file"], 
            sep=",", 
            index_col=spec.get("feature_index_col"),
            header=header,
        )
        x_test_df = pd.read_csv(
            data_dir / spec["test_file"], 
            sep=",", 
            index_col=spec.get("feature_index_col"),
            header=header,
        )

        for col_name in spec.get("drop_feature_columns", []):
            if col_name in x_train_df.columns:
                x_train_df = x_train_df.drop(columns=[col_name])
            if col_name in x_test_df.columns:
                x_test_df = x_test_df.drop(columns=[col_name])

        y_test_df = pd.read_csv(data_dir / spec["label_file"], sep=",", header=header)
        label_col = spec.get("label_column")
        if label_col is not None and label_col in y_test_df.columns:
            y_test = y_test_df[label_col].to_numpy().ravel()
        else:
            y_test = y_test_df.to_numpy().ravel()

        x_train = x_train_df.to_numpy()
        x_test = x_test_df.to_numpy()

    if max_train_samples is not None:
        x_train = x_train[:max_train_samples]
    if max_test_samples is not None:
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    return x_train, x_test, y_test


def evaluate_classifier_on_dataset(clf_name, clf, x_train, x_test, y_test, benchmark_name, dataset_id):
    clf.fit(x_train)
    y_test_scores = clf.decision_function(x_test)
    metric_results, _ = get_ts_eval(y_test_scores, y_test)

    selected_metrics = {
        "auroc": metric_results["auroc"],
        "auprc": metric_results["auprc"],
        "f1": metric_results["f1"],
    }
    print(
        f"[{benchmark_name}/{dataset_id}] {clf_name}: "
        f"auroc={selected_metrics['auroc']:.6f}, "
        f"auprc={selected_metrics['auprc']:.6f}, "
        f"f1={selected_metrics['f1']:.6f}"
    )

    return {
        "benchmark": benchmark_name,
        "dataset_id": dataset_id,
        "clf_name": clf_name,
        **selected_metrics,
    }


def macro_average(per_dataset_df):
    macro_df = (
        per_dataset_df.groupby(["benchmark", "clf_name"], as_index=False)[["auroc", "auprc", "f1"]]
        .mean()
        .sort_values(["benchmark", "clf_name"])
    )
    counts = per_dataset_df.groupby(["benchmark", "clf_name"], as_index=False).size().rename(columns={"size": "num_datasets"})
    macro_df = macro_df.merge(counts, on=["benchmark", "clf_name"], how="left")
    return macro_df


if __name__ == "__main__":
    args = parse_args()
    selected_benchmarks = _select_keys(BENCHMARK_DATASETS, args.benchmarks)
    selected_classifiers = _select_keys(CLASSIFIER_FACTORIES, args.classifiers)

    per_dataset_rows = []
    for benchmark_name in selected_benchmarks:
        dataset_specs = BENCHMARK_DATASETS[benchmark_name]
        for clf_name in selected_classifiers:
            clf_factory = CLASSIFIER_FACTORIES[clf_name]
            for dataset_spec in dataset_specs:
                x_train, x_test, y_test = load_dataset(
                    dataset_spec,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                )
                clf = clf_factory()
                row = evaluate_classifier_on_dataset(
                    clf_name,
                    clf,
                    x_train,
                    x_test,
                    y_test,
                    benchmark_name,
                    dataset_spec["dataset_id"],
                )
                per_dataset_rows.append(row)

    per_dataset_df = pd.DataFrame(per_dataset_rows)
    macro_df = macro_average(per_dataset_df)
    macro_df = macro_df.set_index(["benchmark", "clf_name"])

    output_dir = ROOT_DIR / "out"
    os.makedirs(output_dir, exist_ok=True)

    per_dataset_path = output_dir / "baselines_per_dataset.csv"
    macro_path = output_dir / "baselines.csv"
    per_dataset_df.to_csv(per_dataset_path, index=False)
    macro_df.to_csv(macro_path, index=True)

    print("\nPer-dataset metrics:")
    print(per_dataset_df)
    print("\nMacro-averaged benchmark metrics:")
    print(macro_df)
    print(f"\nSaved per-dataset metrics to {per_dataset_path}")
    print(f"Saved macro-averaged metrics to {macro_path}")

