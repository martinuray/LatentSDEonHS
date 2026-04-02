# -*- coding: utf-8 -*-
"""Evaluate PYOD baselines with optional multi-dataset benchmarks."""

import argparse
import glob
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from anomaly_detection import get_ts_eval


LOGGER = logging.getLogger(__name__)


def build_classifier_factories():
    # Import deepod models lazily so GPU visibility can be configured first.
    from pyod.models.copod import COPOD
    from pyod.models.iforest import IForest
    from pyod.models.knn import KNN
    from pyod.models.lof import LOF
    from pyod.models.ocsvm import OCSVM
    from pyod.models.pca import PCA


    from deepod.models.time_series import (
        AnomalyTransformer,
        COUTA,
        DeepIsolationForestTS,
        DeepSVDDTS,
        TcnED,
        TimesNet,
        TranAD,
        USAD,
        # DCdetector, NCAD
    )

    return {
        "KNN": lambda: KNN(),
        "PCA": lambda: PCA(n_components=3),
        "COPOD": lambda: COPOD(),
        "IForest": lambda: IForest(),
        "LOF": lambda: LOF(),
        "OCSVM": lambda: OCSVM(),
        "TimesNet": lambda: TimesNet(seq_len=100, stride=100),
        "DeepSVDD": lambda: DeepSVDDTS(seq_len=100, stride=100),
        "USAD": lambda: USAD(seq_len=100, stride=100),
        "AnomalyTransformer": lambda: AnomalyTransformer(seq_len=100, stride=100),
        "TcnED": lambda: TcnED(seq_len=100, stride=100),
        "TranAD": lambda: TranAD(seq_len=100, stride=100),
        "DeepIF": lambda: DeepIsolationForestTS(seq_len=100, stride=100),
        "COUTA": lambda: COUTA(seq_len=100, stride=100),
        # "NCAD": lambda: NCAD(seq_len=100, stride=100),
        # "DCdetector": lambda: DCdetector(seq_len=100, stride=100),
    }


def configure_gpu(gpu_id):
    if gpu_id is None:
        LOGGER.info("No --gpu-id provided; keeping CUDA_VISIBLE_DEVICES=%s", os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
        return

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    LOGGER.info("Configured single GPU visibility: CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    try:
        import torch

        if torch.cuda.is_available():
            # After CUDA_VISIBLE_DEVICES remapping, the selected GPU is index 0.
            torch.cuda.set_device(0)
            LOGGER.info("Pinned torch CUDA device to cuda:0 (mapped from physical GPU %s)", gpu_id)
        else:
            LOGGER.warning("--gpu-id=%s set, but torch.cuda is not available; running without CUDA", gpu_id)
    except Exception:
        LOGGER.exception("Failed to pin torch device for --gpu-id=%s", gpu_id)

# One benchmark can contain one or multiple independent datasets.
def _build_smd_datasets():
    """Build dataset specs for all SMD machines (machine-x-n)."""
    smd_base_dir = ROOT_DIR / "data_dir" / "SMD" / "raw"
    smd_datasets = []
    
    # Discover all machine-x-n files and create specs for each
    train_dir = smd_base_dir / "train"
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


def configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


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
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Physical GPU id to use exclusively (sets CUDA_VISIBLE_DEVICES to this single id).",
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
    dataset_id = spec.get("dataset_id", "unknown")

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
            else:
                LOGGER.debug("[%s] train missing drop column '%s'", dataset_id, col_name)
            if col_name in x_test_df.columns:
                x_test_df = x_test_df.drop(columns=[col_name])
            else:
                LOGGER.debug("[%s] test missing drop column '%s'", dataset_id, col_name)

        y_test_df = pd.read_csv(data_dir / spec["label_file"], sep=",", header=header)
        label_col = spec.get("label_column")
        if label_col is not None and label_col in y_test_df.columns:
            y_test = y_test_df[label_col].to_numpy().ravel()
        else:
            y_test = y_test_df.to_numpy().ravel()

        x_train = x_train_df.to_numpy()
        x_test = x_test_df.to_numpy()

    if max_train_samples is not None:
        if x_train.shape[0] > max_train_samples:
            LOGGER.info("[%s] truncating train rows: %s -> %s", dataset_id, x_train.shape[0], max_train_samples)
        x_train = x_train[:max_train_samples]
    if max_test_samples is not None:
        if x_test.shape[0] > max_test_samples:
            LOGGER.info("[%s] truncating test rows: %s -> %s", dataset_id, x_test.shape[0], max_test_samples)
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    LOGGER.info(
        "[%s] loaded dataset: train_shape=%s, test_shape=%s, labels_shape=%s",
        dataset_id,
        x_train.shape,
        x_test.shape,
        y_test.shape,
    )

    return x_train, x_test, y_test


def evaluate_classifier_on_dataset(clf_name, clf, x_train, x_test, y_test, benchmark_name, dataset_id):
    LOGGER.info("[%s/%s] running %s", benchmark_name, dataset_id, clf_name)
    clf.fit(x_train)
    LOGGER.info("[%s/%s] fitted %s", benchmark_name, dataset_id, clf_name)

    y_test_scores = clf.decision_function(x_test)
    metric_results, _ = get_ts_eval(y_test_scores, y_test)

    selected_metrics = {
        "auroc": metric_results["auroc"],
        "auprc": metric_results["auprc"],
        "f1": metric_results["f1"],
    }
    LOGGER.info(
        "[%s/%s] %s: auroc=%.6f, auprc=%.6f, f1=%.6f",
        benchmark_name,
        dataset_id,
        clf_name,
        selected_metrics["auroc"],
        selected_metrics["auprc"],
        selected_metrics["f1"],
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
    configure_logging(args.log_level)
    configure_gpu(args.gpu_id)

    classifier_factories = build_classifier_factories()

    LOGGER.info("Starting baseline evaluation")
    LOGGER.info(
        "Arguments: benchmarks=%s, classifiers=%s, max_train_samples=%s, max_test_samples=%s",
        args.benchmarks,
        args.classifiers,
        args.max_train_samples,
        args.max_test_samples,
    )

    selected_benchmarks = _select_keys(BENCHMARK_DATASETS, args.benchmarks)
    selected_classifiers = _select_keys(classifier_factories, args.classifiers)

    LOGGER.info("Selected benchmarks: %s", selected_benchmarks)
    LOGGER.info("Selected classifiers: %s", selected_classifiers)
    for benchmark_name in selected_benchmarks:
        dataset_count = len(BENCHMARK_DATASETS[benchmark_name])
        if dataset_count == 0:
            LOGGER.warning("Benchmark %s has no discovered datasets", benchmark_name)
        else:
            LOGGER.info("Benchmark %s has %d dataset(s)", benchmark_name, dataset_count)

    per_dataset_rows = []
    failed_runs = []
    for benchmark_name in selected_benchmarks:
        dataset_specs = BENCHMARK_DATASETS[benchmark_name]
        for clf_name in selected_classifiers:
            clf_factory = classifier_factories[clf_name]
            for dataset_spec in dataset_specs:
                dataset_id = dataset_spec["dataset_id"]
                try:
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
                        dataset_id,
                    )
                    per_dataset_rows.append(row)
                except Exception:
                    failed_runs.append((benchmark_name, dataset_id, clf_name))
                    LOGGER.exception("[%s/%s] %s failed", benchmark_name, dataset_id, clf_name)

    if not per_dataset_rows:
        LOGGER.error("No successful runs. Failed runs: %d", len(failed_runs))
        sys.exit(1)

    per_dataset_df = pd.DataFrame(per_dataset_rows)
    macro_df = macro_average(per_dataset_df)
    macro_df = macro_df.set_index(["benchmark", "clf_name"])

    output_dir = ROOT_DIR / "out"
    os.makedirs(output_dir, exist_ok=True)

    per_dataset_path = output_dir / "baselines_per_dataset.csv"
    macro_path = output_dir / "baselines.csv"
    per_dataset_df.to_csv(per_dataset_path, index=False)
    macro_df.to_csv(macro_path, index=True)

    LOGGER.info("Completed %d successful run(s)", len(per_dataset_rows))
    if failed_runs:
        LOGGER.warning("Encountered %d failed run(s); continuing with successful results", len(failed_runs))

    LOGGER.info("Per-dataset metrics:\n%s", per_dataset_df.to_string(index=False))
    LOGGER.info("Macro-averaged benchmark metrics:\n%s", macro_df.to_string())
    LOGGER.info("Saved per-dataset metrics to %s", per_dataset_path)
    LOGGER.info("Saved macro-averaged metrics to %s", macro_path)

