# -*- coding: utf-8 -*-
"""Evaluate PYOD baselines with optional multi-dataset benchmarks."""

import argparse
import ast
import gc
import glob
import logging
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.scoring_functions import get_ts_eval

LOGGER = logging.getLogger(__name__)
CURRENT_ROUND = "-"
_ORIGINAL_LOG_RECORD_FACTORY = logging.getLogRecordFactory()
WADI_REDUCED_BATCH_SIZE = 16
USAD_INFERENCE_BATCH_SIZE = 64
USAD_MIN_INFERENCE_BATCH_SIZE = 8


class RoundContextFilter(logging.Filter):
    """Inject current run/round context into every log record."""

    def filter(self, record):
        record.round = CURRENT_ROUND
        return True


def round_log_record_factory(*args, **kwargs):
    record = _ORIGINAL_LOG_RECORD_FACTORY(*args, **kwargs)
    if not hasattr(record, "round"):
        record.round = CURRENT_ROUND
    return record


def set_round_context(run_number: int | None = None, total_runs: int | None = None):
    global CURRENT_ROUND
    if run_number is None or total_runs is None:
        CURRENT_ROUND = "-"
    else:
        CURRENT_ROUND = f"{run_number}/{total_runs}"


def build_classifier_factories(device: str = "cpu", random_state: int | None = None):
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
        "PCA": lambda: PCA(n_components=3, random_state=random_state),
        "COPOD": lambda: COPOD(),
        "IForest": lambda: IForest(random_state=random_state),
        "LOF": lambda: LOF(),
        "OCSVM": lambda: OCSVM(),
        "TimesNet": lambda: TimesNet(seq_len=100, stride=100, device=device, random_state=random_state),
        "DeepSVDD": lambda: DeepSVDDTS(seq_len=100, stride=100, device=device, random_state=random_state),
        "USAD": lambda: USAD(seq_len=100, stride=100, batch_size=2048, device=device, random_state=random_state),
        "AnomalyTransformer": lambda: AnomalyTransformer(seq_len=100, stride=100, device=device, random_state=random_state),
        "TcnED": lambda: TcnED(seq_len=100, stride=100, device=device, verbose=1, batch_size=16, random_state=random_state),
        "TranAD": lambda: TranAD(seq_len=100, stride=100, device=device, random_state=random_state),
        "DeepIF": lambda: DeepIsolationForestTS(seq_len=100, stride=100, device=device, random_state=random_state),
        "COUTA": lambda: COUTA(seq_len=100, stride=100, device=device, batch_size=16, random_state=random_state),
        # "NCAD": lambda: NCAD(seq_len=100, stride=100),
        # "DCdetector": lambda: DCdetector(seq_len=100, stride=100),
    }


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        LOGGER.debug("Torch seed setup failed for seed=%s", seed, exc_info=True)


def configure_gpu(gpu_id):
    if gpu_id is None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        LOGGER.info("No --gpu-id provided; forcing CPU-only mode (CUDA_VISIBLE_DEVICES hidden)")
        return "cpu"

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    LOGGER.info("Configured single GPU visibility: CUDA_VISIBLE_DEVICES=%s", os.environ["CUDA_VISIBLE_DEVICES"])

    try:
        if torch.cuda.is_available():
            # After CUDA_VISIBLE_DEVICES remapping, the selected GPU is index 0.
            torch.cuda.set_device(0)
            LOGGER.info("Pinned torch CUDA device to cuda:0 (mapped from physical GPU %s)", gpu_id)
            return "cuda"
        else:
            LOGGER.warning("--gpu-id=%s set, but torch.cuda is not available; running without CUDA", gpu_id)
            return "cpu"
    except Exception:
        LOGGER.exception("Failed to pin torch device for --gpu-id=%s", gpu_id)
        return "cpu"

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
    qad_base_dir = _resolve_qad_raw_subdir()
    qad_datasets = []

    # Discover all train_*.txt files and create specs for each.
    train_files = sorted(glob.glob(str(qad_base_dir / "train_*.txt")))

    for train_file in train_files:
        file_name = Path(train_file).name
        match = re.match(r"^train_(\d+)\.txt$", file_name)
        if match is None:
            continue

        dataset_num = match.group(1)
        qad_datasets.append({
            "dataset_id": f"qad_{dataset_num}",
            "data_dir": qad_base_dir,
            "train_file": f"train_{dataset_num}.txt",
            "test_file": f"test_{dataset_num}.txt",
            "label_file": f"test_label_{dataset_num}.txt",
            "file_format": "qad_txt",
        })

    return qad_datasets


def _resolve_qad_raw_subdir(raw_subdir: str = "qad_clean_txt_100Hz"):
    requested = ROOT_DIR / "data_dir" / "QAD" / "raw" / raw_subdir
    if requested.is_dir():
        return requested

    fallback = ROOT_DIR / "data_dir" / "QAD" / "raw" / "qad_clean_txt_100Hz"
    if fallback.is_dir():
        LOGGER.warning(
            "Requested QAD folder '%s' not found. Falling back to '%s'.",
            raw_subdir,
            fallback.name,
        )
        return fallback

    return requested


def _load_qad_txt(dataset_path: Path, is_label: bool = False):
    # sep=None lets pandas infer comma/tab separators from converted TXT files.
    kwargs = {}
    if not is_label:
        kwargs["sep"] = None
        kwargs["engine"] = "python"

    data = pd.read_csv(dataset_path, **kwargs)

    if isinstance(data, pd.Series):
        data = data.to_frame(name="labels")

    # Label files should always expose a canonical `labels` column.
    if is_label and isinstance(data, pd.DataFrame) and len(data.columns) == 1 and "labels" not in data.columns:
        data.columns = ["labels"]

    return data


def _build_nasa_datasets(spacecraft: str):
    """Build dataset specs for NASA benchmarks (SMAP/MSL), one per channel id."""
    nasa_base_dir = ROOT_DIR / "data_dir" / "nasa" / "raw"
    anomalies_csv = nasa_base_dir / "labeled_anomalies.csv"
    nasa_datasets = []

    if not anomalies_csv.exists():
        return nasa_datasets

    anomalies_df = pd.read_csv(anomalies_csv)
    anomalies_df = anomalies_df[anomalies_df["spacecraft"] == spacecraft]

    for _, row in anomalies_df.iterrows():
        chan_id = row["chan_id"]
        train_file = nasa_base_dir / "train" / f"{chan_id}.npy"
        test_file = nasa_base_dir / "test" / f"{chan_id}.npy"

        if not train_file.exists() or not test_file.exists():
            continue

        anomaly_sequences = ast.literal_eval(row["anomaly_sequences"])
        nasa_datasets.append(
            {
                "dataset_id": str(chan_id),
                "data_dir": nasa_base_dir,
                "train_file": f"train/{chan_id}.npy",
                "test_file": f"test/{chan_id}.npy",
                "file_format": "nasa_npy",
                "anomaly_sequences": anomaly_sequences,
                "num_values": int(row["num_values"]),
            }
        )

    return nasa_datasets


def configure_logging(level_name: str):
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | round=%(round)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    root_logger = logging.getLogger()
    logging.setLogRecordFactory(round_log_record_factory)
    for handler in root_logger.handlers:
        handler.addFilter(RoundContextFilter())
    set_round_context()


BENCHMARK_DATASETS = {
    "SWaT": [
        {
            "dataset_id": "SWaT",
            "data_dir": ROOT_DIR / "data_dir" / "SWaT" / "raw",
            "train_file": "train.csv",
            "test_file": "test.csv",
            "label_file": "labels.csv",
            "feature_index_col": 0,
            "label_column": None,
        }
    ],
    "WaDi": [
        {
            "dataset_id": "WaDi",
            "data_dir": ROOT_DIR / "data_dir" / "WaDi" / "raw" / "v2",
            "train_file": "WADI_14days.csv",
            "test_file_candidates": ["attackdata_labbelled.csv", "WADI_attackdata_labelled.csv"],
            "file_format": "wadi_v2",
            "label_column_candidates": [
                "Arrack LABLE",
                "Attack LABLE",
                "Attack LABLE (1:No Attack, -1:Attack)",
            ],
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
    "SMAP": _build_nasa_datasets("SMAP"),
    "MSL": _build_nasa_datasets("MSL"),
    "SMD": _build_smd_datasets(),
    "QAD": _build_qad_datasets(),
}


def parse_args():
    def positive_int(value):
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("--runs must be >= 1")
        return parsed

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
    parser.add_argument(
        "--runs",
        type=positive_int,
        default=1,
        help="How many repeated evaluation runs to execute per benchmark/classifier/dataset combination.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed; run i uses seed + i.",
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

    # Handle WaDi v2 rawraw with second-row test headers and in-file labels.
    if spec.get("file_format") == "wadi_v2":
        train_df = pd.read_csv(data_dir / spec["train_file"], sep=",", header=0)

        test_file = None
        for candidate in spec.get("test_file_candidates", []):
            candidate_path = data_dir / candidate
            if candidate_path.exists():
                test_file = candidate
                break
        if test_file is None:
            raise FileNotFoundError(
                f"[{dataset_id}] none of test file candidates exist: {spec.get('test_file_candidates', [])}"
            )

        # In WaDi v2 attack file, sensor names are stored in row 2 (header=1).
        test_df = pd.read_csv(data_dir / test_file, sep=",", header=1)

        train_df.columns = [str(col).strip() for col in train_df.columns]
        test_df.columns = [str(col).strip() for col in test_df.columns]

        label_col = None
        label_candidates = [c.strip().upper() for c in spec.get("label_column_candidates", [])]
        for col in test_df.columns:
            col_norm = str(col).strip().upper()
            if col_norm in label_candidates:
                label_col = col
                break
            if "LABLE" in col_norm and "ATTACK" in col_norm:
                label_col = col
                break

        if label_col is None:
            raise ValueError(f"[{dataset_id}] could not find WaDi label column in test data")

        # WaDi labels are typically 1 for no-attack and -1 for attack.
        raw_labels = pd.to_numeric(test_df[label_col], errors="coerce")
        y_test = (raw_labels != 1).astype(float).to_numpy().ravel()

        # Remove metadata and label columns from features.
        metadata_cols = {"ROW", "ROW ", "DATE", "DATE ", "TIME", "TIME "}
        train_drop_cols = [c for c in train_df.columns if str(c).strip().upper() in metadata_cols]
        test_drop_cols = [c for c in test_df.columns if str(c).strip().upper() in metadata_cols]
        train_df = train_df.drop(columns=train_drop_cols, errors="ignore")
        test_df = test_df.drop(columns=test_drop_cols + [label_col], errors="ignore")

        common_cols = [c for c in train_df.columns if c in test_df.columns]
        if not common_cols:
            raise ValueError(f"[{dataset_id}] no common feature columns between train and test")

        x_train_df = train_df[common_cols].apply(pd.to_numeric, errors="coerce")
        x_test_df = test_df[common_cols].apply(pd.to_numeric, errors="coerce")
        x_train = x_train_df.to_numpy(dtype=float)
        x_test = x_test_df.to_numpy(dtype=float)
    # Handle QAD TXT files.
    elif spec.get("file_format") == "qad_txt":
        x_train_df = _load_qad_txt(data_dir / spec["train_file"])
        x_test_df = _load_qad_txt(data_dir / spec["test_file"])
        y_test_df = _load_qad_txt(data_dir / spec["label_file"], is_label=True)

        x_train = x_train_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        x_test = x_test_df.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

        if isinstance(y_test_df, pd.DataFrame):
            if y_test_df.shape[1] > 1:
                LOGGER.warning(
                    "[%s] QAD label file has %d columns; using only first column '%s'",
                    dataset_id,
                    y_test_df.shape[1],
                    y_test_df.columns[0],
                )
            y_test_series = y_test_df.iloc[:, 0]
        else:
            y_test_series = y_test_df

        y_test = pd.to_numeric(y_test_series, errors="coerce").to_numpy(dtype=float).ravel()

        if x_test.shape[0] != y_test.shape[0]:
            aligned_len = min(x_test.shape[0], y_test.shape[0])
            LOGGER.warning(
                "[%s] QAD test/label length mismatch (x_test=%d, y_test=%d); truncating both to %d",
                dataset_id,
                x_test.shape[0],
                y_test.shape[0],
                aligned_len,
            )
            x_test = x_test[:aligned_len]
            y_test = y_test[:aligned_len]
    elif spec.get("file_format") == "nasa_npy":
        x_train = np.load(data_dir / spec["train_file"])
        x_test = np.load(data_dir / spec["test_file"])

        if x_train.ndim == 1:
            x_train = x_train.reshape(-1, 1)
        elif x_train.ndim > 2:
            x_train = x_train.reshape(x_train.shape[0], -1)

        if x_test.ndim == 1:
            x_test = x_test.reshape(-1, 1)
        elif x_test.ndim > 2:
            x_test = x_test.reshape(x_test.shape[0], -1)

        y_len = int(spec.get("num_values", x_test.shape[0]))
        y_len = max(y_len, 0)
        y_test = np.zeros(y_len, dtype=float)

        for start_idx, stop_idx in spec.get("anomaly_sequences", []):
            start = max(0, min(int(start_idx), y_len))
            stop = max(start, min(int(stop_idx), y_len))
            y_test[start:stop] = 1.0

        if x_test.shape[0] != y_test.shape[0]:
            aligned_len = min(x_test.shape[0], y_test.shape[0])
            LOGGER.warning(
                "[%s] NASA test/label length mismatch (x_test=%d, y_test=%d); truncating both to %d",
                dataset_id,
                x_test.shape[0],
                y_test.shape[0],
                aligned_len,
            )
            x_test = x_test[:aligned_len]
            y_test = y_test[:aligned_len]
    else:
        # Handle CSV files (SWaT, PSM, SMD, WaDi, ...)
        header = spec.get("header", "infer")  # Default to "infer", can be None for no header
        # Allow per-split index column overrides (e.g. WaDi train has "index" col, test does not)
        train_index_col = spec.get("train_index_col", spec.get("feature_index_col"))
        test_index_col  = spec.get("test_index_col",  spec.get("feature_index_col"))

        x_train_df = pd.read_csv(
            data_dir / spec["train_file"],
            sep=",",
            index_col=train_index_col,
            header=header,
        )
        x_test_df = pd.read_csv(
            data_dir / spec["test_file"],
            sep=",",
            index_col=test_index_col,
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

    x_train = _impute_nan_windowed(x_train, dataset_id, "train")
    x_test = _impute_nan_windowed(x_test, dataset_id, "test")

    y_test = _impute_nan_windowed(y_test, dataset_id, "test")
    y_test[y_test < 0.5] = 0
    y_test[y_test >= 0.5] = 1

    if max_train_samples is not None:
        if x_train.shape[0] > max_train_samples:
            LOGGER.info("[%s] truncating train rows: %s -> %s", dataset_id, x_train.shape[0], max_train_samples)
        x_train = x_train[:max_train_samples]
    if max_test_samples is not None:
        if x_test.shape[0] > max_test_samples:
            LOGGER.info("[%s] truncating test rows: %s -> %s", dataset_id, x_test.shape[0], max_test_samples)
        x_test = x_test[:max_test_samples]
        y_test = y_test[:max_test_samples]

    x_train, x_test = _standard_scale_features(x_train, x_test, dataset_id)

    LOGGER.info(
        "[%s] loaded dataset: train_shape=%s, test_shape=%s, labels_shape=%s",
        dataset_id,
        x_train.shape,
        x_test.shape,
        y_test.shape,
    )

    return x_train, x_test, y_test


def evaluate_classifier_on_dataset(
    clf_name,
    clf,
    x_train,
    x_test,
    y_test,
    benchmark_name,
    dataset_id,
):
    if benchmark_name in ["WaDi", "SWaT"] and hasattr(clf, "batch_size") and False:
        original_batch_size = getattr(clf, "batch_size", None)
        if original_batch_size is None or original_batch_size > WADI_REDUCED_BATCH_SIZE:
            clf.batch_size = WADI_REDUCED_BATCH_SIZE
            LOGGER.info(
                "[%s/%s] %s batch size reduced for WaDi: %s -> %s",
                benchmark_name,
                dataset_id,
                clf_name,
                original_batch_size,
                clf.batch_size,
            )

    if hasattr(clf, "val_pc"):
        clf.val_pc = 0.1
        LOGGER.info(
            "[%s/%s] %s validation split set to %.0f%% (val_pc=%.2f)",
            benchmark_name,
            dataset_id,
            clf_name,
            clf.val_pc * 100,
            clf.val_pc,
        )
    elif hasattr(clf, "train_val_pc"):
        clf.train_val_pc = 0.1
        LOGGER.info(
            "[%s/%s] %s validation split set to %.0f%% (train_val_pc=%.2f)",
            benchmark_name,
            dataset_id,
            clf_name,
            clf.train_val_pc * 100,
            clf.train_val_pc,
        )

    LOGGER.info(
        "[%s/%s] running %s (device=%s, batch_size=%s)",
        benchmark_name,
        dataset_id,
        clf_name,
        getattr(clf, "device", "n/a"),
        getattr(clf, "batch_size", "n/a"),
    )

    clf.fit(x_train)
    LOGGER.info("[%s/%s] fitted %s", benchmark_name, dataset_id, clf_name)
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.debug("[%s/%s] torch cleanup skipped", benchmark_name, dataset_id, exc_info=True)


    if clf_name in ["TcnED"]:
        LOGGER.warning(f"Running on {clf_name}, moving net to cpu for inference.")
        if hasattr(clf, "net") and clf.net is not None:    # TcnED
            clf.net.to('cpu')

        clf.device = "cpu"
        configure_gpu(None)

    if clf_name == "USAD":
        original_batch_size = getattr(clf, "batch_size", None)
        if original_batch_size is not None and original_batch_size > USAD_INFERENCE_BATCH_SIZE:
            clf.batch_size = USAD_INFERENCE_BATCH_SIZE
            LOGGER.info(
                "[%s/%s] %s inference batch size reduced: %s -> %s",
                benchmark_name,
                dataset_id,
                clf_name,
                original_batch_size,
                clf.batch_size,
            )

    max_attempts = 1
    if clf_name == "USAD" and hasattr(clf, "batch_size"):
        max_attempts = 4

    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            with torch.inference_mode():
                y_test_scores = np.asarray(clf.decision_function(x_test)).ravel()
            break
        except RuntimeError as error:
            last_error = error
            message = str(error).lower()
            is_cuda_oom = "cuda" in message and "out of memory" in message
            if not is_cuda_oom or clf_name != "USAD" or not hasattr(clf, "batch_size") or attempt == max_attempts:
                raise

            old_batch_size = int(getattr(clf, "batch_size"))
            new_batch_size = max(USAD_MIN_INFERENCE_BATCH_SIZE, old_batch_size // 2)
            if new_batch_size >= old_batch_size:
                raise

            LOGGER.warning(
                "[%s/%s] %s inference OOM on attempt %d/%d. Reducing batch_size: %d -> %d and retrying.",
                benchmark_name,
                dataset_id,
                clf_name,
                attempt,
                max_attempts,
                old_batch_size,
                new_batch_size,
            )
            clf.batch_size = new_batch_size
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        raise last_error if last_error is not None else RuntimeError("Inference failed without a captured error")

    metric_results, _ = get_ts_eval(y_test_scores, y_test)

    del clf
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.debug("[%s/%s] torch cleanup skipped", benchmark_name, dataset_id, exc_info=True)


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


def _impute_nan_windowed(X, dataset_id: str, split: str, window: int = 5):
    """Replace NaNs with the mean of a ±window context window, then column mean, then 0."""
    arr = np.asarray(X, dtype=float)
    original_shape = arr.shape
    nan_count = int(np.isnan(arr).sum())
    if nan_count == 0:
        return arr

    LOGGER.warning(
        "[%s] %s: found %d NaN value(s) in shape %s — imputing with ±%d window mean",
        dataset_id, split, nan_count, original_shape, window,
    )

    if arr.ndim == 1:
        df = pd.DataFrame(arr, dtype=float)
    else:
        df = pd.DataFrame(arr, dtype=float)

    rolling_mean = df.rolling(window=2 * window + 1, min_periods=1, center=True).mean()
    df = df.where(df.notna(), rolling_mean)   # fill NaNs with rolling mean
    df = df.fillna(df.mean())                 # fallback: column mean
    df = df.fillna(0.0)                       # last resort: zero
    out = df.to_numpy()
    if len(original_shape) == 1:
        return out.ravel()
    return out.reshape(original_shape)


def _standard_scale_features(x_train, x_test, dataset_id: str):
    """Fit a per-feature StandardScaler on train and apply to train/test."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)
    LOGGER.info("[%s] applied StandardScaler feature-wise using train statistics", dataset_id)
    return x_train_scaled, x_test_scaled


def append_df_to_csv(df, csv_path, index=False):
    file_exists = csv_path.exists() and csv_path.stat().st_size > 0
    df.to_csv(csv_path, mode="a", header=not file_exists, index=index)


def aggregate_mean_std(df, group_cols):
    metrics = ["auroc", "auprc", "f1"]
    mean_df = df.groupby(group_cols, as_index=False)[metrics].mean().rename(columns={m: f"{m}_mean" for m in metrics})
    std_df = (
        df.groupby(group_cols, as_index=False)[metrics]
        .std(ddof=0)
        .fillna(0.0)
        .rename(columns={m: f"{m}_std" for m in metrics})
    )
    counts_df = df.groupby(group_cols, as_index=False).size().rename(columns={"size": "num_runs"})
    return mean_df.merge(std_df, on=group_cols, how="inner").merge(counts_df, on=group_cols, how="inner")


def build_mean_std_report(df, group_cols):
    report_df = df[group_cols + ["num_runs"]].copy()
    for metric in ["auroc", "auprc", "f1"]:
        report_df[metric] = (
            df[f"{metric}_mean"].map(lambda value: f"{value:.6f}")
            + " +- "
            + df[f"{metric}_std"].map(lambda value: f"{value:.6f}")
        )
    return report_df


if __name__ == "__main__":
    args = parse_args()
    configure_logging(args.log_level)
    runtime_device = configure_gpu(args.gpu_id)

    classifier_factories = build_classifier_factories(device=runtime_device, random_state=args.seed)

    output_dir = ROOT_DIR / "out"
    os.makedirs(output_dir, exist_ok=True)
    per_dataset_path = output_dir / "baselines_per_dataset.csv"
    macro_path = output_dir / "baselines.csv"
    per_dataset_summary_path = output_dir / "baselines_per_dataset_mean_std.csv"
    macro_summary_path = output_dir / "baselines_macro_mean_std.csv"

    LOGGER.info("Starting baseline evaluation")
    LOGGER.info(
        "Arguments: benchmarks=%s, classifiers=%s, max_train_samples=%s, max_test_samples=%s, runs=%s, seed=%s, device=%s",
        args.benchmarks,
        args.classifiers,
        args.max_train_samples,
        args.max_test_samples,
        args.runs,
        args.seed,
        runtime_device,
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
    per_run_rows = []
    failed_runs = []
    for run_idx in range(args.runs):
        run_number = run_idx + 1
        run_seed = args.seed + run_idx
        set_round_context(run_number, args.runs)
        set_global_seed(run_seed)
        classifier_factories = build_classifier_factories(device=runtime_device, random_state=run_seed)
        LOGGER.info("Starting run %d/%d with seed=%d", run_number, args.runs, run_seed)
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
                        per_run_rows.append({**row, "run": run_number, "seed": run_seed})
                        append_df_to_csv(pd.DataFrame([row]), per_dataset_path, index=False)
                    except Exception:
                        failed_runs.append((run_number, run_seed, benchmark_name, dataset_id, clf_name))
                        LOGGER.exception(
                            "[seed=%d][%s/%s] %s failed",
                            run_seed,
                            benchmark_name,
                            dataset_id,
                            clf_name,
                        )

    set_round_context()

    if not per_dataset_rows:
        LOGGER.error("No successful runs. Failed runs: %d", len(failed_runs))
        sys.exit(1)

    per_dataset_df = pd.DataFrame(per_dataset_rows)
    per_run_df = pd.DataFrame(per_run_rows)

    macro_df = macro_average(per_dataset_df)
    macro_df = macro_df.set_index(["benchmark", "clf_name"])
    append_df_to_csv(macro_df.reset_index(), macro_path, index=False)

    per_dataset_summary_df = aggregate_mean_std(per_run_df, ["benchmark", "dataset_id", "clf_name"])
    append_df_to_csv(per_dataset_summary_df, per_dataset_summary_path, index=False)

    per_run_macro_df = (
        per_run_df.groupby(["run", "benchmark", "clf_name"], as_index=False)[["auroc", "auprc", "f1"]]
        .mean()
    )
    macro_summary_df = aggregate_mean_std(per_run_macro_df, ["benchmark", "clf_name"])
    append_df_to_csv(macro_summary_df, macro_summary_path, index=False)

    LOGGER.info("Completed %d successful run(s)", len(per_dataset_rows))
    if failed_runs:
        LOGGER.warning("Encountered %d failed run(s); continuing with successful results", len(failed_runs))

    LOGGER.info("Per-dataset metrics:\n%s", per_dataset_df.to_string(index=False))
    LOGGER.info("Macro-averaged benchmark metrics:\n%s", macro_df.to_string())
    LOGGER.info(
        "Per-dataset mean +- std across runs:\n%s",
        build_mean_std_report(per_dataset_summary_df, ["benchmark", "dataset_id", "clf_name"]).to_string(index=False),
    )
    LOGGER.info(
        "Macro mean +- std across runs:\n%s",
        build_mean_std_report(macro_summary_df, ["benchmark", "clf_name"]).to_string(index=False),
    )
    LOGGER.info("Appended per-dataset metrics to %s", per_dataset_path)
    LOGGER.info("Appended macro-averaged metrics to %s", macro_path)
    LOGGER.info("Appended per-dataset mean/std metrics to %s", per_dataset_summary_path)
    LOGGER.info("Appended macro mean/std metrics to %s", macro_summary_path)

