# -*- coding: utf-8 -*-
"""Evaluate PYOD baselines with optional multi-dataset benchmarks."""

import argparse
import ast
import gc
import glob
import logging
import os
import pickle
import random
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:64")
import torch

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.scoring_functions import get_ts_eval, smooth_scores

LOGGER = logging.getLogger(__name__)
CURRENT_ROUND = "-"
_ORIGINAL_LOG_RECORD_FACTORY = logging.getLogRecordFactory()
WADI_REDUCED_BATCH_SIZE = 16
USAD_INFERENCE_BATCH_SIZE = 64
COUTA_INFERENCE_BATCH_SIZE = 256
# Any deep model may OOM during inference; halve its batch size down to this
# floor before falling back to CPU (see _score_with_oom_recovery).
MIN_INFERENCE_BATCH_SIZE = 8
INFERENCE_OOM_MAX_ATTEMPTS = 5
# COUTA derives its synthetic-negative count as int(batch_size * neg_batch_ratio),
# so a small batch size silently drops the calibration term that stops the
# one-class objective collapsing - see _warn_if_couta_calibration_disabled.
COUTA_TRAIN_BATCH_SIZE = 64
# Scores whose spread is this small relative to their scale carry no ranking
# information - the model has collapsed (see _warn_on_degenerate_scores).
DEGENERATE_SCORE_REL_STD = 1e-6
DEFAULT_SEQ_LEN = 200
DEFAULT_STRIDE = 20
DEFAULT_DECIMATION = 1
DEFAULT_SCORE_SMOOTHING = 10
DEFAULT_EVAL_WINDOW = 200

# Per-benchmark window/scoring defaults, used unless overridden on the CLI.
# QAD mirrors cfg/anomaly_detection/QAD.json so the baselines see the same
# data (100 Hz decimated to 10 Hz), the same 20 s windows with a 2 s stride,
# and the same score smoothing as the latent-SDE model.
BENCHMARK_WINDOW_DEFAULTS = {
    "QAD": {
        "seq_len": 200,
        "stride": 20,
        "decimation": 10,
        "score_smoothing_window": 10,
        "eval_window": 200,
    },
}


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


_DEEPOD_INFERENCE_PATCHED = False
_DEEPOD_WINDOWING_PATCHED = False
_DEEPOD_COUTA_BATCH_PATCHED = False


def _patch_deepod_inference_memory():
    """Patch deepod's inference loops to move each batch off the GPU immediately.

    deepod's decision_function() always windows its input with stride=1
    regardless of the configured `stride` (see deepod.utils.utility.get_sub_seqs),
    including the internal call `fit()` makes on the *full* training set at the
    end of training. For large benchmarks like SWaT/WaDi this produces hundreds
    of thousands of overlapping windows. The stock inference loops append every
    batch's representation/score tensor to a Python list and only move them to
    CPU once, after the whole test_loader has been consumed - so peak GPU memory
    grows with the entire windowed dataset size, not with batch_size, which
    reliably causes CUDA OOM on SWaT/WaDi. Detaching and moving each batch to
    CPU right away bounds peak GPU memory by batch_size instead.

    Two independent code paths need this:
    - `BaseDeepAD._inference` (used by TcnED, DeepSVDDTS, COUTA, TranAD,
      TimesNet, AnomalyTransformer, DeepIsolationForestTS, ...).
    - `USAD.testing` (USAD overrides fit/decision_function entirely and has
      its own accumulation loop with the same bug).
    """
    global _DEEPOD_INFERENCE_PATCHED
    if _DEEPOD_INFERENCE_PATCHED:
        return

    from deepod.core.base_model import BaseDeepAD
    from deepod.models.time_series import USAD
    from tqdm import tqdm

    def _inference(self):
        self.net.eval()
        with torch.no_grad():
            z_lst = []
            score_lst = []

            if self.verbose >= 2:
                _iter_ = tqdm(self.test_loader, desc="testing: ")
            else:
                _iter_ = self.test_loader

            for batch_x in _iter_:
                batch_z, s = self.inference_forward(batch_x, self.net, self.criterion)
                z_lst.append(batch_z.detach().cpu())
                score_lst.append(s.detach().cpu())

        z = torch.cat(z_lst).numpy()
        scores = torch.cat(score_lst).numpy()

        return z, scores

    def _usad_testing(self, test_loader, alpha=0.5, beta=0.5):
        results = []
        for [batch] in test_loader:
            batch = batch.to(self.device)
            w1 = self.model.decoder1(self.model.encoder(batch))
            w2 = self.model.decoder2(self.model.encoder(w1))
            score = alpha * torch.mean((batch - w1) ** 2, axis=1) + beta * torch.mean((batch - w2) ** 2, axis=1)
            results.append(score.detach().cpu())
        return results

    BaseDeepAD._inference = _inference
    USAD.testing = _usad_testing
    _DEEPOD_INFERENCE_PATCHED = True
    LOGGER.info(
        "Patched deepod BaseDeepAD._inference and USAD.testing to move batches off "
        "the GPU immediately (avoids GPU memory accumulating over the whole dataset)."
    )


def _patch_deepod_windowing_memory(max_materialised_bytes: int = 4 * 1024**3):
    """Patch deepod's `get_sub_seqs` to window large arrays lazily instead of copying.

    `get_sub_seqs` materialises every window with
    `np.array([x_arr[i:i + seq_len] for i in seq_starts])`, i.e. `seq_len` copies
    of the input. `decision_function()` always windows with stride=1 - including
    the call `fit()` makes on the *full* training set - so for WaDi
    (784372 x 127, seq_len=200) that single array is 148 GiB of float64 and the
    process dies with an ArrayMemoryError before a single batch is scored.

    For the common case (no `start_discont`, plain strided starts) the same
    windows are exactly representable as a `sliding_window_view` over the input,
    which is a *view*: host memory stays O(n_samples x n_features) and only the
    per-batch copies made by the DataLoader are materialised. Small arrays keep
    the stock behaviour so nothing downstream sees a read-only/overlapping view
    unless it is the only way to fit in RAM.
    """
    global _DEEPOD_WINDOWING_PATCHED
    if _DEEPOD_WINDOWING_PATCHED:
        return

    from deepod.utils import utility

    original_get_sub_seqs = utility.get_sub_seqs

    def get_sub_seqs(x_arr, seq_len=100, stride=1, start_discont=np.array([])):
        n_windows = max(0, x_arr.shape[0] - seq_len + 1)
        n_windows = len(range(0, n_windows, stride)) if stride else n_windows
        materialised = n_windows * seq_len * int(np.prod(x_arr.shape[1:])) * x_arr.dtype.itemsize

        simple = len(start_discont) == 0 and stride is not None and stride >= 1
        if not simple or materialised <= max_materialised_bytes:
            return original_get_sub_seqs(x_arr, seq_len=seq_len, stride=stride, start_discont=start_discont)

        LOGGER.info(
            "get_sub_seqs: windowing %s lazily (seq_len=%s, stride=%s, %s windows, "
            "%.1f GiB if materialised)",
            x_arr.shape,
            seq_len,
            stride,
            n_windows,
            materialised / 1024**3,
        )
        # (n, seq_len, n_features) view over x_arr - no copy, no fancy indexing.
        view = np.lib.stride_tricks.sliding_window_view(x_arr, seq_len, axis=0)
        return np.moveaxis(view, -1, 1)[::stride]

    patched_modules = []
    for module_name, module in list(sys.modules.items()):
        if not module_name.startswith("deepod") or module is None:
            continue
        if getattr(module, "get_sub_seqs", None) is original_get_sub_seqs:
            module.get_sub_seqs = get_sub_seqs
            patched_modules.append(module_name)

    _DEEPOD_WINDOWING_PATCHED = True
    LOGGER.info(
        "Patched deepod get_sub_seqs in %s module(s) to window arrays larger than "
        "%.1f GiB as a zero-copy sliding-window view.",
        len(patched_modules),
        max_materialised_bytes / 1024**3,
    )


def _patch_deepod_couta_inference_batch(inference_batch_size: int = COUTA_INFERENCE_BATCH_SIZE):
    """Patch `COUTA.decision_function` to score with a larger batch size than it trains with.

    `decision_function()` reuses the *training* `self.batch_size` for its scoring
    DataLoader. Since scoring always windows with stride=1, WaDi yields 784372
    windows - at COUTA_TRAIN_BATCH_SIZE that is ~12k forward passes per scoring
    pass, and there are two: the one `fit()` makes on the training set and the
    one on the test set. Scoring keeps no autograd graph, so it can use a much
    larger batch than training safely.

    That cannot be fixed at the call site: the training-set pass happens *inside*
    `clf.fit()`, and raising `batch_size` before `fit()` would change training.
    So swap the batch size only for the duration of `decision_function`, then
    restore it. No autograd graph is kept during scoring, so a larger batch is
    cheap. The train-set scores only feed `threshold_`/`labels_`, which this
    benchmark does not use - batching them differently cannot move the reported
    metrics.
    """
    global _DEEPOD_COUTA_BATCH_PATCHED
    if _DEEPOD_COUTA_BATCH_PATCHED:
        return

    from deepod.models.time_series import COUTA

    original_decision_function = COUTA.decision_function

    def decision_function(self, X, *args, **kwargs):
        training_batch_size = self.batch_size
        self.batch_size = max(training_batch_size, inference_batch_size)
        if self.batch_size != training_batch_size:
            LOGGER.info(
                "COUTA.decision_function: scoring with batch_size=%s instead of the "
                "training batch_size=%s",
                self.batch_size,
                training_batch_size,
            )
        try:
            return original_decision_function(self, X, *args, **kwargs)
        finally:
            self.batch_size = training_batch_size

    COUTA.decision_function = decision_function
    _DEEPOD_COUTA_BATCH_PATCHED = True
    LOGGER.info(
        "Patched COUTA.decision_function to score with batch_size>=%s (training "
        "batch size is left untouched).",
        inference_batch_size,
    )


def build_classifier_factories(
    device: str = "cpu",
    random_state: int | None = None,
    seq_len: int = DEFAULT_SEQ_LEN,
    stride: int = DEFAULT_STRIDE,
):
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

    _patch_deepod_inference_memory()
    _patch_deepod_windowing_memory()
    _patch_deepod_couta_inference_batch()

    ts_kwargs = {"seq_len": seq_len, "stride": stride, "device": device, "random_state": random_state, "verbose": 1}

    return {
        "KNN": lambda: KNN(),
        "PCA": lambda: PCA(n_components=3, random_state=random_state, weighted=False),
        "COPOD": lambda: COPOD(),
        "IForest": lambda: IForest(random_state=random_state),
        "LOF": lambda: LOF(),
        "OCSVM": lambda: OCSVM(),
        "TimesNet": lambda: TimesNet(batch_size=16, **ts_kwargs),
        "DeepSVDD": lambda: DeepSVDDTS(**ts_kwargs),
        "USAD": lambda: USAD(batch_size=512, **ts_kwargs),
        "AnomalyTransformer": lambda: AnomalyTransformer(batch_size=16, **ts_kwargs),
        "TcnED": lambda: TcnED(batch_size=16, **ts_kwargs),
        "TranAD": lambda: TranAD(**ts_kwargs),
        "DeepIF": lambda: DeepIsolationForestTS(batch_size=256, **ts_kwargs),
        # batch_size must stay well above 1/neg_batch_ratio (deepod default 0.2):
        # at batch_size=2 COUTA generates int(2 * 0.2) == 0 synthetic negatives per
        # batch, which removes the calibration half of the objective and lets the
        # one-class term collapse onto the hypersphere centre (losses -> 0).
        "COUTA": lambda: COUTA(batch_size=COUTA_TRAIN_BATCH_SIZE, **ts_kwargs),
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
    """Build dataset specs for all QAD 100Hz traces (train_*.pkl / test_*.pkl / test_label_*.pkl).

    Mirrors the raw layout consumed by ``data/qad_provider.py``: pickled pandas
    payloads sit directly in ``data_dir/QAD/raw`` (one train/test/label triple per
    numeric trace id).
    """
    qad_base_dir = _resolve_qad_raw_dir()
    qad_datasets = []

    # Discover all train_*.pkl files and create specs for each.
    train_files = sorted(glob.glob(str(qad_base_dir / "train_*.pkl")))

    for train_file in train_files:
        file_name = Path(train_file).name
        match = re.match(r"^train_(\d+)\.pkl$", file_name)
        if match is None:
            continue

        dataset_num = match.group(1)
        qad_datasets.append({
            "dataset_id": f"qad_{dataset_num}",
            "data_dir": qad_base_dir,
            "train_file": f"train_{dataset_num}.pkl",
            "test_file": f"test_{dataset_num}.pkl",
            "label_file": f"test_label_{dataset_num}.pkl",
            "file_format": "qad_pkl",
            # Raw traces are 100 Hz; keep every n-th row (overridable via
            # --benchmark-decimation / BENCHMARK_WINDOW_DEFAULTS).
            "decimation_factor": BENCHMARK_WINDOW_DEFAULTS["QAD"]["decimation"],
        })

    return qad_datasets


def _resolve_qad_raw_dir():
    """Locate the QAD raw folder holding the pickled traces.

    Newer datasets store the pickles straight in ``data_dir/QAD/raw``; older
    checkouts kept them under a ``qad_clean_pkl_100Hz`` subfolder, so fall back to
    that when the flat layout has no ``train_*.pkl``.
    """
    flat = ROOT_DIR / "data_dir" / "QAD" / "raw"
    if list(flat.glob("train_*.pkl")):
        return flat

    legacy = flat / "qad_clean_pkl_100Hz"
    if list(legacy.glob("train_*.pkl")):
        LOGGER.warning("No QAD pickles in '%s'. Falling back to '%s'.", flat, legacy)
        return legacy

    return flat


class _QADCompatUnpickler(pickle.Unpickler):
    """Load QAD pickles written with numpy>=2 under an older numpy at runtime."""

    _MODULE_REMAPS = {
        "numpy._core.numeric": "numpy.core.numeric",
        "numpy._core.multiarray": "numpy.core.multiarray",
        "numpy._core.umath": "numpy.core.umath",
    }

    def find_class(self, module: str, name: str):
        module = self._MODULE_REMAPS.get(module, module)
        if module.startswith("numpy._core."):
            module = module.replace("numpy._core.", "numpy.core.", 1)
        return super().find_class(module, name)


def _load_qad_pkl(dataset_path: Path, is_label: bool = False):
    with open(dataset_path, "rb") as f:
        loaded_data = _QADCompatUnpickler(f).load()

    if isinstance(loaded_data, pd.Series):
        data = loaded_data.to_frame(name="labels")
    elif isinstance(loaded_data, pd.DataFrame):
        data = loaded_data.copy()
    else:
        data = pd.DataFrame(loaded_data)

    # Label files should always expose a canonical `labels` column.
    if is_label and len(data.columns) == 1 and "labels" not in data.columns:
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


def _wandb_is_available(args) -> bool:
    return wandb is not None and not getattr(args, "wandb_disabled", False) and getattr(args, "wandb_mode", "online") != "disabled"


def _wandb_json_safe(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): _wandb_json_safe(val) for key, val in value.items()}
    if isinstance(value, (list, tuple, set)):
        seq = list(value)
        return [_wandb_json_safe(item) for item in seq]
    return value


def _wandb_table_from_dataframe(df: pd.DataFrame):
    if wandb is None:
        return None
    return wandb.Table(dataframe=df.reset_index(drop=True))


def _wandb_summary_from_dataframe(df: pd.DataFrame, prefix: str, key_cols: list[str], metric_cols: list[str]):
    summary = {}
    for _, row in df.iterrows():
        key_suffix = "/".join(str(row[col]) for col in key_cols)
        for metric in metric_cols:
            value = row.get(metric)
            if isinstance(value, (int, float, np.integer, np.floating)):
                summary[f"{prefix}/{key_suffix}/{metric}"] = float(value)
    return summary


def _wandb_init_run(
    args,
    runtime_device: str,
    run_number: int,
    run_seed: int,
    clf_name: str,
    selected_benchmarks: list[str],
    selected_classifiers: list[str],
    benchmark_window_settings: dict[str, dict[str, int]],
    benchmark_dataset_counts: dict[str, int],
    benchmark_dataset_ids: dict[str, list[str]],
    output_paths: dict[str, Path],
    classifier_defaults: dict[str, object],
):
    if not _wandb_is_available(args):
        if wandb is None:
            LOGGER.info("W&B logging disabled because wandb is not installed.")
        else:
            LOGGER.info("W&B logging disabled via CLI.")
        return None

    try:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        benchmark_slug = "-".join(selected_benchmarks)
        classifier_slug = "-".join(selected_classifiers)
        base_run_name = args.wandb_name or f"baseline__{benchmark_slug}__{classifier_slug}__r{args.runs}__seed{args.seed}__{timestamp}"
        run_name = f"{base_run_name}__clf={clf_name}__run{run_number:02d}_seed{run_seed}"
        run_group = args.wandb_group or f"benchmarks={benchmark_slug}__classifiers={classifier_slug}"
        tags = list(dict.fromkeys((args.wandb_tags or []) + selected_benchmarks + [clf_name, runtime_device]))

        config = {
            "args": _wandb_json_safe(vars(args)),
            "run_context": {
                "run_number": run_number,
                "run_seed": run_seed,
                "clf_name": clf_name,
            },
            "selection": {
                "benchmarks": selected_benchmarks,
                "classifiers": selected_classifiers,
            },
            "runtime": {
                "device": runtime_device,
                "python": sys.version.split()[0],
                "torch": torch.__version__,
                "cuda_available": bool(torch.cuda.is_available()),
            },
            "benchmark_window_settings": benchmark_window_settings,
            "benchmark_dataset_counts": benchmark_dataset_counts,
            "benchmark_dataset_ids": benchmark_dataset_ids,
            "classifier_defaults": classifier_defaults,
            "output_paths": {name: str(path) for name, path in output_paths.items()},
            "wandb": {
                "project": args.wandb_project,
                "entity": args.wandb_entity,
                "name": run_name,
                "group": run_group,
                "mode": args.wandb_mode,
            },
        }

        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            group=run_group,
            tags=tags,
            mode=args.wandb_mode,
            config=config,
            reinit=True,
        )
        wandb.define_metric("global_step")
        wandb.define_metric("evaluation/*", step_metric="global_step")
        wandb.define_metric("summary/*", summary="last")
        LOGGER.info("Initialized W&B run: project=%s, name=%s, group=%s, mode=%s", args.wandb_project, run_name, run_group, args.wandb_mode)
        return run
    except Exception as exc:
        LOGGER.warning("W&B initialization failed; continuing without W&B logging: %s", exc, exc_info=True)
        return None


def _wandb_log_evaluation(run, global_step: int, row: dict, run_number: int, seed: int, elapsed_seconds: float, status: str, metrics: dict | None = None):
    if run is None:
        return True

    try:
        payload = {
            "global_step": global_step,
            "evaluation/run_number": run_number,
            "evaluation/seed": seed,
            "evaluation/duration_sec": float(elapsed_seconds),
            "evaluation/success": 1.0 if status == "success" else 0.0,
            "evaluation/failure": 1.0 if status != "success" else 0.0,
        }
        combined_metrics = {}
        if metrics:
            combined_metrics.update(metrics)
        combined_metrics.update({key: value for key, value in row.items() if key not in {"benchmark", "dataset_id", "clf_name"}})
        for metric_name, metric_value in combined_metrics.items():
            if isinstance(metric_value, (int, float, np.integer, np.floating)):
                payload[f"evaluation/{metric_name}"] = float(metric_value)
        run.log(payload, step=global_step)
        return True
    except Exception as exc:
        LOGGER.warning("W&B evaluation logging failed; disabling further W&B logging: %s", exc, exc_info=True)
        return False


def _wandb_log_final_outputs(run, per_dataset_df, per_run_df, macro_df, per_dataset_summary_df, macro_summary_df, runtime_df, failed_runs, output_paths):
    if run is None:
        return True

    try:
        summary_payload = {
            "summary/num_successful_rows": int(len(per_dataset_df)),
            "summary/num_unique_runs": int(per_run_df["run"].nunique()) if (not per_run_df.empty and "run" in per_run_df.columns) else 0,
            "summary/num_failed_runs": int(len(failed_runs)),
            "summary/num_runtime_rows": int(len(runtime_df)) if runtime_df is not None else 0,
        }

        summary_payload.update(_wandb_summary_from_dataframe(macro_df, "summary/macro", ["benchmark", "clf_name"], ["auc_roc", "auc_pr", "f1"]))
        summary_payload.update(_wandb_summary_from_dataframe(per_dataset_summary_df, "summary/per_dataset_mean_std", ["benchmark", "dataset_id", "clf_name"], ["auc_roc_mean", "auc_roc_std", "auc_pr_mean", "auc_pr_std", "f1_mean", "f1_std", "num_runs"]))
        summary_payload.update(_wandb_summary_from_dataframe(macro_summary_df, "summary/macro_mean_std", ["benchmark", "clf_name"], ["auc_roc_mean", "auc_roc_std", "auc_pr_mean", "auc_pr_std", "f1_mean", "f1_std", "num_runs"]))

        for key, value in summary_payload.items():
            run.summary[key] = value

        tables = {
            "tables/per_dataset": per_dataset_df,
            "tables/per_run": per_run_df,
            "tables/macro": macro_df.reset_index(),
            "tables/per_dataset_mean_std": per_dataset_summary_df,
            "tables/macro_mean_std": macro_summary_df,
        }
        if runtime_df is not None and not runtime_df.empty:
            tables["tables/runtime"] = runtime_df

        run.log({name: _wandb_table_from_dataframe(df) for name, df in tables.items() if df is not None})

        artifact = wandb.Artifact(name=f"{run.name}-results".replace("=", "_").replace(":", "-"), type="results")
        for path in output_paths.values():
            if path.exists():
                artifact.add_file(str(path))
        run.log_artifact(artifact)
        return True
    except Exception as exc:
        LOGGER.warning("Final W&B logging failed; continuing without W&B artifacts: %s", exc, exc_info=True)
        return False


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
    parser.add_argument(
        "--seq-len-default",
        type=positive_int,
        default=DEFAULT_SEQ_LEN,
        help="Default sequence length for time-series deep models; stride is set equal to seq_len.",
    )
    parser.add_argument(
        "--benchmark-seq-lens",
        type=str,
        default="",
        help="Optional benchmark-specific seq lens, e.g. 'SWaT:200,WaDi:128'.",
    )
    parser.add_argument(
        "--stride-default",
        type=positive_int,
        default=DEFAULT_STRIDE,
        help="Default window stride for time-series deep models.",
    )
    parser.add_argument(
        "--benchmark-strides",
        type=str,
        default="",
        help="Optional benchmark-specific strides, e.g. 'QAD:20'.",
    )
    parser.add_argument(
        "--benchmark-decimation",
        type=str,
        default="",
        help=(
            "Optional benchmark-specific temporal decimation (keep every n-th raw row of "
            "train/test/labels), e.g. 'QAD:10'. Only honoured by benchmarks whose loader "
            "supports it (currently QAD)."
        ),
    )
    parser.add_argument(
        "--score-smoothing-window-default",
        type=int,
        default=DEFAULT_SCORE_SMOOTHING,
        help=(
            "Half-width (time steps) of the moving average applied to the test scores "
            "before computing metrics (same smoothing as anomaly_detection.py); 0 disables."
        ),
    )
    parser.add_argument(
        "--benchmark-score-smoothing",
        type=str,
        default="",
        help="Optional benchmark-specific score smoothing half-widths, e.g. 'QAD:10'.",
    )
    parser.add_argument(
        "--no-benchmark-window-defaults",
        action="store_true",
        help=(
            "Ignore BENCHMARK_WINDOW_DEFAULTS (per-benchmark seq_len/stride/decimation/"
            "smoothing presets) and use only the global defaults plus explicit overrides."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help=(
            "Directory for the baselines*.csv result files (relative paths resolve against the "
            "repository root). Rows are appended, so use a fresh directory when the evaluation "
            "protocol changes (e.g. QAD windows/smoothing) to avoid mixing results."
        ),
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="latent-sde-on-hs-baselines",
        help="Weights & Biases project name.",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Optional W&B entity / team name.",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Optional explicit W&B run name. If omitted, a descriptive name is generated.",
    )
    parser.add_argument(
        "--wandb-group",
        type=str,
        default=None,
        help="Optional W&B group name. Defaults to the selected benchmark/classifier combination.",
    )
    parser.add_argument(
        "--wandb-tags",
        nargs="*",
        default=[],
        help="Optional W&B tags.",
    )
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default="online",
        help="W&B mode.",
    )
    parser.add_argument(
        "--wandb-disabled",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable W&B logging entirely.",
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


def _parse_benchmark_int_mapping(
    mapping_csv: str,
    available_benchmarks: dict[str, list[dict]],
    option_name: str,
    value_name: str,
    min_value: int = 1,
):
    """Parse 'BENCHMARK:INT,BENCHMARK:INT' CLI mappings."""
    if mapping_csv is None or not mapping_csv.strip():
        return {}

    parsed = {}
    for raw_entry in mapping_csv.split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        if ":" not in entry:
            raise ValueError(
                f"Invalid {option_name} entry '{entry}'. Expected format BENCHMARK:{value_name.upper()}."
            )

        benchmark, value_text = entry.split(":", 1)
        benchmark = benchmark.strip()
        value_text = value_text.strip()
        if benchmark not in available_benchmarks:
            raise ValueError(
                f"Unknown benchmark '{benchmark}' in {option_name}. "
                f"Available: {list(available_benchmarks.keys())}"
            )

        try:
            value = int(value_text)
        except ValueError as exc:
            raise ValueError(
                f"Invalid {value_name} '{value_text}' for benchmark '{benchmark}'. Must be an integer."
            ) from exc

        if value < min_value:
            raise ValueError(
                f"Invalid {value_name} '{value}' for benchmark '{benchmark}'. Must be >= {min_value}."
            )
        parsed[benchmark] = value

    return parsed


def _parse_benchmark_seq_lens(mapping_csv: str, available_benchmarks: dict[str, list[dict]]):
    return _parse_benchmark_int_mapping(mapping_csv, available_benchmarks, "--benchmark-seq-lens", "seq_len")


def resolve_benchmark_window_settings(args, available_benchmarks: dict[str, list[dict]]) -> dict[str, dict[str, int]]:
    """Resolve seq_len / stride / decimation / score smoothing / eval window per benchmark.

    Precedence: explicit CLI mapping > BENCHMARK_WINDOW_DEFAULTS (unless
    --no-benchmark-window-defaults) > global CLI default.
    """
    seq_len_overrides = _parse_benchmark_seq_lens(args.benchmark_seq_lens, available_benchmarks)
    stride_overrides = _parse_benchmark_int_mapping(args.benchmark_strides, available_benchmarks, "--benchmark-strides", "stride")
    decimation_overrides = _parse_benchmark_int_mapping(args.benchmark_decimation, available_benchmarks, "--benchmark-decimation", "decimation")
    smoothing_overrides = _parse_benchmark_int_mapping(
        args.benchmark_score_smoothing, available_benchmarks, "--benchmark-score-smoothing", "score_smoothing_window", min_value=0
    )

    settings = {}
    for benchmark_name in available_benchmarks:
        presets = {} if args.no_benchmark_window_defaults else BENCHMARK_WINDOW_DEFAULTS.get(benchmark_name, {})
        settings[benchmark_name] = {
            "seq_len": seq_len_overrides.get(benchmark_name, presets.get("seq_len", args.seq_len_default)),
            "stride": stride_overrides.get(benchmark_name, presets.get("stride", args.stride_default)),
            "decimation": decimation_overrides.get(benchmark_name, presets.get("decimation", DEFAULT_DECIMATION)),
            "score_smoothing_window": smoothing_overrides.get(
                benchmark_name, presets.get("score_smoothing_window", args.score_smoothing_window_default)
            ),
            "eval_window": presets.get("eval_window", DEFAULT_EVAL_WINDOW),
        }
    return settings


def load_dataset(spec, max_train_samples=None, max_test_samples=None, decimation_factor=None):
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
    # Handle QAD pickle files (see data/qad_provider.py for the raw layout).
    elif spec.get("file_format") == "qad_pkl":
        x_train_df = _load_qad_pkl(data_dir / spec["train_file"])
        x_test_df = _load_qad_pkl(data_dir / spec["test_file"])
        y_test_df = _load_qad_pkl(data_dir / spec["label_file"], is_label=True)

        # The provider drops the non-sensor `Enable` flag before training; keep
        # the baseline feature space identical.
        x_train_df = x_train_df.drop(columns=["Enable"], errors="ignore")
        x_test_df = x_test_df.drop(columns=["Enable"], errors="ignore")

        # Temporal decimation (raw QAD is 100 Hz). Mirrors the
        # `decimation_factor` of data/qad_provider.py so both pipelines see the
        # same sampling rate; labels are decimated identically.
        factor = decimation_factor if decimation_factor is not None else spec.get("decimation_factor", 1)
        factor = max(1, int(factor))
        if factor > 1:
            x_train_df = x_train_df[::factor]
            x_test_df = x_test_df[::factor]
            y_test_df = y_test_df[::factor]
        LOGGER.info("[%s] QAD decimation factor %d -> %d train rows, %d test rows", dataset_id, factor, len(x_train_df), len(x_test_df))

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


def _warn_if_couta_calibration_disabled(clf, benchmark_name, dataset_id):
    """Check that COUTA will actually generate synthetic negatives.

    COUTA's calibration term is built from `int(batch_size * neg_batch_ratio)`
    negatives per batch. When that rounds to 0 the term contributes nothing, the
    model degenerates to plain Deep SVDD, and the one-class objective collapses
    onto the centre - which is what produced `loss: 0.000000, loss_oc: 0.000000,
    val_loss: 0.000000` on SWaT/WaDi. The batch size is chosen to avoid this, but
    the ratio lives in deepod, so verify it here instead of assuming.
    """
    ratio = getattr(clf, "neg_batch_ratio", None)
    batch_size = getattr(clf, "batch_size", None)
    if ratio is None or batch_size is None:
        LOGGER.warning(
            "[%s/%s] COUTA: cannot read neg_batch_ratio/batch_size; unable to verify "
            "that the calibration term is active",
            benchmark_name,
            dataset_id,
        )
        return

    n_negatives = int(batch_size * ratio)
    if n_negatives < 1:
        LOGGER.error(
            "[%s/%s] COUTA: batch_size=%s x neg_batch_ratio=%s yields %d synthetic "
            "negatives per batch. The calibration term is inactive and the one-class "
            "objective will collapse. Raise COUTA_TRAIN_BATCH_SIZE above %d.",
            benchmark_name,
            dataset_id,
            batch_size,
            ratio,
            n_negatives,
            int(np.ceil(1.0 / ratio)) if ratio > 0 else 0,
        )
    else:
        LOGGER.info(
            "[%s/%s] COUTA: %d synthetic negatives per batch (batch_size=%s, neg_batch_ratio=%s)",
            benchmark_name,
            dataset_id,
            n_negatives,
            batch_size,
            ratio,
        )


def _move_classifier_to_cpu(clf, clf_name, benchmark_name, dataset_id):
    """Move a fitted deepod model onto the CPU so scoring can continue after a CUDA OOM.

    Deliberately does *not* call `configure_gpu(None)`: that sets
    CUDA_VISIBLE_DEVICES="" process-wide and never restores it, so one model's
    fallback would quietly push every model evaluated after it onto the CPU too.
    Moving the modules and retargeting `clf.device` is enough and stays local to
    this classifier.
    """
    moved = []
    for attribute in ("net", "model"):  # deepod models use one or the other
        module = getattr(clf, attribute, None)
        if isinstance(module, torch.nn.Module):
            module.to("cpu")
            moved.append(attribute)
    clf.device = "cpu"
    LOGGER.warning(
        "[%s/%s] %s moved to CPU for inference (modules: %s)",
        benchmark_name,
        dataset_id,
        clf_name,
        ", ".join(moved) if moved else "none found",
    )


def _score_with_oom_recovery(clf, clf_name, x_test, benchmark_name, dataset_id):
    """Score `x_test`, recovering from CUDA OOM by shrinking the batch, then by CPU.

    Previously only USAD retried on OOM, and TcnED was unconditionally moved to
    the CPU before scoring whether it needed to be or not - which made TcnED far
    slower than necessary on every benchmark to work around a failure on one.
    Any deep model can OOM here (scoring windows with stride=1, so it sees far
    more samples than training did), and a model that OOMs loses its entire cell
    in the results table, so the recovery ladder now applies to all of them:
    halve the batch size down to MIN_INFERENCE_BATCH_SIZE, then fall back to the
    CPU once, then give up.
    """
    can_shrink = hasattr(clf, "batch_size")
    on_cpu = str(getattr(clf, "device", "cpu")).startswith("cpu")
    last_error = None

    for attempt in range(1, INFERENCE_OOM_MAX_ATTEMPTS + 1):
        try:
            with torch.inference_mode():
                return np.asarray(clf.decision_function(x_test)).ravel()
        except RuntimeError as error:
            last_error = error
            message = str(error).lower()
            if "cuda" not in message or "out of memory" not in message:
                raise

            old_batch_size = int(getattr(clf, "batch_size")) if can_shrink else 0
            new_batch_size = max(MIN_INFERENCE_BATCH_SIZE, old_batch_size // 2)

            if can_shrink and new_batch_size < old_batch_size:
                LOGGER.warning(
                    "[%s/%s] %s inference OOM on attempt %d/%d. Reducing batch_size: %d -> %d and retrying.",
                    benchmark_name,
                    dataset_id,
                    clf_name,
                    attempt,
                    INFERENCE_OOM_MAX_ATTEMPTS,
                    old_batch_size,
                    new_batch_size,
                )
                clf.batch_size = new_batch_size
            elif not on_cpu:
                LOGGER.warning(
                    "[%s/%s] %s inference OOM on attempt %d/%d at batch_size=%s; falling back to CPU.",
                    benchmark_name,
                    dataset_id,
                    clf_name,
                    attempt,
                    INFERENCE_OOM_MAX_ATTEMPTS,
                    getattr(clf, "batch_size", "n/a"),
                )
                _move_classifier_to_cpu(clf, clf_name, benchmark_name, dataset_id)
                on_cpu = True
            else:
                raise

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    raise last_error


def _warn_on_degenerate_scores(scores, clf_name, benchmark_name, dataset_id):
    """Flag scores that carry no ranking information.

    One-class models (COUTA, DeepSVDD) fail by collapsing: the encoder learns the
    constant map onto the hypersphere centre, every training loss goes to ~0, and
    decision_function() then returns near-identical scores for every window. The
    run still "succeeds" and still produces an AUC - a meaningless one, decided by
    floating-point noise and whatever tie-breaking the metric does. Nothing in the
    pipeline noticed, so this is checked explicitly rather than left to be spotted
    as a chance-level number in the results table weeks later.
    """
    finite = scores[np.isfinite(scores)]
    if finite.size != scores.size:
        LOGGER.error(
            "[%s/%s] %s produced %d non-finite scores out of %d",
            benchmark_name,
            dataset_id,
            clf_name,
            scores.size - finite.size,
            scores.size,
        )
    if finite.size == 0:
        return

    spread = float(np.std(finite))
    scale = max(float(np.mean(np.abs(finite))), np.finfo(np.float64).tiny)
    if spread / scale < DEGENERATE_SCORE_REL_STD:
        LOGGER.error(
            "[%s/%s] %s scores are degenerate (std=%.3e, mean|score|=%.3e, "
            "%d distinct values): the model has almost certainly collapsed and "
            "the metrics below are meaningless.",
            benchmark_name,
            dataset_id,
            clf_name,
            spread,
            scale,
            np.unique(finite).size,
        )


def evaluate_classifier_on_dataset(
    clf_name,
    clf,
    x_train,
    x_test,
    y_test,
    benchmark_name,
    dataset_id,
    score_smoothing_window: int = DEFAULT_SCORE_SMOOTHING,
    eval_window_length: int = DEFAULT_EVAL_WINDOW,
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

    if clf_name == "COUTA":
        _warn_if_couta_calibration_disabled(clf, benchmark_name, dataset_id)

    clf.fit(x_train)
    LOGGER.info("[%s/%s] fitted %s", benchmark_name, dataset_id, clf_name)
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.debug("[%s/%s] torch cleanup skipped", benchmark_name, dataset_id, exc_info=True)


    if clf_name == "USAD":
        # USAD trains at batch_size=512, which is far too large for scoring:
        # decision_function() windows with stride=1 and so sees orders of
        # magnitude more samples than training did.
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

    y_test_scores = _score_with_oom_recovery(clf, clf_name, x_test, benchmark_name, dataset_id)
    _warn_on_degenerate_scores(y_test_scores, clf_name, benchmark_name, dataset_id)

    if y_test_scores.shape[0] != y_test.shape[0]:
        aligned_len = min(y_test_scores.shape[0], y_test.shape[0])
        LOGGER.warning(
            "[%s/%s] %s score/label length mismatch (%d vs %d); truncating both to %d",
            benchmark_name, dataset_id, clf_name, y_test_scores.shape[0], y_test.shape[0], aligned_len,
        )
        y_test_scores = y_test_scores[:aligned_len]
        y_test = y_test[:aligned_len]

    if score_smoothing_window and score_smoothing_window > 0:
        # Same moving-average post-processing as anomaly_detection.py's
        # normalise_scores(), so model and baselines are scored identically.
        y_test_scores = smooth_scores(y_test_scores, score_smoothing_window)
        LOGGER.info("[%s/%s] %s scores smoothed with half-window %d", benchmark_name, dataset_id, clf_name, score_smoothing_window)

    metric_results = get_ts_eval(y_test_scores, y_test, window_length=eval_window_length)

    del clf
    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        LOGGER.debug("[%s/%s] torch cleanup skipped", benchmark_name, dataset_id, exc_info=True)


    LOGGER.info(
        "[%s/%s] %s: auc_roc=%.6f, auc_pr=%.6f, f1=%.6f",
        benchmark_name,
        dataset_id,
        clf_name,
        metric_results["auc_roc"],
        metric_results["auc_pr"],
        metric_results["f1"],
    )

    return {
        "benchmark": benchmark_name,
        "dataset_id": dataset_id,
        "clf_name": clf_name,
        **metric_results,
    }, metric_results


def macro_average(per_dataset_df):
    macro_df = (
        per_dataset_df.groupby(["benchmark", "clf_name"], as_index=False)[["auc_roc", "auc_pr", "f1"]]
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
    metrics = ["auc_roc", "auc_pr", "f1"]
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
    for metric in ["auc_roc", "auc_pr", "f1"]:
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
    benchmark_window_settings = resolve_benchmark_window_settings(args, BENCHMARK_DATASETS)

    classifier_factories = build_classifier_factories(
        device=runtime_device,
        random_state=args.seed,
        seq_len=args.seq_len_default,
        stride=args.stride_default,
    )

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = ROOT_DIR / output_dir
    os.makedirs(output_dir, exist_ok=True)
    LOGGER.info("Writing result CSVs to %s", output_dir)
    per_dataset_path = output_dir / "baselines_per_dataset.csv"
    macro_path = output_dir / "baselines.csv"
    per_dataset_summary_path = output_dir / "baselines_per_dataset_mean_std.csv"
    macro_summary_path = output_dir / "baselines_macro_mean_std.csv"
    runtime_path = output_dir / "baselines_runtime_per_dataset.csv"

    LOGGER.info("Starting baseline evaluation")
    LOGGER.info(
        "Arguments: benchmarks=%s, classifiers=%s, max_train_samples=%s, max_test_samples=%s, runs=%s, seed=%s, device=%s, seq_len_default=%s, stride_default=%s, score_smoothing_window_default=%s",
        args.benchmarks,
        args.classifiers,
        args.max_train_samples,
        args.max_test_samples,
        args.runs,
        args.seed,
        runtime_device,
        args.seq_len_default,
        args.stride_default,
        args.score_smoothing_window_default,
    )

    selected_benchmarks = _select_keys(BENCHMARK_DATASETS, args.benchmarks)
    selected_classifiers = _select_keys(classifier_factories, args.classifiers)
    for benchmark_name in selected_benchmarks:
        LOGGER.info("Window settings for %s: %s", benchmark_name, benchmark_window_settings[benchmark_name])

    LOGGER.info("Selected benchmarks: %s", selected_benchmarks)
    LOGGER.info("Selected classifiers: %s", selected_classifiers)
    for benchmark_name in selected_benchmarks:
        dataset_count = len(BENCHMARK_DATASETS[benchmark_name])
        if dataset_count == 0:
            LOGGER.warning("Benchmark %s has no discovered datasets", benchmark_name)
        else:
            LOGGER.info("Benchmark %s has %d dataset(s)", benchmark_name, dataset_count)

    benchmark_dataset_counts = {benchmark_name: len(BENCHMARK_DATASETS[benchmark_name]) for benchmark_name in selected_benchmarks}
    benchmark_dataset_ids = {
        benchmark_name: [spec["dataset_id"] for spec in BENCHMARK_DATASETS[benchmark_name]]
        for benchmark_name in selected_benchmarks
    }
    classifier_defaults = {
        "device": runtime_device,
        "random_state": args.seed,
        "seq_len_default": args.seq_len_default,
        "stride_default": args.stride_default,
        "score_smoothing_window_default": args.score_smoothing_window_default,
        "runs": args.runs,
        "max_train_samples": args.max_train_samples,
        "max_test_samples": args.max_test_samples,
    }
    output_paths = {
        "per_dataset": per_dataset_path,
        "macro": macro_path,
        "per_dataset_mean_std": per_dataset_summary_path,
        "macro_mean_std": macro_summary_path,
        "runtime": runtime_path,
    }
    per_dataset_rows = []
    per_run_rows = []
    failed_runs = []
    runtime_rows = []
    for run_idx in range(args.runs):
        run_number = run_idx + 1
        run_seed = args.seed + run_idx
        set_round_context(run_number, args.runs)
        set_global_seed(run_seed)
        LOGGER.info("Starting run %d/%d with seed=%d", run_number, args.runs, run_seed)

        for clf_name in selected_classifiers:
            for benchmark_name in selected_benchmarks:
                window_settings = benchmark_window_settings[benchmark_name]
                seq_len_for_benchmark = window_settings["seq_len"]
                stride_for_benchmark = window_settings["stride"]
                single_benchmark_dataset_counts = {benchmark_name: benchmark_dataset_counts[benchmark_name]}
                single_benchmark_dataset_ids = {benchmark_name: benchmark_dataset_ids[benchmark_name]}

                run_wandb = _wandb_init_run(
                    args=args,
                    runtime_device=runtime_device,
                    run_number=run_number,
                    run_seed=run_seed,
                    clf_name=clf_name,
                    selected_benchmarks=[benchmark_name],
                    selected_classifiers=selected_classifiers,
                    benchmark_window_settings={benchmark_name: window_settings},
                    benchmark_dataset_counts=single_benchmark_dataset_counts,
                    benchmark_dataset_ids=single_benchmark_dataset_ids,
                    output_paths=output_paths,
                    classifier_defaults=classifier_defaults,
                )
                run_wandb_step = 0
                run_per_dataset_rows = []
                run_per_run_rows = []
                run_failed_runs = []
                run_runtime_rows = []

                try:
                    LOGGER.info(
                        "Run %d/%d clf=%s benchmark=%s seq_len=%d stride=%d decimation=%d score_smoothing=%d",
                        run_number, args.runs, clf_name, benchmark_name, seq_len_for_benchmark,
                        stride_for_benchmark, window_settings["decimation"], window_settings["score_smoothing_window"],
                    )
                    clf_factories_for_benchmark = build_classifier_factories(
                        device=runtime_device,
                        random_state=run_seed,
                        seq_len=seq_len_for_benchmark,
                        stride=stride_for_benchmark,
                    )
                    clf_factory = clf_factories_for_benchmark[clf_name]
                    dataset_specs = BENCHMARK_DATASETS[benchmark_name]
                    for dataset_spec in dataset_specs:
                        dataset_id = dataset_spec["dataset_id"]
                        started_at = time.perf_counter()
                        try:
                            x_train, x_test, y_test = load_dataset(
                                dataset_spec,
                                max_train_samples=args.max_train_samples,
                                max_test_samples=args.max_test_samples,
                                decimation_factor=window_settings["decimation"],
                            )
                            clf = clf_factory()
                            row, metric_results = evaluate_classifier_on_dataset(
                                clf_name,
                                clf,
                                x_train,
                                x_test,
                                y_test,
                                benchmark_name,
                                dataset_id,
                                score_smoothing_window=window_settings["score_smoothing_window"],
                                eval_window_length=window_settings["eval_window"],
                            )
                            per_dataset_rows.append(row)
                            run_per_dataset_rows.append(row)

                            row_with_run = {**row, "run": run_number, "seed": run_seed}
                            per_run_rows.append(row_with_run)
                            run_per_run_rows.append(row_with_run)
                            append_df_to_csv(pd.DataFrame([row]), per_dataset_path, index=False)

                            elapsed_seconds = time.perf_counter() - started_at
                            run_wandb_step += 1
                            if run_wandb is not None and not _wandb_log_evaluation(
                                run_wandb,
                                run_wandb_step,
                                row,
                                run_number,
                                run_seed,
                                elapsed_seconds,
                                status="success",
                                metrics=metric_results,
                            ):
                                run_wandb = None

                            runtime_row = {
                                "run": run_number,
                                "seed": run_seed,
                                "benchmark": benchmark_name,
                                "dataset_id": dataset_id,
                                "clf_name": clf_name,
                                "status": "success",
                                "duration_sec": elapsed_seconds,
                                "error_type": "",
                                "error_message": "",
                            }
                            runtime_rows.append(runtime_row)
                            run_runtime_rows.append(runtime_row)
                            append_df_to_csv(pd.DataFrame([runtime_row]), runtime_path, index=False)
                            LOGGER.info(
                                "[seed=%d][%s/%s] %s completed in %.3f sec",
                                run_seed, benchmark_name, dataset_id, clf_name, elapsed_seconds,
                            )
                        except Exception:
                            elapsed_seconds = time.perf_counter() - started_at
                            error_type, error_message, _ = sys.exc_info()
                            runtime_row = {
                                "run": run_number,
                                "seed": run_seed,
                                "benchmark": benchmark_name,
                                "dataset_id": dataset_id,
                                "clf_name": clf_name,
                                "status": "failed",
                                "duration_sec": elapsed_seconds,
                                "error_type": error_type.__name__ if error_type is not None else "Exception",
                                "error_message": str(error_message) if error_message is not None else "",
                            }
                            runtime_rows.append(runtime_row)
                            run_runtime_rows.append(runtime_row)
                            append_df_to_csv(pd.DataFrame([runtime_row]), runtime_path, index=False)

                            run_wandb_step += 1
                            if run_wandb is not None and not _wandb_log_evaluation(
                                run_wandb,
                                run_wandb_step,
                                {"auc_roc": np.nan, "auc_pr": np.nan, "f1": np.nan},
                                run_number,
                                run_seed,
                                elapsed_seconds,
                                status="failed",
                                metrics=None,
                            ):
                                run_wandb = None

                            failed_run = (run_number, run_seed, benchmark_name, dataset_id, clf_name)
                            failed_runs.append(failed_run)
                            run_failed_runs.append(failed_run)
                            LOGGER.exception(
                                "[seed=%d][%s/%s] %s failed",
                                run_seed, benchmark_name, dataset_id, clf_name,
                            )
                finally:
                    if run_wandb is not None:
                        run_per_dataset_df = pd.DataFrame(run_per_dataset_rows)
                        run_per_run_df = pd.DataFrame(run_per_run_rows)
                        run_runtime_df = pd.DataFrame(run_runtime_rows)

                        if not run_per_dataset_df.empty:
                            run_macro_df = macro_average(run_per_dataset_df)
                            run_per_dataset_summary_df = aggregate_mean_std(run_per_run_df, ["benchmark", "dataset_id", "clf_name"])
                            run_per_run_macro_df = (
                                run_per_run_df.groupby(["run", "benchmark", "clf_name"], as_index=False)[["auc_roc", "auc_pr", "f1"]]
                                .mean()
                            )
                            run_macro_summary_df = aggregate_mean_std(run_per_run_macro_df, ["benchmark", "clf_name"])
                        else:
                            run_macro_df = pd.DataFrame(columns=["benchmark", "clf_name", "auc_roc", "auc_pr", "f1", "num_datasets"])
                            run_per_dataset_summary_df = pd.DataFrame(columns=["benchmark", "dataset_id", "clf_name", "auc_roc_mean", "auc_roc_std", "auc_pr_mean", "auc_pr_std", "f1_mean", "f1_std", "num_runs"])
                            run_macro_summary_df = pd.DataFrame(columns=["benchmark", "clf_name", "auc_roc_mean", "auc_roc_std", "auc_pr_mean", "auc_pr_std", "f1_mean", "f1_std", "num_runs"])

                        if not _wandb_log_final_outputs(
                            run_wandb,
                            run_per_dataset_df,
                            run_per_run_df,
                            run_macro_df,
                            run_per_dataset_summary_df,
                            run_macro_summary_df,
                            run_runtime_df,
                            run_failed_runs,
                            output_paths,
                        ):
                            run_wandb = None

                    if run_wandb is not None:
                        run_wandb.finish()

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
        per_run_df.groupby(["run", "benchmark", "clf_name"], as_index=False)[["auc_roc", "auc_pr", "f1"]]
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
    if runtime_rows:
        runtime_df = pd.DataFrame(runtime_rows)
        LOGGER.info("Runtime tracking rows: %d", len(runtime_df))
        LOGGER.info(
            "Runtime by status: %s",
            runtime_df.groupby("status", as_index=False).size().to_dict(orient="records"),
        )
    LOGGER.info("Appended per-dataset runtime metrics to %s", runtime_path)

    runtime_df = pd.DataFrame(runtime_rows) if runtime_rows else pd.DataFrame()

