#!/usr/bin/env python3
"""CLI utility to summarize anomaly-detection benchmark datasets.

Reports per-subdataset statistics (lengths, feature counts, anomaly ratios) and
aggregate benchmark-level statistics.
"""

import argparse
import ast
import glob
import json
import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table


CONSOLE = Console()


def _safe_load_txt(path: str) -> np.ndarray:
    data = np.loadtxt(path, delimiter=",")
    if data.ndim == 1:
        data = data[:, None]
    return data


def _load_pickle_array(path: str) -> np.ndarray:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if isinstance(data, pd.Series):
        return data.to_numpy()
    if isinstance(data, pd.DataFrame):
        return data.to_numpy()
    return np.asarray(data)


def _flatten_numeric(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.dtype == object:
        flat = []
        for x in arr.ravel():
            if isinstance(x, (list, tuple, np.ndarray)):
                flat.extend(np.asarray(x).ravel().tolist())
            else:
                flat.append(x)
        arr = np.asarray(flat)
    return arr.astype(float).ravel()


def _ratio_from_labels(labels: np.ndarray) -> float:
    labels = _flatten_numeric(labels)
    if labels.size == 0:
        return 0.0
    return float((labels > 0).mean())


def _rows_to_summary(benchmark: str, rows: List[Dict]) -> Dict:
    if not rows:
        return {
            "benchmark": benchmark,
            "num_datasets": 0,
            "total_train_length": 0,
            "total_test_length": 0,
            "weighted_anomaly_ratio": 0.0,
            "mean_anomaly_ratio": 0.0,
            "datasets": [],
        }

    total_test = int(sum(r["test_length"] for r in rows))
    weighted = 0.0
    if total_test > 0:
        weighted = float(sum(r["anomaly_ratio"] * r["test_length"] for r in rows) / total_test)

    return {
        "benchmark": benchmark,
        "num_datasets": len(rows),
        "total_train_length": int(sum(r["train_length"] for r in rows)),
        "total_test_length": total_test,
        "weighted_anomaly_ratio": weighted,
        "mean_anomaly_ratio": float(np.mean([r["anomaly_ratio"] for r in rows])),
        "datasets": rows,
    }


def analyze_smd(data_dir: str) -> Dict:
    train_dir = os.path.join(data_dir, "SMD", "raw", "train")
    test_dir = os.path.join(data_dir, "SMD", "raw", "test")
    label_dir = os.path.join(data_dir, "SMD", "raw", "test_label")

    rows = []
    for train_path in sorted(glob.glob(os.path.join(train_dir, "*.txt"))):
        machine = os.path.basename(train_path).replace(".txt", "")
        test_path = os.path.join(test_dir, f"{machine}.txt")
        label_path = os.path.join(label_dir, f"{machine}.txt")
        if not (os.path.isfile(test_path) and os.path.isfile(label_path)):
            continue

        train = _safe_load_txt(train_path)
        test = _safe_load_txt(test_path)
        labels = _safe_load_txt(label_path)

        rows.append(
            {
                "dataset_id": machine,
                "num_features": int(train.shape[1]),
                "train_length": int(train.shape[0]),
                "test_length": int(test.shape[0]),
                "anomaly_ratio": _ratio_from_labels(labels),
            }
        )

    return _rows_to_summary("SMD", rows)


def analyze_qad(data_dir: str, qad_subdir: str) -> Dict:
    root = os.path.join(data_dir, "QAD", "raw", qad_subdir)
    rows = []

    for train_path in sorted(glob.glob(os.path.join(root, "train_*.pkl"))):
        dsid = os.path.basename(train_path).replace("train_", "").replace(".pkl", "")
        test_path = os.path.join(root, f"test_{dsid}.pkl")
        label_path = os.path.join(root, f"test_label_{dsid}.pkl")
        if not (os.path.isfile(test_path) and os.path.isfile(label_path)):
            continue

        train = _load_pickle_array(train_path)
        test = _load_pickle_array(test_path)
        labels = _load_pickle_array(label_path)

        num_features = int(train.shape[1]) if train.ndim > 1 else 1
        rows.append(
            {
                "dataset_id": dsid,
                "num_features": num_features,
                "train_length": int(train.shape[0]),
                "test_length": int(test.shape[0]),
                "anomaly_ratio": _ratio_from_labels(labels),
            }
        )

    return _rows_to_summary(f"QAD:{qad_subdir}", rows)


def _anomaly_ratio_nasa(num_values: int, anomaly_sequences: str) -> float:
    intervals = ast.literal_eval(anomaly_sequences)
    anom = 0
    for start, stop in intervals:
        anom += max(0, int(stop) - int(start))
    if num_values <= 0:
        return 0.0
    return float(anom / num_values)


def analyze_nasa(data_dir: str, spacecraft: str) -> Dict:
    raw_root = os.path.join(data_dir, "nasa", "raw")
    train_root = os.path.join(raw_root, "train")
    test_root = os.path.join(raw_root, "test")
    labels_csv = os.path.join(raw_root, "labeled_anomalies.csv")

    labels_df = pd.read_csv(labels_csv)
    labels_df = labels_df[labels_df["spacecraft"] == spacecraft]

    rows = []
    for _, row in labels_df.iterrows():
        chan_id = row["chan_id"]
        train_path = os.path.join(train_root, f"{chan_id}.npy")
        test_path = os.path.join(test_root, f"{chan_id}.npy")
        if not (os.path.isfile(train_path) and os.path.isfile(test_path)):
            continue

        train = np.load(train_path)
        test = np.load(test_path)
        num_features = int(train.shape[1]) if train.ndim > 1 else 1

        rows.append(
            {
                "dataset_id": chan_id,
                "num_features": num_features,
                "train_length": int(train.shape[0]),
                "test_length": int(test.shape[0]),
                "anomaly_ratio": _anomaly_ratio_nasa(int(row["num_values"]), row["anomaly_sequences"]),
            }
        )

    return _rows_to_summary(spacecraft, rows)


def _count_timesteps(arr: np.ndarray) -> int:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return int(arr.shape[0] * arr.shape[1])
    if arr.ndim >= 1:
        return int(arr.shape[0])
    return 0


def _num_features(arr: np.ndarray) -> int:
    arr = np.asarray(arr)
    if arr.ndim == 3:
        return int(arr.shape[2])
    if arr.ndim == 2:
        return int(arr.shape[1])
    return 1


def analyze_single_ad_dataset(data_dir: str, benchmark: str) -> Dict:
    root = os.path.join(data_dir, benchmark, "raw")
    train = np.load(os.path.join(root, "train.npy"), allow_pickle=True)
    test = np.load(os.path.join(root, "test.npy"), allow_pickle=True)
    labels = np.load(os.path.join(root, "labels.npy"), allow_pickle=True)

    rows = [
        {
            "dataset_id": benchmark,
            "num_features": _num_features(train),
            "train_length": _count_timesteps(train),
            "test_length": _count_timesteps(test),
            "anomaly_ratio": _ratio_from_labels(labels),
        }
    ]
    return _rows_to_summary(benchmark, rows)


def analyze_psm(data_dir: str) -> Dict:
    """PSM: single dataset with train.csv / test.csv / test_label.csv.

    The first column (``timestamp_(min)``) is a row index and is excluded from
    the feature count.
    """
    raw_root = os.path.join(data_dir, "PSM", "raw")
    train_path = os.path.join(raw_root, "train.csv")
    test_path = os.path.join(raw_root, "test.csv")
    label_path = os.path.join(raw_root, "test_label.csv")

    for p in (train_path, test_path, label_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(f"PSM file not found: {p}")

    def _load_csv_features(path: str) -> np.ndarray:
        df = pd.read_csv(path)
        # Drop any timestamp / index column.
        ts_cols = [c for c in df.columns if "timestamp" in c.lower()]
        df = df.drop(columns=ts_cols, errors="ignore")
        return df.to_numpy(dtype=float)

    train = _load_csv_features(train_path)
    test = _load_csv_features(test_path)

    label_df = pd.read_csv(label_path)
    labels = label_df["label"].to_numpy(dtype=float)

    rows = [
        {
            "dataset_id": "PSM",
            "num_features": int(train.shape[1]),
            "train_length": int(train.shape[0]),
            "test_length": int(test.shape[0]),
            "anomaly_ratio": _ratio_from_labels(labels),
        }
    ]
    return _rows_to_summary("PSM", rows)


def analyze_neurips(data_dir: str, dataset: str) -> Dict:
    """NeurIPS helper datasets (creditcard, gecco).

    Looks for the preprocessed CSV produced by download_neurips_datasets.py at
    ``data_dir/raw/{dataset}/{output_name}``.
    Falls back to the legacy repo-root location for backward compatibility.
    """
    _OUTPUT_NAMES = {
        "creditcard": "creditcard.csv",
        "gecco": "water_quality.csv",
    }
    _LABEL_COLS = {
        "creditcard": "Class",
        "gecco": "label",
    }

    output_name = _OUTPUT_NAMES[dataset]
    label_col = _LABEL_COLS[dataset]

    csv_path = os.path.join(data_dir, "NeurIPS", "raw", output_name)
    df = pd.read_csv(csv_path)
    labels = df[label_col].to_numpy(dtype=float)
    features = df.drop(columns=[label_col]).to_numpy(dtype=float)

    rows = [
        {
            "dataset_id": dataset,
            "num_features": int(features.shape[1]),
            "train_length": int(len(df)),   # no explicit split; report total
            "test_length": int(len(df)),
            "anomaly_ratio": _ratio_from_labels(labels),
        }
    ]
    return _rows_to_summary(f"NeurIPS:{dataset}", rows)


def analyze_benchmark(data_dir: str, benchmark: str, qad_subdir: str) -> Dict:
    if benchmark == "SMD":
        return analyze_smd(data_dir)
    if benchmark == "QAD":
        return analyze_qad(data_dir, qad_subdir=qad_subdir)
    if benchmark in ["SMAP", "MSL"]:
        return analyze_nasa(data_dir, spacecraft=benchmark)
    if benchmark in ["SWaT", "WaDi"]:
        return analyze_single_ad_dataset(data_dir, benchmark)
    if benchmark == "PSM":
        return analyze_psm(data_dir)
    if benchmark in ["creditcard", "gecco"]:
        return analyze_neurips(data_dir, benchmark)
    raise ValueError(f"Unsupported benchmark: {benchmark}")


def _print_summary(summary: Dict, limit: int):
    header = (
        f"[bold cyan]{summary['benchmark']}[/bold cyan]  "
        f"datasets={summary['num_datasets']}  "
        f"train={summary['total_train_length']}  "
        f"test={summary['total_test_length']}"
    )
    CONSOLE.print(Panel(header, box=box.ROUNDED, expand=False))

    agg_table = Table(title="Aggregate Metrics", box=box.SIMPLE_HEAVY)
    agg_table.add_column("Metric", style="bold")
    agg_table.add_column("Value", justify="right")
    agg_table.add_row("Mean anomaly ratio", f"{summary['mean_anomaly_ratio']:.6f}")
    agg_table.add_row("Weighted anomaly ratio", f"{summary['weighted_anomaly_ratio']:.6f}")
    CONSOLE.print(agg_table)

    df = pd.DataFrame(summary["datasets"])
    if df.empty:
        CONSOLE.print("[yellow]No sub-datasets found.[/yellow]")
        return

    df = df.sort_values("dataset_id").reset_index(drop=True)
    if limit > 0:
        df = df.head(limit)

    ds_table = Table(title="Per-Dataset Stats", box=box.MINIMAL_DOUBLE_HEAD)
    ds_table.add_column("dataset_id", style="cyan")
    ds_table.add_column("num_features", justify="right")
    ds_table.add_column("train_length", justify="right")
    ds_table.add_column("test_length", justify="right")
    ds_table.add_column("anomaly_ratio", justify="right", style="magenta")

    for row in df.to_dict(orient="records"):
        ds_table.add_row(
            str(row["dataset_id"]),
            str(int(row["num_features"])),
            str(int(row["train_length"])),
            str(int(row["test_length"])),
            f"{float(row['anomaly_ratio']):.6f}",
        )
    CONSOLE.print(ds_table)


def main():
    parser = argparse.ArgumentParser(description="Analyze anomaly-detection benchmark datasets.")
    parser.add_argument(
        "--benchmark",
        required=True,
        choices=["SMD", "QAD", "SWaT", "WaDi", "SMAP", "MSL", "PSM", "creditcard", "gecco", "all"],
        help="Benchmark to analyze.",
    )
    parser.add_argument("--data-dir", default="data_dir", help="Root data directory.")
    parser.add_argument(
        "--qad-subdir",
        default="qad_clean_pkl_100Hz",
        help="QAD raw subfolder name under data_dir/QAD/raw/.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit printed rows per benchmark (0 = all).")
    parser.add_argument("--json-out", default="", help="Optional path to save JSON summary.")
    args = parser.parse_args()

    benchmarks = (
        [args.benchmark]
        if args.benchmark != "all"
        else ["SMD", "QAD", "SWaT", "WaDi", "SMAP", "MSL", "PSM", "creditcard", "gecco"]
    )

    summaries = {}
    for b in benchmarks:
        try:
            summaries[b] = analyze_benchmark(args.data_dir, b, args.qad_subdir)
            _print_summary(summaries[b], args.limit)
        except FileNotFoundError as exc:
            CONSOLE.print(f"[yellow]Skipping {b}: {exc}[/yellow]")


    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

