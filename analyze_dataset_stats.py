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


def _safe_load_qad_txt(path: str) -> np.ndarray:
    """Load QAD txt robustly (headers/separators/non-numeric cells)."""
    df = pd.read_csv(path, sep=r"[,\s]+", engine="python")
    df = df.loc[:, ~df.columns.astype(str).str.startswith("Unnamed")]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.empty:
        raise ValueError(f"QAD file has no numeric content: {path}")
    data = df.fillna(0.0).to_numpy(dtype=float)
    if data.ndim == 1:
        data = data[:, None]
    return data


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

    for train_path in sorted(glob.glob(os.path.join(root, "train_*.txt"))):
        dsid = os.path.basename(train_path).replace("train_", "").replace(".txt", "")
        test_path = os.path.join(root, f"test_{dsid}.txt")
        label_path = os.path.join(root, f"test_label_{dsid}.txt")
        if not (os.path.isfile(test_path) and os.path.isfile(label_path)):
            continue

        train = _safe_load_qad_txt(train_path)
        test = _safe_load_qad_txt(test_path)
        labels = _safe_load_qad_txt(label_path)

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


def analyze_swat(data_dir: str) -> Dict:
    """SWaT: train.csv / test.csv / labels.csv under raw/."""
    raw_root = os.path.join(data_dir, "SWaT", "raw")
    train = pd.read_csv(os.path.join(raw_root, "train.csv")).to_numpy(dtype=float)
    test  = pd.read_csv(os.path.join(raw_root, "test.csv")).to_numpy(dtype=float)
    labels = pd.read_csv(os.path.join(raw_root, "labels.csv"))["labels"].to_numpy(dtype=float)
    rows = [{
        "dataset_id": "SWaT",
        "num_features": int(train.shape[1]),
        "train_length": int(train.shape[0]),
        "test_length": int(test.shape[0]),
        "anomaly_ratio": _ratio_from_labels(labels),
    }]
    return _rows_to_summary("SWaT", rows)


def analyze_wadi(data_dir: str) -> Dict:
    """WaDi: WADI_14days.csv (train) + WADI_attackdata_labelled.csv (test) under raw/v2/.

    Both files have two header rows (numeric indices + actual names).
    Row, Date, Time columns are dropped. For test the last column is the attack
    label (1 = normal, -1 = attack).
    """
    _META_COLS = {"Row", "Row ", "Date", "Date ", "Time", "Time "}
    raw_root = os.path.join(data_dir, "WaDi", "raw", "v2")

    def _load_wadi_csv(path: str, label_col: str | None = None):
        df = pd.read_csv(path, header=[0, 1])
        # Flatten multi-level columns to their second level (actual names).
        df.columns = [str(b).strip() for _, b in df.columns]
        # Drop metadata columns.
        drop = [c for c in df.columns if c in _META_COLS]
        df = df.drop(columns=drop, errors="ignore")
        if label_col and label_col in df.columns:
            lbl = pd.to_numeric(df[label_col], errors="coerce").fillna(1).to_numpy()
            df = df.drop(columns=[label_col])
        else:
            lbl = None
        data = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
        return data, lbl

    train_data, _ = _load_wadi_csv(os.path.join(raw_root, "WADI_14days.csv"))
    test_data, lbl = _load_wadi_csv(
        os.path.join(raw_root, "WADI_attackdata_labelled.csv"),
        label_col="Attack LABLE (1:No Attack, -1:Attack)",
    )
    # label convention: -1 = attack → convert to 0/1
    labels = (lbl == -1).astype(float) if lbl is not None else np.zeros(test_data.shape[0])

    rows = [{
        "dataset_id": "WaDi",
        "num_features": int(train_data.shape[1]),
        "train_length": int(train_data.shape[0]),
        "test_length": int(test_data.shape[0]),
        "anomaly_ratio": _ratio_from_labels(labels),
    }]
    return _rows_to_summary("WaDi", rows)


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
    if benchmark == "SWaT":
        return analyze_swat(data_dir)
    if benchmark == "WaDi":
        return analyze_wadi(data_dir)
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


def _latex_escape(text: str) -> str:
    """Escape basic LaTeX special chars in table cells."""
    repl = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def _print_final_table(summaries: Dict):
    """Print a single consolidated table across all analyzed benchmarks."""
    table = Table(title="[bold]All Benchmarks — Summary[/bold]")#, box=box.HEAVY_OUTLINE)
    table.add_column("Benchmark", style="bold cyan")
    table.add_column("Traces", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Train points", justify="right")
    table.add_column("Test points", justify="right")
    table.add_column("Anomaly ratio", justify="right", style="magenta")

    final_rows = []
    for name, summary in summaries.items():
        if not summary["datasets"]:
            table.add_row(name, "0", "-", "-", "-", "-")
            final_rows.append(
                {
                    "benchmark": name,
                    "traces": "0",
                    "features": "-",
                    "train_points": "-",
                    "test_points": "-",
                    "anomaly_ratio": "-",
                }
            )
            continue

        df = pd.DataFrame(summary["datasets"])
        num_traces = summary["num_datasets"]
        # features: report as range if not all the same, otherwise single value
        feat_vals = df["num_features"].unique()
        features_str = str(int(feat_vals[0])) if len(feat_vals) == 1 else f"{int(df['num_features'].min())}-{int(df['num_features'].max())}"
        train_total = summary["total_train_length"]
        test_total = summary["total_test_length"]
        anom = summary["weighted_anomaly_ratio"]

        table.add_row(
            name,
            str(num_traces),
            features_str,
            f"{train_total:,}",
            f"{test_total:,}",
            f"{anom:.4f}",
        )
        final_rows.append(
            {
                "benchmark": name,
                "traces": str(num_traces),
                "features": features_str,
                "train_points": str(train_total),
                "test_points": str(test_total),
                "anomaly_ratio": f"{anom:.4f}",
            }
        )

    CONSOLE.print()
    CONSOLE.print(table)

    print("\nLaTeX source:")
    print(r"\begin{tabular}{lrrrrr}")
    print(r"\hline")
    print(r"Benchmark & Traces & Features & Train points & Test points & Anomaly ratio \\")
    print(r"\hline")
    for row in final_rows:
        print(
            f"{_latex_escape(row['benchmark'])} & "
            f"{_latex_escape(row['traces'])} & "
            f"{_latex_escape(row['features'])} & "
            f"{_latex_escape(row['train_points'])} & "
            f"{_latex_escape(row['test_points'])} & "
            f"{_latex_escape(row['anomaly_ratio'])} \\\\"
        )
    print(r"\hline")
    print(r"\end{tabular}")


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
        default="qad_clean_txt_100Hz",
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

    if summaries:
        _print_final_table(summaries)

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

