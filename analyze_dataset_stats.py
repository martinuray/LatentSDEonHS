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
import re
from typing import Dict, List, Tuple

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


class _QADCompatUnpickler(pickle.Unpickler):
    """Load QAD pickles written with numpy>=2 under an older numpy at runtime.

    Same remapping as ``data/qad_provider.py`` / ``baselines/baseline.py``;
    duplicated here so this CLI stays importable without torch/sklearn.
    """

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


def _load_qad_pkl(path: str, is_label: bool = False) -> pd.DataFrame:
    """Load one pickled QAD trace as a DataFrame (mirrors qad_provider.load_qad_pkl)."""
    with open(path, "rb") as f:
        loaded = _QADCompatUnpickler(f).load()

    if isinstance(loaded, pd.Series):
        data = loaded.to_frame(name="labels")
    elif isinstance(loaded, pd.DataFrame):
        data = loaded.copy()
    else:
        data = pd.DataFrame(loaded)

    if is_label and len(data.columns) == 1 and "labels" not in data.columns:
        data.columns = ["labels"]
    return data


def _resolve_qad_raw_dir(data_dir: str, qad_subdir: str) -> Tuple[str, str]:
    """Locate the QAD raw folder and report which file format it holds.

    The current datasets ship pickled pandas payloads straight in
    ``data_dir/QAD/raw`` (``train_<id>.pkl`` / ``test_<id>.pkl`` /
    ``test_label_<id>.pkl``), matching data/qad_provider.py and
    baselines/baseline.py. Older checkouts kept the pickles under
    ``qad_clean_pkl_100Hz`` or plain-text traces under `--qad-subdir`, so both
    are still accepted as fallbacks.

    Returns (root, file_format) with file_format in {"pkl", "txt"}.
    """
    flat = os.path.join(data_dir, "QAD", "raw")
    candidates = [
        (flat, "pkl"),
        (os.path.join(flat, "qad_clean_pkl_100Hz"), "pkl"),
        (os.path.join(flat, qad_subdir), "pkl"),
        (os.path.join(flat, qad_subdir), "txt"),
    ]
    for root, file_format in candidates:
        if glob.glob(os.path.join(root, f"train_*.{file_format}")):
            return root, file_format

    raise FileNotFoundError(
        f"No QAD traces found: looked for train_*.pkl in '{flat}' (and "
        f"'qad_clean_pkl_100Hz'/'{qad_subdir}' below it), and train_*.txt in "
        f"'{os.path.join(flat, qad_subdir)}'"
    )


def analyze_qad(data_dir: str, qad_subdir: str, decimation: int = 1) -> Dict:
    """QAD/QAPPD: one train/test/label triple per numeric trace id.

    Feature and length counts are taken the way the models see them: the
    non-sensor ``Enable`` flag is dropped and, for ``decimation`` > 1, every
    n-th sample is kept (the raw traces are 100 Hz; the experiments run at
    decimation 10, i.e. 10 Hz) -- the same treatment
    data/qad_provider.py and baselines/baseline.py apply.
    """
    root, file_format = _resolve_qad_raw_dir(data_dir, qad_subdir)
    decimation = max(1, int(decimation))
    rows = []

    for train_path in sorted(glob.glob(os.path.join(root, f"train_*.{file_format}"))):
        match = re.match(rf"^train_(\d+)\.{file_format}$", os.path.basename(train_path))
        if match is None:
            continue
        dsid = match.group(1)
        test_path = os.path.join(root, f"test_{dsid}.{file_format}")
        label_path = os.path.join(root, f"test_label_{dsid}.{file_format}")
        if not (os.path.isfile(test_path) and os.path.isfile(label_path)):
            continue

        if file_format == "pkl":
            train_df = _load_qad_pkl(train_path).drop(columns=["Enable"], errors="ignore")
            test_df = _load_qad_pkl(test_path).drop(columns=["Enable"], errors="ignore")
            label_df = _load_qad_pkl(label_path, is_label=True)
            train = train_df[::decimation].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            test = test_df[::decimation].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            labels = pd.to_numeric(label_df.iloc[::decimation, 0], errors="coerce").to_numpy(dtype=float)
        else:
            train = _safe_load_qad_txt(train_path)[::decimation]
            test = _safe_load_qad_txt(test_path)[::decimation]
            labels = _safe_load_qad_txt(label_path)[::decimation]

        # The provider truncates test and labels to their common length.
        n_aligned = min(test.shape[0], labels.shape[0])

        num_features = int(train.shape[1]) if train.ndim > 1 else 1
        rows.append(
            {
                "dataset_id": dsid,
                "num_features": num_features,
                "train_length": int(train.shape[0]),
                "test_length": int(n_aligned),
                "anomaly_ratio": _ratio_from_labels(labels[:n_aligned]),
            }
        )

    label = "QAD" if decimation == 1 else f"QAD (decimation={decimation})"
    return _rows_to_summary(label, rows)


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


def analyze_benchmark(data_dir: str, benchmark: str, qad_subdir: str, qad_decimation: int = 1) -> Dict:
    if benchmark == "SMD":
        return analyze_smd(data_dir)
    if benchmark == "QAD":
        return analyze_qad(data_dir, qad_subdir=qad_subdir, decimation=qad_decimation)
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
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


# Display names for the LaTeX table, mirroring BENCHMARK_LABELS in
# baselines/wandb_results_to_latex.py (where QAD is reported as QAPPD). The
# sub-trace counts that file appends to its labels are left out here -- this
# table has an explicit "Traces" column.
LATEX_BENCHMARK_LABELS = {"QAD": "QAPPD"}
# Filler for a benchmark whose files weren't found, as in the results tables.
MISSING_CELL = "--"


def _latex_benchmark_label(name: str) -> str:
    """``QAD`` -> ``QAPPD``, keeping any parenthesized suffix intact.

    ``analyze_qad`` labels a decimated run ``QAD (decimation=10)``, which has to
    come out as ``QAPPD (decimation=10)``.
    """
    head, sep, tail = name.partition(" (")
    return LATEX_BENCHMARK_LABELS.get(head, head) + sep + tail


def _build_final_rows(summaries: Dict) -> List[Dict]:
    """One row per analyzed benchmark, sorted alphabetically by display label.

    Values stay numeric (``None`` where a benchmark yielded no datasets) so the
    console and LaTeX renderers can format them independently.
    """
    rows = []
    for name, summary in summaries.items():
        if not summary["datasets"]:
            rows.append({
                "benchmark": name,
                "traces": 0,
                "features": None,
                "train_points": None,
                "test_points": None,
                "anomaly_ratio": None,
            })
            continue

        df = pd.DataFrame(summary["datasets"])
        # features: report as a range if the sub-datasets disagree.
        feat_vals = df["num_features"].unique()
        features = (
            str(int(feat_vals[0])) if len(feat_vals) == 1
            else f"{int(df['num_features'].min())}-{int(df['num_features'].max())}"
        )
        rows.append({
            "benchmark": name,
            "traces": summary["num_datasets"],
            "features": features,
            "train_points": summary["total_train_length"],
            "test_points": summary["total_test_length"],
            "anomaly_ratio": summary["weighted_anomaly_ratio"],
        })

    return sorted(rows, key=lambda r: _latex_benchmark_label(r["benchmark"]).lower())


def _print_final_table(rows: List[Dict]):
    """Print a single consolidated console table across all analyzed benchmarks."""
    table = Table(title="[bold]All Benchmarks — Summary[/bold]")#, box=box.HEAVY_OUTLINE)
    table.add_column("Benchmark", style="bold cyan")
    table.add_column("Traces", justify="right")
    table.add_column("Features", justify="right")
    table.add_column("Train points", justify="right")
    table.add_column("Test points", justify="right")
    table.add_column("Anomaly ratio", justify="right", style="magenta")

    for row in rows:
        if row["train_points"] is None:
            table.add_row(row["benchmark"], "0", "-", "-", "-", "-")
            continue
        table.add_row(
            row["benchmark"],
            str(row["traces"]),
            row["features"],
            f"{row['train_points']:,}",
            f"{row['test_points']:,}",
            f"{row['anomaly_ratio']:.4f}",
        )

    CONSOLE.print()
    CONSOLE.print(table)


def to_latex_summary_table(rows: List[Dict]) -> str:
    """Render the per-benchmark summary as a booktabs tabular.

    Styled like the result tables in baselines/wandb_results_to_latex.py:
    ``\\toprule`` / ``\\midrule`` / ``\\bottomrule``, a ``\\small \\textbf{...}``
    column group with a partial ``\\cmidrule(lr)`` under it, bold header cells,
    and data cells wrapped in ``{...}``. Anomaly ratios are reported as
    percentages (x100), as every metric in that file is. Needs only booktabs in
    the preamble -- there is nothing to rank here, so no ``\\cellcolor`` /
    ``\\rankbox`` macros are used.
    """
    header = "\n".join([
        r"&&& \multicolumn{2}{c}{\small \textbf{Length}} & \\",
        r"\cmidrule(lr){4-5}",
        " & ".join([
            r"\textbf{Dataset}",
            r"\textbf{Traces}",
            r"\textbf{Features}",
            r"\textbf{Train}",
            r"\textbf{Test}",
            r"\textbf{Anomalies (\%)}",
        ]) + r" \\",
    ])

    lines = []
    for row in rows:
        cells = [_latex_escape(_latex_benchmark_label(row["benchmark"]))]
        if row["train_points"] is None:
            cells += [f"{{{row['traces']}}}"] + [MISSING_CELL] * 4
        else:
            cells += [
                f"{{{row['traces']}}}",
                f"{{{_latex_escape(row['features'])}}}",
                f"{{{row['train_points']:,}}}",
                f"{{{row['test_points']:,}}}",
                f"{{{row['anomaly_ratio'] * 100:.2f}}}",
            ]
        lines.append(" & ".join(cells) + r" \\")

    return (
        "\\begin{tabular}{l rr rr r}\n"
        "\\toprule\n"
        f"{header}\n"
        "\\midrule\n"
        + "\n".join(lines) + "\n"
        "\\bottomrule\n"
        "\\end{tabular}"
    )


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
        help=(
            "Fallback QAD raw subfolder under data_dir/QAD/raw/, for legacy layouts. "
            "Current datasets keep train_<id>.pkl / test_<id>.pkl / test_label_<id>.pkl "
            "directly in data_dir/QAD/raw/, which is used whenever present."
        ),
    )
    parser.add_argument(
        "--qad-decimation",
        type=int,
        default=1,
        help=(
            "Keep every n-th QAD sample before counting, as data/qad_provider.py does. "
            "Raw traces are 100 Hz; pass 10 to report the 10 Hz setting the experiments "
            "use. Default 1 (raw)."
        ),
    )
    parser.add_argument("--limit", type=int, default=0, help="Limit printed rows per benchmark (0 = all).")
    parser.add_argument("--json-out", default="", help="Optional path to save JSON summary.")
    parser.add_argument(
        "--latex-out",
        default="",
        help=(
            "Optional path to save the LaTeX summary table (it is always printed). "
            "E.g. out/doc/dataset_stats_table.tex, next to the tables written by "
            "baselines/wandb_results_to_latex.py."
        ),
    )
    args = parser.parse_args()

    benchmarks = (
        [args.benchmark]
        if args.benchmark != "all"
        else ["SMD", "QAD", "SWaT", "WaDi", "SMAP", "MSL", "PSM", "creditcard", "gecco"]
    )

    summaries = {}
    for b in benchmarks:
        try:
            summaries[b] = analyze_benchmark(args.data_dir, b, args.qad_subdir, args.qad_decimation)
            _print_summary(summaries[b], args.limit)
        except FileNotFoundError as exc:
            CONSOLE.print(f"[yellow]Skipping {b}: {exc}[/yellow]")

    if summaries:
        final_rows = _build_final_rows(summaries)
        _print_final_table(final_rows)

        latex = to_latex_summary_table(final_rows)
        print("\nLaTeX table (dataset statistics)")
        print(latex)

        if args.latex_out:
            latex_dir = os.path.dirname(args.latex_out)
            if latex_dir:
                os.makedirs(latex_dir, exist_ok=True)
            with open(args.latex_out, "w", encoding="utf-8") as f:
                f.write(latex)
            print(f"\nSaved LaTeX table to: {args.latex_out}")

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(summaries, f, indent=2)
        print(f"\nSaved JSON summary to: {args.json_out}")


if __name__ == "__main__":
    main()

