#!/usr/bin/env python3
"""Summarize datasets in `data_dir/TSB-AD-M`.

The expected filename format is:

    [index]_[Dataset Name]_id_[id]_[Domain]_tr_[Train Index]_1st_[First Anomaly Index].csv

Examples:
    001_Genesis_id_1_Sensor_tr_4055_1st_15538.csv
    129_OPPORTUNITY_id_1_HumanActivity_tr_1801_1st_1901.csv

The script iterates over all CSV files with tqdm, parses metadata from the
filename, loads the CSV to compute simple anomaly statistics, and prints both
per-file and grouped summaries.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

FILENAME_RE = re.compile(
    r"^(?P<index>\d+)_(?P<dataset_name>.+?)_id_(?P<dataset_id>\d+)_(?P<domain>.+?)_tr_(?P<train_index>\d+)_1st_(?P<first_anomaly_index>\d+)\.csv$"
)


def parse_filename(path: Path) -> Optional[Dict[str, Any]]:
    """Parse metadata from a TSB-AD-M filename.

    The regex is intentionally non-greedy for ``dataset_name`` and ``domain`` so
    it works even when these fields contain underscores or spaces.
    """

    match = FILENAME_RE.match(path.name)
    if not match:
        return None

    info = match.groupdict()
    return {
        "file_index": int(info["index"]),
        "dataset_name": info["dataset_name"],
        "dataset_id": int(info["dataset_id"]),
        "domain": info["domain"],
        "train_index": int(info["train_index"]),
        "first_anomaly_index": int(info["first_anomaly_index"]),
    }


def _pick_label_column(df: pd.DataFrame) -> str:
    for col in reversed(list(df.columns)):
        if "label" in str(col).strip().lower():
            return str(col)
    return str(df.columns[-1])


def _anomaly_segments(labels: np.ndarray) -> List[Dict[str, int]]:
    labels = np.asarray(labels).astype(int).ravel()
    if labels.size == 0:
        return []

    padded = np.pad(labels, (1, 1))
    diff = np.diff(padded)
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [{"start": int(s), "end": int(e), "length": int(e - s)} for s, e in zip(starts, ends)]


def analyze_file(path: Path) -> Dict[str, Any]:
    meta = parse_filename(path)
    if meta is None:
        raise ValueError(f"Filename does not match expected pattern: {path.name}")

    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Empty CSV file: {path}")

    label_col = _pick_label_column(df)
    features_df = df.drop(columns=[label_col], errors="ignore")
    labels = pd.to_numeric(df[label_col], errors="coerce").fillna(0).to_numpy()
    labels_bin = (labels > 0).astype(int)

    segments = _anomaly_segments(labels_bin)
    anomaly_points = int(labels_bin.sum())
    total_rows = int(len(df))
    num_features = int(features_df.shape[1])

    first_actual_anomaly = int(segments[0]["start"]) if segments else None
    last_actual_anomaly = int(segments[-1]["end"] - 1) if segments else None
    anomaly_ratio = float(anomaly_points / total_rows) if total_rows else 0.0

    train_index = meta["train_index"]
    first_anomaly_index = meta["first_anomaly_index"]

    return {
        **meta,
        "filepath": str(path),
        "label_column": label_col,
        "total_rows": total_rows,
        "num_features": num_features,
        "train_rows_expected": int(train_index),
        "test_rows_expected": int(max(total_rows - train_index, 0)),
        "train_fraction_expected": float(train_index / total_rows) if total_rows else 0.0,
        "first_anomaly_expected_in_test": int(max(first_anomaly_index - train_index, 0)),
        "anomaly_points": anomaly_points,
        "anomaly_ratio": anomaly_ratio,
        "num_anomaly_segments": int(len(segments)),
        "first_actual_anomaly": first_actual_anomaly,
        "last_actual_anomaly": last_actual_anomaly,
        "segments": segments,
    }


def _summarize_group(df: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    agg = df.groupby(group_cols, as_index=False).agg(
        files=("filepath", "count"),
        total_rows=("total_rows", "sum"),
        num_features_min=("num_features", "min"),
        num_features_max=("num_features", "max"),
        train_index_mean=("train_index", "mean"),
        train_index_min=("train_index", "min"),
        train_index_max=("train_index", "max"),
        first_anomaly_index_mean=("first_anomaly_index", "mean"),
        anomaly_ratio_mean=("anomaly_ratio", "mean"),
        anomaly_ratio_min=("anomaly_ratio", "min"),
        anomaly_ratio_max=("anomaly_ratio", "max"),
    )
    return agg.sort_values(group_cols).reset_index(drop=True)


def _print_df(title: str, df: pd.DataFrame, max_rows: int):
    print(f"\n=== {title} ===")
    if df.empty:
        print("(no rows)")
        return
    if max_rows > 0 and len(df) > max_rows:
        print(df.head(max_rows).to_string(index=False))
        print(f"... ({len(df) - max_rows} more rows)")
    else:
        print(df.to_string(index=False))


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, list):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    return value


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize TSB-AD-M CSV anomaly datasets.")
    parser.add_argument(
        "--data-dir",
        default="data_dir/TSB-AD-M/raw",
        help="Directory containing the TSB-AD-M CSV files.",
    )
    parser.add_argument(
        "--pattern",
        default="*.csv",
        help="Glob pattern relative to --data-dir (default: *.csv).",
    )
    parser.add_argument(
        "--group-by",
        nargs="+",
        default=["domain"],
        choices=["domain", "dataset_name", "dataset_id"],
        help="Columns to group by for summary tables.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=0,
        help="Limit rows shown in printed tables (0 = show all).",
    )
    parser.add_argument(
        "--per-file-csv",
        default="",
        help="Optional output CSV path for the per-file stats table.",
    )
    parser.add_argument(
        "--summary-csv",
        default="",
        help="Optional output CSV path for the grouped summary table.",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional output path for the per-file stats as JSON.",
    )
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")

    files = sorted(data_dir.glob(args.pattern))
    if not files:
        print(f"No files matched {args.pattern!r} in {data_dir}")
        return

    rows: List[Dict[str, Any]] = []
    skipped: List[str] = []
    for path in tqdm(files, desc="TSB-AD-M files", unit="file"):
        if not path.is_file():
            continue
        meta = parse_filename(path)
        if meta is None:
            skipped.append(path.name)
            continue
        try:
            rows.append(analyze_file(path))
        except Exception as exc:
            skipped.append(f"{path.name} ({exc})")

    if not rows:
        print("No valid TSB-AD-M files were analyzed.")
        if skipped:
            print("Skipped files:")
            for item in skipped:
                print(f"  - {item}")
        return

    per_file_df = pd.DataFrame(rows)
    group_cols = list(dict.fromkeys(args.group_by))
    summary_df = _summarize_group(per_file_df, group_cols)

    print("\nTSB-AD-M dataset stats")
    print(f"Analyzed files: {len(per_file_df)}")
    print(f"Skipped files : {len(skipped)}")
    _print_df("Per-file stats", per_file_df[[
        "file_index",
        "dataset_name",
        "dataset_id",
        "domain",
        "train_index",
        "first_anomaly_index",
        "total_rows",
        "num_features",
        "train_rows_expected",
        "test_rows_expected",
        "anomaly_points",
        "anomaly_ratio",
        "num_anomaly_segments",
        "first_actual_anomaly",
        "last_actual_anomaly",
        "filepath",
    ]], args.max_rows)
    _print_df(f"Grouped summary by {group_cols}", summary_df, args.max_rows)

    print("\nOverall totals")
    print(f"  total rows         : {int(per_file_df['total_rows'].sum())}")
    print(f"  total features     : {int(per_file_df['num_features'].sum())}")
    print(f"  total anomaly pts  : {int(per_file_df['anomaly_points'].sum())}")
    print(f"  mean anomaly ratio : {per_file_df['anomaly_ratio'].mean():.6f}")
    print(f"  weighted ratio     : {(per_file_df['anomaly_points'].sum() / per_file_df['total_rows'].sum()):.6f}")

    if skipped:
        print("\nSkipped files:")
        for item in skipped:
            print(f"  - {item}")

    if args.per_file_csv:
        out_path = Path(args.per_file_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        per_file_df.to_csv(out_path, index=False)
        print(f"\nSaved per-file stats to: {out_path}")

    if args.summary_csv:
        out_path = Path(args.summary_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"Saved grouped summary to: {out_path}")

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(_to_jsonable(rows), f, indent=2)
        print(f"Saved JSON stats to: {out_path}")


if __name__ == "__main__":
    main()

