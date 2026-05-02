"""
aggregate_qad_results.py

Reads the shared final_metrics.csv produced by anomaly_detection.py, filters
rows that belong to a given dataset (default: QAD), computes per-trace summary
statistics (mean ± std over repeated runs if present), and writes a compact
aggregation CSV.

Usage
-----
    python aggregate_qad_results.py \
        --csv logs/final_metrics.csv \
        --dataset QAD \
        --n-traces 16 \
        --out logs/qad_aggregated_results.csv

The script can also be imported and called programmatically:
    from aggregate_qad_results import aggregate_results
    df = aggregate_results("logs/final_metrics.csv", dataset="QAD")
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


METRIC_COLS = ["f1", "prec", "rec", "auc", "auprc", "loss"]
# also try mean/std suffixes produced by aggregate_run_metrics
MEAN_COLS   = [f"{m}_mean" for m in METRIC_COLS]
STD_COLS    = [f"{m}_std"  for m in METRIC_COLS]


def _best_available_cols(df: pd.DataFrame, candidates: list[str]) -> list[str]:
    """Return subset of candidates that actually exist in df."""
    return [c for c in candidates if c in df.columns]


def aggregate_results(
    csv_path: str,
    dataset: str = "QAD",
    n_traces: int | None = None,
    out_path: str | None = None,
) -> pd.DataFrame:
    """
    Load *csv_path*, filter rows for *dataset*, compute macro statistics,
    and optionally write to *out_path*.

    Returns
    -------
    pd.DataFrame  – one row per trace + one macro summary row
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Normalise benchmark column
    if "benchmark" not in df.columns:
        raise ValueError("CSV must have a 'benchmark' column.")

    # Accept both "QAD" (whole dataset) and "QAD:13" (single trace) rows.
    mask = df["benchmark"].str.startswith(dataset)
    df_ds = df[mask].copy()

    if df_ds.empty:
        raise ValueError(
            f"No rows found for dataset '{dataset}' in {csv_path}. "
            f"Available benchmarks: {df['benchmark'].unique().tolist()}"
        )

    print(f"Found {len(df_ds)} row(s) for dataset prefix '{dataset}'.")

    # Extract trace ID from 'benchmark' column (e.g. "QAD:13" → "13", "QAD" → "all")
    df_ds["trace_id"] = df_ds["benchmark"].apply(
        lambda b: b.split(":", 1)[1] if ":" in b else "all"
    )

    # Prefer *_mean columns (written by aggregate_run_metrics) over raw columns
    metric_src = _best_available_cols(df_ds, MEAN_COLS) or _best_available_cols(df_ds, METRIC_COLS)
    std_src    = _best_available_cols(df_ds, STD_COLS)

    if not metric_src:
        print(
            "WARNING: None of the expected metric columns found. "
            f"Available columns: {df_ds.columns.tolist()}"
        )

    # --- Per-trace summary ---
    # Group by trace_id; take the latest row (by run_datetime) per trace.
    per_trace = (
        df_ds
        .sort_values("run_datetime", ascending=True)
        .groupby("trace_id", as_index=False)
        .last()
    )

    # --- Macro aggregation across traces ---
    numeric_cols = [c for c in per_trace.columns if pd.api.types.is_numeric_dtype(per_trace[c])]
    macro_values = per_trace[numeric_cols].mean(numeric_only=True)
    macro_row = macro_values.to_dict()
    macro_row["trace_id"] = "MACRO_MEAN"
    macro_row["benchmark"] = dataset
    macro_row["run_datetime"] = "aggregated"

    # For macro std, use std of the per-trace mean values
    macro_std = per_trace[metric_src].std(ddof=0) if metric_src else pd.Series(dtype=float)
    for col in metric_src:
        base = col.replace("_mean", "")
        macro_row[f"macro_{base}_std"] = float(macro_std.get(col, float("nan")))

    summary_df = pd.concat(
        [per_trace, pd.DataFrame([macro_row])],
        ignore_index=True,
        sort=False,
    )

    # Check coverage
    unique_traces = [t for t in per_trace["trace_id"].tolist() if t != "all"]
    if n_traces is not None and len(unique_traces) < n_traces:
        missing = sorted(
            set(str(i) for i in range(1, n_traces + 1)) - set(unique_traces)
        )
        print(
            f"WARNING: Expected {n_traces} traces but only found {len(unique_traces)}. "
            f"Missing trace IDs: {missing}"
        )

    # Print summary to console
    print("\n=== Per-trace results ===")
    display_cols = ["trace_id"] + [c for c in metric_src + std_src if c in per_trace.columns]
    print(per_trace[display_cols].to_string(index=False))

    print("\n=== Macro statistics ===")
    for col in metric_src:
        base = col.replace("_mean", "")
        val = macro_row.get(col, float("nan"))
        std = macro_row.get(f"macro_{base}_std", float("nan"))
        print(f"  {base:10s}: {val:.4f} ± {std:.4f}")

    # Write output
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(out_path, index=False)
        print(f"\nAggregated results written to: {out_path}")

    return summary_df


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-trace QAD results from final_metrics.csv"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="logs/final_metrics.csv",
        help="Path to the final_metrics.csv produced by anomaly_detection.py",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="QAD",
        help="Dataset prefix to filter rows (e.g. 'QAD').",
    )
    parser.add_argument(
        "--n-traces",
        type=int,
        default=None,
        help="Expected total number of traces (used to warn about missing results).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional output CSV path for the aggregated summary.",
    )
    args = parser.parse_args()

    try:
        aggregate_results(
            csv_path=args.csv,
            dataset=args.dataset,
            n_traces=args.n_traces,
            out_path=args.out,
        )
    except (FileNotFoundError, ValueError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

