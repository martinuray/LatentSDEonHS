#!/usr/bin/env python3
"""Plot max AUROC per benchmark from the final metrics CSV.

This script expects a CSV produced by the run-finalization flow
(e.g., benchmark/run_datetime plus metric columns).
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


AUROC_CANDIDATES = [
    "aucroc",
    "auroc",
    "auc",
    "macro_aucroc",
    "macro_auroc",
    "macro_auc",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot best AUROC per dataset and companion metrics.")
    parser.add_argument("csv_path", type=str, help="Path to final metrics CSV.")
    parser.add_argument("--output-dir", type=str, default="out/plots", help="Directory for outputs.")
    parser.add_argument("--prefix", type=str, default="final_scores", help="Output filename prefix.")
    parser.add_argument("--benchmark-column", type=str, default="benchmark", help="Benchmark column name.")
    parser.add_argument("--datetime-column", type=str, default="run_datetime", help="Datetime column name.")
    parser.add_argument(
        "--auroc-column",
        type=str,
        default=None,
        help="Explicit AUROC column. If omitted, a known alias is auto-detected.",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default="all",
        help="Comma-separated metric columns to include in the secondary plot, or 'all'.",
    )
    return parser.parse_args()


def _resolve_auroc_column(df: pd.DataFrame, explicit_col: str = None) -> str:
    if explicit_col:
        if explicit_col not in df.columns:
            raise ValueError(f"Requested AUROC column '{explicit_col}' not in CSV columns.")
        return explicit_col

    for candidate in AUROC_CANDIDATES:
        if candidate in df.columns:
            numeric_col = pd.to_numeric(df[candidate], errors="coerce")
            if numeric_col.notna().any():
                return candidate

    raise ValueError(
        "Could not find an AUROC-like column. Tried: " + ", ".join(AUROC_CANDIDATES)
    )


def _get_numeric_metric_columns(df: pd.DataFrame, excluded: List[str]) -> List[str]:
    metric_cols = []
    for col in df.columns:
        if col in excluded:
            continue
        numeric_col = pd.to_numeric(df[col], errors="coerce")
        if numeric_col.notna().any():
            metric_cols.append(col)
    return metric_cols


def _select_best_per_benchmark(
    df: pd.DataFrame,
    benchmark_col: str,
    datetime_col: str,
    auroc_col: str,
) -> pd.DataFrame:
    tmp = df.copy()
    tmp[auroc_col] = pd.to_numeric(tmp[auroc_col], errors="coerce")
    tmp = tmp[tmp[auroc_col].notna()].copy()
    if tmp.empty:
        raise ValueError(f"AUROC column '{auroc_col}' has no numeric values.")

    # Deterministic tie-break: latest datetime wins when AUROC is equal.
    tmp[datetime_col] = pd.to_datetime(tmp[datetime_col], errors="coerce")
    tmp = tmp.sort_values([benchmark_col, auroc_col, datetime_col], ascending=[True, False, False])
    best = tmp.groupby(benchmark_col, as_index=False).head(1)
    best = best.sort_values(benchmark_col).reset_index(drop=True)
    return best


def _plot_max_auroc(best_df: pd.DataFrame, benchmark_col: str, auroc_col: str, output_path: str) -> None:
    labels = best_df[benchmark_col].astype(str).tolist()
    values = pd.to_numeric(best_df[auroc_col], errors="coerce").to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color="#4472C4")
    ax.set_ylabel(auroc_col)
    ax.set_xlabel("Dataset / benchmark")
    ax.set_title("Best AUROC per dataset")
    ax.set_ylim(0.0, min(1.05, max(1.0, np.nanmax(values) * 1.05)))
    ax.grid(axis="y", alpha=0.25)

    for bar, val in zip(bars, values):
        if np.isnan(val):
            continue
        ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_metrics_heatmap(
    best_df: pd.DataFrame,
    benchmark_col: str,
    metric_cols: List[str],
    output_path: str,
) -> None:
    if not metric_cols:
        return

    data = best_df[[benchmark_col] + metric_cols].copy()
    for col in metric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    matrix = data[metric_cols].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(max(8, len(metric_cols) * 1.4), max(4.5, len(data) * 0.8)))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Metric value")

    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_xticklabels(metric_cols, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data[benchmark_col].astype(str).tolist())
    ax.set_title("Best-run metrics per dataset")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isnan(matrix[i, j]):
                txt = "nan"
            else:
                txt = f"{matrix[i, j]:.3f}"
            ax.text(j, i, txt, ha="center", va="center", color="white", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    df = pd.read_csv(args.csv_path)
    required_cols = [args.benchmark_column, args.datetime_column]
    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns in CSV: {missing_required}")

    auroc_col = _resolve_auroc_column(df, args.auroc_column)
    best_df = _select_best_per_benchmark(
        df=df,
        benchmark_col=args.benchmark_column,
        datetime_col=args.datetime_column,
        auroc_col=auroc_col,
    )

    excluded = [args.benchmark_column, args.datetime_column]
    numeric_metrics = _get_numeric_metric_columns(best_df, excluded=excluded)

    if args.metrics.strip().lower() == "all":
        requested_metrics = numeric_metrics
    else:
        requested_metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
        missing = [m for m in requested_metrics if m not in numeric_metrics]
        if missing:
            print(f"Warning: skipping non-numeric or missing metrics: {missing}")
        requested_metrics = [m for m in requested_metrics if m in numeric_metrics]

    # Keep AUROC first, then all other requested metrics.
    plot_metrics = [auroc_col] + [m for m in requested_metrics if m != auroc_col]

    os.makedirs(args.output_dir, exist_ok=True)

    best_csv_path = os.path.join(args.output_dir, f"{args.prefix}_best_per_dataset.csv")
    best_df[[args.benchmark_column, args.datetime_column] + plot_metrics].to_csv(best_csv_path, index=False)

    auroc_plot_path = os.path.join(args.output_dir, f"{args.prefix}_max_auroc_per_dataset.png")
    _plot_max_auroc(best_df, args.benchmark_column, auroc_col, auroc_plot_path)

    heatmap_plot_path = os.path.join(args.output_dir, f"{args.prefix}_best_metrics_heatmap.png")
    _plot_metrics_heatmap(best_df, args.benchmark_column, plot_metrics, heatmap_plot_path)

    print(f"AUROC column: {auroc_col}")
    print(f"Best-per-dataset CSV: {best_csv_path}")
    print(f"Max-AUROC plot: {auroc_plot_path}")
    print(f"Metrics heatmap: {heatmap_plot_path}")


if __name__ == "__main__":
    main()

