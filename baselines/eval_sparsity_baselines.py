# -*- coding: utf-8 -*-
"""Evaluate PYOD / DeepOD baselines under varying training-data sparsity.

Sparsity is introduced via burst masks over time-points. Masked values are then
linearly interpolated per feature between the last and next available values,
so training arrays keep their original length while retaining realistic
contiguous missing bursts.

Modes
-----
single
    Run one ``(seed, subsample)`` pair identified by ``--task-id`` (or explicitly
    via ``--subsample`` + ``--seed``). Saves a JSON file with the results.
all
    Run the full ``num_seeds × subsamples`` sweep and persist one JSON file per
    task using the same layout as ``single`` mode.
aggregate
    Collect all ``task_*.json`` files from ``--results-dir``, build a summary CSV,
    and produce a per-metric line plot.

Flexibility
-----------
* ``--benchmark``   choose which benchmark (SWaT / WaDi / PSM / SMD / QAD / MSL / SMAP)
* ``--dataset-id``  optionally restrict to a single dataset within a multi-dataset benchmark
* ``--classifiers`` comma-separated list, or ``all``
* ``--subsamples``  comma-separated fractions to *keep*, e.g. ``0.1,0.3,0.5,0.9``
* ``--num-seeds``   how many repeated seeds to try per subsample level

Usage examples
--------------
# Run task 0 (first seed × first subsample) on SWaT with KNN:
python baselines/eval_sparsity_baselines.py \\
    --mode single --task-id 0 \\
    --benchmark SWaT --classifiers KNN \\
    --results-dir out/sparsity_SWaT

# Aggregate results into CSV + plots:
python baselines/eval_sparsity_baselines.py \\
    --mode aggregate --results-dir out/sparsity_SWaT

# Run the full seed × subsample sweep:
python baselines/eval_sparsity_baselines.py \
    --mode all --benchmark SWaT --classifiers KNN \
    --results-dir out/sparsity_SWaT
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Re-use dataset registry and loading helpers from the main baseline module.
from baselines.baseline import (
    BENCHMARK_DATASETS,
    WADI_REDUCED_BATCH_SIZE,
    _select_keys,
    aggregate_mean_std,
    append_df_to_csv,
    build_classifier_factories,
    build_mean_std_report,
    configure_gpu,
    configure_logging,
    evaluate_classifier_on_dataset,
    load_dataset,
    set_global_seed,
    set_round_context,
)
from utils.anomaly_detection import create_random_burst_mask

LOGGER = logging.getLogger(__name__)

DEFAULT_SUBSAMPLES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
DEFAULT_NUM_SEEDS = 5


# ---------------------------------------------------------------------------
# Burst-mask based interpolation
# ---------------------------------------------------------------------------

def _interpolate_masked_values(x: np.ndarray, keep_mask: np.ndarray) -> np.ndarray:
    """Linearly interpolate masked time points per feature.

    ``keep_mask`` marks available points. Missing interior points are linearly
    interpolated, while edge segments are filled with the nearest available
    value.
    """
    x_interp = np.array(x, copy=True)
    n_features = x_interp.shape[1]
    for feat in range(n_features):
        feat_keep = keep_mask[feat]
        masked_idx = np.flatnonzero(~feat_keep)
        if masked_idx.size == 0:
            continue

        keep_idx = np.flatnonzero(feat_keep)
        if keep_idx.size == 0:
            LOGGER.warning("Feature %d has no kept time-points in mask; keeping original values.", feat)
            continue

        vals = x_interp[:, feat]
        x_interp[masked_idx, feat] = np.interp(masked_idx, keep_idx, vals[keep_idx])
    return x_interp


def apply_burst_sparsity(
    x_train: np.ndarray,
    subsample: float,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply burst masking and interpolate masked values in ``x_train``.

    Parameters
    ----------
    x_train:
        Full training data of shape ``(n_time, n_features)``.
    subsample:
        Fraction of time-steps to **keep** (0 < subsample <= 1).
    seed:
        Optional RNG seed for reproducibility.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        Interpolated training array (same shape as input) and keep mask
        of shape ``(n_features, n_time)`` where ``True`` means observed.
    """
    if subsample >= 1.0:
        n_time, n_features = x_train.shape
        return x_train, np.ones((n_features, n_time), dtype=bool)

    rng_state = np.random.get_state()
    if seed is not None:
        np.random.seed(seed)

    n_time, n_features = x_train.shape
    masked_ratio = 1.0 - subsample

    try:
        # create_random_burst_mask returns (n_features, x_len) bool array.
        # Here we use one independent burst mask per feature.
        burst_mask = create_random_burst_mask(
            n_features=n_features,
            x_len=n_time,
            seed=seed,
            masked_ratio=masked_ratio,
        )  # (n_features, n_time), True = keep
        keep_mask = burst_mask
    except (ValueError, RuntimeError) as exc:
        LOGGER.warning(
            "Burst mask generation failed (subsample=%.3f, n_time=%d): %s. "
            "Falling back to uniform random sub-sampling.",
            subsample,
            n_time,
            exc,
        )
        # Fallback: independent uniform random sampling per feature
        n_keep = max(1, int(round(subsample * n_time)))
        keep_mask = np.zeros((n_features, n_time), dtype=bool)
        for feat in range(n_features):
            indices = np.sort(np.random.choice(n_time, size=n_keep, replace=False))
            keep_mask[feat, indices] = True
    finally:
        np.random.set_state(rng_state)

    x_interp = _interpolate_masked_values(x_train, keep_mask)
    n_kept = int(keep_mask.sum())
    n_total = int(keep_mask.size)
    LOGGER.info(
        "Burst sparsity applied: subsample=%.3f -> kept %d / %d values (%.1f%%), masked values interpolated",
        subsample,
        n_kept,
        n_total,
        100.0 * n_kept / n_total,
    )
    return x_interp, keep_mask


# ---------------------------------------------------------------------------
# Task-ID encoding
# ---------------------------------------------------------------------------

def _decode_task_id(task_id: int, subsamples: list[float]) -> tuple[int, float]:
    """Map a 0-based task id to ``(seed_idx, subsample)``."""
    n_sub = len(subsamples)
    seed_idx = task_id // n_sub
    sub_idx = task_id % n_sub
    return seed_idx, subsamples[sub_idx]


def _total_tasks(num_seeds: int, subsamples: list[float]) -> int:
    return num_seeds * len(subsamples)


def _load_existing_result_rows(path: str) -> list[dict]:
    """Load previously saved result rows from ``path``.

    Returns an empty list when the file does not exist or is empty. Legacy
    single-dict payloads are normalized to a one-element list.
    """
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return []

    with open(path) as fh:
        payload = json.load(fh)

    if payload is None:
        return []
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        return [payload]

    raise ValueError(
        f"Expected JSON list/dict in existing results file {path}, got {type(payload).__name__}."
    )


# ---------------------------------------------------------------------------
# Single run
# ---------------------------------------------------------------------------

def run_single(args, subsamples: list[float], device: str | None = None):
    """Run one ``(seed, subsample)`` pair and persist results as JSON."""
    out_dir = args.results_dir
    os.makedirs(out_dir, exist_ok=True)

    # Resolve seed / subsample either from task_id or explicit flags.
    if args.task_id is not None:
        seed_idx, subsample = _decode_task_id(args.task_id, subsamples)
        task_label = f"task_{args.task_id:04d}"
    else:
        seed_idx = args.seed
        subsample = args.subsample_value
        task_label = f"seed{seed_idx}_sub{subsample:.3f}"

    run_seed = args.seed_base + seed_idx
    set_global_seed(run_seed)
    set_round_context(seed_idx + 1, args.num_seeds)

    LOGGER.info(
        "Running: benchmark=%s, seed_idx=%d, seed=%d, subsample=%.4f",
        args.benchmark,
        seed_idx,
        run_seed,
        subsample,
    )

    if device is None:
        device = configure_gpu(args.gpu_id)
    classifier_factories = build_classifier_factories(device=device, random_state=run_seed)
    selected_classifiers = _select_keys(classifier_factories, args.classifiers)

    benchmark_name = args.benchmark
    dataset_specs = BENCHMARK_DATASETS[benchmark_name]

    # Optionally filter to a single dataset within the benchmark.
    if args.dataset_id:
        dataset_specs = [s for s in dataset_specs if s["dataset_id"] == args.dataset_id]
        if not dataset_specs:
            raise ValueError(
                f"dataset_id '{args.dataset_id}' not found in benchmark '{benchmark_name}'. "
                f"Available: {[s['dataset_id'] for s in BENCHMARK_DATASETS[benchmark_name]]}"
            )

    rows = []
    for clf_name in selected_classifiers:
        clf_factory = classifier_factories[clf_name]
        for dataset_spec in dataset_specs:
            dataset_id = dataset_spec["dataset_id"]
            try:
                x_train_full, x_test_full, y_test = load_dataset(
                    dataset_spec,
                    max_train_samples=args.max_train_samples,
                    max_test_samples=args.max_test_samples,
                )

                # Apply burst-mask sparsity to training data only.
                mask_seed = run_seed * 1000 + int(subsample * 1000)
                x_train_sparse, keep_mask_train = apply_burst_sparsity(
                    x_train_full,
                    subsample=subsample,
                    seed=mask_seed,
                )

                x_test_sparse, keep_mask_test = apply_burst_sparsity(
                    x_test_full,
                    subsample=subsample,
                    seed=mask_seed,
                )

                clf = clf_factory()
                result = evaluate_classifier_on_dataset(
                    clf_name=clf_name,
                    clf=clf,
                    x_train=x_train_sparse,
                    x_test=x_test_sparse,
                    y_test=y_test,
                    benchmark_name=benchmark_name,
                    dataset_id=dataset_id,
                )
                rows.append({
                    **result,
                    "subsample": subsample,
                    "seed_idx": seed_idx,
                    "run_seed": run_seed,
                    "n_train_full": x_train_full.shape[0],
                    "n_train_sparse": int(keep_mask_train.sum()),
                    "n_train_interpolated": int((~keep_mask_train).sum()),
                    "n_test_full": x_test_full.shape[0],
                    "n_test_sparse": int(keep_mask_test.sum()),
                    "n_test_interpolated": int((~keep_mask_test).sum()),
                })
            except Exception:
                LOGGER.exception(
                    "[%s/%s] %s failed (subsample=%.3f, seed=%d)",
                    benchmark_name, dataset_id, clf_name, subsample, run_seed,
                )

    set_round_context()

    out_file = os.path.join(out_dir, f"{task_label}.json")
    existing_rows = _load_existing_result_rows(out_file)
    combined_rows = existing_rows + rows
    with open(out_file, "w") as fh:
        json.dump(combined_rows, fh, indent=2)
    LOGGER.info(
        "Saved %d new result(s) to %s (%d total)",
        len(rows),
        out_file,
        len(combined_rows),
    )
    return rows


def run_all(args, subsamples: list[float]):
    """Run the full seed × subsample sweep and persist per-task JSON files."""
    all_rows = []
    total_tasks = _total_tasks(args.num_seeds, subsamples)
    LOGGER.info("Running full sweep across %d task(s)", total_tasks)

    # Configure GPU once for the entire sweep so CUDA_VISIBLE_DEVICES is set
    # consistently across all tasks rather than being reset each iteration.
    device = configure_gpu(args.gpu_id)
    LOGGER.info("Using device: %s (gpu_id=%s)", device, args.gpu_id)

    for task_id in range(total_tasks):
        task_args = argparse.Namespace(**vars(args))
        task_args.task_id = task_id
        task_args.subsample_value = None
        all_rows.extend(run_single(task_args, subsamples, device=device))

    LOGGER.info("Completed full sweep across %d task(s)", total_tasks)
    return all_rows


# ---------------------------------------------------------------------------
# Aggregate
# ---------------------------------------------------------------------------

def aggregate(out_dir: str, benchmark: str | None = None):
    """Collect task JSON files, optionally filter by benchmark, and plot."""
    os.makedirs(out_dir, exist_ok=True)
    json_files = sorted(
        f for f in os.listdir(out_dir) if f.endswith(".json")
    )
    if not json_files:
        raise FileNotFoundError(f"No JSON result files found in {out_dir}")

    all_rows = []
    for fname in json_files:
        with open(os.path.join(out_dir, fname)) as fh:
            data = json.load(fh)
        if isinstance(data, list):
            all_rows.extend(data)
        else:
            all_rows.append(data)

    if not all_rows:
        raise ValueError("All JSON files were empty.")

    results_df = pd.DataFrame(all_rows)
    if benchmark is not None:
        if "benchmark" not in results_df.columns:
            raise ValueError("Cannot filter aggregate results: 'benchmark' column is missing in input JSON rows.")
        results_df = results_df[results_df["benchmark"] == benchmark].copy()
        if results_df.empty:
            raise ValueError(
                f"No rows found for benchmark '{benchmark}' in {out_dir}. "
                "Check --benchmark or results-dir contents."
            )
        LOGGER.info("Aggregate filter active: benchmark=%s (rows=%d)", benchmark, len(results_df))
    csv_path = os.path.join(out_dir, "sparsity_baselines_raw.csv")
    results_df.to_csv(csv_path, index=False)
    print(f"Saved raw results CSV to {csv_path}")

    # For multi-trace benchmarks, first average metrics over traces per run
    # (seed), then compute mean/std over these run-level means.
    collapsed_df = _collapse_traces_to_run_means(results_df)

    group_cols = ["benchmark", "clf_name", "subsample"]
    summary_df = aggregate_mean_std(collapsed_df, group_cols)
    summary_csv = os.path.join(out_dir, "sparsity_baselines_summary.csv")
    summary_df.to_csv(summary_csv, index=False)
    print(f"Saved summary CSV (mean ± std) to {summary_csv}")

    print(
        "\nMean ± std report:\n",
        build_mean_std_report(summary_df, group_cols).to_string(index=False),
    )

    _plot_results(collapsed_df, out_dir, group_cols)


def _collapse_traces_to_run_means(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse per-trace rows to one row per run by averaging metric columns.

    This ensures mean/std are computed across runs (seeds) of benchmark-level
    scores instead of mixing trace-level variability into run variability.
    """
    metric_cols = [c for c in ["auroc", "auprc", "f1"] if c in df.columns]
    if not metric_cols:
        return df

    # Prefer run_seed if available; fall back to seed_idx.
    run_id_cols = [c for c in ["run_seed", "seed_idx"] if c in df.columns]
    if not run_id_cols:
        LOGGER.warning("No run identifier column found (run_seed/seed_idx); skipping trace collapsing.")
        return df

    group_cols = ["benchmark", "clf_name", "subsample", run_id_cols[0]]
    if "dataset_id" in df.columns and df["dataset_id"].nunique() > 1:
        LOGGER.info(
            "Collapsing %d trace-level rows to run-level means using %s.",
            len(df),
            run_id_cols[0],
        )

    collapsed = df.groupby(group_cols, as_index=False)[metric_cols].mean()
    return collapsed


def _plot_results(df: pd.DataFrame, out_dir: str, group_cols: list[str]):
    """Produce per-metric line plots averaged over the whole benchmark.

    For each sparsity level, values are aggregated across all datasets within a
    benchmark (and all seeds), so each line reflects benchmark-level behavior.
    """
    metrics = ["auroc", "auprc", "f1"]
    available_metrics = [m for m in metrics if m in df.columns]
    if not available_metrics:
        LOGGER.warning("No metric columns found in results; skipping plots.")
        return

    # One line per classifier (per benchmark if multiple benchmarks are present).
    df = df.copy()
    if "benchmark" in df.columns and df["benchmark"].nunique() > 1:
        df["_line_key"] = df["benchmark"].astype(str) + "/" + df["clf_name"].astype(str)
    else:
        df["_line_key"] = df["clf_name"].astype(str)

    line_labels = sorted(df["_line_key"].unique())

    n_metrics = len(available_metrics)
    fig, axes = plt.subplots(n_metrics, 1, figsize=(10, 4 * n_metrics), sharex=True)
    if n_metrics == 1:
        axes = [axes]

    cmap = plt.get_cmap("tab20")
    colors = [cmap(i / max(len(line_labels), 1)) for i in range(len(line_labels))]

    for ax, metric in zip(axes, available_metrics):
        for label, color in zip(line_labels, colors):
            sub_df = df[df["_line_key"] == label]
            agg = sub_df.groupby("subsample")[metric].agg(["mean", "std"]).sort_index()
            mean = agg["mean"] * 100
            std = agg["std"].fillna(0.0) * 100
            x = mean.index.to_numpy()

            (line,) = ax.plot(x, mean.to_numpy(), marker="o", color=color, label=label)
            ax.fill_between(
                x,
                (mean - std).to_numpy(),
                (mean + std).to_numpy(),
                alpha=0.08,
                color=color,
            )

            if mean.notna().any():
                best_x = mean.idxmax()
                best_y = mean.loc[best_x]
                ax.scatter([best_x], [best_y], s=80, color=color, edgecolors="black", linewidths=0.8, zorder=4)
                ax.annotate(
                    f"{best_y:.3f}@{best_x:.2f}",
                    xy=(best_x, best_y),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )

        ax.set_ylabel(metric.upper())
        ax.set_title(metric.upper())
        ax.grid(axis='y')
        ax.legend(fontsize=7, loc="lower right", ncol=max(1, len(line_labels) // 6))

    axes[-1].set_xlabel("subsample (fraction kept)")

    plt.tight_layout()
    png_path = os.path.join(out_dir, "sparsity_baselines.png")
    plt.savefig(png_path, dpi=150)
    plt.close("all")
    print(f"Saved plot to {png_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    def pos_float(v):
        f = float(v)
        if not 0.0 < f <= 1.0:
            raise argparse.ArgumentTypeError("Must be in (0, 1].")
        return f

    def pos_int(v):
        i = int(v)
        if i < 1:
            raise argparse.ArgumentTypeError("Must be >= 1.")
        return i

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate PYOD / DeepOD baselines under burst-mask row sparsity. "
            "Use --mode single for one (seed, subsample) pair, aggregate to collect results."
        )
    )

    parser.add_argument(
        "--mode",
        choices=["single", "all", "aggregate"],
        default="aggregate",
        help="'single': run one experiment (requires --task-id or --subsample-value + --seed); "
             "'all': run the full seeds × subsamples sweep; "
             "'aggregate': collect all JSON results and plot.",
    )
    parser.add_argument(
        "--task-id",
        type=int,
        default=None,
        help=(
            "0-based task index encoding (seed_idx, subsample_idx) for --mode single. "
            "Range: 0 .. num_seeds × len(subsamples) - 1."
        ),
    )
    parser.add_argument(
        "--subsample-value",
        type=pos_float,
        default=None,
        help="Explicit subsample fraction to keep for --mode single (alternative to --task-id).",
    )

    # Benchmark / dataset selection
    parser.add_argument(
        "--benchmark",
        choices=list(BENCHMARK_DATASETS.keys()),
        default="SWaT",
        help="Benchmark to evaluate; in aggregate mode, used to filter loaded results (default: SWaT).",
    )
    parser.add_argument(
        "--dataset-id",
        type=str,
        default=None,
        help=(
            "Restrict to a specific dataset within a multi-dataset benchmark "
            "(e.g. 'machine-1-1' for SMD). Leave unset to run all datasets."
        ),
    )

    # Classifier selection
    parser.add_argument(
        "--classifiers",
        type=str,
        default="all",
        help="Comma-separated classifier names, or 'all'.",
    )

    # Sub-sampling grid
    parser.add_argument(
        "--subsamples",
        type=str,
        default=",".join(str(s) for s in DEFAULT_SUBSAMPLES),
        help=(
            "Comma-separated list of fractions to KEEP "
            f"(default: {DEFAULT_SUBSAMPLES})."
        ),
    )
    parser.add_argument(
        "--num-seeds",
        type=pos_int,
        default=DEFAULT_NUM_SEEDS,
        help=f"Number of independent seeds per subsample level (default: {DEFAULT_NUM_SEEDS}).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Explicit seed_idx for single mode when not using --task-id.",
    )
    parser.add_argument(
        "--seed-base",
        type=int,
        default=42,
        help="Base random seed; run i uses seed_base + i.",
    )

    # Data caps
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Cap on training rows (applied before sparsity masking).",
    )
    parser.add_argument(
        "--max-test-samples",
        type=int,
        default=None,
        help="Cap on test rows.",
    )

    # Infrastructure
    parser.add_argument(
        "--results-dir",
        default="out/sparsity_baselines",
        help="Directory for per-task JSON files and aggregated outputs.",
    )
    parser.add_argument(
        "--gpu-id",
        type=int,
        default=None,
        help="Physical GPU id to use (sets CUDA_VISIBLE_DEVICES).",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    configure_logging(args.log_level)

    # Parse subsample grid.
    subsamples = [float(s.strip()) for s in args.subsamples.split(",") if s.strip()]
    if not subsamples:
        raise ValueError("--subsamples must contain at least one value.")

    LOGGER.info(
        "eval_sparsity_baselines | benchmark=%s | dataset_id=%s | classifiers=%s",
        args.benchmark,
        args.dataset_id or "(all)",
        args.classifiers,
    )
    LOGGER.info(
        "subsamples=%s | num_seeds=%d | seed_base=%d",
        subsamples,
        args.num_seeds,
        args.seed_base,
    )
    total = _total_tasks(args.num_seeds, subsamples)
    LOGGER.info("Total tasks (seeds × subsamples): %d", total)

    if args.mode == "single":
        if args.task_id is None and args.subsample_value is None:
            raise SystemExit(
                "In --mode single you must provide either --task-id or --subsample-value."
            )
        run_single(args, subsamples)

    elif args.mode == "all":
        if args.task_id is not None or args.subsample_value is not None:
            raise SystemExit(
                "In --mode all do not provide --task-id or --subsample-value; the full sweep is run automatically."
            )
        run_all(args, subsamples)

    elif args.mode == "aggregate":
        aggregate(args.results_dir, benchmark=args.benchmark)

    else:
        raise ValueError(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

