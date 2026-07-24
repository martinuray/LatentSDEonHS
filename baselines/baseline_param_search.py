# -*- coding: utf-8 -*-
"""Grid-search hyperparameters for the KNN and TcnED baselines.

Reuses almost all of ``baseline.py`` (dataset loading, evaluation loop,
metric aggregation, W&B logging, ...); only the classifier construction is
replaced with a version driven by the hardcoded ``PARAM_GRIDS`` below. Each
classifier's grid is searched independently (no cross-product across
classifiers) in a single run of this script: ``python baseline_param_search.py``.
"""

import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from baselines import baseline as bl

LOGGER = bl.LOGGER

# Hyperparameter search space per classifier. Edit these to change what gets
# searched; every combination (cartesian product) of a classifier's grid is
# evaluated for that classifier, independently of the other classifiers.
PARAM_GRIDS: dict[str, dict[str, list]] = {
    "KNN": {
        "n_neighbors": [5, 10, 20, 30, 50],
        "method": ["largest", "mean", "median"],
    },
    "TcnED": {
        "lr": [1e-3, 1e-4],
        "hidden_dims": [16, 32, 64],
        "rep_dim": [16, 32],
        "kernel_size": [3, 5],
        "dropout": [0.0, 0.2],
    },
}

# Fixed (non-searched) hyperparameters, merged underneath each grid combination.
CLASSIFIER_BASE_KWARGS: dict[str, dict] = {
    "KNN": {
        "metric": "minkowski",
        "p": 2,
        "algorithm": "auto",
        "leaf_size": 30,
        "contamination": 0.1,
    },
    "TcnED": {
        "batch_size": 16,
        "act": "ReLU",
        "bias": False,
        "epochs": 10,
    },
}


def _param_combinations(grid: dict) -> list[dict]:
    if not grid:
        return [{}]
    keys = sorted(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[key] for key in keys))]


def _param_slug(params: dict) -> str:
    return "-".join(f"{key}={value}" for key, value in sorted(params.items()))


def build_classifier_factory(clf_name: str, params: dict, device: str, random_state: int | None, seq_len: int):
    """Build a single classifier factory for one grid point of one classifier."""
    from pyod.models.knn import KNN
    from deepod.models.time_series import TcnED

    merged_params = {**CLASSIFIER_BASE_KWARGS.get(clf_name, {}), **params}

    if clf_name == "KNN":
        return lambda: KNN(**merged_params)
    if clf_name == "TcnED":
        ts_kwargs = {"seq_len": seq_len, "stride": seq_len, "device": device, "random_state": random_state, "verbose": 1}
        return lambda: TcnED(**{**ts_kwargs, **merged_params})
    raise ValueError(f"Unsupported classifier for param search: {clf_name}")


def parse_args():
    import argparse

    def positive_int(value):
        parsed = int(value)
        if parsed < 1:
            raise argparse.ArgumentTypeError("--runs must be >= 1")
        return parsed

    parser = argparse.ArgumentParser(
        description="Grid-search KNN/TcnED hyperparameters (PARAM_GRIDS below) and macro-average metrics across datasets."
    )
    parser.add_argument("--benchmarks", type=str, default="all", help="Comma-separated benchmark names, or 'all'.")
    parser.add_argument("--classifiers", type=str, default="all", help="Comma-separated subset of {KNN, TcnED}, or 'all'.")
    parser.add_argument("--max-train-samples", type=int, default=None, help="Optional cap on training rows for quick checks.")
    parser.add_argument("--max-test-samples", type=int, default=None, help="Optional cap on test rows (and labels) for quick checks.")
    parser.add_argument("--log-level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Logging verbosity.")
    parser.add_argument("--gpu-id", type=int, default=None, help="Physical GPU id to use exclusively (sets CUDA_VISIBLE_DEVICES to this single id).")
    parser.add_argument("--runs", type=positive_int, default=1, help="How many repeated evaluation runs to execute per benchmark/classifier/param-combination/dataset.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed; run i uses seed + i.")
    parser.add_argument("--seq-len-default", type=positive_int, default=bl.DEFAULT_SEQ_LEN, help="Default sequence length for TcnED; stride is set equal to seq_len.")
    parser.add_argument("--benchmark-seq-lens", type=str, default="", help="Optional benchmark-specific seq lens, e.g. 'SWaT:200,WaDi:128'.")

    parser.add_argument("--wandb-project", type=str, default="latent-sde-on-hs-baselines-param-search", help="Weights & Biases project name.")
    parser.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity / team name.")
    parser.add_argument("--wandb-name", type=str, default=None, help="Optional explicit W&B run name. If omitted, a descriptive name is generated.")
    parser.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group name. Defaults to the selected benchmark/classifier combination.")
    parser.add_argument("--wandb-tags", nargs="*", default=[], help="Optional W&B tags.")
    parser.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online", help="W&B mode.")
    parser.add_argument("--wandb-disabled", action=argparse.BooleanOptionalAction, default=False, help="Disable W&B logging entirely.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bl.configure_logging(args.log_level)
    runtime_device = bl.configure_gpu(args.gpu_id)
    benchmark_seq_len_overrides = bl._parse_benchmark_seq_lens(args.benchmark_seq_lens, bl.BENCHMARK_DATASETS)

    selected_benchmarks = bl._select_keys(bl.BENCHMARK_DATASETS, args.benchmarks)
    selected_classifiers = bl._select_keys(PARAM_GRIDS, args.classifiers)

    LOGGER.info("Starting baseline hyperparameter grid search")
    LOGGER.info(
        "Arguments: benchmarks=%s, classifiers=%s, max_train_samples=%s, max_test_samples=%s, runs=%s, seed=%s, device=%s, seq_len_default=%s, benchmark_seq_lens=%s",
        args.benchmarks,
        args.classifiers,
        args.max_train_samples,
        args.max_test_samples,
        args.runs,
        args.seed,
        runtime_device,
        args.seq_len_default,
        benchmark_seq_len_overrides,
    )

    combinations_by_classifier = {clf_name: _param_combinations(PARAM_GRIDS[clf_name]) for clf_name in selected_classifiers}
    for clf_name in selected_classifiers:
        LOGGER.info("Classifier %s: %d hyperparameter combination(s) to search: %s", clf_name, len(combinations_by_classifier[clf_name]), PARAM_GRIDS[clf_name])

    LOGGER.info("Selected benchmarks: %s", selected_benchmarks)
    for benchmark_name in selected_benchmarks:
        dataset_count = len(bl.BENCHMARK_DATASETS[benchmark_name])
        if dataset_count == 0:
            LOGGER.warning("Benchmark %s has no discovered datasets", benchmark_name)
        else:
            LOGGER.info("Benchmark %s has %d dataset(s)", benchmark_name, dataset_count)

    benchmark_dataset_counts = {benchmark_name: len(bl.BENCHMARK_DATASETS[benchmark_name]) for benchmark_name in selected_benchmarks}
    benchmark_dataset_ids = {
        benchmark_name: [spec["dataset_id"] for spec in bl.BENCHMARK_DATASETS[benchmark_name]]
        for benchmark_name in selected_benchmarks
    }

    output_dir = ROOT_DIR / "out"
    os.makedirs(output_dir, exist_ok=True)
    per_dataset_path = output_dir / "baseline_param_search_per_dataset.csv"
    macro_path = output_dir / "baseline_param_search_macro.csv"
    per_dataset_summary_path = output_dir / "baseline_param_search_per_dataset_mean_std.csv"
    macro_summary_path = output_dir / "baseline_param_search_macro_mean_std.csv"
    runtime_path = output_dir / "baseline_param_search_runtime_per_dataset.csv"
    best_params_path = output_dir / "baseline_param_search_best_params.csv"
    output_paths = {
        "per_dataset": per_dataset_path,
        "macro": macro_path,
        "per_dataset_mean_std": per_dataset_summary_path,
        "macro_mean_std": macro_summary_path,
        "runtime": runtime_path,
    }

    total_evaluations = args.runs * sum(
        len(combinations_by_classifier[clf_name]) * sum(benchmark_dataset_counts[b] for b in selected_benchmarks)
        for clf_name in selected_classifiers
    )
    progress_bar = tqdm(total=total_evaluations, desc="Param search", unit="eval")

    per_dataset_rows = []
    per_run_rows = []
    failed_runs = []
    runtime_rows = []
    for run_idx in range(args.runs):
        run_number = run_idx + 1
        run_seed = args.seed + run_idx
        bl.set_round_context(run_number, args.runs)
        bl.set_global_seed(run_seed)
        LOGGER.info("Starting run %d/%d with seed=%d", run_number, args.runs, run_seed)

        for clf_name in selected_classifiers:
            for param_id, params in enumerate(combinations_by_classifier[clf_name]):
                params_json = json.dumps(params, sort_keys=True)
                param_slug = _param_slug(params)

                for benchmark_name in selected_benchmarks:
                    seq_len_for_benchmark = benchmark_seq_len_overrides.get(benchmark_name, args.seq_len_default)
                    single_benchmark_dataset_counts = {benchmark_name: benchmark_dataset_counts[benchmark_name]}
                    single_benchmark_dataset_ids = {benchmark_name: benchmark_dataset_ids[benchmark_name]}

                    run_wandb = bl._wandb_init_run(
                        args=args,
                        runtime_device=runtime_device,
                        run_number=run_number,
                        run_seed=run_seed,
                        clf_name=f"{clf_name}#{param_id}",
                        selected_benchmarks=[benchmark_name],
                        selected_classifiers=selected_classifiers,
                        benchmark_seq_len_overrides=benchmark_seq_len_overrides,
                        benchmark_dataset_counts=single_benchmark_dataset_counts,
                        benchmark_dataset_ids=single_benchmark_dataset_ids,
                        output_paths=output_paths,
                        classifier_defaults={
                            "device": runtime_device,
                            "random_state": args.seed,
                            "seq_len_default": args.seq_len_default,
                            "runs": args.runs,
                            "max_train_samples": args.max_train_samples,
                            "max_test_samples": args.max_test_samples,
                            "param_id": param_id,
                            "params": params,
                            "base_kwargs": CLASSIFIER_BASE_KWARGS.get(clf_name, {}),
                        },
                    )
                    run_wandb_step = 0
                    run_per_dataset_rows = []
                    run_per_run_rows = []
                    run_failed_runs = []
                    run_runtime_rows = []

                    try:
                        LOGGER.info(
                            "Run %d/%d clf=%s param_id=%d params=%s benchmark=%s seq_len=%d",
                            run_number, args.runs, clf_name, param_id, param_slug, benchmark_name, seq_len_for_benchmark,
                        )
                        clf_factory = build_classifier_factory(
                            clf_name, params, device=runtime_device, random_state=run_seed, seq_len=seq_len_for_benchmark
                        )
                        dataset_specs = bl.BENCHMARK_DATASETS[benchmark_name]
                        for dataset_spec in dataset_specs:
                            dataset_id = dataset_spec["dataset_id"]
                            started_at = time.perf_counter()
                            progress_bar.set_postfix(
                                run=run_number, clf=clf_name, param_id=param_id, bench=benchmark_name, ds=dataset_id, refresh=False
                            )
                            try:
                                x_train, x_test, y_test = bl.load_dataset(
                                    dataset_spec,
                                    max_train_samples=args.max_train_samples,
                                    max_test_samples=args.max_test_samples,
                                )
                                clf = clf_factory()
                                row, metric_results = bl.evaluate_classifier_on_dataset(
                                    clf_name,
                                    clf,
                                    x_train,
                                    x_test,
                                    y_test,
                                    benchmark_name,
                                    dataset_id,
                                )
                                row["param_id"] = param_id
                                row["params"] = params_json
                                per_dataset_rows.append(row)
                                run_per_dataset_rows.append(row)

                                row_with_run = {**row, "run": run_number, "seed": run_seed}
                                per_run_rows.append(row_with_run)
                                run_per_run_rows.append(row_with_run)
                                bl.append_df_to_csv(pd.DataFrame([row]), per_dataset_path, index=False)

                                elapsed_seconds = time.perf_counter() - started_at
                                run_wandb_step += 1
                                if run_wandb is not None and not bl._wandb_log_evaluation(
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
                                    "param_id": param_id,
                                    "params": params_json,
                                    "status": "success",
                                    "duration_sec": elapsed_seconds,
                                    "error_type": "",
                                    "error_message": "",
                                }
                                runtime_rows.append(runtime_row)
                                run_runtime_rows.append(runtime_row)
                                bl.append_df_to_csv(pd.DataFrame([runtime_row]), runtime_path, index=False)
                                LOGGER.info(
                                    "[seed=%d][%s/%s] %s (%s) completed in %.3f sec",
                                    run_seed, benchmark_name, dataset_id, clf_name, param_slug, elapsed_seconds,
                                )
                                progress_bar.update(1)
                            except Exception:
                                elapsed_seconds = time.perf_counter() - started_at
                                error_type, error_message, _ = sys.exc_info()
                                runtime_row = {
                                    "run": run_number,
                                    "seed": run_seed,
                                    "benchmark": benchmark_name,
                                    "dataset_id": dataset_id,
                                    "clf_name": clf_name,
                                    "param_id": param_id,
                                    "params": params_json,
                                    "status": "failed",
                                    "duration_sec": elapsed_seconds,
                                    "error_type": error_type.__name__ if error_type is not None else "Exception",
                                    "error_message": str(error_message) if error_message is not None else "",
                                }
                                runtime_rows.append(runtime_row)
                                run_runtime_rows.append(runtime_row)
                                bl.append_df_to_csv(pd.DataFrame([runtime_row]), runtime_path, index=False)

                                run_wandb_step += 1
                                if run_wandb is not None and not bl._wandb_log_evaluation(
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

                                failed_run = (run_number, run_seed, benchmark_name, dataset_id, clf_name, param_id, param_slug)
                                failed_runs.append(failed_run)
                                run_failed_runs.append(failed_run)
                                LOGGER.exception(
                                    "[seed=%d][%s/%s] %s (%s) failed",
                                    run_seed, benchmark_name, dataset_id, clf_name, param_slug,
                                )
                                progress_bar.update(1)
                    finally:
                        if run_wandb is not None:
                            run_per_dataset_df = pd.DataFrame(run_per_dataset_rows)
                            run_per_run_df = pd.DataFrame(run_per_run_rows)
                            run_runtime_df = pd.DataFrame(run_runtime_rows)

                            if not run_per_dataset_df.empty:
                                run_macro_df = bl.macro_average(run_per_dataset_df)
                                run_per_dataset_summary_df = bl.aggregate_mean_std(run_per_run_df, ["benchmark", "dataset_id", "clf_name"])
                                run_per_run_macro_df = (
                                    run_per_run_df.groupby(["run", "benchmark", "clf_name"], as_index=False)[["auc_roc", "auc_pr", "f1"]]
                                    .mean()
                                )
                                run_macro_summary_df = bl.aggregate_mean_std(run_per_run_macro_df, ["benchmark", "clf_name"])
                            else:
                                run_macro_df = pd.DataFrame(columns=["benchmark", "clf_name", "auc_roc", "auc_pr", "f1", "num_datasets"])
                                run_per_dataset_summary_df = pd.DataFrame(columns=["benchmark", "dataset_id", "clf_name", "auc_roc_mean", "auc_roc_std", "auc_pr_mean", "auc_pr_std", "f1_mean", "f1_std", "num_runs"])
                                run_macro_summary_df = pd.DataFrame(columns=["benchmark", "clf_name", "auc_roc_mean", "auc_roc_std", "auc_pr_mean", "auc_pr_std", "f1_mean", "f1_std", "num_runs"])

                            if not bl._wandb_log_final_outputs(
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

    progress_bar.close()
    bl.set_round_context()

    if not per_dataset_rows:
        LOGGER.error("No successful runs. Failed runs: %d", len(failed_runs))
        sys.exit(1)

    per_dataset_df = pd.DataFrame(per_dataset_rows)
    per_run_df = pd.DataFrame(per_run_rows)

    # Group cols include param_id/params so distinct grid points don't get averaged together.
    group_cols_dataset = ["benchmark", "dataset_id", "clf_name", "param_id", "params"]
    group_cols_macro = ["benchmark", "clf_name", "param_id", "params"]

    macro_df = (
        per_dataset_df.groupby(group_cols_macro, as_index=False)[["auc_roc", "auc_pr", "f1"]]
        .mean()
        .sort_values(group_cols_macro)
    )
    macro_counts = per_dataset_df.groupby(group_cols_macro, as_index=False).size().rename(columns={"size": "num_datasets"})
    macro_df = macro_df.merge(macro_counts, on=group_cols_macro, how="left")
    bl.append_df_to_csv(macro_df, macro_path, index=False)

    per_dataset_summary_df = bl.aggregate_mean_std(per_run_df, group_cols_dataset)
    bl.append_df_to_csv(per_dataset_summary_df, per_dataset_summary_path, index=False)

    per_run_macro_df = (
        per_run_df.groupby(["run"] + group_cols_macro, as_index=False)[["auc_roc", "auc_pr", "f1"]]
        .mean()
    )
    macro_summary_df = bl.aggregate_mean_std(per_run_macro_df, group_cols_macro)
    bl.append_df_to_csv(macro_summary_df, macro_summary_path, index=False)

    LOGGER.info("Completed %d successful run(s)", len(per_dataset_rows))
    if failed_runs:
        LOGGER.warning("Encountered %d failed run(s); continuing with successful results", len(failed_runs))

    LOGGER.info("Per-dataset metrics:\n%s", per_dataset_df.to_string(index=False))
    LOGGER.info("Macro-averaged benchmark metrics:\n%s", macro_df.to_string(index=False))
    LOGGER.info(
        "Macro mean +- std across runs:\n%s",
        bl.build_mean_std_report(macro_summary_df, group_cols_macro).to_string(index=False),
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

    # Best hyperparameter combination per benchmark/classifier, ranked by mean auc_roc across runs.
    best_idx = macro_summary_df.groupby(["benchmark", "clf_name"])["auc_roc_mean"].idxmax()
    best_params_df = macro_summary_df.loc[best_idx].sort_values(["benchmark", "clf_name"])
    bl.append_df_to_csv(best_params_df, best_params_path, index=False)
    LOGGER.info("Best hyperparameters per benchmark/classifier (by mean auc_roc):\n%s", best_params_df.to_string(index=False))
    LOGGER.info("Appended best-hyperparameter summary to %s", best_params_path)