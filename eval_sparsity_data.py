import copy
import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from anomaly_detection import (
    _extract_bootstrap_args,
    _load_dataset_config,
    _validate_config_keys,
    extend_argparse,
    start_experiment,
)
from utils.parser import generic_parser, get_partition_batch_size

DEFAULT_SUBSAMPLES = [0.01, 0.05]
DEFAULT_NUM_SEEDS = 3


def _parse_subsamples(spec: str):
    """Parse a comma-separated list of subsample fractions."""
    subsamples = [float(s.strip()) for s in spec.split(',') if s.strip()]
    if not subsamples:
        raise ValueError("--subsamples must contain at least one value")
    for s in subsamples:
        if not 0.0 < s <= 1.0:
            raise ValueError(f"subsample fractions must lie in (0, 1], got {s}")
    return subsamples


def _decode_task_id(task_id: int, subsamples):
    """Map a 0-based SLURM array task id to (seed_idx, subsample)."""
    seed_idx = task_id // len(subsamples)
    sub_idx  = task_id %  len(subsamples)
    return seed_idx, subsamples[sub_idx]


def run_single(args, out_dir: str, subsamples):
    """Run one (seed, subsample) pair and save the result as JSON."""
    os.makedirs(out_dir, exist_ok=True)

    # The pair is either encoded in the SLURM array task id or given explicitly
    # via --subsample-value (+ --seed-idx).
    if args.task_id is not None:
        task_id = int(args.task_id)
        seed_idx, subsample = _decode_task_id(task_id, subsamples)
        task_label = f"task_{task_id:04d}"
    else:
        seed_idx = int(args.seed_idx)
        subsample = float(args.subsample_value)
        task_label = f"seed{seed_idx}_sub{subsample:.3f}"

    logging.info(f"{task_label}: seed={seed_idx}, subsample={subsample}")
    args.subsample = subsample
    args.seed = seed_idx

    best_stats = start_experiment(args, provider=None)

    result = {'subsample': subsample, 'idx': seed_idx, **best_stats}
    out_file = os.path.join(out_dir, f"{task_label}.json")
    with open(out_file, 'w') as f:
        json.dump(result, f)
    logging.info(f"Saved result to {out_file}")


def run_all(args, out_dir: str, subsamples):
    """Run the full num_seeds x subsamples sweep sequentially."""
    for task_id in range(args.num_seeds * len(subsamples)):
        task_args = copy.copy(args)
        task_args.task_id = task_id
        run_single(task_args, out_dir, subsamples)


def aggregate(out_dir: str):
    """Collect all per-task JSON files, build CSV, and produce plots."""
    files = sorted(
        f for f in os.listdir(out_dir)
        if f.endswith('.json') and (f.startswith('task_') or f.startswith('seed'))
    )
    if not files:
        raise FileNotFoundError(f"No task_*.json / seed*.json files found in {out_dir}")

    rows = []
    for fname in files:
        with open(os.path.join(out_dir, fname)) as f:
            rows.append(json.load(f))

    results_df = pd.DataFrame(rows)
    csv_path = os.path.join(out_dir, 'results_sparsity.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"Saved aggregated CSV to {csv_path}")

    metrics = ['auc', 'auprc', 'f1', 'rec', 'prec']
    available = [m for m in metrics if m in results_df.columns]
    if not available:
        raise ValueError(f"None of the expected metrics {metrics} were found in aggregated results")

    agg = results_df.groupby('subsample')[available].agg(['mean', 'std'])

    n_cols = len(available)
    fig, axes = plt.subplots(n_cols, 1, figsize=(8, 3 * n_cols), sharex=True)
    if n_cols == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        display_label = col.upper()
        mean = agg[col]['mean']
        std  = agg[col]['std'].fillna(0.0)
        x    = mean.index
        (line,) = ax.plot(x, mean, marker='o', color='k', label=display_label)
        ax.fill_between(x, mean - std, mean + std, alpha=0.075, color='k', label='±std')

        if mean.notna().any():
            argmax_x = mean.idxmax()
            argmax_y = mean.loc[argmax_x]
            ax.scatter(
                [argmax_x],
                [argmax_y],
                s=90,
                color=line.get_color(),
                edgecolors='black',
                linewidths=1.0,
                zorder=4,
                #label='argmax',
            )
            ax.annotate(
                f"MAX @ {argmax_x:.2f}\n{argmax_y:.3f}",
                xy=(argmax_x, argmax_y),
                xytext=(8, 8),
                textcoords='offset points',
                fontsize=8,
                bbox={'boxstyle': 'round,pad=0.2', 'fc': 'white', 'alpha': 0.8, 'ec': 'none'},
            )

        ax.set_ylabel(display_label)
        ax.set_title(display_label)
        ax.legend(fontsize=8, loc="lower left")

    axes[-1].set_xlabel('subsample')
    plt.tight_layout()
    png_path = os.path.join(out_dir, 'sparsity_results.png')
    plt.savefig(png_path, dpi=300)
    plt.close('all')
    print(f"Saved plot to {png_path}")


def main():
    argv = sys.argv[1:]
    parser = extend_argparse(generic_parser)
    parser.add_argument(
        '--mode', choices=['single', 'all', 'aggregate'], default='aggregate',
        help="'single': run one (seed,subsample) pair (requires --task-id or --subsample-value); "
             "'all': run the full seeds x subsamples sweep sequentially; "
             "'aggregate': collect results and plot.")
    parser.add_argument(
        '--task-id', type=int, default=None,
        help="0-based task index encoding (seed_idx, subsample_idx). "
             "Range: 0 .. num_seeds * len(subsamples) - 1.")
    parser.add_argument(
        '--subsamples', type=str, default=','.join(str(s) for s in DEFAULT_SUBSAMPLES),
        help="Comma-separated subsample fractions forming the sweep grid "
             f"(default: {','.join(str(s) for s in DEFAULT_SUBSAMPLES)}). "
             "Also defines how --task-id is decoded.")
    parser.add_argument(
        '--subsample-value', type=float, default=None,
        help="Explicit subsample fraction to process in --mode single "
             "(alternative to --task-id).")
    parser.add_argument(
        '--seed-idx', type=int, default=0,
        help="Seed index to use in --mode single when --subsample-value is given.")
    parser.add_argument(
        '--num-seeds', type=int, default=DEFAULT_NUM_SEEDS,
        help=f"Number of seeds per subsample level (default: {DEFAULT_NUM_SEEDS}).")
    parser.add_argument(
        '--results-dir', default='out/sparsity_results',
        help="Directory for per-task JSON files and final outputs.")

    # Apply the dataset's JSON config as parser defaults, exactly as
    # anomaly_detection.main() does. Without this the sparsity sweep ignored
    # cfg/anomaly_detection/<dataset>.json entirely and ran on bare parser
    # defaults: for QAD that meant data_decimation_factor=1 (raw 100 Hz) with
    # 100-step non-overlapping windows, while the baseline sparsity sweep
    # decimates QAD to 10 Hz with 200-step windows via BENCHMARK_WINDOW_DEFAULTS.
    # The two curves were therefore not measured on the same data. Explicit CLI
    # values still override the config.
    bootstrap_args = _extract_bootstrap_args(argv)
    dataset_cfg = _load_dataset_config(bootstrap_args.dataset, bootstrap_args.config_file)
    _validate_config_keys(parser, dataset_cfg, bootstrap_args.dataset)
    parser.set_defaults(**dataset_cfg)

    args = parser.parse_args(argv)
    logging.info(
        f"Dataset {args.dataset}: data_decimation_factor={args.data_decimation_factor}, "
        f"data_window_length={args.data_window_length}, "
        f"data_window_overlap={args.data_window_overlap}"
    )

    # important!
    args.fixed_subsample_mask = True
    has_cli_batch_size = any(arg == "--batch-size" or arg.startswith("--batch-size=") for arg in argv)
    if not has_cli_batch_size:
        partition_batch_size = get_partition_batch_size()
        if partition_batch_size is not None:
            args.batch_size = partition_batch_size

    try:
        subsamples = _parse_subsamples(args.subsamples)
    except ValueError as exc:
        parser.error(str(exc))
    logging.info(f"Subsample grid: {subsamples} (num_seeds={args.num_seeds})")

    if args.mode == 'single':
        if args.task_id is None and args.subsample_value is None:
            parser.error("--mode single requires either --task-id or --subsample-value")
        if args.task_id is not None and args.subsample_value is not None:
            parser.error("--task-id and --subsample-value are mutually exclusive")
        if args.task_id is not None and not 0 <= args.task_id < args.num_seeds * len(subsamples):
            parser.error(
                f"--task-id must be in 0 .. {args.num_seeds * len(subsamples) - 1} "
                f"for num_seeds={args.num_seeds} and {len(subsamples)} subsamples")
        run_single(args, args.results_dir, subsamples)
    elif args.mode == 'all':
        if args.task_id is not None or args.subsample_value is not None:
            parser.error("--mode all runs the full sweep; drop --task-id / --subsample-value")
        run_all(args, args.results_dir, subsamples)
    else:
        aggregate(args.results_dir)


if __name__ == "__main__":
    main()
