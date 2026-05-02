import json
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

from anomaly_detection import extend_argparse, start_experiment
from utils.parser import generic_parser, get_partition_batch_size

SUBSAMPLES = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
NUM_SEEDS = 5


def _decode_task_id(task_id: int):
    """Map a 0-based SLURM array task id to (seed_idx, subsample)."""
    seed_idx = task_id // len(SUBSAMPLES)
    sub_idx  = task_id %  len(SUBSAMPLES)
    return seed_idx, SUBSAMPLES[sub_idx]


def run_single(args, out_dir: str):
    """Run one (seed, subsample) pair and save the result as JSON."""
    os.makedirs(out_dir, exist_ok=True)
    task_id = int(args.task_id)
    seed_idx, subsample = _decode_task_id(task_id)

    logging.info(f"Task {task_id}: seed={seed_idx}, subsample={subsample}")
    args.subsample = subsample
    args.seed = seed_idx

    best_stats = start_experiment(args, provider=None)

    result = {'subsample': subsample, 'idx': seed_idx, **best_stats}
    out_file = os.path.join(out_dir, f"task_{task_id:04d}.json")
    with open(out_file, 'w') as f:
        json.dump(result, f)
    logging.info(f"Saved result to {out_file}")


def aggregate(out_dir: str):
    """Collect all per-task JSON files, build CSV, and produce plots."""
    files = sorted(f for f in os.listdir(out_dir) if f.startswith('task_') and f.endswith('.json'))
    if not files:
        raise FileNotFoundError(f"No task_*.json files found in {out_dir}")

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
        '--mode', choices=['single', 'aggregate'], default='aggregate',
        help="'single': run one (seed,subsample) pair (requires --task-id); "
             "'aggregate': collect results and plot.")
    parser.add_argument(
        '--task-id', type=int, default=None,
        help="0-based task index encoding (seed, subsample). "
             f"Range: 0 .. {NUM_SEEDS * len(SUBSAMPLES) - 1}.")
    parser.add_argument(
        '--results-dir', default='out/sparsity_results',
        help="Directory for per-task JSON files and final outputs.")
    args = parser.parse_args(argv)

    # important!
    args.fixed_subsample_mask = True
    has_cli_batch_size = any(arg == "--batch-size" or arg.startswith("--batch-size=") for arg in argv)
    if not has_cli_batch_size:
        partition_batch_size = get_partition_batch_size()
        if partition_batch_size is not None:
            args.batch_size = partition_batch_size

    if args.mode == 'single':
        if args.task_id is None:
            parser.error("--task-id is required for --mode single")
        run_single(args, args.results_dir)
    else:
        aggregate(args.results_dir)


if __name__ == "__main__":
    main()
