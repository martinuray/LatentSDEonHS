import logging
import matplotlib.pyplot as plt
import pandas as pd

from basic_data_anomaly_detection import extend_argparse, start_experiment
from utils.parser import generic_parser


def main():
    parser = extend_argparse(generic_parser)
    args_ = parser.parse_args()
    print(args_)

    rows = []
    for idx in range(5):
        for subsample in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
            logging.info(f"Running with subsample={subsample}")
            args_.subsample = subsample
            args_.seed = idx
            best_stats_run = start_experiment(args_, provider=None)
            rows.append({'subsample': subsample, 'idx': idx, **best_stats_run})

    results_df = pd.DataFrame(rows).set_index('subsample')
    results_df.to_csv('out/results_sparsity.csv')

if __name__ == "__main__":
    main()

    results_df = pd.read_csv('out/results_sparsity.csv')
    results_df = results_df[['subsample', 'f1', 'prec', 'rec', 'auprc', 'auc']]

    metrics = ['f1', 'prec', 'rec', 'auprc', 'auc']
    agg = results_df.groupby('subsample')[metrics].agg(['mean', 'std'])

    n_cols = len(metrics)
    fig, axes = plt.subplots(n_cols, 1, figsize=(8, 3 * n_cols), sharex=True)

    for ax, col in zip(axes, metrics):
        mean = agg[col]['mean']
        std  = agg[col]['std']
        x    = mean.index

        ax.plot(x, mean, marker='o', label=col)
        ax.fill_between(x, mean - std, mean + std, alpha=0.25, label='±1 std')
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel('subsample')
    plt.tight_layout()
    plt.savefig('out/sparsity_results.png', dpi=150)
    plt.show()
    plt.close('all')
