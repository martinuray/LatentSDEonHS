import logging
import pandas as pd

from basic_data_anomaly_detection import extend_argparse, start_experiment
from utils.parser import generic_parser


def main():
    parser = extend_argparse(generic_parser)
    args_ = parser.parse_args()
    print(args_)

    rows = []
    for subsample in [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        logging.info(f"Running with subsample={subsample}")
        args_.subsample = subsample
        best_stats_run = start_experiment(args_, provider=None)
        rows.append({'subsample': subsample, **best_stats_run})

    results_df = pd.DataFrame(rows).set_index('subsample')
    results_df.to_csv('out/results_sparsity.csv')

if __name__ == "__main__":
    main()