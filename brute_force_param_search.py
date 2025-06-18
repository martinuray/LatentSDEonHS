import os
from random import SystemRandom
from itertools import product

from basic_data_anomaly_detection import start_experiment, extend_argparse
from data.ad_provider import ADProvider
from utils.parser import generic_parser


params = {
    #'lr' : [1e-3, 1e-4], # set different for each dataset
    'use_atanh': [False],
    'non_linear_decoder': [True],
    'z_dim' : [8],  # experiments 26.-27.4.
    'kl0_weight' : [1e-3], #[1e-1, 1e-2, 1e-3],  # 3
    'klp_weight' : [1e-1], #[1e-1, 1e-2, 1e-3],  # 3
    'pxz_weight' : [1],
    'h_dim' : [32], #[16, 32],                  # 2
    'n_deg' : [8], #[4, 6, 8],                 # 2
    'data-window-length': [50, 75],
    'data-window-overlap': [0.25, 0.50, 0.75]
}


def generate_param_combinations(params_):
    keys = params_.keys()
    values = product(*params_.values())
    return [dict(zip(keys, v)) for v in values]


def main():
    parser = extend_argparse(generic_parser)
    args_ = parser.parse_args()
    args_dict_ = vars(args_)

    if args_.dataset == "SMD":
        args_.lr = 1e-4

    provider = ADProvider(data_dir='data_dir', dataset=args_.dataset,
                          window_length=args_.data_window_length,
                          window_overlap=args_.data_window_overlap,
                          n_samples=1000 if args_.debug else None)


    os.makedirs('runs/', exist_ok=True)

    combinations = generate_param_combinations(params)
    print(f"Starting grid search with {len(combinations)} combinations")

    for combo in combinations:

        for key_, value_ in combo.items():
            if key_ not in args_dict_.keys():
                print(f"Shiiit! {key_}")
            args_dict_[key_] = value_

        print(args_)
        start_experiment(args_, provider=provider)


if __name__ == "__main__":
    main()