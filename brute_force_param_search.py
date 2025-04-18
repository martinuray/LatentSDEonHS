import os
from random import SystemRandom
from itertools import product

from basic_data_anomaly_detection import start_experiment, extend_argparse
from data.ad_provider import ADProvider
from utils.parser import generic_parser

params = {
    'lr' : [10e-5, 10e-6],
    'kl0_weight' : [10e-1, 10e-2, 10e-3],
    'klp_weight' : [10e-1, 10e-2, 10e-3],
    'pxz_weight' : [1],
    'z_dim' : [4, 8],
    'h_dim' : [16, 32],
    'n_deg' : [4, 6, 8],
    'use_atanh' : [True, False],
    'non_linear_decoder': [True, False],
    }



def generate_param_combinations(params_):
    keys = params_.keys()
    values = product(*params_.values())
    return [dict(zip(keys, v)) for v in values]


def main():
    parser = extend_argparse(generic_parser)
    args_ = parser.parse_args()
    args_dict_ = vars(args_)

    provider = ADProvider(data_dir='data_dir', dataset=args_.dataset, n_samples=1000 if args_.debug else None)


    os.makedirs('runs/', exist_ok=True)

    combinations = generate_param_combinations(params)
    print(f"Generated {len(combinations)} combinations")

    for combo in combinations:  # Print only first 5 combinations for preview

        for key_, value_ in combo.items():
            if key_ not in args_dict_.keys():
                print(f"Shiiit! {key_}")
            args_dict_[key_] = value_

        print(args_)
        start_experiment(args_, provider=provider)


if __name__ == "__main__":
    main()