import argparse
import datetime
import json
import logging
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from core.models import (
    PathToGaussianDecoder,
    ELBO,
    default_SOnPathDistributionEncoder,
    PhysioNetRecogNetwork, GenericMLP, default_GLnPathDistributionEncoder,
)
from core.training import generic_train
from data.ad_provider import ADProvider
from data.nasa_provider import NASAProvider
from data.qad_provider import QADProvider
from data.smd_provider import SMDProvider
from data.psm_provider import PSMProvider
from utils.scoring_functions import get_ts_eval
from utils.logger import set_up_logging
from utils.misc import (
    set_seed,
    count_parameters,
    ProgressMessage,
    save_checkpoint,
    save_stats,
    append_final_metrics_csv)
from utils.parser import generic_parser


DATASET_CHOICES = ["SWaT", "WaDi", "SMD", "QAD", "MSL", "SMAP", "PSM"]
DEFAULT_CFG_DIR = Path("cfg") / "anomaly_detection"


def _extract_bootstrap_args(argv):
    """Parse only args needed to resolve dataset-specific defaults."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="SWaT")
    parser.add_argument("--config-file", type=str, default=None)
    return parser.parse_known_args(argv)[0]


def _load_dataset_config(dataset: str, config_file: str | None = None) -> dict:
    """Load JSON config for a dataset; return empty dict when unavailable."""
    cfg_path = Path(config_file) if config_file else (DEFAULT_CFG_DIR / f"{dataset}.json")
    if not cfg_path.exists():
        logging.warning("No config found at %s. Falling back to parser defaults.", cfg_path)
        return {}

    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {cfg_path}")

    logging.info("Loaded dataset defaults from %s", cfg_path)
    return cfg


def _validate_config_keys(parser: argparse.ArgumentParser, cfg: dict, dataset: str):
    valid_keys = {action.dest for action in parser._actions if action.dest != "help"}
    unknown = sorted(set(cfg.keys()) - valid_keys)
    if unknown:
        raise ValueError(
            f"Unknown config keys for dataset '{dataset}': {unknown}. "
            f"Expected keys from parser destinations."
        )


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--subsample", type=float, default=0.4)
    group.add_argument("--normalize-score", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--data-normalization-strategy", choices=["none", "std", "min-max"], default="min-max")
    group.add_argument("--dec-hidden-dim", type=int, default=32)
    group.add_argument("--n-dec-layers", type=int, default=2)
    group.add_argument("--early-stopping-min-delta", type=float, default=0)
    group.add_argument("--non-linear-decoder", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--dataset", choices=DATASET_CHOICES, default="SWaT")
    group.add_argument(
        "--config-file",
        type=str,
        default=None,
        help=(
            "Path to dataset config JSON file. If omitted, "
            "uses cfg/anomaly_detection/<dataset>.json."
        ),
    )
    group.add_argument("--runs", type=int, default=1, help="Number of repeated experiment runs to aggregate.")
    group.add_argument("--delete-processed-data", action=argparse.BooleanOptionalAction, default=False, help="Delete processed data after each run.")
    group.add_argument(
        "--fixed-subsample-mask",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="If set, sample subsampling masks once at dataset load time for train/val instead of resampling every iteration.",
    )
    group.add_argument(
        "--sphere-embedding",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use SOn path-distribution encoder (sphere embedding). Disable to use GLn encoder.",
    )
    return parser


def stats2tensorboard(trn_stats, val_stats, tst_stats, writer, epoch):
    # trn_stats, val_stats, tst_stats, epoch
    trn_keys, tst_keys = set(trn_stats.keys()), set(tst_stats.keys())
    val_keys = set(val_stats.keys()) if val_stats else {}

    key_intersection = trn_keys.intersection(tst_keys)
    if val_stats:
        key_intersection = key_intersection.intersection(val_keys)

    for key_ in key_intersection:
        writer.add_scalar(f'{key_}/trn', trn_stats[key_], epoch)
        writer.add_scalar(f'{key_}/tst', tst_stats[key_], epoch)
        if val_stats:
            writer.add_scalar(f'{key_}/val', val_stats[key_], epoch)

    non_intersecting_stats2tensorboard(trn_stats, key_intersection, 'trn', writer, epoch)
    non_intersecting_stats2tensorboard(tst_stats, key_intersection, 'tst', writer, epoch)
    if val_stats:
        non_intersecting_stats2tensorboard(val_stats, key_intersection, 'val', writer, epoch)


def non_intersecting_stats2tensorboard(stats, keys_intersected, prefix, writer, epoch):
    keys_non_intersected = set(stats.keys()) - keys_intersected
    for key_ in keys_non_intersected:
        writer.add_scalar(f'{prefix}/{key_}', stats[key_], epoch)


class DatasetSlice(Dataset):
    """View on one sub-dataset using flattened global indexing."""

    def __init__(self, base_dataset: Dataset, ds_idx: int):
        self.base_dataset = base_dataset
        self.ds_idx = ds_idx

        ds = base_dataset.get_dataset(ds_idx)
        self.input_dim = ds['input_dim']
        self.num_timepoints = ds['num_timepoints']
        self.indcs = ds['indcs']
        self.dataset_id = ds.get('dataset_id', str(ds_idx))

        self.start_idx = base_dataset._cumulative[ds_idx]
        self.length = base_dataset._lengths[ds_idx]

    @property
    def has_aux(self):
        return self.base_dataset.has_aux

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(f"Index {idx} out of range for size {self.length}")
        return self.base_dataset[self.start_idx + idx]


def build_modules_and_optim(args, input_dim, desired_t):
    recog_net = PhysioNetRecogNetwork(
        mtan_input_dim=input_dim,
        mtan_hidden_dim=args.h_dim,
        use_atanh=args.use_atanh
    )

    recon_net = GenericMLP(
        inp_dim=args.z_dim,
        out_dim=input_dim,
        n_hidden=args.dec_hidden_dim,
        n_layers=args.n_dec_layers,
        non_linear=args.non_linear_decoder
    )

    pxz_net = PathToGaussianDecoder(
        mu_map=recon_net,
        sigma_map=None,
        initial_sigma=args.initial_sigma)

    if args.sphere_embedding:
        qzx_net = default_SOnPathDistributionEncoder(
            h_dim=args.h_dim,
            z_dim=args.z_dim,
            n_deg=args.n_deg,
            learnable_prior=args.learnable_prior,
            time_min=0.0,
            time_max=2.0 * desired_t[-1].item(),
        )
        logging.debug("Using default_SOnPathDistributionEncoder (sphere embedding)")
    else:
        qzx_net = default_GLnPathDistributionEncoder(
            h_dim=args.h_dim,
            z_dim=args.z_dim,
            n_deg=args.n_deg,
            learnable_prior=args.learnable_prior,
            time_min=0.0,
            time_max=2.0 * desired_t[-1].item(),
        )
        logging.debug("Using default_GLnPathDistributionEncoder")

    if args.freeze_sigma:
        logging.debug("Froze sigma when computing PathToGaussianDecoder")
        pxz_net.sigma.requires_grad = False

    modules = nn.ModuleDict(
        {
            "recog_net": recog_net,
            "recon_net": recon_net,
            "pxz_net": pxz_net,
            "qzx_net": qzx_net,
        }
    ).to(args.device)

    optimizer = optim.Adam(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1)
    elbo_loss = ELBO(reduction="mean")

    logging.debug(f"Number of model parameters={count_parameters(modules)}")
    return modules, optimizer, scheduler, elbo_loss


def train_one_dataset(
    args,
    dl_trn,
    dl_tst,
    dl_val,
    input_dim,
    num_timepoints,
    writer,
    stats_prefix,
    experiment_id_str,
):
    desired_t = torch.linspace(0, 1.00, num_timepoints, device=args.device).float()
    modules, optimizer, scheduler, elbo_loss = build_modules_and_optim(args, input_dim, desired_t)

    stats = defaultdict(list)
    stats_mask = {
        "oth": ["esc", "lr"],
        "trn": ["log_pxz", "kl0", "klp", "loss"],
        "val": ["log_pxz", "kl0", "klp", "loss"],
        "tst": ["loss", "auc", "auprc", "prec", "rec", "f1"],
    }
    if not args.freeze_sigma:
        stats_mask["oth"].append("sig")

    pm = ProgressMessage(stats_mask)
    best_stats = None
    es_counter = 0
    best_val_loss = np.inf

    for epoch in range(1, args.n_epochs + 1):
        trn_stats = generic_train(
            args, dl_trn, modules, elbo_loss, None,
            optimizer, desired_t, args.device
        )

        normalization_scores = None
        if args.normalize_score:
            normalization_scores = calculate_z_normalization_values(
                args, dl_trn, modules, desired_t, args.device)

        tst_stats = evaluate(
            args, dl_tst, modules, elbo_loss, desired_t, args.device,
            normalization_stats=normalization_scores, epoch=epoch
        )

        val_stats = evaluate(
            args, dl_val, modules, elbo_loss, desired_t, args.device,
            normalization_stats=normalization_scores, epoch=epoch, test=False
        )

        val_loss = val_stats["loss"]
        if val_loss < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_loss
            best_stats = tst_stats
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= 2*args.restart: # early stopping patience shall be longer than one cosine sheduling
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                stats["trn"].append(trn_stats)
                stats["tst"].append(tst_stats)
                stats["val"].append(val_stats)
                stats2tensorboard(trn_stats, val_stats, tst_stats, writer, epoch)
                break

        to_append = {"lr": scheduler.get_last_lr()[-1],
                     "esc": es_counter,
                     "sig": modules['pxz_net'].sigma.item()}

        stats["oth"].append(to_append)
        scheduler.step()

        stats["trn"].append(trn_stats)
        stats["tst"].append(tst_stats)
        stats["val"].append(val_stats)
        stats2tensorboard(trn_stats, val_stats, tst_stats, writer, epoch)

        if args.checkpoint_at and (epoch in args.checkpoint_at):
            ckpt_name = f"{experiment_id_str}_{stats_prefix}" if stats_prefix else experiment_id_str
            save_checkpoint(args, epoch, ckpt_name, modules, desired_t)

        msg = pm.build_progress_message(stats, epoch)
        if stats_prefix:
            msg = f"[{stats_prefix}] {msg}"
        logging.debug(msg)

        if args.enable_file_logging:
            fname = os.path.join(args.log_dir, f"{experiment_id_str}.json")
            save_stats(args, stats, fname)

    return (best_stats if best_stats is not None else tst_stats), stats


def compute_macro_metrics(per_dataset_stats):
    metric_keys = ["f1", "prec", "rec", "auc", "auprc", "loss", "elbo", "kl0", "klp", "log_pxz"]
    macro = {}

    for key in metric_keys:
        values = [stats[key] for stats in per_dataset_stats.values() if key in stats]
        if values:
            macro[f"macro_{key}"] = float(np.mean(values))
    return macro


def _is_numeric_scalar(value):
    return isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)


def _flatten_numeric_metrics(metrics, prefix="", out=None):
    if out is None:
        out = {}

    if not isinstance(metrics, dict):
        return out

    for key, value in metrics.items():
        flat_key = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, dict):
            _flatten_numeric_metrics(value, flat_key, out)
        elif _is_numeric_scalar(value):
            out[flat_key] = float(value)
    return out


def aggregate_run_metrics(run_results):
    values_by_key = defaultdict(list)
    for result in run_results:
        flat_metrics = _flatten_numeric_metrics(result)
        for key, value in flat_metrics.items():
            values_by_key[key].append(value)

    aggregated = {}
    for key, values in values_by_key.items():
        arr = np.asarray(values, dtype=float)
        aggregated[f"{key}_mean"] = float(arr.mean())
        aggregated[f"{key}_std"] = float(arr.std(ddof=0))

    aggregated["num_runs"] = len(run_results)
    return aggregated

def finalstats2tensorboard(writer_, params_: dict, stats: dict, args):
    f1, f1_ts = 0, 0
    for ep_dict in stats[::-1]:
        if 'f1' in ep_dict.keys() and 'ts_f1' in ep_dict.keys():
            f1 = ep_dict['f1']
            f1_ts = ep_dict['ts_f1']
            break

    param2store = ['lr', 'kl0_weight', 'klp_weight', 'pxz_weight', 'z_dim',
                   'h_dim', 'n_deg', 'use_atanh', 'non_linear_decoder',
                   'dataset', 'n_dec_layers', 'subsample', 'freeze-sigma', 'initial_sigma',
                   'sphere_embedding']

    params_ = {key: value for key, value in params_.items() if key in param2store}

    writer_.add_hparams(
        hparam_dict=params_,
        metric_dict={
            "fin_test_f1": f1,
            "fin_test_f1_ts": f1_ts,
        },
    )


def normalise_scores(test_delta, norm="median-iqr", smooth=True,
                     smooth_window=5):
    """
    Args:
        norm: None, "mean-std" or "median-iqr"
    """
    if norm == "mean-std":
        err_scores = StandardScaler().fit_transform(test_delta)
    elif norm == "median-iqr":
        n_err_mid = np.median(test_delta, axis=0)
        n_err_iqr = iqr(test_delta, axis=0)
        epsilon = 1e-2

        err_scores = (test_delta - n_err_mid) / (np.abs(n_err_iqr) + epsilon)
    elif norm is None:
        err_scores = test_delta
    else:
        raise ValueError(
            f'specified normalisation ({norm}) not implemented, please use one of {None, "mean-std", "median-iqr"}')

    if smooth:
        smoothed_err_scores = np.zeros(err_scores.shape)

        for i in range(smooth_window, len(err_scores)):
            smoothed_err_scores[i] = np.mean(
                err_scores[i - smooth_window: i + smooth_window - 1], axis=0)
        return smoothed_err_scores
    else:
        return err_scores

def get_results_for_all_score_normalizations(
        scores: np.ndarray,
        test_labels: np.ndarray):

    # Options
    #normalisations = ["median-iqr", "mean-std", None] # TODO: then for better performance in the end ?
    normalisations = [None]
    #aggregation_strategies = ["l1", "l2", "linfty", "mean", "max", "median", "p75", "p95"] # TODO: then for better performance in the end
    aggregation_strategies = ["l1"]

    AGG_STRATEGIES = {
        'l1': lambda x: np.linalg.norm(x, ord=1, axis=1),
        'l2': lambda x: np.linalg.norm(x, ord=2, axis=1),
        'linfty': lambda x: np.linalg.norm(x, ord=np.inf, axis=1),
        'mean': lambda x: x.mean(1),
        'max': lambda x: x.max(1),
        'median': lambda x: np.median(x, axis=1),
        'p75': lambda x: np.percentile(x, 75, axis=1),
        'p95': lambda x: np.percentile(x, 95, axis=1),
    }

    agg_scores = {}
    df_list = []

    for n in normalisations:
        normed_scores = normalise_scores(scores, norm=n, smooth=True)
        n_key = n if n is not None else 'no'

        for aggregation_strategy in aggregation_strategies:
            normed_agg_scores = AGG_STRATEGIES[aggregation_strategy](normed_scores)
            r, d = get_ts_eval(normed_agg_scores, test_labels.flatten())

            agg_scores[(aggregation_strategy, n_key)] = r['f1']
            df_list.append((n_key, (aggregation_strategy, d)))

    best_strategy = max(agg_scores, key=agg_scores.get)
    #logging.debug(f"Best score through {best_strategy[0]} and {best_strategy[1]}: {agg_scores[best_strategy]}")
    best_idx = next(i for i, (n, (s, _)) in enumerate(df_list) if (s, n) == best_strategy)
    return dict(df_list), df_list[best_idx][1][1]


def logprob2f1s(scores, true_labels):

    if type(scores) is list:
        scores = torch.cat(scores, dim=0)

    if type(true_labels) is list:
        true_labels = torch.cat(true_labels, dim=0)

    all_metrics_, best_metrics_ = get_results_for_all_score_normalizations(scores, true_labels)
    return best_metrics_


def calculate_z_normalization_values(args, dl, modules, desired_t, device):
    stats = defaultdict(list)

    modules.eval()
    with (torch.no_grad()):
        all_labels, all_scores_list = [], []
        for _, batch in enumerate(dl):
            parts = {key: val.to(device) for key, val in batch.items()}

            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            h = modules["recog_net"](inp)
            qzx, pz = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((args.mc_eval_samples,))
            pxz = modules["pxz_net"](zis)

            aux_log_prob = -pxz.log_prob(parts["evd_obs"])

            # make sure that log_prob is in the right shape
            if aux_log_prob.dim() >= 4:
                aux_log_prob = aux_log_prob.squeeze()
            if aux_log_prob.dim() == 2:
                aux_log_prob = aux_log_prob[None, :, :]

            # aux_log_prob = aux_log_prob.mean(dim=0)
            #aux_log_prob = aux_log_prob.mean(dim=2)
            aux_log_prob = aux_log_prob.mean(dim=0)
            all_scores_list.append(aux_log_prob)

    try:
        all_scores = torch.cat(all_scores_list, dim=0)
    except RuntimeError:
        pass
        raise RuntimeError

    stats['mu'] = all_scores.mean(dim=0)
    stats['sigma'] = all_scores.std(dim=0)
    stats['max'] = all_scores.max(dim=0).values
    stats['min'] = all_scores.min(dim=0).values
    return stats


def start_experiment(args, provider=None, store_final_metrics=True):
    experiment_id = datetime.datetime.now().strftime('%y%m%d-%H:%M:%S')
    experiment_log_file_string = 'DEBUG' if args.debug else f'AD_{args.dataset}'
    experiment_id_str = f'{experiment_log_file_string}_{experiment_id}'

    if args.debug:
        args.n_epochs = 1

    writer = SummaryWriter(f'runs/{experiment_id_str}')

    set_up_logging(
        console_log_level=args.loglevel,
        console_log_color=True,
        logfile_file=os.path.join(args.log_dir, f"{experiment_id_str}.txt")
        if args.log_dir is not None
        else None,
        logfile_log_level=args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(created)d] [%(levelname)-8s] %(message)s%(color_off)s",
    )

    # temp
    #processed_dir = f'data_dir/{args.dataset}/processed'
    #if os.path.exists(processed_dir):
    #    shutil.rmtree(processed_dir)

    logging.debug(f"{experiment_log_file_string} -- Experiment ID={experiment_id}")
    if args.seed > 0:
        set_seed(args.seed)
    logging.debug(f"Seed set to {args.seed}")
    logging.debug(f'Parameters set: {vars(args)}')

    def _store_final_metrics(final_metrics: dict):
        append_final_metrics_csv(
            csv_path=getattr(args, "final_metrics_csv", "logs/final_metrics.csv"),
            benchmark=args.dataset,
            run_datetime=experiment_id,
            metrics=final_metrics,
        )

    data_dir = getattr(args, "data_dir", "data_dir")

    if provider is None:
        logging.info("Instantiating data provider")
        if args.dataset in ['SWaT', 'WaDi']:
            provider = ADProvider(
                data_dir=data_dir, dataset=args.dataset,
                window_length=args.data_window_length, window_overlap=args.data_window_overlap,
                n_samples=1000 if args.debug else None,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy
            )
        elif args.dataset == 'SMD':
            provider = SMDProvider(
                data_dir=data_dir,
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy,
            )
        elif args.dataset == 'QAD':
            provider = QADProvider(
                data_dir=data_dir,
                dataset_number=None,
                window_length=args.data_window_length,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy,
                raw_subdir="qad_clean_txt_100Hz",
            )
        elif args.dataset in ['SMAP', 'MSL']:
            provider = NASAProvider(
                data_dir=data_dir, dataset=args.dataset,
                window_length=args.data_window_length,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask)
        elif args.dataset == 'PSM':
            provider = PSMProvider(
                data_dir=data_dir,
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy,
            )
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
    else:
        logging.info("Using provided data provider")

    def _run_with_provider(active_provider):
        has_hybrid_layout = all(
            hasattr(active_provider, attr) for attr in ["num_datasets", "input_dims", "num_timepoints_list"]
        ) and all(hasattr(active_provider, attr) for attr in ["_ds_trn", "_ds_tst", "_ds_val"])

        if has_hybrid_layout:
            per_dataset_stats = {}
            per_dataset_histories = {}

            for ds_idx in range(active_provider.num_datasets):
                trn_slice = DatasetSlice(active_provider._ds_trn, ds_idx)
                tst_slice = DatasetSlice(active_provider._ds_tst, ds_idx)
                val_slice = DatasetSlice(active_provider._ds_val, ds_idx)

                dataset_id = str(trn_slice.dataset_id)
                logging.info(
                    f"Training on sub-dataset {dataset_id} ({ds_idx + 1}/{active_provider.num_datasets})"
                )

                dl_trn = DataLoader(
                    trn_slice,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=None,
                    num_workers=8,
                    pin_memory=True,
                    drop_last=False,
                )
                dl_tst = DataLoader(
                    tst_slice,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=None,
                    num_workers=8,
                    pin_memory=True,
                )
                dl_val = DataLoader(
                    val_slice,
                    batch_size=args.batch_size,
                    shuffle=False,
                    collate_fn=None,
                    num_workers=8,
                    pin_memory=True,
                )

                tst_stats, hist_stats = train_one_dataset(
                    args=args,
                    dl_trn=dl_trn,
                    dl_tst=dl_tst,
                    dl_val=dl_val,
                    input_dim=active_provider.input_dims[ds_idx],
                    num_timepoints=active_provider.num_timepoints_list[ds_idx],
                    writer=writer,
                    stats_prefix=dataset_id,
                    experiment_id_str=experiment_id_str,
                )

                per_dataset_stats[dataset_id] = tst_stats
                per_dataset_histories[dataset_id] = hist_stats

            if active_provider.num_datasets == 1:
                only_id = next(iter(per_dataset_stats.keys()))
                stats2pass = per_dataset_histories[only_id]["tst"]
                best_stats2pass = per_dataset_stats[only_id]
                finalstats2tensorboard(
                    writer_=writer,
                    params_=vars(args),
                    stats=stats2pass,
                    args=args,
                )
                logging.shutdown()
                writer.close()
                if store_final_metrics:
                    _store_final_metrics(stats2pass[0])
                logging.info(f"Final metrics across {active_provider.num_datasets} datasets: {best_stats2pass}")
                return per_dataset_stats[only_id]

            macro_stats = compute_macro_metrics(per_dataset_stats)
            for key, value in macro_stats.items():
                writer.add_scalar(key, value, 0)

            combined_stats = {
                "per_dataset": per_dataset_stats,
                **macro_stats,
            }

            if args.enable_file_logging:
                fname = os.path.join(args.log_dir, f"{experiment_id_str}.json")
                save_stats(args, combined_stats, fname)

            logging.info(f"Macro metrics across {active_provider.num_datasets} datasets: {macro_stats}")
            if store_final_metrics:
                _store_final_metrics(combined_stats)
            logging.shutdown()
            writer.close()
            return combined_stats

        # Fallback path for providers without hybrid layout.
        dl_trn = active_provider.get_train_loader(
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=None,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )
        dl_tst = active_provider.get_test_loader(
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=8,
            pin_memory=True,
        )
        dl_val = active_provider.get_val_loader(
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=8,
            pin_memory=True,
        )

        tst_stats, stats = train_one_dataset(
            args=args,
            dl_trn=dl_trn,
            dl_tst=dl_tst,
            dl_val=dl_val,
            input_dim=active_provider.input_dim,
            num_timepoints=active_provider.num_timepoints,
            writer=writer,
            stats_prefix="",
            experiment_id_str=experiment_id_str,
        )

        finalstats2tensorboard(writer_=writer, params_=vars(args), stats=stats["tst"], args=args)
        if store_final_metrics:
            _store_final_metrics(tst_stats)
        logging.shutdown()
        writer.close()
        return tst_stats

    try:
        return _run_with_provider(provider)
    finally:
        if args.delete_processed_data:
            delete_processed_data(args.dataset, data_dir=data_dir)
        if provider is not None and hasattr(provider, "cleanup"):
            try:
                provider.cleanup()
            except Exception as err:
                logging.warning(f"Provider cleanup failed: {err}")


def evaluate(
    args,
    dl: torch.utils.data.DataLoader,
    modules: nn.ModuleDict,
    elbo_loss: nn.Module,
    desired_t: torch.Tensor,
    device: str,
    normalization_stats=None,
    epoch: int = 1,
    test=True,
):
    stats = defaultdict(list)

    all_scores = np.zeros(
        (int(dl.dataset.indcs.max().detach().numpy().tolist()) + 1,
         dl.dataset.input_dim))
    all_labels = np.zeros(all_scores.shape[0])
    normalize_counts = np.zeros(all_scores.shape[0])

    modules.eval()
    with ((torch.no_grad())):
        for _, batch in enumerate(dl):
            parts = {key: val.to(device) for key, val in batch.items()}

            indcs = parts["inp_indcs"].cpu().numpy().astype(int)
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            h = modules["recog_net"](inp)
            qzx, pz = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((args.mc_eval_samples,))
            pxz = modules["pxz_net"](zis)

            elbo_val, elbo_parts = elbo_loss(
                qzx,
                pz,
                pxz,
                parts["evd_obs"],
                parts["evd_tid"],
                parts["evd_msk"],
                {
                    "kl0_weight": args.kl0_weight,
                    "klp_weight": args.klp_weight,
                    "pxz_weight": args.pxz_weight,
                },
            )
            loss = elbo_val

            aux_log_prob = -pxz.log_prob(parts["evd_obs"])

            # make sure that log_prob is in the right shape
            if aux_log_prob.dim() >= 4:
                aux_log_prob = aux_log_prob.squeeze()
            if aux_log_prob.dim() == 2:
                aux_log_prob = aux_log_prob[None, :, :]

            #aux_log_prob = aux_log_prob.mean(dim=0)
            if normalization_stats is not None:
                #aux_log_prob = (aux_log_prob - normalization_stats['mu']) / \
                #               normalization_stats['sigma']
                aux_log_prob = (aux_log_prob - normalization_stats['min']) / (normalization_stats['max'] - normalization_stats['min'])

            if aux_log_prob.dim() == 4:
                aux_log_prob = aux_log_prob.mean(axis=0)

            for idx in range(aux_log_prob.shape[0]):
                all_scores[indcs[idx, :], :] += aux_log_prob[idx, :, :].cpu().numpy()

            values, counts = np.unique(indcs, return_counts=True)
            for key, value in zip(values, counts):
                normalize_counts[key] += value

            for idx in range(parts["aux_tgt"].shape[0]):
                all_labels[indcs[idx, :]] = parts["aux_tgt"][idx].cpu().numpy().ravel()

            batch_len = parts["evd_obs"].shape[0]
            stats["loss"].append(loss.item() * batch_len)

            stats["elbo"].append(elbo_val.item() * batch_len)
            stats["kl0"].append(elbo_parts["kl0"].item() * batch_len)
            stats["klp"].append(elbo_parts["klp"].item() * batch_len)
            stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)

    stats = {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}

    all_scores = np.divide(
        all_scores,
        normalize_counts[:, None],
        out=np.zeros_like(all_scores),
        where=normalize_counts[:, None] > 0,
    )

    #if epoch % 10 == 0:
    if test:
        best_metrics = logprob2f1s(all_scores, all_labels)
        best_metrics.set_index(best_metrics.columns[0], inplace=True)
        stats['f1'] = best_metrics.loc["F1", "point_wise"]
        stats['prec'] = best_metrics.loc["Precision", "point_wise"]
        stats['rec'] = best_metrics.loc["Recall", "point_wise"]
        stats['auc'] = best_metrics.loc["AUROC", "point_wise"] #roc_auc_score(all_labels, all_scores.mean(1))
        stats['auprc'] = best_metrics.loc["AUPRC", "point_wise"] #average_precision_score(all_labels, all_scores.mean(1))
    return stats


def delete_processed_data(dataset_name: str, data_dir: str = 'data_dir'):
    """Delete processed data directory for a given dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'SWaT', 'SMD', 'QAD')
        data_dir: Base data directory (default: 'data_dir')
    """
    processed_dirs = []
    
    # Map dataset names to their processed directories
    if dataset_name in ['SWaT', 'WaDi']:
        processed_dirs.append(os.path.join(data_dir, dataset_name, 'processed'))
    elif dataset_name == 'SMD':
        processed_dirs.append(os.path.join(data_dir, 'SMD', 'processed'))
    elif dataset_name == 'QAD':
        processed_dirs.append(os.path.join(data_dir, 'QAD', 'processed'))
    elif dataset_name in ['SMAP', 'MSL']:
        processed_dirs.append(os.path.join(data_dir, dataset_name, 'processed'))
    elif dataset_name == 'PSM':
        processed_dirs.append(os.path.join(data_dir, 'PSM', 'processed'))
    
    # Delete each processed directory if it exists
    for processed_dir in processed_dirs:
        if os.path.exists(processed_dir):
            try:
                shutil.rmtree(processed_dir)
                logging.info(f"Deleted processed data at: {processed_dir}")
            except Exception as e:
                logging.warning(f"Failed to delete processed data at {processed_dir}: {e}")
        else:
            logging.debug(f"Processed data directory not found: {processed_dir}")


def main():
    argv = sys.argv[1:]
    bootstrap_args = _extract_bootstrap_args(argv)

    parser = extend_argparse(generic_parser)
    dataset_cfg = _load_dataset_config(bootstrap_args.dataset, bootstrap_args.config_file)
    _validate_config_keys(parser, dataset_cfg, bootstrap_args.dataset)
    parser.set_defaults(**dataset_cfg)

    # Final parse: explicit CLI values override dataset config defaults.
    args_ = parser.parse_args(argv)
    if args_.runs < 1:
        parser.error("--runs must be >= 1")

    print(args_)
    #if args_.runs == 1:
    #    _ = start_experiment(args_, provider=None, store_final_metrics=True)
    #    return

    run_results = []
    for run_idx in range(args_.runs):
        delete_processed_data(args_.dataset, data_dir=args_.data_dir)
        logging.info("Starting run %d/%d", run_idx + 1, args_.runs)
        run_result = start_experiment(args_, provider=None, store_final_metrics=False)
        run_results.append(run_result)

    aggregated_metrics = aggregate_run_metrics(run_results)
    append_final_metrics_csv(
        csv_path=getattr(args_, "final_metrics_csv", "logs/final_metrics.csv"),
        benchmark=args_.dataset,
        run_datetime=datetime.datetime.now().strftime('%y%m%d-%H:%M:%S'),
        metrics=aggregated_metrics,
    )
    logging.info("Aggregated metrics over %d run(s): %s", args_.runs, aggregated_metrics)
    print(aggregated_metrics)


if __name__ == "__main__":
    main()
