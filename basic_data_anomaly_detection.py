"""Pendulum angle regression from Sec. 4.1 of

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023
"""

import argparse
import datetime
import logging
import os
from collections import defaultdict

import numpy as np
import pandas as pd
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
    PhysioNetRecogNetwork, GenericMLP,
)
from core.training import generic_train
from data.ad_provider import ADProvider
from data.aero_provider import AeroDataProvider
from data.nasa_provider import NASAProvider
from data.qad_provider import QADProvider
from data.smd_provider import SMDProvider
from data.psm_provider import PSMProvider
from utils.logger import set_up_logging
from utils.misc import (
    set_seed,
    count_parameters,
    ProgressMessage,
    save_checkpoint,
    save_stats,
    append_final_metrics_csv)
from utils.parser import generic_parser
from utils.scoring_functions import Evaluator


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--subsample", type=float, default=0.1)
    group.add_argument("--normalize-score", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--data-normalization-strategy", choices=["none", "std", "min-max"], default="none")
    group.add_argument("--dec-hidden-dim", type=int, default=64)
    group.add_argument("--n-dec-layers", type=int, default=1)
    group.add_argument("--early-stopping-patience", type=int, default=10)
    group.add_argument("--early-stopping-min-delta", type=float, default=0)
    group.add_argument("--non-linear-decoder", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--dataset", choices=["SWaT", "WaDi", "SMD", "aero", "QAD", "MSL", "SMAP", "PSM"], default="SWaT")
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

    qzx_net = default_SOnPathDistributionEncoder(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item())

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

    param_groups = [
        {"params": recon_net.parameters(), "weight_decay": 1e-4},
        {"params": recog_net.parameters()},
        {"params": qzx_net.parameters()},
    ]

    optimizer = optim.Adam(param_groups, lr=args.lr)
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

    pm = ProgressMessage(stats_mask)
    best_auc = 0.0
    best_stats = None
    es_counter = 0
    best_es_loss = np.inf

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

        if tst_stats["auc"] > best_auc:
            best_auc = tst_stats["auc"]
            best_stats = tst_stats

        es_loss = val_stats["loss"]
        if es_loss < best_es_loss - args.early_stopping_min_delta:
            best_es_loss = es_loss
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= args.early_stopping_patience:
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                stats["trn"].append(trn_stats)
                stats["tst"].append(tst_stats)
                stats["val"].append(val_stats)
                stats2tensorboard(trn_stats, val_stats, tst_stats, writer, epoch)
                break

        stats["oth"].append({"lr": scheduler.get_last_lr()[-1],
                             "esc": es_counter})
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

def finalstats2tensorboard(writer_, params_: dict, stats: dict, args):
    f1, f1_ts = 0, 0
    for ep_dict in stats[::-1]:
        if 'f1' in ep_dict.keys() and 'ts_f1' in ep_dict.keys():
            f1 = ep_dict['f1']
            f1_ts = ep_dict['ts_f1']
            break

    param2store = ['lr', 'kl0_weight', 'klp_weight', 'pxz_weight', 'z_dim',
                   'h_dim', 'n_deg', 'use_atanh', 'non_linear_decoder',
                   'dataset', 'n_dec_layers', 'subsample', 'freeze-sigma', 'initial_sigma']

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
    normalisations = ["median-iqr"]
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


def get_ts_eval(scores, targets):
    ts_evalator = Evaluator()
    # targets = torch.from_numpy(targets)
    # scores = torch.from_numpy(scores)

    results = ts_evalator.best_f1_score(targets, scores)

    ## dataframe to display
    metrics_name = ['F1', 'Precision', 'Recall', 'AUPRC', 'AUROC']
    raw = [results['f1'], results['precision'], results['recall'],
           results['auprc'], results['auroc']]
    score_dict = {'': metrics_name, 'point_wise': raw}

    df = pd.DataFrame(score_dict)
    return results, df


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
        all_labels, all_scores = [], []
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

            aux_log_prob = aux_log_prob.mean(dim=2)

            all_scores.append(aux_log_prob)

    all_scores = torch.cat(all_scores, dim=0)
    stats['mu'] = all_scores.mean(dim=0)
    stats['sigma'] = all_scores.std(dim=0)
    return stats


def start_experiment(args, provider=None):
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

    if provider is None:
        logging.info("Instantiating data provider")
        if args.dataset in ['SWaT', 'WaDi']:
            provider = ADProvider(
                data_dir='data_dir', dataset=args.dataset,
                window_length=args.data_window_length, window_overlap=args.data_window_overlap,
                n_samples=1000 if args.debug else None,
                subsample=args.subsample,
                data_normalization_strategy=args.data_normalization_strategy
            )
        elif args.dataset == 'SMD':
            provider = SMDProvider(
                data_dir='data_dir',
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                subsample=args.subsample,
                data_normalization_strategy=args.data_normalization_strategy,
            )
        elif args.dataset == 'aero':
            provider = AeroDataProvider(data_dir="data_dir/aero", subsample=2)
        elif args.dataset == 'QAD':
            provider = QADProvider(
                data_dir="data_dir/",
                dataset_number=None,
                window_length=args.data_window_length,
                subsample=args.subsample,
                data_normalization_strategy=args.data_normalization_strategy,
                raw_subdir="qad_clean_pkl_100Hz",
            )
        elif args.dataset in ['SMAP', 'MSL']:
            provider = NASAProvider(
                data_dir="data_dir/", dataset=args.dataset,
                window_length=args.data_window_length,
                subsample=args.subsample)
        elif args.dataset == 'PSM':
            provider = PSMProvider(
                data_dir='data_dir',
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                subsample=args.subsample,
                data_normalization_strategy=args.data_normalization_strategy,
            )
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
    else:
        logging.info("Using provided data provider")

    has_hybrid_layout = all(
        hasattr(provider, attr) for attr in ["num_datasets", "input_dims", "num_timepoints_list"]
    ) and all(hasattr(provider, attr) for attr in ["_ds_trn", "_ds_tst", "_ds_val"])

    if has_hybrid_layout:
        per_dataset_stats = {}
        per_dataset_histories = {}

        for ds_idx in range(provider.num_datasets):
            trn_slice = DatasetSlice(provider._ds_trn, ds_idx)
            tst_slice = DatasetSlice(provider._ds_tst, ds_idx)
            val_slice = DatasetSlice(provider._ds_val, ds_idx)

            dataset_id = str(trn_slice.dataset_id)
            logging.info(
                f"Training on sub-dataset {dataset_id} ({ds_idx + 1}/{provider.num_datasets})"
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
                input_dim=provider.input_dims[ds_idx],
                num_timepoints=provider.num_timepoints_list[ds_idx],
                writer=writer,
                stats_prefix=dataset_id,
                experiment_id_str=experiment_id_str,
            )

            per_dataset_stats[dataset_id] = tst_stats
            per_dataset_histories[dataset_id] = hist_stats

        if provider.num_datasets == 1:
            only_id = next(iter(per_dataset_stats.keys()))
            finalstats2tensorboard(
                writer_=writer,
                params_=vars(args),
                stats=per_dataset_histories[only_id]["tst"],
                args=args,
            )
            logging.shutdown()
            writer.close()
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

        logging.info(f"Macro metrics across {provider.num_datasets} datasets: {macro_stats}")
        _store_final_metrics(combined_stats)
        logging.shutdown()
        writer.close()
        return combined_stats

    # Fallback path for providers without hybrid layout.
    dl_trn = provider.get_train_loader(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )
    dl_tst = provider.get_test_loader(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
    )
    dl_val = provider.get_val_loader(
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
        input_dim=provider.input_dim,
        num_timepoints=provider.num_timepoints,
        writer=writer,
        stats_prefix="",
        experiment_id_str=experiment_id_str,
    )

    finalstats2tensorboard(writer_=writer, params_=vars(args), stats=stats["tst"], args=args)
    _store_final_metrics(tst_stats)
    logging.shutdown()
    writer.close()
    return tst_stats


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
                aux_log_prob = (aux_log_prob - normalization_stats['mu']) / \
                               normalization_stats['sigma']

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


def main():
    parser = extend_argparse(generic_parser)
    args_ = parser.parse_args()
    print(args_)
    _ = start_experiment(args_, provider=None)


if __name__ == "__main__":
    main()
