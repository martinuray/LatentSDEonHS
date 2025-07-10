"""Pendulum angle regression from Sec. 4.1 of

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023
"""

import datetime
import os
import logging
import argparse
import shutil

import numpy as np
from random import SystemRandom
from collections import defaultdict

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import iqr
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter

from data.ad_provider import ADProvider
from core.models import (
    PathToGaussianDecoder,
    ELBO,
    default_SOnPathDistributionEncoder,
    PhysioNetRecogNetwork, GenericMLP,
)

from core.training import generic_train
from data.aero_provider import AeroDataProvider
from data.imm_provider import IMMProvider
from notebooks.utils.analyze import batch_get_log_prob
from utils.anomaly_detection import anomaly_detection_performances
from utils.logger import set_up_logging
from utils.misc import (
    set_seed,
    count_parameters,
    ProgressMessage,
    save_checkpoint,
    save_stats)
from utils.parser import generic_parser
from utils.scoring_functions import Evaluator


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--debug", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--data-normalization-strategy", choices=["none", "std", "min-max"], default="none")
    group.add_argument("--dec-hidden-dim", type=int, default=64)
    group.add_argument("--n-dec-layers", type=int, default=1)
    group.add_argument("--non-linear-decoder", action=argparse.BooleanOptionalAction, default=True)
    group.add_argument("--dataset", choices=["SWaT", "WaDi", "SMD", "aero"], default="SWaT")
    return parser


def stats2tensorboard(stats_, writer_, epoch_, prefix_=''):
    for key_, value_ in stats_.items():
        if value_ is not None:
            writer_.add_scalar(f'{prefix_}{key_}', value_, epoch_)

def finalstats2tensorboard(writer_, params_: dict, stats: dict, args):
    f1, f1_ts = 0, 0
    for ep_dict in stats[::-1]:
        if 'f1' in ep_dict.keys() and 'ts_f1' in ep_dict.keys():
            f1 = ep_dict['f1']
            f1_ts = ep_dict['ts_f1']
            break

    param2store = ['lr', 'kl0_weight', 'klp_weight', 'pxz_weight', 'z_dim',
                   'h_dim', 'n_deg', 'use_atanh', 'non_linear_decoder',
                   'dataset', 'n_dec_layers']

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
            'specified normalisation not implemented, please use one of {None, "mean-std", "median-iqr"}')

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

    # get score under all three normalizations
    df_list = []
    f1_scores = []
    normalisations = ["median-iqr", "mean-std", None]
    aggregation_strategies = ["mean", "max", "median", "p75", "p95"]
    for n in normalisations:
        normed_scores = normalise_scores(scores, norm=n)

        for aggregation_strategy in aggregation_strategies:
            if aggregation_strategy == 'mean':
                normed_agg_scores = normed_scores.mean(1)
            elif aggregation_strategy == 'max':
                normed_agg_scores = normed_scores.max(1)
            elif aggregation_strategy == 'median':
                normed_agg_scores = np.median(normed_scores, axis=1)
            elif aggregation_strategy == 'p75':
                normed_agg_scores = np.percentile(normed_scores, 75, axis=1)
            elif aggregation_strategy == 'p95':
                normed_agg_scores = np.percentile(normed_scores, 95, axis=1)

            r, d = get_ts_eval(
                normed_agg_scores,
                test_labels.flatten()
            )

            f1_scores.append(r['f1'])
            if n is None:
                n = 'no'
            df_list.append((n, (aggregation_strategy, d)))
    best_score_idx = np.array(f1_scores).argmax()
    return dict(df_list), df_list[best_score_idx][1][1]


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

    scores = scores.detach().cpu().numpy()
    true_labels = true_labels.detach().cpu().numpy()

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
            if parts["aux_tgt"].dim() == 2:
                if args.aggregation_strategy == 'mean':
                    aux_log_prob = aux_log_prob.mean(
                        dim=2)  # TODO: mean, but very basic
                elif args.aggregation_strategy == 'max':
                    aux_log_prob = aux_log_prob.max(dim=2).values
                else:
                    raise NotImplementedError(
                        "Anomaly Aggregation Strategy not implemented.")

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
    processed_dir = f'data_dir/{args.dataset}/processed'
    if os.path.exists(processed_dir):
        shutil.rmtree(processed_dir)

    logging.debug(f"{experiment_log_file_string} -- Experiment ID={experiment_id}")
    if args.seed > 0:
        set_seed(args.seed)
    logging.debug(f"Seed set to {args.seed}")
    logging.debug(f'Parameters set: {vars(args)}')

    if provider is None:
        logging.info("Instantiating data provider")
        if args.dataset in ['SWaT', 'WaDi', 'SMD']:
            provider = ADProvider(
                data_dir='data_dir', dataset=args.dataset,
                window_length=args.data_window_length, window_overlap=args.data_window_overlap,
                n_samples=1000 if args.debug else None,
                data_normalization_strategy=args.data_normalization_strategy
            )
        elif args.dataset == 'aero':
            provider = AeroDataProvider(data_dir="data_dir/aero", subsample=2)
        else:
            raise ValueError(f"Unknown dataset {args.dataset}")
    else:
        logging.info("Using provided data provider")

    dl_trn = provider.get_train_loader(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
        drop_last=False,  #what does this do? Necessary?
    )
    dl_tst = provider.get_test_loader(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=8,
        pin_memory=True,
    )

    use_validation = True
    try:
        dl_val = provider.get_val_loader(
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=8,
            pin_memory=True,
        )
    except NotImplementedError:
        use_validation = False

    desired_t = torch.linspace(0, 1.00, provider.num_timepoints, device=args.device).float()

    recog_net = PhysioNetRecogNetwork(
        mtan_input_dim=provider.input_dim,
        mtan_hidden_dim=args.h_dim,
        use_atanh=args.use_atanh
    )

    recon_net = GenericMLP(
        inp_dim=args.z_dim,
        out_dim=provider.input_dim,
        n_hidden=args.dec_hidden_dim,
        n_layers=args.n_dec_layers,
        non_linear=args.non_linear_decoder
    )

    pxz_net = PathToGaussianDecoder(
        mu_map=recon_net,
        sigma_map=None,
        initial_sigma=args.initial_sigma) # TODO: is this initial sigma ok so?

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
    )
    modules = modules.to(args.device)

    param_groups = [
        {"params": recon_net.parameters(), "weight_decay": 1e-4},
        {"params": recog_net.parameters()},
        #{"params": pxz_net.parameters()},
        {"params": qzx_net.parameters()},
        # Specific weight decay for fc2
    ]


    optimizer = optim.Adam(param_groups, lr=args.lr)
    #optimizer = optim.SGD(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1)

    logging.debug(f"Number of model parameters={count_parameters(modules)}")

    elbo_loss = ELBO(reduction="mean")

    stats = defaultdict(list)
    stats_mask = {
        "trn": ["log_pxz", "kl0", "klp", "loss"],
        "tst": ["loss", "f1"],
        "oth": ["lr"],
    }
    if use_validation:
        stats_mask["val"] = ["loss", "aux_val", "aux_log_prob"]

    pm = ProgressMessage(stats_mask)

    best_epoch_val_aux = np.inf
    best_epoch_tst_aux = np.inf

    for epoch in range(1, args.n_epochs + 1):

        trn_stats = generic_train(
            args,
            dl_trn,
            modules,
            elbo_loss,
            None,
            optimizer,
            desired_t,
            args.device
        )

        tst_stats = evaluate(args, dl_tst, modules, elbo_loss, desired_t, args.device, epoch=epoch)

        if use_validation:
            val_stats = evaluate(args, dl_val, modules, elbo_loss, desired_t, args.device, epoch=epoch)

        stats["oth"].append({"lr": scheduler.get_last_lr()[-1]})
        scheduler.step()

        # if val_stats["aux_val"] < best_epoch_val_aux:
        #     best_epoch_val_aux = val_stats["aux_val"]
        #     best_epoch_tst_aux = tst_stats["aux_val"]
        #     save_checkpoint(
        #         args,
        #         'best',
        #         experiment_id,
        #         modules,
        #         desired_t)
        # val_stats["aux_val*"] = best_epoch_val_aux
        # tst_stats["aux_val*"] = best_epoch_tst_aux

        stats["trn"].append(trn_stats)
        stats2tensorboard(trn_stats, writer, epoch, prefix_='trn')

        stats["tst"].append(tst_stats)
        stats2tensorboard(tst_stats, writer, epoch, prefix_='tst')

        if use_validation:
            stats["val"].append(val_stats)
            stats2tensorboard(val_stats, writer, epoch, prefix_='val')

        if args.checkpoint_at and (epoch in args.checkpoint_at):
            save_checkpoint(
                args,
                epoch,
                experiment_id_str,
                modules,
                desired_t)

        msg = pm.build_progress_message(stats, epoch)
        logging.debug(msg)

        if args.enable_file_logging:
            fname = os.path.join(
                args.log_dir, f"{experiment_id_str}.json"
            )
            save_stats(args, stats, fname)

    finalstats2tensorboard(writer_=writer, params_=vars(args),
                           stats=stats["tst"], args=args)

    logging.shutdown()
    writer.close()


def evaluate(
    args,
    dl: torch.utils.data.DataLoader,
    modules: nn.ModuleDict,
    elbo_loss: nn.Module,
    desired_t: torch.Tensor,
    device: str,
    normalization_stats=None,
    epoch: int = 1,
):
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

            # from batch -> to one ts again
            aux_log_prob = aux_log_prob.reshape(-1, aux_log_prob.shape[-1])

           # if epoch % 10 == 0:
            all_labels.append(parts["aux_tgt"].flatten())
            all_scores.append(aux_log_prob)

            batch_len = parts["evd_obs"].shape[0]
            stats["loss"].append(loss.item() * batch_len)

            stats["elbo"].append(elbo_val.item() * batch_len)
            stats["kl0"].append(elbo_parts["kl0"].item() * batch_len)
            stats["klp"].append(elbo_parts["klp"].item() * batch_len)
            stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)

    stats = {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}

    #if epoch % 10 == 0:
    best_metrics = logprob2f1s(all_scores, all_labels)
    stats['f1'] = best_metrics.iloc[0, 1]
    return stats


def main():
    parser = extend_argparse(generic_parser)
    args_ = parser.parse_args()
    print(args_)
    start_experiment(args_, provider=None)


if __name__ == "__main__":
    main()
