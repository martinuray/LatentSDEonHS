"""Pendulum angle regression from Sec. 4.1 of

    Zeng S., Graf F. and Kwitt, R.
    Latent SDEs on Homogeneous Spaces
    NeurIPS 2023
"""

import os
import logging
import argparse
import numpy as np
from random import SystemRandom
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
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
from utils.anomaly_detection import anomaly_detection_performances
from utils.logger import set_up_logging
from utils.misc import (
    set_seed,
    count_parameters,
    ProgressMessage,
    save_checkpoint,
    save_stats)
from utils.parser import generic_parser


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    group = parser.add_argument_group("Experiment specific arguments")
    #group.add_argument("--aux-weight", type=float, default=10.0)
    group.add_argument("--num-features", type=int, default=4)
    group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    group.add_argument("--dec-hidden-dim", type=int, default=64)
    group.add_argument("--n-dec-layers", type=int, default=1)
    #group.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    return parser


def stats2tensorboard(stats_, writer_, epoch_, prefix_=''):
    for key_, value_ in stats_.items():
        writer_.add_scalar(f'{prefix_}{key_}', value_, epoch_)


def main():
    experiment_id = int(SystemRandom().random() * 100000)
    parser = extend_argparse(generic_parser)
    args = parser.parse_args()
    print(args)

    writer = SummaryWriter()
    #writer.add_scalars('params' ,{pkey_: pvalue_ for pkey_, pvalue_ in vars(args).items() if type(pvalue_) in [float, int]},)
    #writer.add_text('params',{pkey_: pvalue_ for pkey_, pvalue_ in vars(args) if type(pvalue_) in [float, int]},)

    experiment_log_file_string = 'anomaly_detection'

    set_up_logging(
        console_log_level=args.loglevel,
        console_log_color=True,
        logfile_file=os.path.join(args.log_dir, f"{experiment_log_file_string}_{experiment_id}.txt")
        if args.log_dir is not None
        else None,
        logfile_log_level=args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(created)d] [%(levelname)-8s] %(message)s%(color_off)s",
    )

    logging.debug(f"{experiment_log_file_string} -- Experiment ID={experiment_id}")
    if args.seed > 0:
        set_seed(args.seed)
    logging.debug(f"Seed set to {args.seed}")

    #provider = BasicDataProvider(data_dir='data_dir', num_features=args.num_features, sample_tp=1., data_kind=None)
    provider = ADProvider(data_dir='data_dir', data_kind='SWaT')
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
    )

    pxz_net = PathToGaussianDecoder(
        mu_map=recon_net,
        sigma_map=None,
        initial_sigma=0.1) # TODO: is this initial sigma ok so?

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

    optimizer = optim.Adam(modules.parameters(), lr=args.lr)
    #optimizer = optim.SGD(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1)

    logging.debug(f"Number of model parameters={count_parameters(modules)}")

    elbo_loss = ELBO(reduction="mean")

    stats = defaultdict(list)
    stats_mask = {
        "trn": ["log_pxz", "kl0", "klp", "loss"],
        "tst": ["loss"],
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
        tst_stats = evaluate(args, dl_tst, modules, elbo_loss, desired_t, args.device)
        if use_validation:
            val_stats = evaluate(args, dl_val, modules, elbo_loss, desired_t, args.device)

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
                experiment_id,
                modules,
                desired_t)

        msg = pm.build_progress_message(stats, epoch)
        logging.debug(msg)

        if args.enable_file_logging:
            fname = os.path.join(
                args.log_dir, f"{experiment_log_file_string}_{experiment_id}.json"
            )
            save_stats(args, stats, fname)

    writer.flush()


def evaluate(
    args,
    dl: torch.utils.data.DataLoader,
    modules: nn.ModuleDict,
    elbo_loss: nn.Module,
    desired_t: torch.Tensor,
    device: str,
    aux_perf: bool = False
):
    stats = defaultdict(list)

    modules.eval()
    with (torch.no_grad()):
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
            aux_log_prob = aux_log_prob.mean(dim=0)
            if parts["aux_tgt"].dim() == 2:
                aux_log_prob, _ = aux_log_prob.max(dim=2)

            if aux_perf:
                aux_tgt = (aux_log_prob < -torch.log(torch.Tensor([0.75]).to(device))) * 1
                aux_performance = anomaly_detection_performances(aux_tgt.to(device), parts["aux_tgt"].to(device))
                stats["aux_val"].append(aux_performance.item())
                stats["aux_log_prob"].append(aux_log_prob.mean().item())

            batch_len = parts["evd_obs"].shape[0]
            stats["loss"].append(loss.item() * batch_len)

            stats["elbo"].append(elbo_val.item() * batch_len)
            stats["kl0"].append(elbo_parts["kl0"].item() * batch_len)
            stats["klp"].append(elbo_parts["klp"].item() * batch_len)
            stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)

    stats = {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}
    return stats


if __name__ == "__main__":
    main()
