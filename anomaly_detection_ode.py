"""Anomaly detection with a classic Latent ODE model (torchdiffeq), instead of
the Latent SDE on Homogeneous Spaces framework used by anomaly_detection.py.

Only a Euclidean latent space (Rn) is supported here -- there is no sphere
embedding / SO(n) or GL(n) path-distribution machinery. The recognition
network encodes observations into a single Gaussian q(z0|x) over the initial
latent state; the latent trajectory z(t) is obtained deterministically by
integrating a neural ODE dz/dt = f(z) with torchdiffeq.odeint from t=0 to the
reference grid; the trajectory is decoded back to observation space exactly
like in anomaly_detection.py. Because there is no stochastic path process,
the ELBO here only has a kl0 (initial-state) term -- no klp (path) term.

This module deliberately reuses anomaly_detection.py (imported, not modified)
for everything that is dataset-/orchestration-related and model-agnostic:
dataset providers, the DatasetSlice hybrid-layout helper, tensorboard/W&B
plumbing, scoring, and final-metrics aggregation. Only the model itself and
the training/evaluation loops that depend on it are reimplemented.
"""

import argparse
import datetime
import json
import logging
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import reduce
from torch import Tensor
from torch.distributions import Normal, kl_divergence
from torch.nn.modules.loss import _Loss
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchdiffeq import odeint

import anomaly_detection as ad
from core.models import GenericMLP, PathToGaussianDecoder, PhysioNetRecogNetwork
from data.ad_provider import ADProvider
from data.nasa_provider import NASAProvider
from data.psm_provider import PSMProvider
from data.qad_provider import QADProvider
from data.smd_provider import SMDProvider
from data.tsb_ad_m_provider import TSBADMProvider
from utils.logger import set_up_logging
from utils.misc import (
    ProgressMessage,
    append_final_metrics_csv,
    count_parameters,
    save_checkpoint,
    save_stats,
    scatter_obs_and_msk,
    set_seed,
)
from utils.parser import generic_parser, get_partition_batch_size, remove_argument


DATASET_CHOICES = ad.DATASET_CHOICES
DEFAULT_CFG_DIR = Path("cfg") / "anomaly_detection_ode"
FIXED_STEP_SOLVERS = {"euler", "midpoint", "rk4"}
_INV_SOFTPLUS_ONE = math.log(math.expm1(1.0))  # softplus(_INV_SOFTPLUS_ONE) == 1.0


def _load_dataset_config(dataset: str, config_file: str | None = None) -> dict:
    """Load JSON config for a dataset; return empty dict when unavailable.

    Mirrors anomaly_detection._load_dataset_config but reads from
    DEFAULT_CFG_DIR (cfg/anomaly_detection_ode/) since the ODE model has a
    different hyperparameter surface than the SDE model.
    """
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


def extend_argparse(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    # Not applicable to a Euclidean-latent-space Neural ODE model.
    remove_argument(parser, "--n-deg")
    remove_argument(parser, "--klp-weight")

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
            "uses cfg/anomaly_detection_ode/<dataset>.json."
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
        "--trace-ids",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Optional trace/sub-dataset selectors for multi-trace benchmarks. "
            "Each token accepts dataset_id (preferred) or zero-based trace index. "
            "Supports comma-separated values and repeated tokens."
        ),
    )
    group.add_argument(
        "--wandb-project",
        type=str,
        default="latent-sde-on-hs-anomaly-detection-ode",
        help="Weights & Biases project name.",
    )
    group.add_argument("--wandb-entity", type=str, default=None, help="Optional W&B entity / team name.")
    group.add_argument("--wandb-name", type=str, default=None, help="Optional explicit W&B run name. If omitted, a descriptive name is generated.")
    group.add_argument("--wandb-group", type=str, default=None, help="Optional W&B group name. Defaults to the selected dataset/model combination.")
    group.add_argument("--wandb-tags", nargs="*", default=[], help="Optional W&B tags.")
    group.add_argument("--wandb-mode", type=str, choices=["online", "offline", "disabled"], default="online", help="W&B mode.")
    group.add_argument("--wandb-disabled", action=argparse.BooleanOptionalAction, default=False, help="Disable W&B logging entirely.")

    ode_group = parser.add_argument_group("Neural ODE arguments (torchdiffeq)")
    ode_group.add_argument("--ode-hidden-dim", type=int, default=64, help="Hidden width of the latent ODE dynamics MLP dz/dt = f(z).")
    ode_group.add_argument("--ode-n-layers", type=int, default=3, help="Number of layers in the latent ODE dynamics MLP.")
    ode_group.add_argument(
        "--ode-solver",
        type=str,
        choices=["dopri5", "dopri8", "bosh3", "adaptive_heun", "euler", "midpoint", "rk4", "explicit_adams", "implicit_adams"],
        default="dopri5",
        help="torchdiffeq solver used to integrate the latent trajectory z(t).",
    )
    ode_group.add_argument("--ode-rtol", type=float, default=1e-3, help="Relative tolerance (adaptive-step solvers).")
    ode_group.add_argument("--ode-atol", type=float, default=1e-4, help="Absolute tolerance (adaptive-step solvers).")
    ode_group.add_argument(
        "--ode-step-size",
        type=float,
        default=None,
        help="Fixed step size for fixed-step solvers (euler/midpoint/rk4). Defaults to the reference-grid spacing if unset.",
    )

    return parser


class ODEFunc(nn.Module):
    """Neural ODE dynamics dz/dt = f(z) (autonomous, no explicit t dependence)."""

    def __init__(self, z_dim: int, hidden_dim: int = 64, n_layers: int = 3) -> None:
        super().__init__()
        assert n_layers >= 1, "Number of layers needs to be >= 1"

        layers = []
        in_dim = z_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(in_dim, hidden_dim), nn.Tanh()]
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, z_dim))
        self.net = nn.Sequential(*layers)

        # Small init keeps the initial vector field close to zero, which
        # stabilizes early training (standard trick for Neural ODEs).
        for module in self.net:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.1)
                nn.init.zeros_(module.bias)

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        return self.net(z)


class ODEPrior(nn.Module):
    """Prior p(z0) = N(mu, softplus(log_sigma)); optionally learnable."""

    def __init__(self, z_dim: int, learnable: bool = False) -> None:
        super().__init__()
        init_log_sigma = torch.full((z_dim,), _INV_SOFTPLUS_ONE)
        if learnable:
            self.mu = nn.Parameter(torch.zeros(z_dim))
            self.log_sigma = nn.Parameter(init_log_sigma.clone())
        else:
            self.register_buffer("mu", torch.zeros(z_dim))
            self.register_buffer("log_sigma", init_log_sigma)

    def forward(self, batch_size: int) -> Normal:
        mu = self.mu.unsqueeze(0).expand(batch_size, -1)
        sigma = F.softplus(self.log_sigma).unsqueeze(0).expand(batch_size, -1)
        return Normal(mu, sigma)


class LatentODEELBO(_Loss):
    """ELBO for a Latent ODE: only a kl0 (initial-state) term, no path KL."""

    def __init__(self, reduction: str = "mean") -> None:
        super().__init__(reduction=reduction)
        self.reduction = reduction

    def __repr__(self) -> str:
        return f"LatentODEELBO(reduction={self.reduction})"

    def forward(self, qz0, pz0, likelihood, evd_obs, evd_tps, evd_msk, weights=None):
        if weights is None:
            weights = {"pxz_weight": 1.0, "kl0_weight": 1.0}

        kl0 = kl_divergence(qz0, pz0).sum(dim=-1)  # -> [batch_len]

        mc_samples, _, num_tps, *_ = likelihood.mean.shape
        evd_obs, evd_msk = scatter_obs_and_msk(evd_obs, evd_msk, evd_tps, num_tps, mc_samples)

        log_pxz = -likelihood.log_prob(evd_obs)
        log_pxz[evd_msk == 0] = 0
        log_pxz = log_pxz.mean(dim=0)
        log_pxz = reduce(log_pxz, "b ... -> b", "sum")
        numel = reduce(evd_msk[0], "b ... -> b", "sum")

        elbo = (weights["kl0_weight"] * kl0 + weights["pxz_weight"] * log_pxz) / numel

        if self.reduction == "mean":
            return elbo.mean(), {"kl0": kl0.mean(), "log_pxz": log_pxz.mean()}
        elif self.reduction == "sum":
            return elbo.sum(), {"kl0": kl0.sum(), "log_pxz": log_pxz.sum()}
        else:
            return elbo, {"kl0": kl0, "log_pxz": log_pxz}


def build_modules_and_optim(args, input_dim, desired_t):
    recog_net = PhysioNetRecogNetwork(
        mtan_input_dim=input_dim,
        mtan_hidden_dim=args.h_dim,
        use_atanh=args.use_atanh,
    )

    z0_net = nn.Linear(args.h_dim, 2 * args.z_dim)
    ode_func = ODEFunc(z_dim=args.z_dim, hidden_dim=args.ode_hidden_dim, n_layers=args.ode_n_layers)
    prior_net = ODEPrior(z_dim=args.z_dim, learnable=args.learnable_prior)

    recon_net = GenericMLP(
        inp_dim=args.z_dim,
        out_dim=input_dim,
        n_hidden=args.dec_hidden_dim,
        n_layers=args.n_dec_layers,
        non_linear=args.non_linear_decoder,
    )
    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=args.initial_sigma)

    if args.freeze_sigma:
        logging.debug("Froze sigma when computing PathToGaussianDecoder")
        pxz_net.sigma.requires_grad = False

    modules = nn.ModuleDict(
        {
            "recog_net": recog_net,
            "z0_net": z0_net,
            "ode_func": ode_func,
            "prior_net": prior_net,
            "recon_net": recon_net,
            "pxz_net": pxz_net,
        }
    ).to(args.device)

    optimizer = optim.Adam(modules.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, args.restart, eta_min=0, last_epoch=-1)
    elbo_loss = LatentODEELBO(reduction="mean")

    logging.debug(f"Number of model parameters={count_parameters(modules)}")
    return modules, optimizer, scheduler, elbo_loss


def _ode_integration_options(args, desired_t: Tensor):
    if args.ode_solver not in FIXED_STEP_SOLVERS:
        return None
    step_size = args.ode_step_size
    if step_size is None:
        step_size = (desired_t[-1] - desired_t[0]).item() / max(desired_t.numel() - 1, 1)
    return {"step_size": step_size}


def latent_ode_rollout(args, modules, inp, desired_t: Tensor, mc_samples: int):
    """Encode -> sample z0 -> integrate the ODE across desired_t -> decode."""
    h = modules["recog_net"](inp)
    mu0, log_sigma0_raw = modules["z0_net"](h).chunk(2, dim=-1)
    sigma0 = F.softplus(log_sigma0_raw) + 1e-3
    qz0 = Normal(mu0, sigma0)

    batch_size = mu0.shape[0]
    pz0 = modules["prior_net"](batch_size)

    z0 = qz0.rsample((mc_samples,))  # [mc, batch, z_dim]
    z0_flat = z0.reshape(-1, z0.shape[-1])

    options = _ode_integration_options(args, desired_t)
    zt = odeint(
        modules["ode_func"], z0_flat, desired_t,
        method=args.ode_solver, rtol=args.ode_rtol, atol=args.ode_atol,
        options=options,
    )  # [T, mc*batch, z_dim]

    num_t = zt.shape[0]
    zt = zt.view(num_t, mc_samples, batch_size, -1).permute(1, 2, 0, 3)  # [mc, batch, T, z_dim]

    pxz = modules["pxz_net"](zt)
    return qz0, pz0, pxz


def generic_train_ode(args, dl, modules, elbo_loss, optimizer, desired_t, device):
    stats = defaultdict(list)

    modules.train()
    for _, batch in enumerate(dl):
        parts = {key: val.to(device) for key, val in batch.items()}
        inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])
        batch_len = parts["evd_obs"].shape[0]

        qz0, pz0, pxz = latent_ode_rollout(args, modules, inp, desired_t, args.mc_train_samples)

        elbo_val, elbo_parts = elbo_loss(
            qz0, pz0, pxz,
            parts["evd_obs"], parts["evd_tid"], parts["evd_msk"],
            {"kl0_weight": args.kl0_weight, "pxz_weight": args.pxz_weight},
        )
        loss = elbo_val

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        stats["loss"].append(loss.item() * batch_len)
        stats["elbo"].append(elbo_val.item() * batch_len)
        stats["kl0"].append(elbo_parts["kl0"].item() * batch_len)
        stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)

    stats = {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}
    return stats


def calculate_z_normalization_values_ode(args, dl, modules, desired_t, device):
    stats = defaultdict(list)

    modules.eval()
    with torch.no_grad():
        all_scores_list = []
        for _, batch in enumerate(dl):
            parts = {key: val.to(device) for key, val in batch.items()}
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            _, _, pxz = latent_ode_rollout(args, modules, inp, desired_t, args.mc_eval_samples)

            aux_log_prob = -pxz.log_prob(parts["evd_obs"])
            if aux_log_prob.dim() >= 4:
                aux_log_prob = aux_log_prob.squeeze()
            if aux_log_prob.dim() == 2:
                aux_log_prob = aux_log_prob[None, :, :]

            aux_log_prob = aux_log_prob.mean(dim=0)
            all_scores_list.append(aux_log_prob)

    all_scores = torch.cat(all_scores_list, dim=0)

    stats["mu"] = all_scores.mean(dim=0)
    stats["sigma"] = all_scores.std(dim=0)
    stats["max"] = all_scores.max(dim=0).values
    stats["min"] = all_scores.min(dim=0).values
    return stats


def evaluate_ode(
    args,
    dl: torch.utils.data.DataLoader,
    modules: nn.ModuleDict,
    elbo_loss: nn.Module,
    desired_t: Tensor,
    device: str,
    normalization_stats=None,
    epoch: int = 1,
    test=True,
):
    stats = defaultdict(list)

    all_scores = np.zeros(
        (int(dl.dataset.indcs.max().detach().numpy().tolist()) + 1, dl.dataset.input_dim)
    )
    all_labels = np.zeros(all_scores.shape[0])
    normalize_counts = np.zeros(all_scores.shape[0])

    modules.eval()
    with torch.no_grad():
        for _, batch in enumerate(dl):
            parts = {key: val.to(device) for key, val in batch.items()}

            indcs = parts["inp_indcs"].cpu().numpy().astype(int)
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            qz0, pz0, pxz = latent_ode_rollout(args, modules, inp, desired_t, args.mc_eval_samples)

            elbo_val, elbo_parts = elbo_loss(
                qz0, pz0, pxz,
                parts["evd_obs"], parts["evd_tid"], parts["evd_msk"],
                {"kl0_weight": args.kl0_weight, "pxz_weight": args.pxz_weight},
            )
            loss = elbo_val

            aux_log_prob = -pxz.log_prob(parts["evd_obs"])
            if aux_log_prob.dim() >= 4:
                aux_log_prob = aux_log_prob.squeeze()
            if aux_log_prob.dim() == 2:
                aux_log_prob = aux_log_prob[None, :, :]

            if normalization_stats is not None:
                aux_log_prob = (aux_log_prob - normalization_stats["min"]) / (
                    normalization_stats["max"] - normalization_stats["min"]
                )

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
            stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)

    stats = {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}

    all_scores = np.divide(
        all_scores,
        normalize_counts[:, None],
        out=np.zeros_like(all_scores),
        where=normalize_counts[:, None] > 0,
    )

    if test:
        best_metrics = ad.eval_scores(all_scores, all_labels, window_length=args.data_window_length)
        for key, value in best_metrics.items():
            stats[key.lower()] = value

    return stats


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
    wandb_run=None,
):
    desired_t = torch.linspace(0, 1.00, num_timepoints, device=args.device).float()
    modules, optimizer, scheduler, elbo_loss = build_modules_and_optim(args, input_dim, desired_t)

    stats = defaultdict(list)
    stats_mask = {
        "oth": ["esc", "lr"],
        "trn": ["log_pxz", "kl0", "loss"],
        "val": ["log_pxz", "kl0", "loss"],
        "tst": ["loss", "auc_roc", "auc_pr", "f1"],
    }
    if not args.freeze_sigma:
        stats_mask["oth"].append("sig")

    pm = ProgressMessage(stats_mask)
    best_stats = None
    es_counter = 0
    best_val_loss = np.inf

    for epoch in range(1, args.n_epochs + 1):
        trn_stats = generic_train_ode(args, dl_trn, modules, elbo_loss, optimizer, desired_t, args.device)

        normalization_scores = None
        if args.normalize_score:
            normalization_scores = calculate_z_normalization_values_ode(
                args, dl_trn, modules, desired_t, args.device)

        tst_stats = evaluate_ode(
            args, dl_tst, modules, elbo_loss, desired_t, args.device,
            normalization_stats=normalization_scores, epoch=epoch
        )

        val_stats = evaluate_ode(
            args, dl_val, modules, elbo_loss, desired_t, args.device,
            normalization_stats=normalization_scores, epoch=epoch, test=False
        )

        val_loss = val_stats["loss"]
        to_append = {"lr": scheduler.get_last_lr()[-1],
                     "esc": es_counter,
                     "sig": modules['pxz_net'].sigma.item()}
        if val_loss < (best_val_loss - args.early_stopping_min_delta):
            best_val_loss = val_loss
            best_stats = tst_stats
            es_counter = 0
        else:
            es_counter += 1
            if es_counter >= 4 * args.restart:  # early stopping patience shall be longer than one cosine sheduling
                logging.info(f"Early stopping triggered at epoch {epoch}.")
                stats["trn"].append(trn_stats)
                stats["tst"].append(tst_stats)
                stats["val"].append(val_stats)
                ad.stats2tensorboard(trn_stats, val_stats, tst_stats, writer, epoch)
                if wandb_run is not None and not ad._wandb_log_epoch(wandb_run, epoch, stats_prefix, trn_stats, val_stats, tst_stats, to_append):
                    wandb_run = None
                break

        to_append["esc"] = es_counter

        stats["oth"].append(to_append)
        scheduler.step()

        stats["trn"].append(trn_stats)
        stats["tst"].append(tst_stats)
        stats["val"].append(val_stats)
        ad.stats2tensorboard(trn_stats, val_stats, tst_stats, writer, epoch)
        if wandb_run is not None and not ad._wandb_log_epoch(wandb_run, epoch, stats_prefix, trn_stats, val_stats, tst_stats, to_append):
            wandb_run = None

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


def _build_wandb_context(args, run_number: int, total_runs: int, run_seed: int):
    benchmark_name = args.dataset if getattr(args, "trace_ids", None) is None else f"{args.dataset}:{','.join(args.trace_ids)}"
    return {
        "benchmark_name": benchmark_name,
        "model_variant": "Rn",  # Euclidean latent space only -- no sphere embedding for the ODE model.
        "run_number": run_number,
        "total_runs": total_runs,
        "run_seed": run_seed,
    }


def start_experiment(args, provider=None, store_final_metrics=True, run_number: int = 1, total_runs: int = 1):
    experiment_id = datetime.datetime.now().strftime('%y%m%d-%H:%M:%S')
    experiment_log_file_string = 'DEBUG' if args.debug else f'AD_ODE_{args.dataset}'
    experiment_id_str = f'{experiment_log_file_string}_{experiment_id}'

    if args.debug:
        args.n_epochs = 1

    writer = SummaryWriter(f'runs/{experiment_id_str}')
    log_txt_path = Path(args.log_dir) / f"{experiment_id_str}.txt" if args.log_dir is not None else None
    log_json_path = Path(args.log_dir) / f"{experiment_id_str}.json" if args.log_dir is not None else None
    runtime_context = _build_wandb_context(args, run_number=run_number, total_runs=total_runs, run_seed=args.seed)
    output_paths = {
        "log_txt": log_txt_path,
        "log_json": log_json_path,
    }

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

    logging.debug(f"{experiment_log_file_string} -- Experiment ID={experiment_id}")
    if args.seed > 0:
        set_seed(args.seed)
    logging.debug(f"Seed set to {args.seed}")
    logging.debug(f'Parameters set: {vars(args)}')
    wandb_run = ad._wandb_init_run(args, experiment_id_str, runtime_context, output_paths)
    final_result = None

    def _store_final_metrics(final_metrics: dict):
        benchmark_name = args.dataset
        if getattr(args, "trace_ids", None):
            benchmark_name = f"{args.dataset}:{','.join(args.trace_ids)}"
        append_final_metrics_csv(
            csv_path=getattr(args, "final_metrics_csv", "logs/final_metrics.csv"),
            benchmark=benchmark_name,
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
                seed=args.seed,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy
            )
        elif args.dataset == 'SMD':
            provider = SMDProvider(
                data_dir=data_dir,
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                seed=args.seed,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy,
            )
        elif args.dataset == 'QAD':
            dataset_number = None
            if args.trace_ids is not None and len(args.trace_ids) == 1:
                dataset_number = int(args.trace_ids[0])
            provider = QADProvider(
                data_dir=data_dir,
                dataset_number=dataset_number,
                window_length=args.data_window_length,
                seed=args.seed,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy,
                raw_subdir="qad_clean_txt_100Hz",
            )
        elif args.dataset == 'TSB-AD-M':
            dataset_number = None
            if args.trace_ids is not None:
                try:
                    dataset_number = [int(trace_id) for trace_id in args.trace_ids]
                    if len(dataset_number) == 1:
                        dataset_number = dataset_number[0]
                except ValueError as exc:
                    raise ValueError(
                        f"--trace-ids for dataset {args.dataset} must be numeric file indices, got {args.trace_ids}"
                    ) from exc
            provider = TSBADMProvider(
                data_dir=data_dir,
                dataset_number=dataset_number,
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                seed=args.seed,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask,
                data_normalization_strategy=args.data_normalization_strategy,
            )
        elif args.dataset in ['SMAP', 'MSL']:
            provider = NASAProvider(
                data_dir=data_dir, dataset=args.dataset,
                window_length=args.data_window_length,
                seed=args.seed,
                subsample=args.subsample,
                fixed_subsample_mask=args.fixed_subsample_mask)
        elif args.dataset == 'PSM':
            provider = PSMProvider(
                data_dir=data_dir,
                window_length=args.data_window_length,
                window_overlap=args.data_window_overlap,
                seed=args.seed,
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

            selected_indices = list(range(active_provider.num_datasets))
            requested_traces = getattr(args, "trace_ids", None)
            if requested_traces is not None:

                id_to_idx = {}
                for ds_idx in range(active_provider.num_datasets):
                    ds = active_provider._ds_trn.get_dataset(ds_idx)
                    dataset_id = str(ds.get("dataset_id", str(ds_idx)))
                    id_to_idx[dataset_id] = ds_idx

                resolved_indices = []
                unknown = []
                for requested_trace in requested_traces:
                    if requested_trace in id_to_idx:
                        resolved_indices.append(id_to_idx[requested_trace])
                        continue

                    try:
                        req_idx = int(requested_trace)
                        resolved_indices.append(req_idx)
                    except ValueError:
                        unknown.append(requested_trace)

                if unknown:
                    available_ids = sorted(id_to_idx.keys())
                    raise ValueError(
                        f"Unknown trace(s) {unknown} for dataset {args.dataset}. "
                        f"Use one of dataset_ids={available_ids} or an index in "
                        f"[0, {active_provider.num_datasets - 1}]."
                    )

                selected_indices = list(dict.fromkeys(resolved_indices))
                if not selected_indices:
                    raise ValueError("--trace-ids did not resolve to any trace.")

                logging.info(
                    "Restricting run to traces via --trace-ids=%s (resolved idx=%s)",
                    requested_traces,
                    selected_indices,
                )

            for ds_idx, _ in enumerate(selected_indices):
                trn_slice = ad.DatasetSlice(active_provider._ds_trn, ds_idx)
                tst_slice = ad.DatasetSlice(active_provider._ds_tst, ds_idx)
                val_slice = ad.DatasetSlice(active_provider._ds_val, ds_idx)

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
                    wandb_run=wandb_run,
                )

                per_dataset_stats[dataset_id] = tst_stats
                per_dataset_histories[dataset_id] = hist_stats

            if len(per_dataset_stats) == 1:
                only_id = next(iter(per_dataset_stats.keys()))
                stats2pass = per_dataset_histories[only_id]["tst"]
                best_stats2pass = per_dataset_stats[only_id]
                ad.finalstats2tensorboard(
                    writer_=writer,
                    params_=vars(args),
                    stats=stats2pass,
                    args=args,
                )
                if store_final_metrics:
                    _store_final_metrics(best_stats2pass)
                logging.info(f"Final metrics for dataset trace {only_id}: {best_stats2pass}")
                return per_dataset_stats[only_id]

            macro_stats = ad.compute_macro_metrics(per_dataset_stats)
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
            return combined_stats

        # Fallback path for providers without hybrid layout.
        if getattr(args, "trace_ids", None) is not None:
            raise ValueError(
                f"--trace-ids is only supported for multi-trace datasets. "
                f"Dataset {args.dataset} does not expose trace-wise training in this provider."
            )

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
            wandb_run=wandb_run,
        )

        ad.finalstats2tensorboard(writer_=writer, params_=vars(args), stats=stats["tst"], args=args)
        if store_final_metrics:
            _store_final_metrics(tst_stats)
        return tst_stats

    try:
        final_result = _run_with_provider(provider)
        return final_result
    finally:
        if wandb_run is not None:
            if not ad._wandb_log_final_outputs(wandb_run, final_result or {}, output_paths):
                wandb_run = None
        if wandb_run is not None:
            wandb_run.finish()
        writer.close()
        if args.delete_processed_data:
            ad.delete_processed_data(args.dataset, data_dir=data_dir)
        if provider is not None and hasattr(provider, "cleanup"):
            try:
                provider.cleanup()
            except Exception as err:
                logging.warning(f"Provider cleanup failed: {err}")
        logging.shutdown()


def main():
    argv = sys.argv[1:]
    has_cli_batch_size = ad._has_explicit_batch_size_cli_override(argv)
    bootstrap_args = ad._extract_bootstrap_args(argv)

    parser = extend_argparse(generic_parser)
    dataset_cfg = _load_dataset_config(bootstrap_args.dataset, bootstrap_args.config_file)
    ad._validate_config_keys(parser, dataset_cfg, bootstrap_args.dataset)
    parser.set_defaults(**dataset_cfg)

    # Final parse: explicit CLI values override dataset config defaults.
    args_ = parser.parse_args(argv)
    args_.trace_ids = ad._normalize_trace_ids(args_.trace_ids)
    if not has_cli_batch_size:
        partition_batch_size = get_partition_batch_size()
        if partition_batch_size is not None:
            args_.batch_size = partition_batch_size
    if args_.runs < 1:
        parser.error("--runs must be >= 1")

    run_results = []
    for run_idx in range(args_.runs):
        ad.delete_processed_data(args_.dataset, data_dir=args_.data_dir)
        logging.info("Starting run %d/%d", run_idx + 1, args_.runs)
        run_result = start_experiment(
            args_,
            provider=None,
            store_final_metrics=False,
            run_number=run_idx + 1,
            total_runs=args_.runs,
        )
        run_results.append(run_result)

    aggregated_metrics = ad.aggregate_run_metrics(run_results)
    benchmark_name = args_.dataset if args_.trace_ids is None else f"{args_.dataset}:{','.join(args_.trace_ids)}"
    append_final_metrics_csv(
        csv_path=getattr(args_, "final_metrics_csv", "logs/final_metrics.csv"),
        benchmark=benchmark_name,
        run_datetime=datetime.datetime.now().strftime('%y%m%d-%H:%M:%S'),
        metrics=aggregated_metrics,
    )
    logging.info("Aggregated metrics over %d run(s): %s", args_.runs, aggregated_metrics)


if __name__ == "__main__":
    main()