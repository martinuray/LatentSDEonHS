"""MVP fitting script: train the Latent SDE model on a single QAD trace.

This is deliberately scoped to *fitting only* (no evaluation, no anomaly
scoring, no wandb/tensorboard). The idea is to nail down a good fit on one
trace first; anomaly detection is layered on top incrementally afterwards.
"""

import argparse
from collections import defaultdict
import datetime
import importlib
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR

from core.models import (
    ELBO,
    GenericMLP,
    PathToGaussianDecoder,
    PhysioNetRecogNetwork,
    default_GLnPathDistributionEncoder,
    default_SOnPathDistributionEncoder,
)
from core.training import generic_train
from data.qad_provider import QADProvider
from utils.logger import set_up_logging
from utils.misc import count_parameters, save_checkpoint, set_seed


DEFAULT_CONFIG_PATH = "cfg/anomaly_detection/QAD.json"

# Only data/model/training hyperparameters are sourced from the dataset config;
# orchestration concerns (checkpointing, logging, reconstruction plotting) keep
# this script's own defaults since the shared QAD.json wasn't written for them.
CONFIG_ELIGIBLE_KEYS = {
    "data_dir",
    "data_window_length",
    "data_window_overlap",
    "data_normalization_strategy",
    "subsample",
    "fixed_subsample_mask",
    "z_dim",
    "h_dim",
    "n_deg",
    "dec_hidden_dim",
    "n_dec_layers",
    "non_linear_decoder",
    "use_atanh",
    "sphere_embedding",
    "sde",
    "learnable_prior",
    "initial_sigma",
    "freeze_sigma",
    "batch_size",
    "lr",
    "n_epochs",
    "restart",
    "kl0_weight",
    "klp_weight",
    "pxz_weight",
    "mc_train_samples",
    "eval_every_n_epochs",
    "early_stopping_patience",
    "early_stopping_min_delta",
    "seed",
    "device",
}


def load_dataset_config(config_path: str | None) -> dict:
    """Load hyperparameter defaults from a dataset JSON config (e.g. QAD.json).

    Only keys in CONFIG_ELIGIBLE_KEYS are applied; everything else in the file
    (wandb/runs/eval-only settings from the shared anomaly_detection config
    schema) is ignored since it doesn't apply to this fitting-only script.
    """
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        logging.warning("No config found at %s. Falling back to script defaults.", path)
        return {}

    with path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"Config must be a JSON object: {path}")

    applied = {key: value for key, value in cfg.items() if key in CONFIG_ELIGIBLE_KEYS}
    ignored = sorted(set(cfg.keys()) - CONFIG_ELIGIBLE_KEYS)
    if ignored:
        logging.debug("Ignoring config keys not used by fit_qad.py: %s", ignored)

    logging.info("Loaded %d config default(s) from %s", len(applied), path)
    return applied


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit the Latent SDE model on a single QAD trace (training only)."
    )

    cfg = parser.add_argument_group("Config arguments")
    cfg.add_argument(
        "--config-file", type=str, default=DEFAULT_CONFIG_PATH,
        help="Path to a dataset JSON config providing hyperparameter defaults (data/model/training only). Pass an empty string to disable.",
    )

    data = parser.add_argument_group("Data arguments")
    data.add_argument("--data-dir", type=str, default="data_dir")
    data.add_argument("--trace-id", type=int, default=1, help="QAD trace/dataset number to fit (train_<id>.txt).")
    data.add_argument("--data-window-length", type=int, default=200)
    data.add_argument("--data-window-overlap", type=float, default=0.5)
    data.add_argument("--data-normalization-strategy", choices=["none", "std", "min-max"], default="min-max")
    data.add_argument("--subsample", type=float, default=0.5, help="Fraction of input observations kept visible to the encoder.")
    data.add_argument("--fixed-subsample-mask", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--num-workers", type=int, default=8)

    model = parser.add_argument_group("Model arguments")
    model.add_argument("--z-dim", type=int, default=3)
    model.add_argument("--h-dim", type=int, default=12)
    model.add_argument("--n-deg", type=int, default=6)
    model.add_argument("--dec-hidden-dim", type=int, default=11)
    model.add_argument("--n-dec-layers", type=int, default=2)
    model.add_argument("--non-linear-decoder", action=argparse.BooleanOptionalAction, default=True)
    model.add_argument("--use-atanh", action=argparse.BooleanOptionalAction, default=False)
    model.add_argument("--sphere-embedding", action=argparse.BooleanOptionalAction, default=True, help="Use SOn path-distribution encoder. Disable for GLn.")
    model.add_argument("--sde", action=argparse.BooleanOptionalAction, default=True, help="Use SDE, or if not, ODE.")
    model.add_argument("--learnable-prior", action=argparse.BooleanOptionalAction, default=False)
    model.add_argument("--initial-sigma", type=float, default=0.2)
    model.add_argument("--freeze-sigma", action=argparse.BooleanOptionalAction, default=True)

    train = parser.add_argument_group("Training arguments")
    train.add_argument("--batch-size", type=int, default=256)
    train.add_argument("--lr", type=float, default=5e-2)
    train.add_argument("--n-epochs", type=int, default=180)
    train.add_argument("--restart", type=int, default=30, help="Cosine annealing restart period (epochs).")
    train.add_argument("--kl0-weight", type=float, default=1e-4)
    train.add_argument("--klp-weight", type=float, default=1e-2)
    train.add_argument("--pxz-weight", type=float, default=100.0)
    train.add_argument("--mc-train-samples", type=int, default=1)
    train.add_argument("--eval-every-n-epochs", type=int, default=1,
                      help="Run validation evaluation every k epochs (>=1).")
    train.add_argument("--early-stopping-patience", type=int, default=30,
                      help="Stop after this many consecutive validation checks without sufficient improvement. Set <=0 to disable.")
    train.add_argument("--early-stopping-min-delta", type=float, default=0.0,
                      help="Minimum decrease in validation loss to count as an improvement.")
    train.add_argument("--seed", type=int, default=-1)
    train.add_argument("--device", type=str, default="cuda")

    ckpt = parser.add_argument_group("Checkpointing/logging arguments")
    ckpt.add_argument("--enable-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints/qad_fit")
    ckpt.add_argument("--checkpoint-every-n-epochs", type=int, default=0, help="0 disables periodic checkpoints; the final epoch is always saved.")
    ckpt.add_argument("--log-every-n-epochs", type=int, default=10)
    ckpt.add_argument("--loglevel", choices=["debug", "info", "warning", "error", "critical"], default="info")

    recon = parser.add_argument_group("Reconstruction plotting arguments")
    recon.add_argument(
        "--reconstruct-at-k", type=int, default=1500,
        help="If >0, plot a data reconstruction every k-th epoch. 0 disables reconstruction plotting.",
    )
    recon.add_argument("--reconstruct-dir", type=str, default="out/reconstructions/qad_fit")
    recon.add_argument(
        "--reconstruct-n-windows", type=int, default=15,
        help="Number of windows to reconstruct and plot: the (middle) windows for train, "
             "and a contiguous block containing the labeled anomalous segment for test.",
    )
    recon.add_argument(
        "--reconstruct-mc-samples", type=int, default=100,
        help="Number of MC samples drawn when decoding the reconstruction; each is plotted "
             "as its own low-alpha trace (see notebooks/analyze_irregular_sine_exp.ipynb).",
    )
    recon.add_argument(
        "--reconstruct-gif", action=argparse.BooleanOptionalAction, default=True,
        help="Combine all reconstruction PNGs generated during the run into a single chronological GIF at the end.",
    )
    recon.add_argument(
        "--plot-latent-sphere", action=argparse.BooleanOptionalAction, default=True,
        help="Every k-th epoch (same cadence as reconstructions), plot the sampled latent "
             "paths of several train windows projected onto the unit sphere "
             "(cf. notebooks/analyze_irregular_sine_exp.py). Combined into its own GIF at the end.",
    )
    recon.add_argument(
        "--latent-sphere-n-windows", type=int, default=1000,
        help="Number of (middle) training windows drawn on the latent sphere, one colour each.",
    )
    recon.add_argument(
        "--latent-sphere-mc-samples", type=int, default=1,
        help="Number of latent-path samples drawn per window for the latent-sphere plot.",
    )
    recon.add_argument(
        "--reconstruct-gif-duration-ms", type=int, default=400,
        help="Per-frame display duration (milliseconds) for the reconstruction GIF.",
    )

    return parser


def resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available; falling back to CPU.")
        return "cpu"
    return requested


def build_modules_and_optim(args, input_dim, desired_t):
    recog_net = PhysioNetRecogNetwork(
        mtan_input_dim=input_dim,
        mtan_hidden_dim=args.h_dim,
        use_atanh=args.use_atanh,
    )

    recon_net = GenericMLP(
        inp_dim=args.z_dim,
        out_dim=input_dim,
        n_hidden=args.dec_hidden_dim,
        n_layers=args.n_dec_layers,
        non_linear=args.non_linear_decoder,
    )

    pxz_net = PathToGaussianDecoder(mu_map=recon_net, sigma_map=None, initial_sigma=args.initial_sigma)

    encoder_cls = default_SOnPathDistributionEncoder if args.sphere_embedding else default_GLnPathDistributionEncoder
    qzx_net = encoder_cls(
        h_dim=args.h_dim,
        z_dim=args.z_dim,
        n_deg=args.n_deg,
        learnable_prior=args.learnable_prior,
        time_min=0.0,
        time_max=2.0 * desired_t[-1].item(),
        sde=args.sde,
    )

    if args.freeze_sigma:
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

    return modules, optimizer, scheduler, elbo_loss


def evaluate_validation(args, dl, modules, elbo_loss, desired_t, device):
    """Evaluate ELBO-based loss on a validation loader."""
    stats = defaultdict(list)
    was_training = modules.training
    modules.eval()

    with torch.no_grad():
        for _, batch in enumerate(dl):
            parts = {key: val.to(device) for key, val in batch.items()}
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])
            batch_len = parts["evd_obs"].shape[0]

            h = modules["recog_net"](inp)
            qzx, pz = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((args.mc_train_samples,))
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
            stats["loss"].append(loss.item() * batch_len)
            stats["elbo"].append(elbo_val.item() * batch_len)
            stats["kl0"].append(elbo_parts["kl0"].item() * batch_len)
            stats["klp"].append(elbo_parts["klp"].item() * batch_len)
            stats["log_pxz"].append(elbo_parts["log_pxz"].item() * batch_len)

    if was_training:
        modules.train()

    return {key: np.sum(val) / len(dl.dataset) for key, val in stats.items()}


def _select_middle_window_indices(ds, n_select):
    n_windows = len(ds)
    n_select = min(n_select, n_windows)
    start = max(0, (n_windows - n_select) // 2)
    return list(range(start, start + n_select))


def _select_anomalous_window_indices(ds, n_select):
    """Pick a contiguous block of `n_select` windows that contains the
    labeled anomalous segment (if any), centering the block on it when the
    segment itself is shorter than the requested block size.

    Falls back to the middle of the trace when no window is labeled
    anomalous (e.g. a test trace without any injected anomaly).
    """
    n_windows = len(ds)
    n_select = min(n_select, n_windows)

    #anomalous = [i for i in range(n_windows) if bool((ds[i]["aux_tgt"] > 0).any())]
    #if not anomalous:
    #    return _select_middle_window_indices(ds, n_select)

    #first, last = anomalous[0], anomalous[-1]
    first, last = 675, 750
    span = last - first + 1
    start = first if span >= n_select else first - (n_select - span) // 2
    start = max(0, min(start, n_windows - n_select))
    return list(range(start, start + n_select))


def _gather_window_batch(ds, indices, device):
    samples = [ds[i] for i in indices]
    batch = {
        key: torch.stack([sample[key] for sample in samples], dim=0)
        for key in samples[0]
        if isinstance(samples[0][key], torch.Tensor)
    }
    return {key: val.to(device) for key, val in batch.items()}


def _decode_reconstruction(args, modules, desired_t, parts):
    """Encode+decode a batch of windows at `desired_t`, the same decoding
    timepoints used for training, so train/test reconstructions are directly
    comparable.

    Draws `args.reconstruct_mc_samples` decoder samples (kept separate,
    not averaged) so the plot can show the reconstruction spread as a
    spaghetti fan, matching notebooks/analyze_irregular_sine_exp.ipynb.

    Ground truth is `evd_obs` (the complete, unmasked window) rather than the
    subsampled `inp_obs` fed to the encoder, so the plot reflects reconstruction
    quality against the full underlying signal.
    """
    inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

    modules.eval()
    with torch.no_grad():
        h = modules["recog_net"](inp)
        qzx, _ = modules["qzx_net"](h, desired_t)
        zis = qzx.rsample((args.reconstruct_mc_samples,))
        pxz = modules["pxz_net"](zis)
        # Same score as anomaly_detection.py's evaluate(): negative log-likelihood
        # of the ground truth under the decoder, averaged over MC samples.
        nll = (-pxz.log_prob(parts["evd_obs"])).mean(dim=0).detach().cpu()  # (n_windows, n_time, dim)
    modules.train()

    recon_samples = pxz.mean.detach().cpu()  # (mc_samples, n_windows, n_time, dim)
    actual = parts["evd_obs"].detach().cpu()
    return actual, recon_samples, nll


def _highlight_anomalous_regions(ax, anomaly_mask):
    """Shade contiguous stretches of `anomaly_mask` (1D bool array aligned
    with the plotted x-axis) as a low-alpha red background band.
    """
    in_segment = False
    seg_start = 0
    for idx, flagged in enumerate(anomaly_mask.tolist()):
        if flagged and not in_segment:
            in_segment, seg_start = True, idx
        elif not flagged and in_segment:
            in_segment = False
            ax.axvspan(seg_start - 0.5, idx - 0.5, color="red", alpha=0.15, zorder=0)
    if in_segment:
        ax.axvspan(seg_start - 0.5, len(anomaly_mask) - 0.5, color="red", alpha=0.15, zorder=0)


def _plot_actual_vs_reconstructed(actual, recon_samples, anomaly_mask, title, out_path, likelihood=None):
    actual = np.asarray(actual)
    recon_samples = np.asarray(recon_samples)
    if actual.ndim == 1:
        actual = actual[:, None]
    if actual.ndim == 2:
        actual = actual[None, :, :]
    if recon_samples.ndim == 3:
        recon_samples = recon_samples[:, None, :, :]
    if likelihood is not None:
        likelihood = np.asarray(likelihood)
        if likelihood.ndim == 1:
            likelihood = likelihood[:, None]
        if likelihood.ndim == 2:
            likelihood = likelihood[None, :, :]

    input_dim = actual.shape[-1]
    n_mc = recon_samples.shape[0]
    # Same alpha scale as the notebook's spaghetti plot (500 samples @ alpha=0.01).
    recon_alpha = min(0.3, max(0.01, 5.0 / n_mc))

    nrows = max(1, int(np.ceil(input_dim)))
    fig, axs = plt.subplots(nrows=nrows, ncols=1, figsize=(12, 3.6 * nrows), sharex=True)
    axs = axs.flatten()
    # Fix one common y-range across all per-variate twin axes so the
    # likelihood traces stay directly comparable across variates.
    likelihood_range = (likelihood.min().item(), likelihood.max().item()) if likelihood is not None else None

    twin_handles, twin_labels = [], []
    for var_idx in range(input_dim):
        ax = axs[var_idx]

        if anomaly_mask is not None:
            _highlight_anomalous_regions(ax, anomaly_mask)
        for mc_idx in range(n_mc):
            ax.plot(recon_samples[mc_idx, :, :, var_idx].flatten(), color="tab:green",
                     alpha=recon_alpha, linewidth=1.2,
                     label="reconstructed" if var_idx == 0 and mc_idx == 0 else None)
        ax.plot(actual[:, :, var_idx].flatten(), color="tab:blue", linewidth=1.2, alpha=.7,
                 label="actual" if var_idx == 0 else None)
        #ax.set_ylabel(f"var {var_idx}", fontsize=8)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zorder(1)
        ax.patch.set_visible(False)  # keep anomaly shading visible through the twin axis

        if likelihood is not None:
            ax_twin = ax.twinx()
            # Distinct from the anomaly-highlight red so the two don't blend.
            line, = ax_twin.plot(
                likelihood[:, :, var_idx].flatten(), color="tab:orange", linewidth=0.8, alpha=.7,
                label="NLL" if var_idx == 0 else None,
            )
            if likelihood_range is not None:
                ax_twin.set_ylim(*likelihood_range)
            ax_twin.set_ylabel("NLL", fontsize=7, color="tab:orange")
            ax_twin.tick_params(axis="y", labelcolor="tab:orange", labelsize=6)
            if var_idx == 0:
                twin_handles, twin_labels = [line], ["NLL"]

    handles, labels = axs[0].get_legend_handles_labels()
    axs[0].legend(handles + twin_handles, labels + twin_labels, loc="upper right", fontsize=7)
    axs[-1].set_xlabel("timepoint (concatenated windows)")
    fig.suptitle(title)
    fig.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_reconstruction(args, provider, modules, desired_t, epoch, experiment_id):
    """Reconstruct the middle `args.reconstruct_n_windows` windows of the training
    trace, for all variates, and save the actual-vs-reconstructed plot to disk.
    """
    ds = provider._ds_trn
    indices = _select_middle_window_indices(ds, args.reconstruct_n_windows)

    parts = _gather_window_batch(ds, indices, args.device)
    actual, recon, _ = _decode_reconstruction(args, modules, desired_t, parts)

    out_path = os.path.join(args.reconstruct_dir, f"{experiment_id}_epoch{epoch:04d}.png")
    _plot_actual_vs_reconstructed(
        actual, recon, anomaly_mask=None,
        title=f"Reconstruction @ epoch {epoch} (train windows {indices[0]}-{indices[-1]})",
        out_path=out_path,
    )

    logging.info("Saved reconstruction plot to %s", out_path)
    return out_path


def plot_test_reconstruction(args, provider, modules, desired_t, epoch, experiment_id):
    """Reconstruct `args.reconstruct_n_windows` windows of the test trace, at
    the same decoding timepoints (`desired_t`) used for the training
    reconstruction, and save the actual-vs-reconstructed plot to disk.

    The plotted windows are chosen to contain the labeled anomalous segment
    (see `_select_anomalous_window_indices`), which is highlighted as a
    low-alpha red background band. Each variate additionally gets a twin
    y-axis showing the per-timepoint decoder negative log-likelihood (the
    same score used as `aux_log_prob` in anomaly_detection.py's evaluate()),
    with all twin axes sharing one common y-range for comparability.
    """
    ds = provider._ds_tst
    indices = _select_anomalous_window_indices(ds, args.reconstruct_n_windows)

    parts = _gather_window_batch(ds, indices, args.device)
    actual, recon, nll = _decode_reconstruction(args, modules, desired_t, parts)
    anomaly_mask = (parts["aux_tgt"].detach().cpu().flatten() > 0).numpy()

    out_path = os.path.join(args.reconstruct_dir, f"{experiment_id}_test_epoch{epoch:04d}.png")
    _plot_actual_vs_reconstructed(
        actual, recon, anomaly_mask=anomaly_mask,
        title=f"Test reconstruction @ epoch {epoch} (test windows {indices[0]}-{indices[-1]})",
        out_path=out_path,
        #likelihood=nll,
    )

    logging.info("Saved test reconstruction plot to %s", out_path)
    return out_path


def slerp_path(start_point, end_point, n_interp=5):
    """Interpolate the shortest great-circle arc between two points on the unit sphere."""
    p0 = np.asarray(start_point, dtype=float)
    p1 = np.asarray(end_point, dtype=float)

    p0 = p0 / np.linalg.norm(p0)
    p1 = p1 / np.linalg.norm(p1)

    dot = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    theta = float(np.arccos(dot))
    if theta < 1e-8:
        return np.repeat(p0[None, :], n_interp, axis=0)

    if np.isclose(theta, np.pi):
        # Deterministic fallback for antipodal points: rotate through an orthogonal direction.
        basis = np.eye(p0.shape[0])[np.argmin(np.abs(p0))]
        ortho = basis - np.dot(basis, p0) * p0
        ortho_norm = float(np.linalg.norm(ortho))
        if ortho_norm < 1e-8:
            basis = np.eye(p0.shape[0])[0]
            ortho = basis - np.dot(basis, p0) * p0
            ortho_norm = float(np.linalg.norm(ortho))
        ortho = ortho / ortho_norm
        ts = np.linspace(0.0, 1.0, n_interp)
        return np.array([
            np.cos(np.pi * t) * p0 + np.sin(np.pi * t) * ortho
            for t in ts
        ])

    sin_theta = np.sin(theta)
    ts = np.linspace(0.0, 1.0, n_interp)
    return np.array([
        (np.sin((1 - t) * theta) / sin_theta) * p0 + (np.sin(t * theta) / sin_theta) * p1
        for t in ts
    ])


def _prepare_window_scores(values, n_windows, *, default_value=0.0, clip_to_unit=False):
    if values is None:
        scores = np.full(n_windows, default_value, dtype=float)
    else:
        scores = np.asarray(values, dtype=float)
        if scores.ndim == 0:
            scores = np.full(n_windows, float(scores), dtype=float)
        elif scores.ndim == 1:
            if scores.size == 1:
                scores = np.full(n_windows, float(scores[0]), dtype=float)
            elif scores.size != n_windows:
                logging.warning(
                    "Expected %d window scores for latent-sphere coloring but got %d; using default %.3f.",
                    n_windows,
                    scores.size,
                    default_value,
                )
                scores = np.full(n_windows, default_value, dtype=float)
        else:
            if scores.shape[0] != n_windows:
                logging.warning(
                    "Expected first axis of score tensor to match n_windows=%d, got shape=%s; using default %.3f.",
                    n_windows,
                    scores.shape,
                    default_value,
                )
                scores = np.full(n_windows, default_value, dtype=float)
            else:
                scores = np.nanmean(scores, axis=tuple(range(1, scores.ndim)))

    scores = np.asarray(scores, dtype=float)
    invalid = ~np.isfinite(scores)
    if invalid.any():
        scores[invalid] = default_value
    if clip_to_unit:
        scores = np.clip(scores, 0.0, 1.0)
    return scores


def _timeline_color(progress):
    """Return a blue-to-red RGB color for a normalized timeline progress value."""
    t = float(np.clip(progress, 0.0, 1.0))
    start = np.array([66, 133, 244], dtype=float)
    end = np.array([230, 67, 53], dtype=float)
    rgb = np.round(start + (end - start) * t).astype(int)
    return f"rgb({rgb[0]}, {rgb[1]}, {rgb[2]})"


def _score_to_color(score, *, vmin, vmax):
    if not np.isfinite(score):
        score = vmin
    if np.isclose(vmin, vmax):
        ratio = 0.5
    else:
        ratio = (score - vmin) / (vmax - vmin)
    ratio = float(np.clip(ratio, 0.0, 1.0))
    red = int(255 * ratio)
    blue = int(255 * (1.0 - ratio))
    return f"rgb({red}, 0, {blue})"


def _get_plotly_modules():
    go = importlib.import_module("plotly.graph_objects")
    make_subplots = importlib.import_module("plotly.subplots").make_subplots
    return go, make_subplots


def _add_plotly_sphere_surface(fig, row, col):
    go, _ = _get_plotly_modules()
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    fig.add_trace(
        go.Surface(
            x=x,
            y=y,
            z=z,
            opacity=0.15,
            showscale=False,
            hoverinfo="skip",
            colorscale=[[0.0, "lightgray"], [1.0, "lightgray"]],
            name="sphere",
        ),
        row=row,
        col=col,
    )


def _write_plotly_dashboard_with_window_sliders(
        fig,
        out_path,
        window_indices,
        window_trace_indices,
        selection_band_trace_indices,
        start_marker_trace_indices,
        end_marker_trace_indices,
        n_time,
):
    plotly_io = importlib.import_module("plotly.io")

    n_windows = len(window_indices)
    window_labels = [str(idx) for idx in window_indices]
    trace_map_json = json.dumps(window_trace_indices)
    label_json = json.dumps(window_labels)
    selection_band_json = json.dumps(selection_band_trace_indices)
    start_marker_json = json.dumps(start_marker_trace_indices)
    end_marker_json = json.dumps(end_marker_trace_indices)
    div_id = "latent_dashboard_plot"

    plot_html = plotly_io.to_html(
        fig,
        include_plotlyjs="cdn",
        full_html=False,
        div_id=div_id,
        config={"responsive": True},
    )

    controls_html = ""
    if n_windows > 0:
        controls_html = f"""
<div style=\"margin: 10px 0 14px 0; padding: 10px 12px; border: 1px solid #ddd; border-radius: 6px;\">
  <div style=\"font-weight: 600; margin-bottom: 8px;\">Sphere window selection (discrete, non-overlapping)</div>
  <div style=\"display: grid; grid-template-columns: 120px 1fr 120px; gap: 10px; align-items: center; margin-bottom: 8px;\">
    <label for=\"window_start_slider\">Start window</label>
    <input id=\"window_start_slider\" type=\"range\" min=\"0\" max=\"{n_windows - 1}\" step=\"1\" value=\"0\" />
    <span id=\"window_start_label\" style=\"font-family: monospace;\"></span>
  </div>
  <div style=\"display: grid; grid-template-columns: 120px 1fr 120px; gap: 10px; align-items: center;\">
    <label for=\"window_end_slider\">End window</label>
    <input id=\"window_end_slider\" type=\"range\" min=\"0\" max=\"{n_windows - 1}\" step=\"1\" value=\"{n_windows - 1}\" />
    <span id=\"window_end_label\" style=\"font-family: monospace;\"></span>
  </div>
  <div style="margin-top: 10px;">
    <button id="window_apply_button" type="button" style="padding: 6px 12px; cursor: pointer;">Apply selected window range</button>
  </div>
</div>
"""

    script_html = f"""
<script>
(function() {{
  const gd = document.getElementById('{div_id}');
  if (!gd) return;

  const windowTraceMap = {trace_map_json};
  const windowLabels = {label_json};
  const selectionBandTraceIdxs = {selection_band_json};
  const startMarkerTraceIdxs = {start_marker_json};
  const endMarkerTraceIdxs = {end_marker_json};
  const nTime = {int(n_time)};
  const totalTraces = gd.data.length;

  const startSlider = document.getElementById('window_start_slider');
  const endSlider = document.getElementById('window_end_slider');
  const startLabel = document.getElementById('window_start_label');
  const endLabel = document.getElementById('window_end_label');
  const applyButton = document.getElementById('window_apply_button');

  function timelineColor(progress) {{
    const t = Math.max(0.0, Math.min(1.0, progress));
    const start = [66, 133, 244];
    const end = [230, 67, 53];
    const rgb = start.map(function(v, i) {{
      return Math.round(v + (end[i] - v) * t);
    }});
    return 'rgb(' + rgb[0] + ', ' + rgb[1] + ', ' + rgb[2] + ')';
  }}

  const windowLineTraceMap = windowTraceMap.map(function(traceIdxs) {{
    return traceIdxs.filter(function(traceIdx) {{
      const trace = gd.data[traceIdx] || {{}};
      return trace.mode === 'lines';
    }});
  }});
  const windowMarkerTraceMap = windowTraceMap.map(function(traceIdxs) {{
    return traceIdxs.filter(function(traceIdx) {{
      const trace = gd.data[traceIdx] || {{}};
      return trace.mode === 'markers';
    }});
  }});

  function fmt(i) {{
    return '#' + i + ' (id=' + windowLabels[i] + ')';
  }}

  function updateLabels() {{
    if (!startSlider || !endSlider) return;
    if (startLabel) startLabel.textContent = fmt(parseInt(startSlider.value, 10));
    if (endLabel) endLabel.textContent = fmt(parseInt(endSlider.value, 10));
  }}

  function normalizeSliderRange(changed) {{
    if (!startSlider || !endSlider) return;
    let start = parseInt(startSlider.value, 10);
    let end = parseInt(endSlider.value, 10);

    if (changed === 'start' && start > end) {{
        end = start;
        endSlider.value = String(end);
    }}
    if (changed === 'end' && end < start) {{
        start = end;
        startSlider.value = String(start);
    }}

    updateLabels();
  }}

  function applyRange(changed) {{
    if (!startSlider || !endSlider) return;
    let start = parseInt(startSlider.value, 10);
    let end = parseInt(endSlider.value, 10);

    if (changed === 'start' && start > end) {{
        end = start;
        endSlider.value = String(end);
    }}
    if (changed === 'end' && end < start) {{
        start = end;
        startSlider.value = String(start);
    }}

    start = parseInt(startSlider.value, 10);
    end = parseInt(endSlider.value, 10);
    updateLabels();

    const visibleWindowCount = Math.max(1, end - start + 1);

    const vis = Array(totalTraces).fill(true);
    for (let w = 0; w < windowTraceMap.length; w++) {{
        const inRange = (w >= start && w <= end);
        const traceIdxs = windowTraceMap[w] || [];
        for (const traceIdx of traceIdxs) {{
            vis[traceIdx] = inRange;
        }}
    }}

    const traceIndices = Array.from({{length: totalTraces}}, (_, i) => i);
    const visibleValues = vis.map(v => v ? true : false);
    Plotly.restyle(gd, {{visible: visibleValues}}, traceIndices);

    for (let w = start; w <= end; w++) {{
      const localProgress = visibleWindowCount <= 1 ? 0.0 : (w - start) / (visibleWindowCount - 1);
      const trajectoryColor = timelineColor(localProgress);
      if (windowLineTraceMap[w] && windowLineTraceMap[w].length) {{
        Plotly.restyle(gd, {{'line.color': trajectoryColor}}, windowLineTraceMap[w]);
      }}
      if (windowMarkerTraceMap[w] && windowMarkerTraceMap[w].length) {{
        Plotly.restyle(gd, {{'marker.color': trajectoryColor}}, windowMarkerTraceMap[w]);
      }}
    }}

    const xStart = (start * nTime) - 0.5;
    const xEnd = ((end + 1) * nTime) - 0.5;
    Plotly.restyle(gd, {{x: [[xStart, xEnd, xEnd, xStart, xStart]]}}, selectionBandTraceIdxs);
    Plotly.restyle(gd, {{x: [[xStart, xStart]]}}, startMarkerTraceIdxs);
    Plotly.restyle(gd, {{x: [[xEnd, xEnd]]}}, endMarkerTraceIdxs);
  }}

  if (startSlider && endSlider) {{
    startSlider.addEventListener('input', function() {{ normalizeSliderRange('start'); }});
    endSlider.addEventListener('input', function() {{ normalizeSliderRange('end'); }});
    if (applyButton) {{
      applyButton.addEventListener('click', function() {{ applyRange('button'); }});
    }}
    updateLabels();
    applyRange('init');
  }}

}})();
</script>
"""

    full_html = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Latent Dashboard</title></head>"
        "<body style='font-family: Arial, sans-serif; margin: 12px;'>"
        + controls_html
        + plot_html
        + script_html
        + "</body></html>"
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(full_html)


def _plot_latent_path_on_sphere(latents, window_indices, title, out_path, actual, recon_samples,
                                anomaly_mask=None, elev=20, azim=45, anomaly_ratios=None, log_prob=None):
    """Create an interactive Plotly dashboard with a rotatable latent sphere
    and, alongside it, the reconstruction-vs-ground-truth time series for the
    same selected windows.

    Args:
        latents: array of shape (n_windows, n_samples, n_time, z_dim>=3).
        window_indices: dataset indices of the plotted windows.
        actual: array of shape (n_windows, n_time, dim).
        recon_samples: array of shape (n_samples, n_windows, n_time, dim).
    """
    latents = np.asarray(latents)
    actual = np.asarray(actual)
    recon_samples = np.asarray(recon_samples)
    if latents.ndim == 3:
        latents = latents[None, ...]
    if actual.ndim == 1:
        actual = actual[:, None]
    if actual.ndim == 2:
        actual = actual[None, :, :]
    if recon_samples.ndim == 3:
        recon_samples = recon_samples[:, None, :, :]

    n_windows, n_samples = latents.shape[0], latents.shape[1]
    input_dim = actual.shape[-1]
    ts_rows = int(np.ceil(input_dim / 2.0))
    path_alpha = min(0.35, max(0.05, 6.0 / max(1, n_samples)))
    anomaly_scores = _prepare_window_scores(anomaly_ratios, n_windows, clip_to_unit=True)
    go, make_subplots = _get_plotly_modules()

    specs: list[list[object]] = [[{"type": "scene", "rowspan": max(1, ts_rows)}, {"type": "xy"}, {"type": "xy"}]]
    specs.extend([[None, {"type": "xy"}, {"type": "xy"}] for _ in range(max(0, ts_rows - 1))])
    fig = make_subplots(
        rows=max(1, ts_rows),
        cols=3,
        specs=specs,
        horizontal_spacing=0.05,
        vertical_spacing=0.06,
    )

    _add_plotly_sphere_surface(fig, row=1, col=1)
    n_visible_windows = len(window_indices)
    window_trace_indices = [[] for _ in range(n_windows)]
    for w in range(n_windows):
        legend_name = f"window {window_indices[w]}"
        window_progress = 0.0 if n_visible_windows <= 1 else w / (n_visible_windows - 1)
        trajectory_color = _timeline_color(window_progress)
        for s in range(n_samples):
            xs, ys, zs = latents[w, s, :, 0], latents[w, s, :, 1], latents[w, s, :, 2]
            geo_path = slerp_path(np.array([xs[0], ys[0], zs[0]]), np.array([xs[-1], ys[-1], zs[-1]]), n_interp=20)
            for seg_idx in range(geo_path.shape[0] - 1):
                seg_x = [geo_path[seg_idx, 0], geo_path[seg_idx + 1, 0]]
                seg_y = [geo_path[seg_idx, 1], geo_path[seg_idx + 1, 1]]
                seg_z = [geo_path[seg_idx, 2], geo_path[seg_idx + 1, 2]]
                seg_hover = f"{legend_name} | time step {seg_idx + 1}/{max(1, geo_path.shape[0] - 1)}"
                fig.add_trace(
                    go.Scatter3d(
                        x=seg_x,
                        y=seg_y,
                        z=seg_z,
                        mode="lines",
                        line=dict(color=trajectory_color, width=3),
                        opacity=path_alpha,
                        name=legend_name,
                        legendgroup=f"window-{w}",
                        showlegend=(s == 0 and seg_idx == 0),
                        hovertext=[seg_hover, seg_hover],
                        hoverinfo="text",
                    ),
                    row=1,
                    col=1,
                )
                window_trace_indices[w].append(len(fig.data) - 1)
            fig.add_trace(
                go.Scatter3d(
                    x=[xs[0], xs[-1]],
                    y=[ys[0], ys[-1]],
                    z=[zs[0], zs[-1]],
                    mode="markers",
                    marker=dict(color=trajectory_color, size=[3, 4], symbol=["circle", "square"]),
                    legendgroup=f"window-{w}",
                    showlegend=False,
                    hovertext=[f"{legend_name} | start", f"{legend_name} | end"],
                    hoverinfo="text",
                ),
                row=1,
                col=1,
            )
            window_trace_indices[w].append(len(fig.data) - 1)

    sphere_camera = dict(
        eye=dict(
            x=1.8 * np.cos(np.deg2rad(azim)),
            y=1.8 * np.sin(np.deg2rad(azim)),
            z=1.2 * np.sin(np.deg2rad(elev)),
        )
    )
    fig.update_scenes(
        xaxis_title="z0",
        yaxis_title="z1",
        zaxis_title="z2",
        xaxis=dict(range=[-1, 1]),
        yaxis=dict(range=[-1, 1]),
        zaxis=dict(range=[-1, 1]),
        aspectmode="cube",
        camera=sphere_camera,
        row=1,
        col=1,
    )

    recon_mean = recon_samples.mean(axis=0)
    recon_std = recon_samples.std(axis=0)
    n_time = actual.shape[1]
    x_values = np.arange(actual.shape[0] * actual.shape[1])
    start_marker_x = -0.5
    end_marker_x = (n_windows * n_time) - 0.5
    selection_band_trace_indices = []
    start_marker_trace_indices = []
    end_marker_trace_indices = []
    flat_anomaly_mask = None if anomaly_mask is None else np.asarray(anomaly_mask).astype(bool).reshape(-1)
    if flat_anomaly_mask is not None and flat_anomaly_mask.size != x_values.size:
        logging.warning(
            "Anomaly mask length %d does not match plotted time axis %d; disabling anomaly shading.",
            flat_anomaly_mask.size,
            x_values.size,
        )
        flat_anomaly_mask = None

    for var_idx in range(input_dim):
        row = (var_idx // 2) + 1
        col = 2 + (var_idx % 2)
        actual_flat = actual[:, :, var_idx].reshape(-1)
        recon_flat = recon_mean[:, :, var_idx].reshape(-1)
        recon_std_flat = recon_std[:, :, var_idx].reshape(-1)
        lower = recon_flat - recon_std_flat
        upper = recon_flat + recon_std_flat
        y_lower = np.asarray(np.concatenate([actual_flat, lower]), dtype=float)
        y_upper = np.asarray(np.concatenate([actual_flat, upper]), dtype=float)
        y_min = float(np.nanmin(y_lower).item())
        y_max = float(np.nanmax(y_upper).item())
        if np.isclose(y_min, y_max):
            y_max = y_min + 1e-6

        fig.add_trace(
            go.Scatter(
                x=[start_marker_x, end_marker_x, end_marker_x, start_marker_x, start_marker_x],
                y=[y_min, y_min, y_max, y_max, y_min],
                fill="toself",
                fillcolor="rgba(66, 133, 244, 0.10)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        selection_band_trace_indices.append(len(fig.data) - 1)

        # Subtle vertical markers to indicate non-overlapping window boundaries.
        for b in range(1, n_windows):
            x_boundary = (b * n_time) - 0.5
            fig.add_trace(
                go.Scatter(
                    x=[x_boundary, x_boundary],
                    y=[y_min, y_max],
                    mode="lines",
                    line=dict(color="rgba(80, 80, 80, 0.20)", width=1, dash="dot"),
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )

        if flat_anomaly_mask is not None:
            in_segment = False
            seg_start = 0
            for idx, flagged in enumerate(flat_anomaly_mask.tolist()):
                if flagged and not in_segment:
                    in_segment = True
                    seg_start = idx
                elif not flagged and in_segment:
                    in_segment = False
                    fig.add_trace(
                        go.Scatter(
                            x=[seg_start - 0.5, idx - 0.5, idx - 0.5, seg_start - 0.5, seg_start - 0.5],
                            y=[y_min, y_min, y_max, y_max, y_min],
                            fill="toself",
                            fillcolor="rgba(255, 0, 0, 0.12)",
                            line=dict(color="rgba(0,0,0,0)"),
                            hoverinfo="skip",
                            showlegend=False,
                        ),
                        row=row,
                        col=col,
                    )
            if in_segment:
                fig.add_trace(
                    go.Scatter(
                        x=[seg_start - 0.5, x_values[-1] + 0.5, x_values[-1] + 0.5, seg_start - 0.5, seg_start - 0.5],
                        y=[y_min, y_min, y_max, y_max, y_min],
                        fill="toself",
                        fillcolor="rgba(255, 0, 0, 0.12)",
                        line=dict(color="rgba(0,0,0,0)"),
                        hoverinfo="skip",
                        showlegend=False,
                    ),
                    row=row,
                    col=col,
                )

        fig.add_trace(
            go.Scatter(
                x=[start_marker_x, start_marker_x],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="rgba(40, 140, 255, 0.95)", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        start_marker_trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=[end_marker_x, end_marker_x],
                y=[y_min, y_max],
                mode="lines",
                line=dict(color="rgba(230, 80, 20, 0.95)", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            row=row,
            col=col,
        )
        end_marker_trace_indices.append(len(fig.data) - 1)

        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=actual_flat,
                mode="lines",
                line=dict(color="rgb(31, 119, 180)", width=1.5),
                name="ground truth" if var_idx == 0 else None,
                legendgroup="ground-truth",
                showlegend=(var_idx == 0),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=recon_flat,
                mode="lines",
                line=dict(color="rgb(44, 160, 44)", width=1.5),
                name="reconstruction mean" if var_idx == 0 else None,
                legendgroup="reconstruction",
                showlegend=(var_idx == 0),
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=np.concatenate([x_values, x_values[::-1]]),
                y=np.concatenate([upper, lower[::-1]]),
                fill="toself",
                fillcolor="rgba(44, 160, 44, 0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                hoverinfo="skip",
                name="reconstruction ±1 std" if var_idx == 0 else None,
                legendgroup="reconstruction-band",
                showlegend=(var_idx == 0),
            ),
            row=row,
            col=col,
        )

        fig.update_yaxes(title_text=f"var {var_idx}", row=row, col=col)
        if row < ts_rows:
            fig.update_xaxes(showticklabels=False, row=row, col=col)


    fig.update_xaxes(title_text="timepoint (concatenated windows)", row=ts_rows, col=2)
    fig.update_xaxes(title_text="timepoint (concatenated windows)", row=ts_rows, col=3)
    fig.update_layout(
        title=title,
        template="plotly_white",
        width=1900,
        height=max(680, 260 * ts_rows),
        showlegend=False,
        margin=dict(l=40, r=40, t=80, b=40),
    )

    _write_plotly_dashboard_with_window_sliders(
        fig=fig,
        out_path=out_path,
        window_indices=window_indices,
        window_trace_indices=window_trace_indices,
        selection_band_trace_indices=selection_band_trace_indices,
        start_marker_trace_indices=start_marker_trace_indices,
        end_marker_trace_indices=end_marker_trace_indices,
        n_time=n_time,
    )


def plot_latent_sphere(args, provider, modules, desired_t, epoch, experiment_id):
    """Create an interactive Plotly dashboard for selected test windows with
    a rotatable latent sphere on the left and reconstruction vs. ground truth
    time-series panels on the right.
    """
    ds = provider._ds_tst
    n_windows = max(1, args.latent_sphere_n_windows)
    n_windows = min(n_windows, max(1, provider._ds_tst.evd_msk.shape[0] // 4))
    indices = _select_anomalous_window_indices(ds, n_windows)

    parts = _gather_window_batch(ds, indices, args.device)
    inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

    # One anomaly ratio per window: fraction of anomalous timepoints in [0, 1].
    anomaly_ratio = parts["aux_tgt"].float().mean(dim=1).detach().cpu().numpy()

    was_training = modules.training
    modules.eval()
    with torch.no_grad():
        h = modules["recog_net"](inp)
        qzx, _ = modules["qzx_net"](h, desired_t)
        # (mc_samples, n_windows, n_time, z_dim)
        latents = qzx.rsample((args.latent_sphere_mc_samples,))
        pxz = modules["pxz_net"](latents)
        recon_samples = pxz.mean.detach().cpu()
        log_prob = pxz.log_prob(parts["evd_obs"]).detach().cpu()
        latents = latents.detach().cpu()

    if was_training:
        modules.train()

    if latents.shape[-1] < 3:
        logging.warning("z_dim=%d < 3; skipping latent-sphere plot.", latents.shape[-1])
        return None

    # -> (n_windows, mc_samples, n_time, z_dim)
    latents = latents.permute(1, 0, 2, 3).numpy()
    log_prob = log_prob.permute(1, 0, 2, 3).numpy()
    actual = parts["evd_obs"].detach().cpu().numpy()
    recon_samples = recon_samples.numpy()
    anomaly_mask = (parts["aux_tgt"].detach().cpu().numpy() > 0).reshape(-1)

    out_path = os.path.join(args.reconstruct_dir, f"{experiment_id}_sphere_epoch{epoch:04d}.html")
    _plot_latent_path_on_sphere(
        latents,
        window_indices=indices,
        title=f"Latent paths on the sphere @ epoch {epoch} "
              f"(test windows {indices[0]}-{indices[-1]})",
        out_path=out_path,
        actual=actual,
        recon_samples=recon_samples,
        anomaly_mask=anomaly_mask,
        anomaly_ratios=anomaly_ratio,
        log_prob=log_prob,
    )

    logging.info("Saved interactive latent dashboard to %s", out_path)
    return out_path


def make_reconstruction_gif(image_paths, out_path, duration_ms):
    """Combine PNGs (already in chronological order) into a single looping GIF."""
    frames = [Image.open(p).convert("RGB") for p in image_paths]
    frames[0].save(
        out_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def main():
    parser = build_parser()
    bootstrap_args, _ = parser.parse_known_args()

    set_up_logging(
        console_log_level=bootstrap_args.loglevel,
        console_log_color=True,
        logfile_file=None,
        logfile_log_level=bootstrap_args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(created)d] [%(levelname)-8s] %(message)s%(color_off)s",
    )

    cfg = load_dataset_config(bootstrap_args.config_file)
    parser.set_defaults(**cfg)

    args = parser.parse_args()
    args.device = resolve_device(args.device)
    args.eval_every_n_epochs = max(1, int(args.eval_every_n_epochs))

    args.z_dim = 3 # !! setting fix

    if args.seed > 0:
        set_seed(args.seed)

    experiment_id = f"qad_fit_trace{args.trace_id}_{datetime.datetime.now().strftime('%y%m%d-%H%M%S')}"
    logging.info("Fitting QAD trace %d (experiment_id=%s)", args.trace_id, experiment_id)

    # Keep each run's reconstruction images/GIFs in their own subfolder.
    args.reconstruct_dir = os.path.join(args.reconstruct_dir, experiment_id)

    logging.info("Parameters: %s", vars(args))

    provider = QADProvider(
        data_dir=args.data_dir,
        dataset_number=args.trace_id,
        window_length=args.data_window_length,
        window_overlap=args.data_window_overlap,
        data_normalization_strategy=args.data_normalization_strategy,
        subsample=args.subsample,
        seed=args.seed,
        fixed_subsample_mask=args.fixed_subsample_mask,
        train_shuffle = True
    )

    dl_trn = provider.get_train_loader(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    dl_val = provider.get_val_loader(
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    provider_viz = QADProvider(
        data_dir=args.data_dir,
        dataset_number=args.trace_id,
        window_length=args.data_window_length,
        window_overlap=0.0,
        data_normalization_strategy=args.data_normalization_strategy,
        subsample=args.subsample,
        seed=args.seed,
        fixed_subsample_mask=args.fixed_subsample_mask,
        train_shuffle = False
    )

    desired_t = torch.linspace(0, 1.00, provider.num_timepoints, device=args.device).float()
    modules, optimizer, scheduler, elbo_loss = build_modules_and_optim(args, provider.input_dim, desired_t)
    logging.info("Number of model parameters=%d", count_parameters(modules))

    if args.enable_checkpointing:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    reconstruction_paths = []
    test_reconstruction_paths = []
    latent_sphere_paths = []
    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    last_epoch_completed = 0
    stopped_early = False
    try:
        for epoch in range(1, args.n_epochs + 1):
            trn_stats = generic_train(args, dl_trn, modules, elbo_loss, None, optimizer, desired_t, args.device)
            scheduler.step()
            last_epoch_completed = epoch

            run_validation = (epoch % args.eval_every_n_epochs == 0) or (epoch == args.n_epochs)
            val_stats = None
            if run_validation:
                val_stats = evaluate_validation(args, dl_val, modules, elbo_loss, desired_t, args.device)

                improved = val_stats["loss"] < (best_val_loss - args.early_stopping_min_delta)
                if improved:
                    best_val_loss = val_stats["loss"]
                    best_epoch = epoch
                    patience_counter = 0
                    save_checkpoint(args, "best", experiment_id, modules, desired_t)
                else:
                    patience_counter += 1

                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    stopped_early = True
                    logging.info(
                        "Early stopping triggered at epoch=%04d (best_val_loss=%.6f @ epoch=%04d, patience=%d, min_delta=%.6f)",
                        epoch,
                        best_val_loss,
                        best_epoch,
                        args.early_stopping_patience,
                        args.early_stopping_min_delta,
                    )

            if epoch % args.log_every_n_epochs == 0 or epoch == args.n_epochs:
                if val_stats is None:
                    logging.info(
                        "epoch=%04d | trn_loss=%.6f | trn_elbo=%.6f | trn_kl0=%.6f | trn_klp=%.6f | trn_log_pxz=%.6f | lr=%.6g",
                        epoch,
                        trn_stats["loss"],
                        trn_stats["elbo"],
                        trn_stats["kl0"],
                        trn_stats["klp"],
                        trn_stats["log_pxz"],
                        scheduler.get_last_lr()[-1],
                    )
                else:
                    logging.info(
                        "epoch=%04d | trn_loss=%.6f | val_loss=%.6f | trn_elbo=%.6f | val_elbo=%.6f | lr=%.6g | es_wait=%d/%s",
                        epoch,
                        trn_stats["loss"],
                        val_stats["loss"],
                        trn_stats["elbo"],
                        val_stats["elbo"],
                        scheduler.get_last_lr()[-1],
                        patience_counter,
                        "off" if args.early_stopping_patience <= 0 else str(args.early_stopping_patience),
                    )

            if args.checkpoint_every_n_epochs and epoch % args.checkpoint_every_n_epochs == 0:
                save_checkpoint(args, epoch, experiment_id, modules, desired_t)

            if args.reconstruct_at_k and epoch % args.reconstruct_at_k == 0:
                reconstruction_paths.append(
                    plot_reconstruction(args, provider_viz, modules, desired_t, epoch, experiment_id)
                )
                test_reconstruction_paths.append(
                    plot_test_reconstruction(args, provider_viz, modules, desired_t, epoch, experiment_id)
                )
                if args.plot_latent_sphere:
                    sphere_path = plot_latent_sphere(args, provider_viz, modules, desired_t, epoch, experiment_id)
                    if sphere_path is not None:
                        latent_sphere_paths.append(sphere_path)

            if stopped_early:
                break

        if best_epoch > 0:
            logging.info("Best validation checkpoint: epoch=%04d, val_loss=%.6f", best_epoch, best_val_loss)
        save_checkpoint(args, last_epoch_completed, experiment_id, modules, desired_t)
    finally:
        if hasattr(provider, "cleanup"):
            provider.cleanup()

        if args.reconstruct_gif and reconstruction_paths:
            gif_path = os.path.join(args.reconstruct_dir, f"{experiment_id}_reconstruction.gif")
            make_reconstruction_gif(reconstruction_paths, gif_path, args.reconstruct_gif_duration_ms)
            logging.info("Saved reconstruction GIF (%d frames) to %s", len(reconstruction_paths), gif_path)

        if args.reconstruct_gif and test_reconstruction_paths:
            test_gif_path = os.path.join(args.reconstruct_dir, f"{experiment_id}_test_reconstruction.gif")
            make_reconstruction_gif(test_reconstruction_paths, test_gif_path, args.reconstruct_gif_duration_ms)
            logging.info("Saved test reconstruction GIF (%d frames) to %s", len(test_reconstruction_paths), test_gif_path)

        if args.reconstruct_gif and latent_sphere_paths:
            png_latent_paths = [path for path in latent_sphere_paths if str(path).lower().endswith((".png", ".jpg", ".jpeg"))]
            if png_latent_paths:
                sphere_gif_path = os.path.join(args.reconstruct_dir, f"{experiment_id}_latent_sphere.gif")
                make_reconstruction_gif(png_latent_paths, sphere_gif_path, args.reconstruct_gif_duration_ms)
                logging.info("Saved latent-sphere GIF (%d frames) to %s", len(png_latent_paths), sphere_gif_path)
            else:
                logging.info("Skipping latent-sphere GIF because Plotly latent visualizations are saved as HTML.")

    logging.info("Done fitting QAD trace %d.", args.trace_id)


if __name__ == "__main__":
    main()

