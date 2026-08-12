"""MVP fitting script: train the Latent SDE model on a single QAD trace.

This is deliberately scoped to *fitting only* (no evaluation, no anomaly
scoring, no wandb/tensorboard). The idea is to nail down a good fit on one
trace first; anomaly detection is layered on top incrementally afterwards.
"""

import argparse
import datetime
import json
import logging
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    data.add_argument("--raw-subdir", type=str, default="qad_clean_txt_100Hz")
    data.add_argument("--trace-id", type=int, default=1, help="QAD trace/dataset number to fit (train_<id>.txt).")
    data.add_argument("--data-window-length", type=int, default=500)
    data.add_argument("--data-window-overlap", type=float, default=0.0)
    data.add_argument("--data-normalization-strategy", choices=["none", "std", "min-max"], default="min-max")
    data.add_argument("--subsample", type=float, default=0.5, help="Fraction of input observations kept visible to the encoder.")
    data.add_argument("--fixed-subsample-mask", action=argparse.BooleanOptionalAction, default=True)
    data.add_argument("--num-workers", type=int, default=8)

    model = parser.add_argument_group("Model arguments")
    model.add_argument("--z-dim", type=int, default=4)
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
    train.add_argument("--batch-size", type=int, default=1024)
    train.add_argument("--lr", type=float, default=5e-2)
    train.add_argument("--n-epochs", type=int, default=180)
    train.add_argument("--restart", type=int, default=30, help="Cosine annealing restart period (epochs).")
    train.add_argument("--kl0-weight", type=float, default=1e-4)
    train.add_argument("--klp-weight", type=float, default=1e-2)
    train.add_argument("--pxz-weight", type=float, default=100.0)
    train.add_argument("--mc-train-samples", type=int, default=1)
    train.add_argument("--seed", type=int, default=-1)
    train.add_argument("--device", type=str, default="cuda")

    ckpt = parser.add_argument_group("Checkpointing/logging arguments")
    ckpt.add_argument("--enable-checkpointing", action=argparse.BooleanOptionalAction, default=True)
    ckpt.add_argument("--checkpoint-dir", type=str, default="checkpoints/qad_fit")
    ckpt.add_argument("--checkpoint-every-n-epochs", type=int, default=50, help="0 disables periodic checkpoints; the final epoch is always saved.")
    ckpt.add_argument("--log-every-n-epochs", type=int, default=10)
    ckpt.add_argument("--loglevel", choices=["debug", "info", "warning", "error", "critical"], default="info")

    recon = parser.add_argument_group("Reconstruction plotting arguments")
    recon.add_argument(
        "--reconstruct-at-k", type=int, default=5,
        help="If >0, plot a data reconstruction every k-th epoch. 0 disables reconstruction plotting.",
    )
    recon.add_argument("--reconstruct-dir", type=str, default="out/reconstructions/qad_fit")
    recon.add_argument(
        "--reconstruct-n-windows", type=int, default=10,
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

    anomalous = [i for i in range(n_windows) if bool((ds[i]["aux_tgt"] > 0).any())]
    if not anomalous:
        return _select_middle_window_indices(ds, n_select)

    first, last = anomalous[0], anomalous[-1]
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
    input_dim = actual.shape[-1]
    n_mc = recon_samples.shape[0]
    # Same alpha scale as the notebook's spaghetti plot (500 samples @ alpha=0.01).
    recon_alpha = min(0.3, max(0.01, 5.0 / n_mc))

    fig, axs = plt.subplots(nrows=input_dim//2, ncols=2, figsize=(10, 1.6 * input_dim//2), sharex=True)
    if input_dim == 1:
        axs = [axs]

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
        ax.set_ylabel(f"var {var_idx}", fontsize=8)
        ax.set_zorder(1)
        ax.patch.set_visible(False)  # keep anomaly shading visible through the twin axis

        if likelihood is not None:
            ax_twin = ax.twinx()
            # Distinct from the anomaly-highlight red so the two don't blend.
            line, = ax_twin.plot(
                likelihood[:, :, var_idx].flatten(), color="tab:orange", linewidth=0.8, alpha=.7,
                label="NLL" if var_idx == 0 else None,
            )
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
        likelihood=nll,
    )

    logging.info("Saved test reconstruction plot to %s", out_path)
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
        raw_subdir=args.raw_subdir,
        fixed_subsample_mask=args.fixed_subsample_mask,
    )

    dl_trn = provider.get_train_loader(
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    desired_t = torch.linspace(0, 1.00, provider.num_timepoints, device=args.device).float()
    modules, optimizer, scheduler, elbo_loss = build_modules_and_optim(args, provider.input_dim, desired_t)
    logging.info("Number of model parameters=%d", count_parameters(modules))

    if args.enable_checkpointing:
        os.makedirs(args.checkpoint_dir, exist_ok=True)

    reconstruction_paths = []
    test_reconstruction_paths = []
    try:
        for epoch in range(1, args.n_epochs + 1):
            trn_stats = generic_train(args, dl_trn, modules, elbo_loss, None, optimizer, desired_t, args.device)
            scheduler.step()

            if epoch % args.log_every_n_epochs == 0 or epoch == args.n_epochs:
                logging.info(
                    "epoch=%04d | loss=%.6f | elbo=%.6f | kl0=%.6f | klp=%.6f | log_pxz=%.6f | lr=%.6g",
                    epoch, trn_stats["loss"], trn_stats["elbo"], trn_stats["kl0"],
                    trn_stats["klp"], trn_stats["log_pxz"], scheduler.get_last_lr()[-1],
                )

            if args.checkpoint_every_n_epochs and epoch % args.checkpoint_every_n_epochs == 0:
                save_checkpoint(args, epoch, experiment_id, modules, desired_t)

            if args.reconstruct_at_k and epoch % args.reconstruct_at_k == 0:
                reconstruction_paths.append(
                    plot_reconstruction(args, provider, modules, desired_t, epoch, experiment_id)
                )
                test_reconstruction_paths.append(
                    plot_test_reconstruction(args, provider, modules, desired_t, epoch, experiment_id)
                )

        save_checkpoint(args, args.n_epochs, experiment_id, modules, desired_t)
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

    logging.info("Done fitting QAD trace %d.", args.trace_id)


if __name__ == "__main__":
    main()