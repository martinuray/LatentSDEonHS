#!/usr/bin/env python3
"""Visualize a trained QAD checkpoint with matplotlib only.

This script loads a checkpoint produced by `notebooks/fit_qad.py` and creates
static matplotlib figures for:
- train reconstruction,
- test reconstruction (with anomaly shading + NLL),
- latent trajectories on a unit sphere plus reconstruction panels.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from anomaly_detection import calculate_feature_reconstruction_weights

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from data.qad_provider import QADProvider

try:
    # Works when running from `notebooks/`.
    from fit_qad import (  # type: ignore
        _decode_reconstruction,
        _gather_window_batch,
        _plot_actual_vs_reconstructed,
        _select_anomalous_window_indices,
        _select_middle_window_indices,
        build_modules_and_optim,
        resolve_device,
    )
except ImportError:
    # Works when running from repo root.
    from notebooks.fit_qad import (  # type: ignore
        _decode_reconstruction,
        _gather_window_batch,
        _plot_actual_vs_reconstructed,
        _select_anomalous_window_indices,
        _select_middle_window_indices,
        build_modules_and_optim,
        resolve_device,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Load a stored QAD checkpoint and visualize it with matplotlib.")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Direct path to checkpoint_*.h5")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory used when --checkpoint-path is omitted")
    parser.add_argument("--experiment-id", type=str, default=None, help="Optional experiment id used in checkpoint filename")
    parser.add_argument("--epoch", type=str, default=None, help="Optional epoch used in checkpoint filename, e.g. 990 or best")

    parser.add_argument("--data-dir", type=str, default=None, help="Override data dir from checkpoint args")
    parser.add_argument("--trace-id", type=int, default=None, help="Override trace id from checkpoint args")
    parser.add_argument("--device", type=str, default=None, help="Override runtime device (default: checkpoint args device)")
    parser.add_argument("--num-workers", type=int, default=8)

    parser.add_argument("--reconstruct-n-windows", type=int, default=60)
    parser.add_argument("--reconstruct-mc-samples", type=int, default=100)
    parser.add_argument("--latent-sphere-n-windows", type=int, default=100)
    parser.add_argument("--latent-sphere-mc-samples", type=int, default=5)
    parser.add_argument("--latent-sphere-elev", type=float, default=20.0)
    parser.add_argument("--latent-sphere-azim", type=float, default=45.0)
    parser.add_argument(
        "--latent-sphere-color-mode",
        choices=["label", "timeline"],
        default="label",
        help="Color latent paths either by benign/anomalous labels (label) or by age from older to newer windows (timeline).",
    )
    parser.add_argument(
        "--latent-geometry",
        choices=["sphere", "euclidean"],
        default="sphere",
        help="Render latent paths on the unit sphere or in ordinary Euclidean 3D space.",
    )
    parser.add_argument(
        "--pdf",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Export only the test reconstruction and final latent sphere as PDF files, with latent paths rendered in black.",
    )

    parser.add_argument("--out-dir", type=str, default=None, help="Output directory (default: out/reconstructions/qad_checkpoint_viz/<checkpoint-stem>)")
    parser.add_argument("--loglevel", choices=["debug", "info", "warning", "error", "critical"], default="info")
    return parser


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint_path:
        ckpt_path = Path(args.checkpoint_path)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    ckpt_dir = Path(args.checkpoint_dir)
    if args.experiment_id is not None and args.epoch is not None:
        ckpt_path = ckpt_dir / f"checkpoint_{args.experiment_id}_{args.epoch}.h5"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        return ckpt_path

    candidates = sorted(ckpt_dir.glob("checkpoint_*.h5"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints found in {ckpt_dir}. Pass --checkpoint-path or --experiment-id/--epoch."
        )
    return candidates[-1]


def _parse_checkpoint_name(ckpt_path: Path) -> tuple[str, int]:
    match = re.match(r"^checkpoint_(.+)_(best|\d+)\.h5$", ckpt_path.name)
    if not match:
        return ckpt_path.stem, 0

    exp_id, epoch_text = match.group(1), match.group(2)
    epoch = int(epoch_text) if epoch_text.isdigit() else 0
    return exp_id, epoch


def _window_index_color(window_idx: int, n_windows: int):
    if n_windows <= 1:
        t = 0.0
    else:
        t = float(window_idx) / float(n_windows - 1)
    # Custom palette to avoid the default matplotlib look.
    start = np.array([76.0, 110.0, 245.0]) / 255.0   # indigo
    end = np.array([255.0, 107.0, 107.0]) / 255.0    # coral
    rgb = start + (end - start) * t
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]), 1.0)


def _plot_arrow_path(ax3d, xs, ys, zs, *, color, alpha, linewidth, arrow_length_ratio=0.06):
    xs = np.asarray(xs)
    ys = np.asarray(ys)
    zs = np.asarray(zs)
    if xs.size < 2:
        return

    ax3d.quiver(
        xs[:-1],
        ys[:-1],
        zs[:-1],
        xs[1:] - xs[:-1],
        ys[1:] - ys[:-1],
        zs[1:] - zs[:-1],
        color=color,
        alpha=alpha,
        linewidth=linewidth,
        arrow_length_ratio=arrow_length_ratio,
        normalize=False,
        length=1.0,
        pivot="tail",
    )


def _contiguous_label_runs(labels):
    labels = np.asarray(labels).astype(bool)
    if labels.size == 0:
        return

    run_start = 0
    run_value = bool(labels[0])
    for idx in range(1, labels.size):
        current = bool(labels[idx])
        if current != run_value:
            yield run_start, idx, run_value
            run_start = idx
            run_value = current
    yield run_start, labels.size, run_value


def _save_gif_from_pngs(frame_paths, gif_path, duration_ms=350):
    frames = [Image.open(path).convert("RGB") for path in frame_paths]
    if not frames:
        return

    gif_path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        gif_path,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _compute_weighted_anomaly_timeline(eval_args, dl_tst, modules, desired_t, device, feature_weights):
    n_time = int(dl_tst.dataset.indcs.max().item()) + 1
    score_sum = np.zeros(n_time, dtype=np.float64)
    label_sum = np.zeros(n_time, dtype=np.float64)
    counts = np.zeros(n_time, dtype=np.float64)
    weights = torch.as_tensor(feature_weights, device=device, dtype=torch.float32)

    with torch.no_grad():
        for batch in dl_tst:
            parts = {key: val.to(device) for key, val in batch.items()}
            indcs = parts["inp_indcs"].detach().cpu().numpy().astype(int)
            inp = (parts["inp_obs"], parts["inp_msk"], parts["inp_tps"])

            h = modules["recog_net"](inp)
            qzx, _ = modules["qzx_net"](h, desired_t)
            zis = qzx.rsample((eval_args.mc_eval_samples,))
            pxz = modules["pxz_net"](zis)

            aux_log_prob = -pxz.log_prob(parts["evd_obs"])
            if aux_log_prob.dim() >= 4:
                aux_log_prob = aux_log_prob.squeeze()
            if aux_log_prob.dim() == 2:
                aux_log_prob = aux_log_prob[None, :, :]
            if aux_log_prob.dim() == 4:
                aux_log_prob = aux_log_prob.mean(dim=0)

            weighted_scores = (aux_log_prob * weights[None, None, :]).sum(dim=-1)
            weighted_scores = weighted_scores.detach().cpu().numpy()
            labels = parts["aux_tgt"].detach().cpu().numpy()

            for idx in range(weighted_scores.shape[0]):
                np.add.at(score_sum, indcs[idx], weighted_scores[idx])
                np.add.at(label_sum, indcs[idx], labels[idx])
                np.add.at(counts, indcs[idx], np.ones_like(indcs[idx], dtype=np.float64))

    valid = counts > 0
    timeline = np.zeros_like(score_sum)
    timeline[valid] = score_sum[valid] / counts[valid]

    binary_labels = np.zeros_like(label_sum, dtype=bool)
    binary_labels[valid] = (label_sum[valid] / counts[valid]) > 0.5
    return timeline, binary_labels


def _plot_anomaly_score_timeline(scores, labels, out_path, title):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=bool)
    x = np.arange(scores.shape[0])

    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.plot(x, scores, color="black", linewidth=1.1, alpha=0.9, label="weighted anomaly score")

    in_segment = False
    seg_start = 0
    for idx, flagged in enumerate(labels.tolist()):
        if flagged and not in_segment:
            in_segment = True
            seg_start = idx
        elif not flagged and in_segment:
            in_segment = False
            ax.axvspan(seg_start - 0.5, idx - 0.5, color="red", alpha=0.14, linewidth=0)
    if in_segment:
        ax.axvspan(seg_start - 0.5, len(labels) - 0.5, color="red", alpha=0.14, linewidth=0)

    #ax.set_xlabel("timepoint")
   #ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(alpha=0.22)
    #ax.legend(loc="upper right", fontsize=8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax.set_xlim(675*200, 750*200)
    ax.grid(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_motivational_style_test_figure(actual, anomaly_mask, score, score_anomaly_mask, out_path,
                                         channel_labels=None):
    """Motivational-figure-style plot (stacked per-channel traces on top,
    anomaly score panel at the bottom). No new scores or statistics are
    computed here, only re-plotting of existing test-set data.
    """
    LINE_COLOR = "0.2"
    GT_COLOR = "#E8A33D"      # ground-truth (labeled) anomaly regions
    DET_COLOR = "#D62728"     # detected anomaly (score over threshold)
    GT_ALPHA = 0.15
    GRID_ALPHA = 0.25
    LINE_WIDTH = 1.0
    YLABEL_X = -0.05
    THRESHOLD = 0.4           # score panel threshold; drives both the red split and the line

    actual = np.asarray(actual)
    if actual.ndim == 3:
        actual = actual.reshape(-1, actual.shape[-1])
    anomaly_mask = np.asarray(anomaly_mask, dtype=bool)
    score = np.asarray(score, dtype=float)
    score_anomaly_mask = np.asarray(score_anomaly_mask, dtype=bool)

    n_samples = min(actual.shape[0], anomaly_mask.shape[0], score.shape[0], score_anomaly_mask.shape[0])
    channel_idx = [5, 11, 12]
    actual = actual[:n_samples, channel_idx]
    anomaly_mask = anomaly_mask[:n_samples]
    score = score[:n_samples]
    t = np.arange(n_samples)

    # Real QAPPD feature names; falls back to generic labels if not passed.
    if channel_labels is None:
        channel_labels = [f"Feature {j}" for j in channel_idx]

    label_edges = np.diff(np.pad(anomaly_mask.astype(np.int8), (1, 1)))
    anomaly_starts = np.flatnonzero(label_edges == 1)
    anomaly_ends = np.flatnonzero(label_edges == -1)
    anomaly_spans = [
        (int(t[start]), int(t[min(end, n_samples - 1)]) + 1)
        for start, end in zip(anomaly_starts, anomaly_ends)
    ]

    def draw_gt_spans(ax):
        """Mark ground-truth regions identically on every panel so a score
        peak can be traced up to the data (panels share the x-axis)."""
        for span_start, span_end in anomaly_spans:
            ax.axvspan(span_start, span_end, color=GT_COLOR, alpha=GT_ALPHA, linewidth=0)
            ax.axvline(span_start, color=GT_COLOR, alpha=0.6, linewidth=0.8, linestyle="--")
            ax.axvline(span_end, color=GT_COLOR, alpha=0.6, linewidth=0.8, linestyle="--")

    n_channels = actual.shape[1] + 1
    spacer_height = 0.6

    fig = plt.figure(figsize=(15, 1.2 * n_channels + 1.0), constrained_layout=False)
    height_ratios = [1.2] * (n_channels - 1) + [spacer_height, 1.0]
    gs = fig.add_gridspec(nrows=n_channels + 1, ncols=1, height_ratios=height_ratios, hspace=0.06)

    axs = []
    for i in range(n_channels - 1):
        share_ax = axs[0] if axs else None
        axs.append(fig.add_subplot(gs[i, 0], sharex=share_ax))

    sep_ax = fig.add_subplot(gs[n_channels - 1, 0])
    sep_ax.set_axis_off()

    axs.append(fig.add_subplot(gs[n_channels, 0], sharex=axs[0]))

    for i in range(actual.shape[1]):
        ax = axs[i]
        ax.plot(t, actual[:, i], color=LINE_COLOR, linewidth=LINE_WIDTH)
        draw_gt_spans(ax)
        ax.set_ylabel(channel_labels[i], rotation=90, va="center")
        ax.yaxis.set_label_coords(YLABEL_X, 0.5)
        ax.grid(axis="y", alpha=GRID_ALPHA, linewidth=0.6)

    # Color the score above the threshold in red. Extend the red by one sample
    # into the below region on each side so the black and red lines share the
    # boundary point: the last below point joins the first above point, and the
    # last above point joins the first below point, with no gap.
    above = score > THRESHOLD
    above_ext = above.copy()
    above_ext[:-1] |= above[1:]   # include the below-point just before each anomalous run
    above_ext[1:] |= above[:-1]   # include the below-point just after each anomalous run
    score_below = np.where(above, np.nan, score)
    score_above = np.where(above_ext, score, np.nan)

    ax_score = axs[-1]
    ax_score.plot(t, score_below, color=LINE_COLOR, linewidth=LINE_WIDTH, label="anomaly score")
    ax_score.plot(t, score_above, color=DET_COLOR, linewidth=LINE_WIDTH * 1.5)
    draw_gt_spans(ax_score)   # same ground-truth markers as the panels above
    ax_score.set_ylabel(r"Anomaly Score$", rotation=90, va="center")
    ax_score.yaxis.set_label_coords(YLABEL_X, 0.5)
    ax_score.set_xlabel("Timepoint (concatenated test windows)")
    ax_score.set_xlim(t.min(), t.max())
    ax_score.set_title("Anomaly score")
    axs[0].set_title("Sensor data (test)")

    for ax in axs[:-1]:
        ax.tick_params(axis="x", which="both", labelbottom=False)

    for ax in axs:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    ax_score.axhline(y=THRESHOLD, color="purple", linestyle="--", linewidth=1.5)
    ax_score.text(0.995, 0.1, "threshold", color="purple", va="bottom", ha="right",
                  transform=ax_score.get_yaxis_transform())

    # Name the two now-distinct colors so the figure reads without the caption.
    ax_score.annotate("detection",
                      xy=(9500, 4.5), xytext=(8000, 7.5),
                      color=DET_COLOR, ha="left", va="center", fontsize=9,
                      arrowprops=dict(arrowstyle="->", color=DET_COLOR, lw=0.8))
    if anomaly_spans:
        gs0, ge0 = anomaly_spans[0]
        xc = (0.66 * gs0 + 0.33 * ge0)
        axs[0].annotate("ground truth",
                        xy=(xc, 0.2), xycoords=("data", "axes fraction"),
                        xytext=(gs0 - 0.08 * n_samples, 0.20), textcoords=("data", "axes fraction"),
                        color=GT_COLOR, ha="left", va="top", fontsize=9,
                        arrowprops=dict(arrowstyle="->", color=GT_COLOR, lw=0.8))

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(pad=0.2, h_pad=0.1)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def _compute_xyz_bounds(*arrays, padding=0.08):
    stacked = []
    for arr in arrays:
        if arr is None:
            continue
        arr = np.asarray(arr)
        if arr.size == 0:
            continue
        arr = arr.reshape(-1, arr.shape[-1])
        if arr.shape[-1] < 3:
            continue
        stacked.append(arr[:, :3])

    if not stacked:
        return (-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)

    pts = np.concatenate(stacked, axis=0)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    pad = span * padding
    return tuple((float(mins[i] - pad[i]), float(maxs[i] + pad[i])) for i in range(3))


def _style_euclidean_box(ax3d, xlim, ylim, zlim):
    ax3d.set_xlim(*xlim)
    ax3d.set_ylim(*ylim)
    ax3d.set_zlim(*zlim)

    dx = float(xlim[1] - xlim[0])
    dy = float(ylim[1] - ylim[0])
    dz = float(zlim[1] - zlim[0])
    try:
        ax3d.set_box_aspect((dx, dy, dz))
    except Exception:
        try:
            ax3d.set_box_aspect((1.0, 1.0, 1.0))
        except Exception:
            pass

    ax3d.set_axis_on()
    ax3d.grid(False)

    pane_axes = [ax3d.xaxis, ax3d.yaxis, ax3d.zaxis]
    for axis in pane_axes:
        try:
            axis.pane.set_facecolor((1.0, 1.0, 1.0, 1.0))
            axis.pane.set_edgecolor((0.12, 0.12, 0.12, 1.0))
        except Exception:
            pass
        try:
            axis.line.set_color((0.12, 0.12, 0.12, 1.0))
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["linewidth"] = 0.0
        except Exception:
            pass


def _plot_latent_sphere_dashboard_matplotlib(
    latents,
    window_indices,
    title,
    out_path,
    shadow_latents=None,
    elev=20,
    azim=45,
    use_uniform_color=False,
    uniform_color="#0F7173",
    timepoint_labels=None,
    normal_label_color="#4C6EF5",
    anomaly_label_color="#FF6B6B",
    color_mode="label",
    latent_geometry="sphere",
    force_black_paths=False,
):
    """Static matplotlib latent-sphere figure (sphere only, no time-series panels)."""
    latents = np.asarray(latents)

    if latents.ndim == 3:
        latents = latents[None, ...]

    n_windows, n_samples = latents.shape[0], latents.shape[1]
    n_time = latents.shape[2]
    mean_alpha = 0.95
    if force_black_paths:
        mean_alpha = 0.5

    # `shadow_latents` is intentionally ignored: render mean trajectories only.

    labels = None
    if timepoint_labels is not None:
        labels = np.asarray(timepoint_labels)
        if labels.ndim != 2 or labels.shape[0] != n_windows or labels.shape[1] != n_time:
            logging.warning(
                "Ignoring timepoint labels for latent-sphere coloring due to shape mismatch: expected (%d, %d), got %s",
                n_windows,
                n_time,
                labels.shape,
            )
            labels = None

    fig = plt.figure(figsize=(8, 7))
    ax3d = fig.add_subplot(111, projection="3d")

    if latent_geometry == "sphere":
        # Draw unit sphere wireframe (matches analyze_irregular_sine_exp style).
        u, v = np.mgrid[0:2 * np.pi:100j, 0:np.pi:50j]
        x = np.cos(u) * np.sin(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(v)
        ax3d.plot_wireframe(x, y, z, color='0.75', alpha=0.4, linewidth=0.5)

    for w in range(n_windows):
        color = uniform_color if use_uniform_color else _window_index_color(w, n_windows)
        timeline_color = _window_index_color(w, n_windows)
        path_color = timeline_color if color_mode == "timeline" else color
        if force_black_paths:
            path_color = (0.74, 0.74, 0.74, 1)

        # Mean path overlay: emphasized trajectory on top of the shadow.
        for s in range(n_samples):
            xs = latents[w, s, :, 0]
            ys = latents[w, s, :, 1]
            zs = latents[w, s, :, 2]

            if force_black_paths or color_mode == "timeline" or labels is None:
                _plot_arrow_path(
                    ax3d,
                    xs,
                    ys,
                    zs,
                    color=path_color,
                    alpha=mean_alpha,
                    linewidth=1.6,
                    arrow_length_ratio=0.08,
                )
            else:
                for start, stop, label_active in _contiguous_label_runs(labels[w]):
                    seg_color = anomaly_label_color if label_active else normal_label_color
                    _plot_arrow_path(
                        ax3d,
                        xs[start:stop],
                        ys[start:stop],
                        zs[start:stop],
                        color=seg_color,
                        alpha=mean_alpha,
                        linewidth=1.6,
                        arrow_length_ratio=0.08,
                    )

    # Mark the global start of the first window and the end of the full sequence.
    start_xyz = latents[0, 0, 0, :3]
    end_xyz = latents[-1, 0, -1, :3]
    ax3d.scatter(
        start_xyz[0], start_xyz[1], start_xyz[2],
        color="0.18", edgecolor="white", linewidth=0.4, marker="o", s=28, alpha=0.95
    )
    ax3d.scatter(
        end_xyz[0], end_xyz[1], end_xyz[2],
        color="0.08", edgecolor="white", linewidth=0.4, marker="X", s=38, alpha=0.95
    )

    if latent_geometry == "sphere":
        ax3d.set_xlim(-1, 1)
        ax3d.set_ylim(-1, 1)
        ax3d.set_zlim(-1, 1)
        ax3d.set_box_aspect((1, 1, 1))
        ax3d.set_axis_off()
    else:
        xlim, ylim, zlim = _compute_xyz_bounds(latents)
        _style_euclidean_box(ax3d, xlim, ylim, zlim)
        ax3d.set_aspect('equal', 'box')
        ax3d.set_xticks([])
        ax3d.set_yticks([])
        ax3d.set_zticks([])
    ax3d.view_init(elev=elev, azim=azim)
    if latent_geometry == "sphere":
        ax3d.set_axis_off()

    #fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.loglevel.upper(), logging.INFO),
        format="[%(asctime)s] [%(levelname)-8s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ckpt_path = _resolve_checkpoint_path(args)
    logging.info("Loading checkpoint: %s", ckpt_path)

    checkpoint = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    if "args" not in checkpoint or "modules" not in checkpoint:
        raise ValueError("Checkpoint does not contain expected keys: 'args' and 'modules'.")

    train_args = checkpoint["args"]
    experiment_id, epoch = _parse_checkpoint_name(ckpt_path)

    runtime_device = args.device if args.device is not None else getattr(train_args, "device", "cpu")
    runtime_device = resolve_device(runtime_device)
    train_args.device = runtime_device

    if args.data_dir is not None:
        train_args.data_dir = args.data_dir
    if args.trace_id is not None:
        train_args.trace_id = args.trace_id

    out_dir = Path(args.out_dir) if args.out_dir is not None else Path("out/reconstructions/qad_checkpoint_viz") / ckpt_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    provider_viz = QADProvider(
        data_dir=train_args.data_dir,
        dataset_number=train_args.trace_id,
        window_length=train_args.data_window_length,
        window_overlap=0.0,
        data_normalization_strategy=train_args.data_normalization_strategy,
        subsample=train_args.subsample,
        seed=train_args.seed,
        fixed_subsample_mask=train_args.fixed_subsample_mask,
        train_shuffle=False,
    )

    try:
        desired_t_ckpt = checkpoint.get("desired_t")
        if desired_t_ckpt is None:
            desired_t = torch.linspace(0, 1.0, provider_viz.num_timepoints, device=runtime_device).float()
        else:
            desired_t = desired_t_ckpt.detach().to(runtime_device).float()

        modules, _, _, _ = build_modules_and_optim(train_args, provider_viz.input_dim, desired_t)
        modules.load_state_dict(checkpoint["modules"], strict=True)
        modules.eval()

        viz_args = argparse.Namespace(
            reconstruct_n_windows=args.reconstruct_n_windows,
            reconstruct_mc_samples=args.reconstruct_mc_samples,
            device=runtime_device,
        )

        # Train reconstruction.
        trn_ds = provider_viz._ds_trn
        trn_indices = _select_middle_window_indices(trn_ds, args.reconstruct_n_windows)
        trn_parts = _gather_window_batch(trn_ds, trn_indices, runtime_device)
        trn_actual, trn_recon, _ = _decode_reconstruction(viz_args, modules, desired_t, trn_parts)
        train_recon_path = out_dir / f"{experiment_id}_epoch{epoch:04d}_train_reconstruction_matplotlib.{ 'pdf' if args.pdf else 'png' }"
        if not args.pdf:
            _plot_actual_vs_reconstructed(
                trn_actual,
                trn_recon,
                anomaly_mask=None,
                title=f"Train reconstruction @ epoch {epoch} (windows {trn_indices[0]}-{trn_indices[-1]})",
                out_path=str(train_recon_path),
                likelihood=None,
            )
            logging.info("Saved train reconstruction plot to %s", train_recon_path)

        # Test reconstruction.
        tst_ds = provider_viz._ds_tst
        tst_indices = _select_anomalous_window_indices(tst_ds, args.reconstruct_n_windows)
        tst_parts = _gather_window_batch(tst_ds, tst_indices, runtime_device)
        tst_actual, tst_recon, tst_nll = _decode_reconstruction(viz_args, modules, desired_t, tst_parts)
        tst_anomaly_mask = (tst_parts["aux_tgt"].detach().cpu().flatten() > 0).numpy()
        test_recon_path = out_dir / f"{experiment_id}_epoch{epoch:04d}_test_reconstruction_matplotlib.{ 'pdf' if args.pdf else 'png' }"
        _plot_actual_vs_reconstructed(
            tst_actual,
            tst_recon,
            anomaly_mask=tst_anomaly_mask,
            title=f"",
            out_path=str(test_recon_path),
            #likelihood=tst_nll,
        )
        logging.info("Saved test reconstruction plot to %s", test_recon_path)

        # Weighted anomaly score timeline (weights from train reconstruction MSE).
        score_eval_args = argparse.Namespace(
            mc_eval_samples=max(1, int(getattr(train_args, "mc_eval_samples", 1)))
        )
        score_batch_size = max(1, int(getattr(train_args, "batch_size", 256)))
        dl_trn_score = provider_viz.get_train_loader(
            batch_size=score_batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
        )
        dl_tst_score = provider_viz.get_test_loader(
            batch_size=score_batch_size,
            shuffle=False,
            collate_fn=None,
            num_workers=args.num_workers,
            pin_memory=False,
            drop_last=False,
        )

        feature_stats = calculate_feature_reconstruction_weights(
            score_eval_args,
            dl_trn_score,
            modules,
            desired_t,
            runtime_device,
            weighting="inverse",
        )
        weighted_timeline, weighted_labels = _compute_weighted_anomaly_timeline(
            score_eval_args,
            dl_tst_score,
            modules,
            desired_t,
            runtime_device,
            feature_stats["feature_weights"],
        )
        timeline_path = out_dir / f"{experiment_id}_epoch{epoch:04d}_test_anomaly_score_timeline_matplotlib.{ 'pdf' if args.pdf else 'png' }"
        _plot_anomaly_score_timeline(
            weighted_timeline,
            weighted_labels,
            timeline_path,
            title="Test anomaly score timeline (weighted by train reconstruction MSE)",
        )
        logging.info("Saved anomaly score timeline plot to %s", timeline_path)

        # Motivational-figure-style test plot: re-plots the same test actual
        # data and anomaly score already computed above (tst_actual /
        # tst_anomaly_mask / weighted_timeline / weighted_labels) in the
        # stacked-channels-plus-score layout from gen_motiv_figure.py.
        motiv_window_length = int(train_args.data_window_length)
        motiv_start_idx = tst_indices[0] * motiv_window_length
        motiv_end_idx = (tst_indices[-1] + 1) * motiv_window_length
        motiv_fig_path = out_dir / f"motivational_figure.pdf"
        _plot_motivational_style_test_figure(
            actual=tst_actual,
            anomaly_mask=tst_anomaly_mask,
            score=weighted_timeline[motiv_start_idx:motiv_end_idx],
            score_anomaly_mask=weighted_labels[motiv_start_idx:motiv_end_idx],
            out_path=motiv_fig_path,
        )
        logging.info("Saved motivational-style test figure to %s", motiv_fig_path)
        import sys; sys.exit(1)
        def _render_latent_sphere(
            ds,
            sphere_indices,
            split_name,
            use_uniform_color=False,
            color_by_labels=False,
            output_filename=None,
            output_path=None,
            latent_geometry="sphere",
            force_black_paths=False,
        ):
            if len(sphere_indices) == 0:
                logging.warning("No windows selected for latent-sphere figure (%s); skipping.", split_name)
                return
            sphere_parts = _gather_window_batch(ds, sphere_indices, runtime_device)
            inp = (sphere_parts["inp_obs"], sphere_parts["inp_msk"], sphere_parts["inp_tps"])

            with torch.no_grad():
                h = modules["recog_net"](inp)
                qzx, _ = modules["qzx_net"](h, desired_t)
                latent_samples = qzx.rsample((args.latent_sphere_mc_samples,))
                latent_mean = latent_samples.mean(0, keepdim=True)
                latent_samples = latent_samples.detach().cpu()
                latent_mean = latent_mean.detach().cpu()

            if latent_mean.shape[-1] < 3:
                logging.warning("z_dim=%d < 3; skipping latent-sphere figure (%s).", latent_mean.shape[-1], split_name)
                return

            latents = latent_mean.permute(1, 0, 2, 3).numpy()
            #s#hadow_latents = latent_samples.permute(1, 0, 2, 3).numpy()

            if output_path is not None:
                sphere_out = Path(output_path)
            elif output_filename is None:
                sphere_out = out_dir / f"{experiment_id}_epoch{epoch:04d}_latent_sphere_{split_name}_matplotlib.{ 'pdf' if args.pdf else 'png' }"
            else:
                sphere_out = out_dir / output_filename

            labels = None
            if color_by_labels:
                labels = sphere_parts["aux_tgt"].detach().cpu().numpy()

            _plot_latent_sphere_dashboard_matplotlib(
                latents=latents,
                window_indices=sphere_indices,
                title=f"Latent paths on {'sphere' if latent_geometry == 'sphere' else 'euclidean space'} @ epoch {epoch} ({split_name} windows {sphere_indices[0]}-{sphere_indices[-1]})",
                out_path=sphere_out,
                shadow_latents=None,
                elev=args.latent_sphere_elev,
                azim=args.latent_sphere_azim,
                use_uniform_color=use_uniform_color,
                timepoint_labels=labels,
                color_mode=args.latent_sphere_color_mode,
                latent_geometry=latent_geometry,
                force_black_paths=force_black_paths,
            )
            logging.info("Saved latent-sphere matplotlib plot to %s", sphere_out)

        def _render_cumulative_latent_gif(
            ds,
            sphere_indices,
            split_name,
            output_filename,
            use_uniform_color=False,
            color_by_labels=False,
            latent_geometry="sphere",
            force_black_paths=False,
        ):
            if len(sphere_indices) == 0:
                logging.warning("No windows selected for cumulative latent GIF (%s); skipping.", split_name)
                return

            gif_path = out_dir / output_filename
            with tempfile.TemporaryDirectory(prefix="latent_sphere_frames_") as tmp_dir:
                tmp_dir = Path(tmp_dir)
                frame_paths = []
                for end_idx in range(1, len(sphere_indices) + 1):
                    frame_path = tmp_dir / f"{split_name}_frame_{end_idx:04d}.png"
                    _render_latent_sphere(
                        ds=ds,
                        sphere_indices=sphere_indices[:end_idx],
                        split_name=f"{split_name}_frame_{end_idx:04d}",
                        use_uniform_color=use_uniform_color,
                        color_by_labels=color_by_labels,
                        output_path=frame_path,
                        latent_geometry=latent_geometry,
                        force_black_paths=force_black_paths,
                    )
                    frame_paths.append(frame_path)

                _save_gif_from_pngs(frame_paths, gif_path)
                logging.info("Saved cumulative latent GIF to %s", gif_path)

        # Sphere-only latent plots for both test and train windows.
        sphere_tst_indices = _select_anomalous_window_indices(tst_ds, args.latent_sphere_n_windows)
        sphere_tst_indices_all = list(range(len(tst_ds)))
        _render_latent_sphere(
            ds=tst_ds,
            sphere_indices=sphere_tst_indices_all,
            split_name="test",
            use_uniform_color=False,
            color_by_labels=True,
            latent_geometry=args.latent_geometry,
            force_black_paths=args.pdf,
        )
        sphere_trn_indices = range(len(trn_ds))
        _render_latent_sphere(
            ds=trn_ds,
            sphere_indices=sphere_trn_indices,
            split_name="train",
            use_uniform_color=False,
            color_by_labels=True,
            latent_geometry=args.latent_geometry,
            force_black_paths=args.pdf,
        )
        if not args.pdf:
            _render_cumulative_latent_gif(
                ds=tst_ds,
                sphere_indices=sphere_tst_indices,
                split_name="test",
                output_filename=f"{experiment_id}_epoch{epoch:04d}_latent_sphere_test_cumulative.gif",
                use_uniform_color=False,
                color_by_labels=True,
                latent_geometry=args.latent_geometry,
            )

        if not args.pdf:
            sphere_trn_indices = _select_middle_window_indices(trn_ds, args.latent_sphere_n_windows)
            _render_latent_sphere(
                ds=trn_ds,
                sphere_indices=sphere_trn_indices,
                split_name="train",
                use_uniform_color=True,
                color_by_labels=False,
                latent_geometry=args.latent_geometry,
                force_black_paths = args.pdf,
            )

            # Requested paper/slide exports.
            _render_latent_sphere(
                ds=trn_ds,
                sphere_indices=sphere_trn_indices,
                split_name="train",
                use_uniform_color=True,
                color_by_labels=False,
                output_filename="motiv_fig_sphere_latent.pdf",
                latent_geometry=args.latent_geometry,
            )

            sphere_trn_indices_small = _select_middle_window_indices(trn_ds, 5)
            _render_latent_sphere(
                ds=trn_ds,
                sphere_indices=sphere_trn_indices_small,
                split_name="train_5windows",
                use_uniform_color=True,
                color_by_labels=False,
                output_filename="pipeline_fig_sphere_path.pdf",
                latent_geometry=args.latent_geometry,
            )
        else:
            final_test_indices = sphere_tst_indices_all
            _render_latent_sphere(
                ds=tst_ds,
                sphere_indices=final_test_indices,
                split_name="test_final",
                use_uniform_color=False,
                color_by_labels=False,
                output_filename=f"{experiment_id}_epoch{epoch:04d}_latent_sphere_test_final.pdf",
                latent_geometry=args.latent_geometry,
                force_black_paths=True,
            )

    finally:
        if hasattr(provider_viz, "cleanup"):
            provider_viz.cleanup()

    logging.info("Finished visualization. Outputs written to %s", out_dir)


if __name__ == "__main__":
    main()


