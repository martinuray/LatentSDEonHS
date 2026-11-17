#!/usr/bin/env python3
"""Test-set anomaly scores of one seed's per-trace checkpoints, by label and input noise.

A sibling of ``conformal_anomaly_detection.py``, using the same checkpoint
handling: anomaly_detection.py writes one checkpoint per trace
(``..._trace-<id>_best.h5``), and each one carries its trace id and seed. This
script uses one seed only: of all the checkpoints given, it keeps those trained
with ``--select-seed``, one per trace. That can be omitted when all checkpoints
share a single seed.

For every trace, the model and data provider are restored from the args stored
in the checkpoint, and the test split is scored with the same post-processing
as ``anomaly_detection.py`` (smoothing, feature aggregation, z-normalisation),
giving one scalar anomaly score per timepoint. This is done once on the clean
test data and once per ``--noise-levels`` sigma, with i.i.d. Gaussian noise
added to the test inputs. The noise is in the provider's normalised data units
(min-max scaled to [0, 1] for QAD).

Noise model: per trace, one standard-normal pattern ``z[t, f]`` is drawn per
global time index and feature (seeded with the checkpoint's seed), and the
noise at level sigma is ``sigma * z``. This means overlapping windows see the
same noisy value for a timepoint, the model input (``inp_obs``) and the
reconstruction target (``evd_obs``) are perturbed identically, and the columns
differ only in the noise scale, not in the noise realisation. The model
itself, the score normalisation statistics and the feature weights are always
those of the clean training data. The torch RNG (random subsample masks, MC
samples) is reseeded before each pass, so the passes differ only in the noise.

Figure: one axis per trace, stacked. Each axis holds, per noise level (clean
first, top to bottom), a thin horizontal box plot of the benign and one of the
anomalous test timepoints, so all levels of a trace share one score axis.
Scores of different traces come from different models, so each trace gets its
own axis range unless ``--sharex`` is given.

Outputs (in ``--eval-out-dir``):
  * ``SCOREDIST_<dataset>_seed<seed>_<run>.{pdf,png}``: the box plots,
  * ``SCOREDIST_<dataset>_seed<seed>_<run>_density.{pdf,png}``: the same
    layout and data, each trace's axis showing a Gaussian KDE (a smoothed
    histogram normalised to area 1, so the groups are comparable in shape
    despite their very different sizes) per label and noise level -- colour
    for the label, shade for the level (darkest: clean); the density axis is
    left unlabelled. With ``--log-x`` the KDE is fitted on log scores.
  * ``SCOREDIST_<dataset>_seed<seed>_<run>.npz``: per trace and level, the test
    index, scores and labels (keys ``trace<id>__sigma<level>__{index,scores,labels}``),
  * ``SCOREDIST_<dataset>_seed<seed>_<run>.csv``: count, mean and quantiles per
    trace, level and label.
  * ``SCOREDIST_<dataset>_seed<seed>_<run>_snapshots.{pdf,png}`` (unless
    ``--no-snapshots``): per trace (rows) and noise level (columns), a window of
    the test series around a benign->anomalous transition (``--onset``, the
    first by default). It shows all variates, with that level's noise added
    (left axis), and the per-timepoint NLL on a twin axis (right). The NLL is
    the reconstruction ``-log p(x|z)`` summed over the variates, without score
    normalisation or smoothing. The x axis is the time relative to the onset and
    is shared by every panel; anomalous stretches are shaded. The folder
    ``..._snapshots/`` holds the same panels as single figures, with a legend.
    The snapshot NLL also goes into the .npz (``...__nll`` over ``trace<id>__snapshot_t``).

Examples::

    # seed 42, traces 1-10, noise levels 0.1 0.5 1.0 (+ clean)
    python score_distribution_by_label.py --dataset QAD \\
        --checkpoint-dir checkpoints --checkpoint-pattern 'checkpoint_AD_QAD_*_best.h5' \\
        --select-seed 42

    # explicit checkpoints, other noise levels, one score axis for all traces
    python score_distribution_by_label.py --dataset QAD --checkpoints ckpt_trace-*_best.h5 \\
        --noise-levels 0.05 0.2 0.4 --sharex
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402

from anomaly_detection import (  # noqa: E402
    AGG_STRATEGIES,
    _normalize_trace_ids,
    build_provider,
    calculate_feature_reconstruction_weights,
    calculate_z_normalization_values,
    compute_eval_scores,
    delete_processed_data,
)
from conformal_anomaly_detection import (  # noqa: E402
    build_parser as build_conformal_parser,
    collect_candidates,
    covered_timepoints,
    parse_checkpoint_specs,
    scalar_scores,
)
from evaluate_checkpoints import (  # noqa: E402
    _bootstrap_dataset,
    _explicit_cli_dests,
    _make_loaders,
    _resolve_trace_index,
    _restore_modules,
    discover_checkpoints,
)
from utils.logger import set_up_logging  # noqa: E402
from utils.misc import set_seed  # noqa: E402

# (label, label value, colour), drawn top to bottom in each cell.
GROUPS = (("Benign", 0, "#1F77B4"), ("Anomalous", 1, "#D62728"))
DEFAULT_NOISE_LEVELS = [0.1, 0.25, 0.4]
CLEAN = 0.0


def build_parser(dataset: str, config_file: str | None) -> argparse.ArgumentParser:
    # Reuse the conformal parser for the checkpoint/seed/data arguments; its
    # alpha/p-value options are accepted but play no role here.
    parser = build_conformal_parser(dataset, config_file)
    parser.description = ("Score the test split of one seed's per-trace checkpoints, clean and with "
                          "Gaussian input noise, and plot the scores split by label.")
    parser.set_defaults(eval_out_dir="logs/score_distribution")
    group = parser.add_argument_group("Score distribution plot")
    group.add_argument(
        "--select-seed", type=int, default=None,
        help="Use only the checkpoints trained with this seed (read from the checkpoint). "
             "Required when the given checkpoints span several seeds.",
    )
    group.add_argument(
        "--noise-levels", type=float, nargs="+", default=list(DEFAULT_NOISE_LEVELS), metavar="SIGMA",
        help="Std of the Gaussian noise added to the test inputs, one figure column each, after the "
             f"clean column. Default: {' '.join(map(str, DEFAULT_NOISE_LEVELS))}.",
    )
    group.add_argument(
        "--sharex", action="store_true",
        help="Share the score axis across traces. Off by default: the traces' scores come from "
             "different models and need not share a scale.",
    )
    group.add_argument("--log-x", action="store_true", help="Log-scale the score axes.")
    group.add_argument(
        "--show-fliers", action=argparse.BooleanOptionalAction, default=True,
        help="Draw points beyond the whiskers (a test set has many, so they are tiny and translucent).",
    )
    group.add_argument(
        "--whisker-range", type=float, nargs=2, default=None, metavar=("LOW_PCT", "HIGH_PCT"),
        help="Whiskers at these percentiles (e.g. 1 99) instead of 1.5 IQR.",
    )
    group.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Figure file formats.")

    group = parser.add_argument_group("Anomaly-onset snapshots")
    group.add_argument(
        "--snapshots", action=argparse.BooleanOptionalAction, default=True,
        help="Also plot, per trace and noise level, the (noisy) variates and the NLL around an anomaly onset.",
    )
    group.add_argument(
        "--onset", type=int, default=0,
        help="Which benign->anomalous transition of each trace's test split to show (0 = the first; "
             "negative counts from the end, -1 = the last). Default: 0.",
    )
    group.add_argument(
        "--snapshot-length", type=int, default=None,
        help="Timepoints shown per snapshot. Default: twice the model's data_window_length.",
    )
    group.add_argument(
        "--snapshot-before", type=float, default=0.5,
        help="Share of the snapshot placed before the onset. Default: 0.5.",
    )
    return parser


# --------------------------------------------------------------------------- #
# Checkpoint selection
# --------------------------------------------------------------------------- #
def trace_sort_key(trace: str):
    return (0, int(trace), "") if trace.isdigit() else (1, 0, trace)


def select_one_seed(candidates: list[dict], seed: int | None) -> tuple[int, list[dict]]:
    """Keep one checkpoint per trace, all from the same seed, sorted by trace."""
    seeds = sorted({cand["seed"] for cand in candidates})
    if seed is None:
        if len(seeds) > 1:
            raise SystemExit(f"Checkpoints span seeds {seeds}; pick one with --select-seed.")
        seed = seeds[0]
    chosen = [cand for cand in candidates if cand["seed"] == seed]
    if not chosen:
        raise SystemExit(f"No checkpoint was trained with seed {seed}; available seeds: {seeds}.")

    by_trace: dict[str, list[dict]] = defaultdict(list)
    for cand in chosen:
        by_trace[cand["trace_key"]].append(cand)
    selected = []
    for trace in sorted(by_trace, key=trace_sort_key):
        group = by_trace[trace]
        # discover_checkpoints sorts by mtime, so the last one is the newest.
        if len(group) > 1:
            logging.warning("Trace %s has %d checkpoints for seed %d; using the newest, %s.",
                            trace, len(group), seed, group[-1]["path"].name)
        selected.append(group[-1])
    return seed, selected


# --------------------------------------------------------------------------- #
# Noisy test data
# --------------------------------------------------------------------------- #
class NoisyDataset(Dataset):
    """Adds a fixed per-(timepoint, feature) noise to a test split's observations.

    ``noise`` has shape [max global time index + 1, features]; each window gets
    the rows of its own global time indices, so overlapping windows agree.
    Everything else (indcs, input_dim, ...) is delegated to the wrapped split,
    as compute_eval_scores reads those off ``dl.dataset``.
    """

    def __init__(self, base: Dataset, noise: torch.Tensor):
        self.base = base
        self.noise = noise

    def __len__(self):
        return len(self.base)

    def __getattr__(self, name):
        return getattr(self.__dict__["base"], name)

    def __getitem__(self, idx):
        item = dict(self.base[idx])
        noise = self.noise[torch.as_tensor(item["inp_indcs"]).long()]
        item["inp_obs"] = item["inp_obs"] + noise.to(item["inp_obs"].dtype)
        item["evd_obs"] = item["evd_obs"] + noise.to(item["evd_obs"].dtype)
        return item


def noisy_loader(dl: DataLoader, noise: torch.Tensor) -> DataLoader:
    return DataLoader(NoisyDataset(dl.dataset, noise), batch_size=dl.batch_size, shuffle=False,
                      num_workers=dl.num_workers, pin_memory=dl.pin_memory, collate_fn=dl.collate_fn)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def test_series(dataset: Dataset, length: int) -> tuple[np.ndarray, np.ndarray]:
    """The clean test series [time, feature] and the time span of every window.

    The series (``evd_obs``, NaN where no window covers a timepoint) is rebuilt
    from the windows by their global time indices; overlapping windows carry
    the same values, so the order of writing does not matter. ``spans[i]`` is
    the (first, last) global time index of window ``i``.
    """
    series = np.full((length, int(dataset.input_dim)), np.nan)
    spans = np.empty((len(dataset), 2), dtype=int)
    for i in range(len(dataset)):
        item = dataset[i]
        idx = np.asarray(item["inp_indcs"]).astype(int)
        series[idx] = np.asarray(item["evd_obs"], dtype=float)
        spans[i] = idx.min(), idx.max()
    return series, spans


class WindowSubset(Dataset):
    """Some windows of a split; attributes (indcs, input_dim, ...) are the full split's."""

    def __init__(self, base: Dataset, positions: np.ndarray):
        self.base = base
        self.positions = [int(p) for p in positions]

    def __len__(self):
        return len(self.positions)

    def __getattr__(self, name):
        return getattr(self.__dict__["base"], name)

    def __getitem__(self, idx):
        return self.base[self.positions[idx]]


def score_trace(args, ckpt: dict, provider, ds_idx: int, levels: list[float],
                window_fn=None) -> tuple[dict[float, tuple], dict]:
    """Score the test split once per noise level.

    Returns ``({sigma: (global time index, scalar score, label)}, snapshot data)``.
    The snapshot data holds what the onset plots need: the clean series, the
    noise pattern, the labels and covered timepoints, and -- if `window_fn`
    (snapshot data -> (onset, time slice) or None) finds a snapshot window --
    that window and, per level, the per-timepoint NLL on it (summed over the
    variates; NaN outside the window).

    The NLL needs its own pass: compute_eval_scores applies the score
    normalisation per window *position* (its min/max are [window length,
    feature]) before averaging the overlapping windows, so the raw NLL cannot
    be recovered from the normalised scores, nor the other way round. That pass
    only runs the windows overlapping the snapshot, which still gives every
    timepoint in it all of its windows -- the same average as a full pass, up
    to the MC sampling of the decoder.
    """
    device = args.device
    dl_trn, dl_tst, _dl_val, input_dim, num_timepoints = _make_loaders(provider, ds_idx, args)
    modules, elbo_loss, desired_t = _restore_modules(args, ckpt, input_dim, num_timepoints, device)

    # Score normalisation and feature weights come from the clean training data
    # and stay fixed, so only the test inputs change between the levels.
    normalization_stats = None
    if args.normalize_score:
        normalization_stats = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)
    feature_weights = None
    if args.score_aggregation in ("weighted-mse", "weighted-mse-exp"):
        weighting = "exp-inverse" if args.score_aggregation == "weighted-mse-exp" else "inverse"
        feature_weights = calculate_feature_reconstruction_weights(
            args, dl_trn, modules, desired_t, device, weighting=weighting)["feature_weights"]

    covered = covered_timepoints(dl_tst)
    length = int(covered.max()) + 1
    rng = np.random.default_rng(abs(int(args.seed)))
    base_noise = torch.from_numpy(
        rng.standard_normal((length, int(dl_tst.dataset.input_dim))).astype(np.float32))

    def loader_for(sigma, dl):
        return dl if sigma == CLEAN else noisy_loader(dl, sigma * base_noise)

    out = {}
    for sigma in levels:
        # Same random subsample masks / MC samples in every pass.
        torch.manual_seed(abs(int(args.seed)))
        _stats, raw_scores, raw_labels = compute_eval_scores(
            args, loader_for(sigma, dl_tst), modules, elbo_loss, desired_t, device,
            normalization_stats=normalization_stats,
        )
        index, scores = scalar_scores(
            raw_scores, covered, smoothing_window=args.score_smoothing_window,
            aggregation=args.score_aggregation, feature_weights=feature_weights,
        )
        labels = np.asarray(raw_labels)
        out[sigma] = (index, scores, labels[index])

    if window_fn is None:
        return out, {}

    series, spans = test_series(dl_tst.dataset, length)
    snap = {
        "series": series,
        "noise": base_noise.numpy(),
        "covered": covered,
        "labels": labels[:length],  # labels do not depend on the noise
        "feature_names": getattr(getattr(dl_tst.dataset, "base_dataset", None), "feature_names", None),
        "nll": {},
    }
    snap["window"] = window_fn(snap)
    if snap["window"] is None:
        return out, snap

    _onset, window = snap["window"]
    positions = np.flatnonzero((spans[:, 1] >= window.start) & (spans[:, 0] < window.stop))
    dl_snap = DataLoader(WindowSubset(dl_tst.dataset, positions), batch_size=dl_tst.batch_size, shuffle=False,
                         num_workers=dl_tst.num_workers, pin_memory=dl_tst.pin_memory,
                         collate_fn=dl_tst.collate_fn)
    for sigma in levels:
        torch.manual_seed(abs(int(args.seed)))
        _stats, raw_nll, _labels = compute_eval_scores(
            args, loader_for(sigma, dl_snap), modules, elbo_loss, desired_t, device, normalization_stats=None,
        )
        nll = np.full(length, np.nan)
        nll[window] = raw_nll[window].sum(axis=1)
        snap["nll"][sigma] = nll
    return out, snap


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def level_title(sigma: float) -> str:
    return "Clean" if sigma == CLEAN else f"Noise $\\sigma$ = {sigma:g}"


# Shade per noise level: the clean level gets the darkest tone of its label's
# colour map, the strongest noise the lightest.
LABEL_CMAPS = {0: "Blues", 1: "Reds"}


def level_shade(value: int, k: int, n_levels: int):
    frac = 0.0 if n_levels < 2 else k / (n_levels - 1)
    return plt.get_cmap(LABEL_CMAPS[value])(0.95 - 0.5 * frac)


def _split(scores: np.ndarray, labels: np.ndarray, value: int) -> np.ndarray:
    return scores[(labels > 0) == bool(value)]


def _group_counts(per_level: dict, levels: list[float]) -> str:
    """"benign n=..., anomalous n=..." of a trace (identical for every level)."""
    _index, scores, labels = per_level[levels[0]]
    return ", ".join(f"{name.lower()} n={_split(scores, labels, value).size:,}" for name, value, _ in GROUPS)


def _style_axis(ax) -> None:
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def draw_trace_boxes(ax, per_level: dict, levels: list[float], cli_args) -> list:
    """One trace: per noise level (top to bottom) a benign and an anomalous thin horizontal box.

    Returns legend handles (one per label).
    """
    whis = tuple(cli_args.whisker_range) if cli_args.whisker_range else 1.5
    n = len(levels)
    centers = np.arange(n)[::-1].astype(float)  # first level on top
    offset, width = 0.19, 0.3
    for center, sigma in zip(centers, levels):
        _index, scores, labels = per_level[sigma]
        for (name, value, color), pos in zip(GROUPS, (center + offset, center - offset)):
            group = _split(scores, labels, value)
            if group.size == 0:
                continue
            ax.boxplot(
                group, positions=[pos], vert=False, widths=width, whis=whis, showfliers=cli_args.show_fliers,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=0.9),
                medianprops=dict(color=color, linewidth=1.6),
                whiskerprops=dict(color=color, linewidth=0.9), capprops=dict(color=color, linewidth=0.9),
                flierprops=dict(marker="o", markersize=1.2, markerfacecolor=color, markeredgecolor="none",
                                alpha=0.25),
            )
    # boxplot() resets the y ticks to its positions, so label them afterwards.
    ax.set_yticks(centers, [level_title(sigma) for sigma in levels], fontsize=7)
    ax.set_ylim(-0.6, n - 0.4)
    ax.tick_params(axis="y", length=0)
    _style_axis(ax)
    return [matplotlib.patches.Patch(facecolor=color, alpha=0.35, edgecolor=color, label=name)
            for name, _value, color in GROUPS]


def draw_trace_density(ax, per_level: dict, levels: list[float], cli_args) -> list:
    """One trace: a KDE (area 1) per label and noise level, all on one axis.

    Colour = label, shade = noise level (darkest: clean). Returns legend
    handles, interleaved benign/anomalous per level so that a legend with one
    column per level reads benign in its first row and anomalous in its second.
    """
    log_x = cli_args.log_x
    if log_x and min(np.nanmin(per_level[s][1]) for s in levels) <= 0:
        logging.warning("Scores <= 0 present; drawing the KDE on a linear scale despite --log-x.")
        log_x = False
    handles = []
    for k, sigma in enumerate(levels):
        _index, scores, labels = per_level[sigma]
        for name, value, _color in GROUPS:
            color = level_shade(value, k, len(levels))
            label = f"{name}, clean" if sigma == CLEAN else f"{name}, $\\sigma$ = {sigma:g}"
            group = _split(scores, labels, value)
            curve = _kde_curve(group, log_x)
            linewidth = 1.5 if sigma == CLEAN else 1.0
            if curve is None:
                if group.size:  # degenerate: all values equal
                    handles.append(ax.axvline(group[0], color=color, linewidth=linewidth, label=label))
                continue
            x, density = curve
            ax.fill_between(x, density, color=color, alpha=0.06, linewidth=0)
            handles.append(ax.plot(x, density, color=color, linewidth=linewidth, label=label)[0])
    ax.set_ylim(bottom=0)
    ax.set_yticks([])
    _style_axis(ax)
    return handles


KDE_GRID_POINTS = 400
# Points per group the KDE is fitted on; larger groups are subsampled
# (deterministically), which leaves the curve unchanged to the eye but keeps
# many traces x levels x groups of a long test split fast.
KDE_MAX_POINTS = 20000


def _kde_curve(values: np.ndarray, log_x: bool) -> tuple[np.ndarray, np.ndarray] | None:
    """(x, density) of a Gaussian KDE (Scott's bandwidth) over the data range.

    With `log_x` the KDE is fitted on log(values) and the density is per unit
    of log score, so it looks smooth on the log axis it is drawn on.
    """
    from scipy.stats import gaussian_kde

    values = values[np.isfinite(values)]
    if log_x:
        values = np.log(values)
    if values.size < 2 or np.ptp(values) == 0:
        return None
    if values.size > KDE_MAX_POINTS:
        values = np.random.default_rng(0).choice(values, KDE_MAX_POINTS, replace=False)
    kde = gaussian_kde(values)
    pad = 3 * float(np.sqrt(kde.covariance[0, 0]))
    grid = np.linspace(values.min() - pad, values.max() + pad, KDE_GRID_POINTS)
    density = kde(grid)
    return (np.exp(grid) if log_x else grid), density


def plot_per_trace(results: dict, traces: list[str], levels: list[float], cli_args, stem: Path,
                   draw, row_height: float, legend_cols: int):
    """One axis per trace, stacked; `draw` puts every label x noise level of that trace on it."""
    n_rows = len(traces)
    fig, axes = plt.subplots(n_rows//2, 2, figsize=(14.0, row_height * n_rows//2 + 0.6),
                             sharex=cli_args.sharex, squeeze=False)
    axes = axes.flatten()
    handles = []
    for r, trace in enumerate(traces):
        ax = axes[r]
        handles = draw(ax, results[trace], levels, cli_args) or handles
        ax.set_ylabel(f"Trace {trace}", fontsize=8)
        ax.text(1.0, 1.0, _group_counts(results[trace], levels), transform=ax.transAxes,
                ha="right", va="bottom", fontsize=6, color="0.35")
        if cli_args.log_x:
            ax.set_xscale("log")
        if not cli_args.sharex:
            ax.tick_params(axis="x", labelbottom=True)
    axes[-2].set_xlabel("Anomaly score", fontsize=8)
    axes[-1].set_xlabel("Anomaly score", fontsize=8)
    fig.tight_layout()
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=legend_cols,
               frameon=False, fontsize=7)
    for fmt in cli_args.formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


NLL_COLOR = "black"
ANOMALY_SHADE = "#D62728"


def find_onsets(labels: np.ndarray, covered: np.ndarray) -> np.ndarray:
    """Timepoints where the label switches from benign to anomalous (both sides covered)."""
    is_covered = np.zeros(labels.size, dtype=bool)
    is_covered[covered[covered < labels.size]] = True
    anomalous = labels > 0
    return np.flatnonzero(anomalous[1:] & ~anomalous[:-1] & is_covered[1:] & is_covered[:-1]) + 1


def snapshot_window(snap: dict, cli_args, window_length: int) -> tuple[int, slice] | None:
    """(onset, time slice) of the snapshot of one trace, or None if its test split has no onset."""
    onsets = find_onsets(snap["labels"], snap["covered"])
    if onsets.size == 0:
        return None
    k = cli_args.onset if -onsets.size <= cli_args.onset < onsets.size else (0 if cli_args.onset >= 0 else -1)
    if k != cli_args.onset:
        logging.warning("Only %d onset(s) in this trace; showing onset %d instead of %d.",
                        onsets.size, k, cli_args.onset)
    onset = int(onsets[k])
    length = cli_args.snapshot_length or 2 * window_length
    start = max(0, onset - int(round(cli_args.snapshot_before * length)))
    stop = min(snap["labels"].size, start + length)
    return onset, slice(start, stop)


def draw_snapshot(ax, snap: dict, sigma: float, onset: int, window: slice, legend: bool):
    """Variates (with the level's noise) on `ax`, the NLL on a twin axis; x relative to the onset.

    Returns the twin axis.
    """
    t = np.arange(window.start, window.stop)
    x = t - onset
    values = snap["series"][window] + sigma * snap["noise"][window]
    n_var = values.shape[1]
    cmap = plt.get_cmap("tab10" if n_var <= 10 else "tab20" if n_var <= 20 else "viridis")
    names = snap["feature_names"] if snap["feature_names"] is not None and len(snap["feature_names"]) == n_var \
        else [f"x{j}" for j in range(n_var)]
    for j in range(n_var):
        ax.plot(x, values[:, j], color=cmap(j / max(n_var - 1, 1) if n_var > 20 else j), linewidth=0.6,
                alpha=0.85, label=names[j])

    # Shade every anomalous stretch of the window.
    anomalous = snap["labels"][window] > 0
    edges = np.flatnonzero(np.diff(np.r_[0, anomalous.astype(int), 0]))
    for a, b in zip(edges[::2], edges[1::2]):
        ax.axvspan(x[a] - 0.5, x[b - 1] + 0.5, color=ANOMALY_SHADE, alpha=0.1, linewidth=0)
    ax.axvline(0, color=ANOMALY_SHADE, linewidth=0.8, linestyle="--")

    twin = ax.twinx()
    twin.plot(x, snap["nll"][sigma][window], color=NLL_COLOR, linewidth=1.1, label="NLL")
    twin.tick_params(axis="y", labelsize=7, colors=NLL_COLOR)
    ax.tick_params(labelsize=7)
    ax.set_xlim(x[0], x[-1])
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        h2, l2 = twin.get_legend_handles_labels()
        if n_var <= 20:
            ax.legend(handles + h2, labels + l2, fontsize=6, ncol=min(n_var + 1, 6), frameon=False,
                      loc="lower left", bbox_to_anchor=(0, 1.02))
    return twin


def plot_snapshots(snaps: dict, windows: dict, traces: list[str], levels: list[float], cli_args, stem: Path):
    """Grid (rows: traces, columns: noise levels) plus one file per (trace, level).

    The x axis is the time relative to the onset, shared by every panel. The
    variate axis (left) is shared within a row, so the noise levels of a trace
    are on one scale; the NLL axis (right) is per panel, because the NLL grows
    by orders of magnitude with the noise.
    """
    traces = [t for t in traces if windows.get(t) is not None]
    if not traces:
        logging.warning("No trace has a benign->anomalous transition in its test split; no snapshots.")
        return
    n_rows, n_cols = len(traces), len(levels)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.6 * n_cols, 1.7 * n_rows + 0.6),
                             sharex=True, sharey="row", squeeze=False)
    for r, trace in enumerate(traces):
        onset, window = windows[trace]
        for c, sigma in enumerate(levels):
            ax = axes[r, c]
            twin = draw_snapshot(ax, snaps[trace], sigma, onset, window, legend=False)
            if r == 0:
                ax.set_title(level_title(sigma), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Trace {trace}\nvalue", fontsize=8)
            if c == n_cols - 1:
                twin.set_ylabel("NLL", fontsize=8, color=NLL_COLOR)
    for ax in axes[-1]:
        ax.set_xlabel("Time steps relative to anomaly onset", fontsize=8)
    fig.tight_layout()
    for fmt in cli_args.formats:
        fig.savefig(stem.parent / f"{stem.name}_snapshots.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)

    single_dir = stem.parent / f"{stem.name}_snapshots"
    single_dir.mkdir(exist_ok=True)
    for trace in traces:
        onset, window = windows[trace]
        for sigma in levels:
            fig, ax = plt.subplots(figsize=(6.5, 2.6))
            twin = draw_snapshot(ax, snaps[trace], sigma, onset, window, legend=True)
            ax.set_ylabel("Value", fontsize=8)
            twin.set_ylabel("NLL", fontsize=8, color=NLL_COLOR)
            ax.set_xlabel("Time steps relative to anomaly onset", fontsize=8)
            fig.tight_layout()
            for fmt in cli_args.formats:
                fig.savefig(single_dir / f"trace{trace}_sigma{sigma:g}.{fmt}", dpi=200, bbox_inches="tight")
            plt.close(fig)


def summary_rows(trace: str, sigma: float, scores: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for name, value, _ in GROUPS:
        group = scores[(labels > 0) == bool(value)]
        row = {"trace": trace, "sigma": sigma, "label": name, "n": int(group.size)}
        if group.size:
            q = np.quantile(group, [0.05, 0.25, 0.5, 0.75, 0.95])
            row.update(mean=float(group.mean()), q05=q[0], q25=q[1], median=q[2], q75=q[3], q95=q[4])
        rows.append(row)
    return rows


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    argv = sys.argv[1:]
    dataset, config_file = _bootstrap_dataset(argv)
    parser = build_parser(dataset, config_file)
    cli_args = parser.parse_args(argv)
    cli_args.trace_ids = _normalize_trace_ids(cli_args.trace_ids)
    explicit = _explicit_cli_dests(parser, argv)

    if bool(cli_args.checkpoints) == bool(cli_args.checkpoint_dir):
        parser.error("Provide the checkpoints via exactly one of --checkpoints or --checkpoint-dir.")
    if cli_args.score_aggregation not in AGG_STRATEGIES:
        parser.error(f"Unknown --score-aggregation {cli_args.score_aggregation!r}.")
    if any(sigma <= 0 for sigma in cli_args.noise_levels):
        parser.error("--noise-levels must be positive (the clean column is always included).")
    levels = [CLEAN] + sorted(dict.fromkeys(cli_args.noise_levels))
    if cli_args.checkpoints:
        specs = parse_checkpoint_specs(cli_args.checkpoints, cli_args.seeds, parser)
    else:
        paths = discover_checkpoints(Path(cli_args.checkpoint_dir), cli_args.checkpoint_pattern)
        seeds = cli_args.seeds
        if seeds is not None and len(seeds) not in (1, len(paths)):
            parser.error(f"--seeds must give one seed per checkpoint ({len(paths)}) or a single seed.")
        specs = [(path, None if seeds is None else seeds[0] if len(seeds) == 1 else seeds[i])
                 for i, path in enumerate(paths)]

    out_dir = Path(cli_args.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_up_logging(
        console_log_level=cli_args.loglevel, console_log_color=True, logfile_file=None,
        logfile_log_level=cli_args.loglevel, logfile_log_color=False,
        log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s",
    )

    candidates, _failures = collect_candidates(specs, cli_args, explicit)
    if not candidates:
        raise SystemExit(f"None of the {len(specs)} checkpoint(s) matched {cli_args.dataset}; see log above.")
    seed, selected = select_one_seed(candidates, cli_args.select_seed)
    traces = [cand["trace_key"] for cand in selected]
    logging.info("Seed %d: %d trace(s) %s, noise levels %s", seed, len(traces), traces, levels)

    results: dict[str, dict] = {}
    snaps, windows = {}, {}
    rows, arrays = [], {}
    for cand in selected:
        path, args = cand["path"], cand["args"]
        if args.seed > 0:
            set_seed(args.seed)
        if cli_args.regenerate_processed_data:
            delete_processed_data(args.dataset, data_dir=getattr(args, "data_dir", "data_dir"))
        provider = build_provider(args)
        try:
            ds_idx = _resolve_trace_index(provider, cand["trace_id"])
            logging.info("Scoring %s (trace=%s, seed=%d)", path.name, cand["trace_key"], args.seed)
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            window_fn = ((lambda sn, wl=int(args.data_window_length): snapshot_window(sn, cli_args, wl))
                         if cli_args.snapshots else None)
            per_level, snap = score_trace(args, ckpt, provider, ds_idx, levels, window_fn=window_fn)
            del ckpt
        finally:
            if hasattr(provider, "cleanup"):
                provider.cleanup()

        trace = cand["trace_key"]
        results[trace] = per_level
        if cli_args.snapshots:
            snaps[trace] = snap
            windows[trace] = snap["window"]
            if windows[trace] is None:
                logging.warning("Trace %s: no benign->anomalous transition in the test split; no snapshot.", trace)
            else:
                onset, window = windows[trace]
                logging.info("Trace %s: snapshot at onset t=%d (t=%d..%d)", trace, onset, window.start,
                             window.stop - 1)
                arrays[f"trace{trace}__snapshot_t"] = np.arange(window.start, window.stop)
                for sigma in levels:
                    arrays[f"trace{trace}__sigma{sigma:g}__nll"] = snap["nll"][sigma][window]
        for sigma, (index, scores, labels) in per_level.items():
            rows += summary_rows(trace, sigma, scores, labels)
            key = f"trace{trace}__sigma{sigma:g}"
            arrays.update({f"{key}__index": index, f"{key}__scores": scores, f"{key}__labels": labels})
        logging.info("Trace %s:\n%s", trace, pd.DataFrame([r for r in rows if r["trace"] == trace])
                     .drop(columns="trace").to_string(index=False, float_format=lambda v: f"{v:.4g}"))

    run_id = datetime.datetime.now().strftime("%y%m%d-%H%M%S")
    stem = out_dir / f"SCOREDIST_{cli_args.dataset}_seed{seed}_{run_id}"
    np.savez_compressed(stem.with_suffix(".npz"), **arrays)
    pd.DataFrame(rows).to_csv(stem.with_suffix(".csv"), index=False)
    plot_per_trace(results, traces, levels, cli_args, stem, draw=draw_trace_boxes,
                   row_height=0.42 * len(levels) + 0.45, legend_cols=len(GROUPS))
    density_stem = stem.parent / f"{stem.name}_density"
    plot_per_trace(results, traces, levels, cli_args, density_stem, draw=draw_trace_density,
                   row_height=1.4, legend_cols=len(levels))
    logging.info("Wrote %s.{%s,npz,csv} and %s.{%s}", stem, ",".join(cli_args.formats),
                 density_stem, ",".join(cli_args.formats))
    if cli_args.snapshots:
        plot_snapshots(snaps, windows, traces, levels, cli_args, stem)
        logging.info("Wrote %s_snapshots.{%s} and %s_snapshots/", stem, ",".join(cli_args.formats), stem)


if __name__ == "__main__":
    main()
