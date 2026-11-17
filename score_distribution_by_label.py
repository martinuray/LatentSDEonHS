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

Figure: rows = traces, columns = noise levels (clean first). Each cell shows
two thin horizontal box plots, benign and anomalous test timepoints, on one
score axis. ``--sharex`` decides which cells share that axis: ``row`` (the
default) makes the noise levels of one trace directly comparable. Scores of
different traces come from different models and need not share a scale.

Outputs (in ``--eval-out-dir``):
  * ``SCOREDIST_<dataset>_seed<seed>_<run>.{pdf,png}``: the grid,
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
    The NLL series also go into the .npz (``...__nll``).

Examples::

    # seed 42, traces 1-10, noise levels 0.1 0.5 1.0 (+ clean)
    python score_distribution_by_label.py --dataset QAD \\
        --checkpoint-dir checkpoints --checkpoint-pattern 'checkpoint_AD_QAD_*_best.h5' \\
        --select-seed 42

    # explicit checkpoints, other noise levels, one score axis per column
    python score_distribution_by_label.py --dataset QAD --checkpoints ckpt_trace-*_best.h5 \\
        --noise-levels 0.05 0.2 0.4 --sharex col
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
DEFAULT_NOISE_LEVELS = [0.1, 0.5, 1.0]
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
        "--sharex", choices=["row", "col", "all", "none"], default="row",
        help="Which cells share the score axis. Default: row (one trace across noise levels).",
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
def test_series(dataset: Dataset, length: int) -> np.ndarray:
    """The clean test series [time, feature] (``evd_obs``, NaN where no window covers a timepoint).

    Rebuilt from the windows by their global time indices; overlapping windows
    carry the same values, so the order of writing does not matter.
    """
    series = np.full((length, int(dataset.input_dim)), np.nan)
    for i in range(len(dataset)):
        item = dataset[i]
        series[np.asarray(item["inp_indcs"]).astype(int)] = np.asarray(item["evd_obs"], dtype=float)
    return series


def score_trace(args, ckpt: dict, provider, ds_idx: int, levels: list[float]) -> tuple[dict[float, tuple], dict]:
    """Score the test split once per noise level.

    Returns ``({sigma: (global time index, scalar score, label)}, snapshot data)``.
    The snapshot data holds what the onset plots need: the clean series, the
    noise pattern, the labels and covered timepoints, and per level the
    per-timepoint NLL (summed over the variates, NaN where not covered).
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
    snap = {
        "series": test_series(dl_tst.dataset, length),
        "noise": base_noise.numpy(),
        "covered": covered,
        "feature_names": getattr(getattr(dl_tst.dataset, "base_dataset", None), "feature_names", None),
        "nll": {},
    }

    out = {}
    for sigma in levels:
        loader = dl_tst if sigma == CLEAN else noisy_loader(dl_tst, sigma * base_noise)
        # Same random subsample masks / MC samples in every pass.
        torch.manual_seed(abs(int(args.seed)))
        # The pass runs without score normalisation so the raw NLL is kept for
        # the snapshots. compute_eval_scores would apply the normalisation per
        # window before averaging the overlapping windows; it is affine per
        # variate, so applying it to the averaged scores here gives the same result.
        _stats, raw_nll, raw_labels = compute_eval_scores(
            args, loader, modules, elbo_loss, desired_t, device, normalization_stats=None,
        )
        per_feature = raw_nll
        if normalization_stats is not None:
            lo = normalization_stats["min"].detach().cpu().numpy()
            span = (normalization_stats["max"] - normalization_stats["min"]).clamp_min(1e-8).detach().cpu().numpy()
            per_feature = (raw_nll - lo) / span
        index, scores = scalar_scores(
            per_feature, covered, smoothing_window=args.score_smoothing_window,
            aggregation=args.score_aggregation, feature_weights=feature_weights,
        )
        labels = np.asarray(raw_labels)
        out[sigma] = (index, scores, labels[index])
        nll = np.full(length, np.nan)
        nll[covered] = raw_nll[covered].sum(axis=1)
        snap["nll"][sigma] = nll
        snap["labels"] = labels[:length]
    return out, snap


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def level_title(sigma: float) -> str:
    return "Clean" if sigma == CLEAN else f"Noise $\\sigma$ = {sigma:g}"


def draw_cell(ax, scores: np.ndarray, labels: np.ndarray, cli_args, first_col: bool) -> None:
    """Benign and anomalous scores as two thin horizontal boxes on one axis."""
    whis = tuple(cli_args.whisker_range) if cli_args.whisker_range else 1.5
    positions = np.arange(len(GROUPS))[::-1]  # first group on top
    ticklabels = []
    for pos, (name, value, color) in zip(positions, GROUPS):
        group = scores[(labels > 0) == bool(value)]
        ticklabels.append(f"{name} (n={group.size:,})")
        if group.size == 0:
            continue
        ax.boxplot(
            group, positions=[pos], vert=False, widths=0.4, whis=whis, showfliers=cli_args.show_fliers,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=0.9),
            medianprops=dict(color=color, linewidth=1.6),
            whiskerprops=dict(color=color, linewidth=0.9), capprops=dict(color=color, linewidth=0.9),
            flierprops=dict(marker="o", markersize=1.2, markerfacecolor=color, markeredgecolor="none",
                            alpha=0.25),
        )
    # boxplot() resets the y ticks to its positions, so label them afterwards.
    ax.set_yticks(positions, ticklabels if first_col else [""] * len(GROUPS), fontsize=7)
    ax.set_ylim(-0.6, len(GROUPS) - 0.4)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)


def plot_grid(results: dict, traces: list[str], levels: list[float], title: str, cli_args, stem: Path):
    n_rows, n_cols = len(traces), len(levels)
    sharex = False if cli_args.sharex == "none" else cli_args.sharex
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.9 * n_cols + 1.4, 0.85 * n_rows + 0.8),
                             sharex=sharex, squeeze=False)
    for r, trace in enumerate(traces):
        for c, sigma in enumerate(levels):
            ax = axes[r, c]
            _index, scores, labels = results[trace][sigma]
            draw_cell(ax, scores, labels, cli_args, first_col=c == 0)
            if r == 0:
                ax.set_title(level_title(sigma), fontsize=9)
            if c == 0:
                ax.set_ylabel(f"Trace {trace}", fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
            if cli_args.log_x:
                ax.set_xscale("log")
            # With a shared x axis matplotlib hides the inner tick labels; keep
            # them where the axis is not shared with the cell below.
            if sharex in ("row", False) or r == n_rows - 1:
                ax.tick_params(axis="x", labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel("Anomaly score", fontsize=8)
    #fig.suptitle(title, fontsize=10)
    fig.tight_layout()
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
            per_level, snap = score_trace(args, ckpt, provider, ds_idx, levels)
            del ckpt
        finally:
            if hasattr(provider, "cleanup"):
                provider.cleanup()

        trace = cand["trace_key"]
        results[trace] = per_level
        if cli_args.snapshots:
            snaps[trace] = snap
            windows[trace] = snapshot_window(snap, cli_args, int(args.data_window_length))
            if windows[trace] is None:
                logging.warning("Trace %s: no benign->anomalous transition in the test split; no snapshot.", trace)
            else:
                onset, window = windows[trace]
                logging.info("Trace %s: snapshot at onset t=%d (t=%d..%d)", trace, onset, window.start,
                             window.stop - 1)
                for sigma in levels:
                    arrays[f"trace{trace}__sigma{sigma:g}__nll"] = snap["nll"][sigma]
        else:
            del snap
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
    first = selected[0]["args"]
    title = (f"{cli_args.dataset}, seed {seed}: test scores by label "
             f"({first.score_aggregation}, smoothing {first.score_smoothing_window})")
    plot_grid(results, traces, levels, title, cli_args, stem)
    logging.info("Wrote %s.{%s,npz,csv}", stem, ",".join(cli_args.formats))
    if cli_args.snapshots:
        plot_snapshots(snaps, windows, traces, levels, cli_args, stem)
        logging.info("Wrote %s_snapshots.{%s} and %s_snapshots/", stem, ",".join(cli_args.formats), stem)


if __name__ == "__main__":
    main()
