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
def score_trace(args, ckpt: dict, provider, ds_idx: int, levels: list[float]) -> dict[float, tuple]:
    """``{sigma: (global time index, scalar score, label)}`` on the test split, per noise level."""
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
    rng = np.random.default_rng(abs(int(args.seed)))
    base_noise = torch.from_numpy(
        rng.standard_normal((int(covered.max()) + 1, int(dl_tst.dataset.input_dim))).astype(np.float32))

    out = {}
    for sigma in levels:
        loader = dl_tst if sigma == CLEAN else noisy_loader(dl_tst, sigma * base_noise)
        # Same random subsample masks / MC samples in every pass.
        torch.manual_seed(abs(int(args.seed)))
        _stats, raw_scores, raw_labels = compute_eval_scores(
            args, loader, modules, elbo_loss, desired_t, device, normalization_stats=normalization_stats,
        )
        index, scores = scalar_scores(
            raw_scores, covered, smoothing_window=args.score_smoothing_window,
            aggregation=args.score_aggregation, feature_weights=feature_weights,
        )
        out[sigma] = (index, scores, np.asarray(raw_labels)[index])
    return out


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
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    for fmt in cli_args.formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
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
            per_level = score_trace(args, ckpt, provider, ds_idx, levels)
            del ckpt
        finally:
            if hasattr(provider, "cleanup"):
                provider.cleanup()

        trace = cand["trace_key"]
        results[trace] = per_level
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


if __name__ == "__main__":
    main()
