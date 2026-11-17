#!/usr/bin/env python3
"""Test-set anomaly scores of a checkpoint, split by ground-truth label.

A sibling of ``conformal_anomaly_detection.py``, using the same checkpoint
handling: the model and data provider are restored from the args stored in the
checkpoint. The trace id and the seed come from the checkpoint too, unless a
seed is given explicitly (``--seeds`` / ``PATH=SEED``). The test split is
scored with the same post-processing (smoothing, feature aggregation,
z-normalisation) as ``anomaly_detection.py``, so each timepoint gets one scalar
anomaly score.

The scores are then split by the test labels into true anomalies and true
benign timepoints. Each group is drawn as its own thin horizontal box plot,
stacked on a shared score axis so the two can be compared directly.

Outputs (in ``--eval-out-dir``, one set per checkpoint):
  * ``SCOREDIST_<dataset>__<ckpt>__seed<seed>.{pdf,png}``: the figure,
  * ``SCOREDIST_<dataset>__<ckpt>__seed<seed>.npz``: test index, scores and labels,
  * a summary of each group's quantiles in the log.

The seed caveat of ``conformal_anomaly_detection.py`` applies here as well:
pass ``--regenerate-processed-data`` to redraw the processed data under the
checkpoint's seed. For the test split this rarely matters, because the
providers draw only the train/val split at random.

Examples::

    # seed taken from the checkpoint
    python score_distribution_by_label.py --dataset PSM \\
        --checkpoints checkpoints/checkpoint_AD_PSM_250901-10:00:00_best.h5

    # every QAD checkpoint in a directory, log-scaled score axis
    python score_distribution_by_label.py --dataset QAD \\
        --checkpoint-dir checkpoints --checkpoint-pattern 'checkpoint_AD_QAD_*_best.h5' --log-x
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

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

GROUPS = (("Benign", 0, "#1F77B4"), ("Anomalous", 1, "#D62728"))


def build_parser(dataset: str, config_file: str | None) -> argparse.ArgumentParser:
    # Reuse the conformal parser for the checkpoint/seed/data arguments; its
    # alpha/p-value options are accepted but play no role here.
    parser = build_conformal_parser(dataset, config_file)
    parser.description = "Score the test split of checkpoints and plot the scores split by label."
    parser.set_defaults(eval_out_dir="logs/score_distribution")
    group = parser.add_argument_group("Score distribution plot")
    group.add_argument("--log-x", action="store_true", help="Log-scale the shared score axis.")
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


def test_scores(args, ckpt: dict, provider, ds_idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(global time index, scalar score, label) of every scored test timepoint."""
    device = args.device
    dl_trn, dl_tst, _dl_val, input_dim, num_timepoints = _make_loaders(provider, ds_idx, args)
    modules, elbo_loss, desired_t = _restore_modules(args, ckpt, input_dim, num_timepoints, device)

    normalization_stats = None
    if args.normalize_score:
        normalization_stats = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)
    feature_weights = None
    if args.score_aggregation in ("weighted-mse", "weighted-mse-exp"):
        weighting = "exp-inverse" if args.score_aggregation == "weighted-mse-exp" else "inverse"
        feature_weights = calculate_feature_reconstruction_weights(
            args, dl_trn, modules, desired_t, device, weighting=weighting)["feature_weights"]

    _stats, raw_scores, raw_labels = compute_eval_scores(
        args, dl_tst, modules, elbo_loss, desired_t, device, normalization_stats=normalization_stats,
    )
    index, scores = scalar_scores(
        raw_scores, covered_timepoints(dl_tst), smoothing_window=args.score_smoothing_window,
        aggregation=args.score_aggregation, feature_weights=feature_weights,
    )
    return index, scores, np.asarray(raw_labels)[index]


def plot_by_label(scores: np.ndarray, labels: np.ndarray, title: str, cli_args, stem: Path) -> None:
    """One thin horizontal box plot per label group, stacked on a shared score axis."""
    fig, axes = plt.subplots(len(GROUPS), 1, figsize=(6.5, 0.75 * len(GROUPS) + 0.7), sharex=True)
    whis = tuple(cli_args.whisker_range) if cli_args.whisker_range else 1.5
    for ax, (name, value, color) in zip(axes, GROUPS):
        group = scores[(labels > 0) == bool(value)]
        ax.set_yticks([0], [f"{name}\n(n={group.size:,})"])
        ax.set_ylim(-0.5, 0.5)
        ax.grid(axis="x", alpha=0.3)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
        if group.size == 0:
            ax.text(0.5, 0.5, "no timepoints", transform=ax.transAxes, ha="center", va="center", color="0.4")
            continue
        ax.boxplot(
            group, positions=[0], vert=False, widths=0.35, whis=whis, showfliers=cli_args.show_fliers,
            patch_artist=True,
            boxprops=dict(facecolor=color, alpha=0.35, edgecolor=color, linewidth=1),
            medianprops=dict(color=color, linewidth=1.8),
            whiskerprops=dict(color=color, linewidth=1), capprops=dict(color=color, linewidth=1),
            flierprops=dict(marker="o", markersize=1.5, markerfacecolor=color, markeredgecolor="none",
                            alpha=0.25),
        )
        # boxplot() resets the y ticks to its positions, so label them afterwards.
        ax.set_yticks([0], [f"{name}\n(n={group.size:,})"])
        ax.set_ylim(-0.5, 0.5)
    if cli_args.log_x:
        axes[-1].set_xscale("log")
    axes[-1].set_xlabel("Anomaly score")
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    for fmt in cli_args.formats:
        fig.savefig(stem.with_suffix(f".{fmt}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def log_summary(scores: np.ndarray, labels: np.ndarray) -> None:
    for name, value, _ in GROUPS:
        group = scores[(labels > 0) == bool(value)]
        if group.size == 0:
            logging.info("  %-9s n=0", name)
            continue
        q = np.quantile(group, [0.05, 0.25, 0.5, 0.75, 0.95])
        logging.info("  %-9s n=%-8d mean=%.4g  q05=%.4g q25=%.4g median=%.4g q75=%.4g q95=%.4g",
                     name, group.size, group.mean(), *q)


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

    for cand in candidates:
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
            index, scores, labels = test_scores(args, ckpt, provider, ds_idx)
            del ckpt
        finally:
            if hasattr(provider, "cleanup"):
                provider.cleanup()

        logging.info("%d test timepoints, %.2f%% anomalous:", scores.size, 100 * np.mean(labels > 0))
        log_summary(scores, labels)

        stem = out_dir / f"SCOREDIST_{args.dataset}__{path.stem}__seed{args.seed}"
        np.savez_compressed(stem.with_suffix(".npz"), test_index=index, test_scores=scores, test_labels=labels)
        trace = "" if cand["trace_id"] is None else f", trace {cand['trace_key']}"
        title = (f"{args.dataset}{trace}, seed {args.seed} -- test scores "
                 f"({args.score_aggregation}, smoothing {args.score_smoothing_window})")
        plot_by_label(scores, labels, title, cli_args, stem)
        logging.info("Wrote %s.{%s,npz}", stem, ",".join(cli_args.formats))


if __name__ == "__main__":
    main()
