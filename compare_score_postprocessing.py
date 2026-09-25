"""Compare the four score post-processing variants on stored checkpoints.

A sibling of ``evaluate_checkpoints.py``: same checkpoint discovery, same
provider/model restoration, same evaluation pipeline -- but instead of scoring
each checkpoint once with the settings it was trained under, every checkpoint
is scored under all four combinations of the two post-processing steps:

============================  =========  =============================
variant                       smoothing  aggregation
============================  =========  =============================
``smoothing+weighting``       yes        ``--weighted-aggregation``
``smoothing-only``            yes        ``max``
``weighting-only``            no         ``--weighted-aggregation``
``none``                      no         ``max``
============================  =========  =============================

"smoothing" is the moving average over the per-feature anomaly scores, with a
half-width of ``--smoothing-half-window`` (default 5, i.e. the historical
``--score-smoothing-window 5``); disabled means half-width 0. "weighting" is
the soft-voting aggregation that weights each feature by the inverse of its
training reconstruction MSE (``--score-aggregation weighted-mse``), as opposed
to the plain per-timepoint ``max`` across features.

Both steps are pure post-processing of the per-timepoint/per-feature scores, so
the expensive part -- the forward pass over the test set -- runs **once** per
checkpoint and feeds all four variants (see
``anomaly_detection.compute_eval_scores``). The four variants therefore see bit
-identical model outputs, and any metric difference between them is attributable
to the post-processing alone. Note that ``--normalize-score`` is applied inside
that shared pass and is *not* part of the grid: it stays at whatever the
checkpoint/CLI says and is held constant across the four variants.

Only test-set metrics are compared. The validation pass computes no score
metrics in this codebase (``evaluate(..., test=False)`` returns ELBO terms
only), so it cannot distinguish the variants and is skipped entirely.

Example::

    python compare_score_postprocessing.py --checkpoint-dir checkpoints \
        --dataset QAD --one-per-trace \
        --final-metrics-csv logs/eval/qad_postprocessing.csv
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch

from anomaly_detection import (
    _load_dataset_config,
    _normalize_trace_ids,
    _validate_config_keys,
    build_provider,
    calculate_feature_reconstruction_weights,
    calculate_z_normalization_values,
    compute_eval_scores,
    eval_scores,
    extend_argparse,
)
from evaluate_checkpoints import (
    DATA_ARG_KEYS,
    _bootstrap_dataset,
    _explicit_cli_dests,
    _make_loaders,
    _mean_of_stats,
    _resolve_trace_index,
    _restore_modules,
    _select_device,
    _to_python,
    _trace_from_checkpoint,
    discover_checkpoints,
    resolve_effective_args,
)
from utils.logger import set_up_logging
from utils.misc import append_final_metrics_csv, set_seed
from utils.parser import generic_parser

# Aggregation used whenever MSE feature weighting is switched *off*; this is
# anomaly_detection.py's own default (per-timepoint max across features).
UNWEIGHTED_AGGREGATION = "max"

# Reported in this order, baseline last, so the log table reads as "what does
# each step add on top of nothing".
VARIANT_ORDER = ("smoothing+weighting", "smoothing-only", "weighting-only", "none")
BASELINE_VARIANT = "none"

# Columns of the comparison tables, in the order metrics are worth reading.
METRIC_COLS = ("f1", "precision", "recall", "auc_roc", "auc_pr", "vus_roc", "vus_pr")

# These two are what the grid varies, so taking them from the checkpoint or the
# command line would be meaningless here.
GRID_CONTROLLED_ARGS = ("score_aggregation", "score_smoothing_window")


# --------------------------------------------------------------------------- #
# Variant grid
# --------------------------------------------------------------------------- #
def build_variants(half_window: int, weighted_aggregation: str) -> list[dict]:
    """The 2x2 grid: score smoothing on/off  x  MSE feature weighting on/off."""
    grid = {
        "smoothing+weighting": (True, True),
        "smoothing-only": (True, False),
        "weighting-only": (False, True),
        "none": (False, False),
    }
    return [
        {
            "name": name,
            "smoothing": smoothing,
            "weighting": weighting,
            "score_smoothing_window": half_window if smoothing else 0,
            "score_aggregation": weighted_aggregation if weighting else UNWEIGHTED_AGGREGATION,
        }
        for name, (smoothing, weighting) in ((n, grid[n]) for n in VARIANT_ORDER)
    ]


def variant_label(variant: dict) -> str:
    return (f"{variant['name']} (smooth_window={variant['score_smoothing_window']}, "
            f"aggregation={variant['score_aggregation']})")


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #
def build_parser(dataset: str, config_file: str | None) -> argparse.ArgumentParser:
    parser = extend_argparse(generic_parser)
    parser.description = (
        "Compare score smoothing and MSE feature weighting (2x2 grid) on stored checkpoints."
    )
    dataset_cfg = _load_dataset_config(dataset, config_file)
    _validate_config_keys(parser, dataset_cfg, dataset)
    parser.set_defaults(**dataset_cfg)
    parser.set_defaults(dataset=dataset, final_metrics_csv=None, loglevel="info")

    group = parser.add_argument_group("Post-processing comparison arguments")
    group.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="checkpoint_*_best.h5",
        help="Glob (relative to --checkpoint-dir) selecting the checkpoints to evaluate.",
    )
    group.add_argument(
        "--one-per-trace",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep only the most recent best checkpoint per trace instead of evaluating all of them.",
    )
    group.add_argument(
        "--eval-out-dir",
        type=str,
        default="logs/eval",
        help="Directory for the JSON report (per-checkpoint, per-trace and macro metrics).",
    )
    group.add_argument(
        "--smoothing-half-window",
        type=int,
        default=5,
        help=(
            "Half-width of the moving average used by the two smoothing variants "
            "(same meaning as --score-smoothing-window). The non-smoothing variants "
            "always use 0. Default 5."
        ),
    )
    group.add_argument(
        "--weighted-aggregation",
        choices=["weighted-mse", "weighted-mse-exp"],
        default="weighted-mse",
        help=(
            "Aggregation used by the two weighting variants. 'weighted-mse' (default) "
            "weights each feature by the inverse of its training reconstruction MSE; "
            "'weighted-mse-exp' uses exp(1/MSE) instead. The non-weighting variants "
            f"always use '{UNWEIGHTED_AGGREGATION}'."
        ),
    )
    return parser


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def compare_variants_for_checkpoint(args, ckpt: dict, provider, ds_idx: int, variants: list[dict]) -> dict:
    """Score one checkpoint under every variant, from a single forward pass.

    Returns ``{variant_name: stats}``. The ELBO terms in `stats` come from that
    shared pass and are therefore identical across variants; only the metrics
    derived from `eval_scores` differ.
    """
    device = args.device
    dl_trn, dl_tst, _, input_dim, num_timepoints = _make_loaders(provider, ds_idx, args)
    modules, elbo_loss, desired_t = _restore_modules(args, ckpt, input_dim, num_timepoints, device)

    normalization_scores = None
    if args.normalize_score:
        normalization_scores = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)

    # The one expensive pass. Everything below is post-processing of its output.
    base_stats, all_scores, all_labels = compute_eval_scores(
        args, dl_tst, modules, elbo_loss, desired_t, device,
        normalization_stats=normalization_scores,
    )

    # One training pass for the weights, shared by both weighting variants.
    feature_weights = None
    if any(variant["weighting"] for variant in variants):
        weighting = "exp-inverse" if args.weighted_aggregation == "weighted-mse-exp" else "inverse"
        weight_stats = calculate_feature_reconstruction_weights(
            args, dl_trn, modules, desired_t, device, weighting=weighting)
        feature_weights = weight_stats["feature_weights"]

    results = {}
    for variant in variants:
        logging.debug("  variant %s", variant_label(variant))
        metrics = eval_scores(
            all_scores, all_labels, window_length=args.data_window_length,
            aggregation_strategy=variant["score_aggregation"],
            feature_weights=feature_weights if variant["weighting"] else None,
            smoothing_window=variant["score_smoothing_window"],
        )
        stats = dict(base_stats)
        stats.update({key.lower(): value for key, value in metrics.items()})
        results[variant["name"]] = _to_python(stats)
    return results


def _metrics_frame(stats_by_variant: dict[str, dict]) -> pd.DataFrame:
    """Variants as rows (in VARIANT_ORDER), METRIC_COLS as columns."""
    frame = pd.DataFrame.from_dict(stats_by_variant, orient="index")
    frame = frame.reindex([name for name in VARIANT_ORDER if name in frame.index])
    frame = frame[[col for col in METRIC_COLS if col in frame.columns]]
    frame.index.name = "variant"
    return frame


def _delta_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Per-metric difference of every variant against the BASELINE_VARIANT row."""
    if BASELINE_VARIANT not in frame.index:
        return pd.DataFrame(index=frame.index, columns=frame.columns, dtype=float)
    return frame.subtract(frame.loc[BASELINE_VARIANT], axis=1)


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

    if cli_args.checkpoint_dir is None:
        parser.error("--checkpoint-dir is required")
    checkpoint_dir = Path(cli_args.checkpoint_dir)
    if not checkpoint_dir.is_dir():
        parser.error(f"--checkpoint-dir {checkpoint_dir} is not a directory")
    if cli_args.smoothing_half_window <= 0:
        parser.error("--smoothing-half-window must be > 0 (the no-smoothing variants use 0 by definition)")

    ignored_overrides = sorted(explicit & set(GRID_CONTROLLED_ARGS))
    explicit = explicit - set(GRID_CONTROLLED_ARGS)
    variants = build_variants(cli_args.smoothing_half_window, cli_args.weighted_aggregation)

    run_id = datetime.datetime.now().strftime("%y%m%d-%H:%M:%S")
    out_dir = Path(cli_args.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_up_logging(
        console_log_level=cli_args.loglevel,
        console_log_color=True,
        logfile_file=str(out_dir / f"POSTPROC_{cli_args.dataset}_{run_id}.txt"),
        logfile_log_level=cli_args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s",
    )

    for key in ignored_overrides:
        logging.warning(
            "Ignoring --%s: this script varies it across the four variants "
            "(see --smoothing-half-window / --weighted-aggregation).",
            key.replace("_", "-"),
        )

    # ---- discover + filter checkpoints -------------------------------------
    candidates = []
    for path in discover_checkpoints(checkpoint_dir, cli_args.checkpoint_pattern):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args")
        if ckpt_args is None:
            logging.warning("Skipping %s: no stored args.", path.name)
            continue
        if isinstance(ckpt_args, dict):
            ckpt_args = Namespace(**ckpt_args)
        ckpt_dataset = getattr(ckpt_args, "dataset", None)
        if ckpt_dataset != cli_args.dataset:
            logging.debug("Skipping %s: dataset %s != %s.", path.name, ckpt_dataset, cli_args.dataset)
            continue
        trace_id = _trace_from_checkpoint(ckpt, ckpt_args, path)
        if cli_args.trace_ids is not None and trace_id not in cli_args.trace_ids:
            logging.debug("Skipping %s: trace %s not in --trace-ids.", path.name, trace_id)
            continue
        candidates.append({"path": path, "ckpt": ckpt, "ckpt_args": ckpt_args, "trace_id": trace_id})

    if not candidates:
        raise SystemExit(f"No {cli_args.dataset} checkpoints to evaluate in {checkpoint_dir}.")

    if cli_args.one_per_trace:
        newest = {}
        for cand in candidates:  # candidates are sorted by mtime ascending
            newest[cand["trace_id"]] = cand
        candidates = list(newest.values())

    logging.info(
        "Comparing %d post-processing variant(s) on %d checkpoint(s) for %s from %s (traces=%s)",
        len(variants), len(candidates), cli_args.dataset, checkpoint_dir,
        sorted({c["trace_id"] for c in candidates}, key=lambda t: (len(str(t)), str(t))),
    )
    for variant in variants:
        logging.info("  variant: %s", variant_label(variant))

    # ---- evaluate -----------------------------------------------------------
    provider_cache: dict[tuple, object] = {}
    results = []
    failures = []
    for cand in candidates:
        path, ckpt, trace_id = cand["path"], cand["ckpt"], cand["trace_id"]
        args = resolve_effective_args(cand["ckpt_args"], cli_args, explicit)
        args.device = _select_device(args.device)
        args.trace_ids = [trace_id] if trace_id is not None else None
        # resolve_effective_args copies the checkpoint's stored args wholesale,
        # so pin the grid's own knobs afterwards in case a checkpoint ever
        # carries keys of these names.
        args.smoothing_half_window = cli_args.smoothing_half_window
        args.weighted_aggregation = cli_args.weighted_aggregation

        if args.seed > 0:
            set_seed(args.seed)

        cache_key = tuple(getattr(args, key, None) for key in DATA_ARG_KEYS) + (trace_id,)
        provider = provider_cache.get(cache_key)
        if provider is None:
            provider = build_provider(args)
            provider_cache[cache_key] = provider
        ds_idx = _resolve_trace_index(provider, trace_id)

        logging.info("Evaluating %s (trace=%s, idx=%d)", path.name, trace_id, ds_idx)
        logging.debug("Effective evaluation args: %s", vars(args))
        try:
            stats_by_variant = compare_variants_for_checkpoint(args, ckpt, provider, ds_idx, variants)
        except RuntimeError as exc:
            logging.error("Skipping %s: %s", path.name, exc)
            failures.append({"checkpoint": str(path), "trace_id": trace_id, "error": str(exc)})
            continue

        frame = _metrics_frame(stats_by_variant)
        logging.info(
            "[%s] %s:\n%s", trace_id, path.name,
            frame.to_string(float_format=lambda v: f"{v:.4f}"),
        )
        for variant in variants:
            results.append({
                "checkpoint": str(path),
                "trace_id": trace_id,
                "dataset": args.dataset,
                "variant": variant["name"],
                "smoothing": variant["smoothing"],
                "weighting": variant["weighting"],
                "score_smoothing_window": variant["score_smoothing_window"],
                "score_aggregation": variant["score_aggregation"],
                "normalize_score": args.normalize_score,
                "mc_eval_samples": args.mc_eval_samples,
                "tst": stats_by_variant[variant["name"]],
            })

    if not results:
        raise SystemExit(f"All {len(failures)} checkpoint(s) failed to evaluate; see log above.")

    # ---- aggregate ----------------------------------------------------------
    # per_trace[trace][variant] = mean over that trace's checkpoints
    by_trace_variant = defaultdict(list)
    for res in results:
        by_trace_variant[(str(res["trace_id"]), res["variant"])].append(res["tst"])
    per_trace: dict[str, dict[str, dict]] = defaultdict(dict)
    for (trace, variant), stats_list in by_trace_variant.items():
        per_trace[trace][variant] = {**_mean_of_stats(stats_list), "num_checkpoints": len(stats_list)}

    # macro[variant] = unweighted mean over traces
    macro = {
        variant["name"]: {
            f"macro_{key}": value
            for key, value in _mean_of_stats(
                [per_trace[trace][variant["name"]] for trace in per_trace
                 if variant["name"] in per_trace[trace]]
            ).items()
            if key != "num_checkpoints"
        }
        for variant in variants
    }

    macro_frame = _metrics_frame({
        name: {key[len("macro_"):]: value for key, value in stats.items()}
        for name, stats in macro.items()
    })
    delta_frame = _delta_frame(macro_frame)

    report = {
        "run_id": run_id,
        "dataset": cli_args.dataset,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_pattern": cli_args.checkpoint_pattern,
        "smoothing_half_window": cli_args.smoothing_half_window,
        "weighted_aggregation": cli_args.weighted_aggregation,
        "variants": variants,
        "cli_overrides": {key: getattr(cli_args, key) for key in sorted(explicit)},
        "per_checkpoint": results,
        "failed_checkpoints": failures,
        "per_trace": {trace: dict(stats) for trace, stats in per_trace.items()},
        "macro": macro,
        "macro_delta_vs_none": {
            name: {col: (None if pd.isna(val) else float(val)) for col, val in row.items()}
            for name, row in delta_frame.iterrows()
        },
    }
    report_path = out_dir / f"POSTPROC_{cli_args.dataset}_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    # ---- report -------------------------------------------------------------
    for trace in sorted(per_trace, key=lambda t: (len(t), t)):
        logging.info(
            "Trace %s -- test metrics per variant (mean over %d checkpoint(s)):\n%s",
            trace,
            max(stats.get("num_checkpoints", 0) for stats in per_trace[trace].values()),
            _metrics_frame(per_trace[trace]).to_string(float_format=lambda v: f"{v:.4f}"),
        )
    logging.info(
        "Macro test metrics over %d trace(s), per variant:\n%s",
        len(per_trace), macro_frame.to_string(float_format=lambda v: f"{v:.4f}"),
    )
    logging.info(
        "Macro delta vs. '%s' (positive = the post-processing helps):\n%s",
        BASELINE_VARIANT, delta_frame.to_string(float_format=lambda v: f"{v:+.4f}"),
    )
    if "f1" in macro_frame.columns and macro_frame["f1"].notna().any():
        best = macro_frame["f1"].idxmax()
        logging.info(
            "Best variant by macro F1: %s (%.4f, %+.4f vs. '%s')",
            best, macro_frame.loc[best, "f1"], delta_frame.loc[best, "f1"], BASELINE_VARIANT,
        )
    if failures:
        logging.warning(
            "%d checkpoint(s) could not be evaluated: %s",
            len(failures), [f["checkpoint"] for f in failures],
        )
    logging.info("Report written to %s", report_path)

    if cli_args.final_metrics_csv:
        for trace in per_trace:
            for variant_name, stats in per_trace[trace].items():
                append_final_metrics_csv(
                    csv_path=cli_args.final_metrics_csv,
                    benchmark=f"{cli_args.dataset}:{trace}:{variant_name}",
                    run_datetime=run_id,
                    metrics=stats,
                )
        for variant_name, stats in macro.items():
            append_final_metrics_csv(
                csv_path=cli_args.final_metrics_csv,
                benchmark=f"{cli_args.dataset}:macro:{variant_name}",
                run_datetime=run_id,
                metrics={**stats, "num_traces": len(per_trace)},
            )
        logging.info("Appended per-trace and macro rows to %s", cli_args.final_metrics_csv)


if __name__ == "__main__":
    main()