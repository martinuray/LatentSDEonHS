"""Turn per-timepoint anomaly scores into calibrated anomaly probabilities.

A sibling of ``evaluate_checkpoints.py`` / ``compare_score_postprocessing.py``:
same checkpoint handling, same provider/model restoration, same score
post-processing -- but instead of reporting threshold-free ranking metrics it
calibrates every score against the nominal (anomaly-free) validation split and
reports, for each single test timepoint, the probability that it is anomalous.

Given a list of ``(checkpoint, seed)`` pairs, each pair is processed on its own:

1. restore the model and the data provider from the checkpoint's stored args
   (the seed decides the train/val split permutation and the subsampling masks,
   hence it has to be the seed the checkpoint was trained with -- it is taken
   from the checkpoint unless given explicitly),
2. score the **validation** split and keep the resulting per-timepoint scores as
   the *calibration set* (also called the conformity/compliance scores). The
   validation split is a held-out slice of the nominal training data in every
   provider of this code base, so these are scores the model assigns to data it
   considers normal,
3. score the **test** split with the identical pipeline, and
4. convert each test score ``s`` into a conformal p-value

       p(s) = (1 + #{c in calibration : c >= s}) / (n_calibration + 1)

   i.e. the fraction of nominal scores that are at least as extreme as ``s``.
   The reported anomaly probability is ``1 - p(s)``.

``p`` is a proper (marginally valid) p-value for the null hypothesis "this
timepoint is nominal": if test timepoints are exchangeable with the calibration
timepoints, ``P(p <= alpha) <= alpha``. Flagging ``p <= alpha`` therefore
controls the per-timepoint false-positive rate at ``alpha`` *without* tuning a
score threshold on test labels, which is what the ``best_f1`` style metrics of
the other scripts do. How well that holds in practice is reported as
``empirical_fpr`` (the observed rate of ``p <= alpha`` among the test timepoints
labelled nominal) next to the nominal ``alpha``; a large gap means the
validation split is not exchangeable with the nominal part of the test split
(distribution drift between the train-time and test-time segments of the
series), not that the arithmetic is off.

Because ``p`` is a monotone transform of the score, the ranking metrics
(AUC-ROC/PR, VUS, best F1) are those of the underlying score -- they are
reported for comparability with ``evaluate_checkpoints.py``. The metrics that
are new here are the ones at fixed levels ``--alphas``: no threshold search, no
test labels involved in the decision.

Multi-trace benchmarks (QAD, TSB-AD-M, SMD, ...) are trained one model per
trace, so the natural input here is one checkpoint per (trace, seed). Each
checkpoint carries its trace id and its seed, hence the whole grid can simply
be handed over at once (or discovered with ``--checkpoint-dir``); the run is
then *ordered by trace*, and each trace is calibrated against its own
validation split -- calibration scores are never pooled across traces, which
would compare scores of models that saw different data. Reported are

* per checkpoint: calibration size, test size, metrics per alpha,
* per trace: mean/std over that trace's seeds, plus a seed-ensemble that
  combines the per-seed p-values of the same timepoint with the Vovk-Wang
  average ``p = min(1, 2 * mean_i p_i)`` (valid under arbitrary dependence
  between the seeds),
* macro: unweighted mean over traces, as in ``evaluate_checkpoints.py``,
* pooled: all traces evaluated together per seed, then averaged over seeds.
  Pooling raw scores across traces is meaningless, but p-values are already
  per-trace calibrated and so share one scale -- a single alpha therefore
  applies benchmark-wide (``--no-pool-traces`` to skip).

One caveat about the seed: the providers draw the train/val permutation while
*processing* the raw data and then cache the result under
``<data-dir>/<dataset>/processed``. A seed therefore only reproduces the split
it produced during training if the processed cache is regenerated under it,
which is what ``anomaly_detection.py`` does (it deletes the processed data
before every run) and what ``--regenerate-processed-data`` does here. Without
that flag all seeds share whatever split currently sits in the cache -- fine
for a quick look, and it keeps the calibration sets comparable across seeds,
but then the calibration set is not necessarily the one the checkpoint never
saw during training.

Examples::

    # two seeds of one trace, seeds given explicitly
    python conformal_anomaly_detection.py --dataset QAD \\
        --checkpoints checkpoints/checkpoint_AD_QAD_250901-10:00:00_trace-1_best.h5 \\
                      checkpoints/checkpoint_AD_QAD_250901-12:00:00_trace-1_best.h5 \\
        --seeds 1 2 --alphas 0.01 0.05 0.1

    # one checkpoint per (trace, seed): trace and seed come from the checkpoints
    python conformal_anomaly_detection.py --dataset QAD \\
        --checkpoint-dir checkpoints --checkpoint-pattern 'checkpoint_AD_QAD_*_best.h5' \\
        --regenerate-processed-data --final-metrics-csv logs/conformal/qad.csv

    # same, restricted to three traces, seeds spelled out per checkpoint
    python conformal_anomaly_detection.py --dataset QAD --trace-ids 1 2 3 \\
        --checkpoints 'ckpt_trace-1_best.h5=1' 'ckpt_trace-2_best.h5=1' 'ckpt_trace-3_best.h5=1'

    # single-trace dataset
    python conformal_anomaly_detection.py --dataset PSM \\
        --checkpoint-dir checkpoints --checkpoint-pattern 'checkpoint_AD_PSM_*_best.h5'
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

import numpy as np
import pandas as pd
import torch

from anomaly_detection import (
    AGG_STRATEGIES,
    _load_dataset_config,
    _normalize_trace_ids,
    _validate_config_keys,
    aggregate_feature_scores,
    build_provider,
    calculate_feature_reconstruction_weights,
    calculate_z_normalization_values,
    compute_eval_scores,
    delete_processed_data,
    extend_argparse,
    normalise_scores,
)
from evaluate_checkpoints import (
    DATA_ARG_KEYS,
    _bootstrap_dataset,
    _explicit_cli_dests,
    _make_loaders,
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
from utils.scoring_functions import get_ts_eval, improved_cardinality_fn, ts_precision_and_recall
from utils.parser import generic_parser

# Separator of the inline ``PATH=SEED`` form of --checkpoints. '=' and not ':'
# because checkpoint names carry the experiment timestamp (``..._250901-10:00:00_...``).
SEED_SEPARATOR = "="

# Report key for checkpoints without a trace id (single-trace datasets such as
# SWaT/PSM), so reports and CSV rows read "PSM:single" instead of "PSM:None".
SINGLE_TRACE_KEY = "single"

# Columns of the per-alpha tables, in the order they are worth reading.
ALPHA_METRIC_COLS = (
    "alpha", "flagged_rate", "empirical_fpr", "calibration_gap",
    "precision", "recall", "f1", "ts_precision", "ts_recall", "ts_f1",
)

# Columns of the ranking-metric tables (same metrics evaluate_checkpoints.py reports).
RANKING_METRIC_COLS = ("f1", "precision", "recall", "auc_roc", "auc_pr", "vus_roc", "vus_pr")


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #
def build_parser(dataset: str, config_file: str | None) -> argparse.ArgumentParser:
    parser = extend_argparse(generic_parser)
    parser.description = (
        "Calibrate anomaly scores of stored checkpoints against the nominal validation "
        "split and report per-timepoint anomaly probabilities (conformal p-values)."
    )
    dataset_cfg = _load_dataset_config(dataset, config_file)
    _validate_config_keys(parser, dataset_cfg, dataset)
    parser.set_defaults(**dataset_cfg)
    parser.set_defaults(dataset=dataset, final_metrics_csv=None, loglevel="info")

    group = parser.add_argument_group("Conformal calibration arguments")
    group.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Checkpoints to calibrate, as PATH or PATH=SEED. Without a seed (and "
            "without --seeds) the seed stored in the checkpoint is used. Mutually "
            "exclusive with --checkpoint-dir."
        ),
    )
    group.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=None,
        help=(
            "Seeds for --checkpoints: either one seed per checkpoint or a single "
            "seed used for all of them. The seed must be the one the checkpoint was "
            "trained with, since it determines the train/val split and the "
            "subsampling masks."
        ),
    )
    group.add_argument(
        "--checkpoint-pattern",
        type=str,
        default="checkpoint_*_best.h5",
        help="Glob (relative to --checkpoint-dir) used when --checkpoints is not given.",
    )
    group.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.01, 0.05, 0.1],
        help=(
            "Significance levels at which a timepoint is flagged anomalous "
            "(p <= alpha). Each level comes with its observed false-positive rate, "
            "so the calibration can be checked against the nominal level."
        ),
    )
    group.add_argument(
        "--p-value-mode",
        choices=["conformal", "randomized"],
        default="conformal",
        help=(
            "'conformal' (default): p = (1 + #{cal >= s}) / (n + 1), the standard "
            "conservative conformal p-value. 'randomized': break ties uniformly at "
            "random, which makes p exactly uniform under the null but no longer a "
            "deterministic function of the score."
        ),
    )
    group.add_argument(
        "--ranking-metrics",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Also compute the threshold-free metric suite (AUC-ROC/PR, VUS, best F1) "
            "on the anomaly probabilities, for comparison with evaluate_checkpoints.py. "
            "The VUS metrics dominate the runtime; disable for a quick calibration check."
        ),
    )
    group.add_argument(
        "--combine-seeds",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Per trace, additionally combine the per-seed p-values of each timepoint "
            "into one ensemble p-value (Vovk-Wang: min(1, 2 * mean))."
        ),
    )
    group.add_argument(
        "--pool-traces",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "On multi-trace benchmarks, additionally evaluate all traces together: "
            "every trace's p-values come from its own calibration set, so they share "
            "one scale and one alpha applies benchmark-wide. Pooled per seed, then "
            "averaged over seeds (see pool_over_traces)."
        ),
    )
    group.add_argument(
        "--regenerate-processed-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Delete the cached processed data before building each provider, so the "
            "train/val split is drawn again under that checkpoint's seed -- the way "
            "anomaly_detection.py does it for every run. Without this the cached split "
            "on disk is reused for all seeds, whoever wrote it last (see module docstring)."
        ),
    )
    group.add_argument(
        "--eval-out-dir",
        type=str,
        default="logs/conformal",
        help="Directory for the JSON report and (unless --no-save-scores) the score arrays.",
    )
    group.add_argument(
        "--save-scores",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Store calibration scores, test scores, p-values, probabilities and labels "
            "of every checkpoint as .npz next to the report."
        ),
    )
    return parser


def parse_checkpoint_specs(
    tokens: list[str], seeds: list[int] | None, parser: argparse.ArgumentParser
) -> list[tuple[Path, int | None]]:
    """Resolve the ``--checkpoints``/``--seeds`` pair into ``(path, seed|None)``."""
    specs: list[tuple[Path, int | None]] = []
    for token in tokens:
        path_str, sep, seed_str = token.rpartition(SEED_SEPARATOR)
        # Only a trailing integer counts as a seed; a '=' anywhere else belongs
        # to the path.
        if sep and path_str and seed_str.lstrip("-").isdigit():
            specs.append((Path(path_str), int(seed_str)))
            continue
        specs.append((Path(token), None))

    if seeds is not None:
        inline = [seed for _, seed in specs if seed is not None]
        if inline:
            parser.error("Use either --seeds or the inline PATH=SEED form, not both.")
        if len(seeds) == 1:
            specs = [(path, seeds[0]) for path, _ in specs]
        elif len(seeds) == len(specs):
            specs = [(path, seed) for (path, _), seed in zip(specs, seeds)]
        else:
            parser.error(
                f"--seeds must provide one seed per checkpoint ({len(specs)}) or a single "
                f"seed for all of them, got {len(seeds)}."
            )

    for path, _ in specs:
        if not path.is_file():
            parser.error(f"Checkpoint {path} does not exist.")
    return specs


# --------------------------------------------------------------------------- #
# Scores -> per-timepoint scalar scores
# --------------------------------------------------------------------------- #
def contiguous_runs(indices: np.ndarray) -> list[tuple[int, int]]:
    """Split sorted unique integers into half-open runs of consecutive values."""
    if indices.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.concatenate([[0], breaks + 1])
    stops = np.concatenate([breaks + 1, [indices.size]])
    return [(int(indices[start]), int(indices[stop - 1]) + 1) for start, stop in zip(starts, stops)]


def covered_timepoints(dataloader) -> np.ndarray:
    """Global time indices actually covered by a split's windows.

    ``compute_eval_scores`` scatters the per-window scores into an array that
    spans ``0 .. max(index)``; for a split made of a *subset* of the windows
    (the validation split is a random 10% of the training windows in most
    providers) the rows in between stay at their zero initialisation. Those
    rows are not scores and must not enter the calibration set.
    """
    return np.unique(np.asarray(dataloader.dataset.indcs).ravel().astype(int))


def scalar_scores(
    all_scores: np.ndarray,
    covered: np.ndarray,
    smoothing_window: int,
    aggregation: str,
    feature_weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Post-process per-feature scores into one score per timepoint.

    Applies exactly the post-processing of ``eval_scores`` (moving average over
    time, then the per-feature reduction) but returns the aggregated scores
    themselves, together with the global time index of each one.

    Smoothing is done per contiguous run of covered timepoints, so a gap in the
    covered index (validation split!) never averages a real score with the zero
    rows next to it. ``smooth_scores`` leaves its first ``smoothing_window``
    outputs at 0 by construction, so those leading timepoints of every run are
    dropped instead of entering the result as fake zero scores.
    """
    kept_index, kept_scores = [], []
    for start, stop in contiguous_runs(covered):
        block = all_scores[start:stop]
        offset = start
        if smoothing_window and smoothing_window > 0:
            if block.shape[0] <= smoothing_window:
                continue
            block = normalise_scores(block, norm=None, smooth=True, smooth_window=smoothing_window)
            block = block[smoothing_window:]
            offset = start + smoothing_window
        kept_index.append(np.arange(offset, offset + block.shape[0]))
        kept_scores.append(aggregate_feature_scores(block, aggregation, feature_weights))

    if not kept_index:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)
    return np.concatenate(kept_index), np.concatenate(kept_scores)


# --------------------------------------------------------------------------- #
# Conformal p-values
# --------------------------------------------------------------------------- #
def conformal_p_values(
    calibration_scores: np.ndarray,
    test_scores: np.ndarray,
    mode: str = "conformal",
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """p-value of each test score w.r.t. the nominal calibration scores.

    ``p = (1 + #{c >= s}) / (n + 1)``: the share of nominal scores at least as
    extreme as ``s``, with the ``+1`` accounting for the test point itself (it
    keeps the p-value valid in finite samples and bounds it below by
    ``1 / (n + 1)``). ``mode='randomized'`` spreads ties uniformly, which turns
    the inequality ``P(p <= alpha) <= alpha`` into an equality.
    """
    calibration_scores = np.sort(np.asarray(calibration_scores, dtype=float))
    test_scores = np.asarray(test_scores, dtype=float)
    n = calibration_scores.size
    if n == 0:
        raise ValueError("Empty calibration set; cannot compute conformal p-values.")

    n_greater_equal = n - np.searchsorted(calibration_scores, test_scores, side="left")
    if mode == "randomized":
        n_greater = n - np.searchsorted(calibration_scores, test_scores, side="right")
        n_equal = n_greater_equal - n_greater
        if rng is None:
            rng = np.random.default_rng(0)
        return (n_greater + rng.random(test_scores.shape) * (n_equal + 1)) / (n + 1.0)
    if mode != "conformal":
        raise ValueError(f"Unknown p-value mode '{mode}'. Expected 'conformal' or 'randomized'.")
    return (1.0 + n_greater_equal) / (n + 1.0)


def _point_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    tp = float(np.sum(predictions & (labels > 0)))
    fp = float(np.sum(predictions & (labels == 0)))
    fn = float(np.sum(~predictions & (labels > 0)))
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def metrics_at_alpha(p_values: np.ndarray, labels: np.ndarray, alpha: float,
                     include_ts: bool = True) -> dict:
    """Decision quality and calibration validity at one significance level.

    ``empirical_fpr`` is the observed rate of ``p <= alpha`` among the nominal
    test timepoints; under exchangeability with the calibration set it should be
    at most ``alpha``, so ``calibration_gap = empirical_fpr - alpha`` tells how
    far the guarantee is from holding on this trace.

    ``include_ts=False`` drops the range-based (time-series) precision/recall.
    They are only meaningful on one continuous series, so the pooled view over
    several traces -- a concatenation, whose trace boundaries would merge
    unrelated ranges -- leaves them out.
    """
    predictions = p_values <= alpha
    nominal = labels == 0
    empirical_fpr = float(predictions[nominal].mean()) if nominal.any() else float("nan")

    stats = {
        "alpha": float(alpha),
        "flagged_rate": float(predictions.mean()),
        "empirical_fpr": empirical_fpr,
        "calibration_gap": empirical_fpr - float(alpha),
        **_point_metrics(labels, predictions),
    }
    if not include_ts:
        return stats

    ts_precision, ts_recall = ts_precision_and_recall(
        labels.astype(int), predictions.astype(int), alpha=0,
        recall_cardinality_fn=improved_cardinality_fn,
    )
    ts_f1 = (2 * ts_precision * ts_recall / (ts_precision + ts_recall)
             if ts_precision + ts_recall > 0 else 0.0)
    stats.update({"ts_precision": float(ts_precision), "ts_recall": float(ts_recall), "ts_f1": float(ts_f1)})
    return stats


def calibrate_and_score(args, ckpt: dict, provider, ds_idx: int) -> dict:
    """Calibrate one checkpoint on its validation split and score the test split."""
    device = args.device
    dl_trn, dl_tst, dl_val, input_dim, num_timepoints = _make_loaders(provider, ds_idx, args)
    modules, elbo_loss, desired_t = _restore_modules(args, ckpt, input_dim, num_timepoints, device)

    normalization_scores = None
    if args.normalize_score:
        normalization_scores = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)

    feature_weights = None
    if args.score_aggregation in ("weighted-mse", "weighted-mse-exp"):
        weighting = "exp-inverse" if args.score_aggregation == "weighted-mse-exp" else "inverse"
        feature_weights = calculate_feature_reconstruction_weights(
            args, dl_trn, modules, desired_t, device, weighting=weighting)["feature_weights"]

    post = dict(
        smoothing_window=args.score_smoothing_window,
        aggregation=args.score_aggregation,
        feature_weights=feature_weights,
    )

    # ---- calibration (nominal validation split) ----------------------------
    val_stats, val_scores, val_labels = compute_eval_scores(
        args, dl_val, modules, elbo_loss, desired_t, device,
        normalization_stats=normalization_scores,
    )
    cal_index, cal_scores = scalar_scores(val_scores, covered_timepoints(dl_val), **post)
    if cal_scores.size == 0:
        raise RuntimeError(
            "Validation split yields no usable calibration scores (all windows shorter "
            f"than the smoothing window {args.score_smoothing_window})."
        )
    contaminated = float(np.mean(val_labels[cal_index] > 0))
    if contaminated > 0:
        logging.warning(
            "Calibration set is contaminated: %.2f%% of the validation timepoints are "
            "labelled anomalous. Conformal p-values assume a nominal calibration set.",
            100 * contaminated,
        )

    # ---- test --------------------------------------------------------------
    tst_stats, tst_scores_raw, tst_labels_raw = compute_eval_scores(
        args, dl_tst, modules, elbo_loss, desired_t, device,
        normalization_stats=normalization_scores,
    )
    tst_index, tst_scores = scalar_scores(tst_scores_raw, covered_timepoints(dl_tst), **post)
    tst_labels = tst_labels_raw[tst_index]

    rng = np.random.default_rng(abs(int(args.seed)) if args.seed is not None else 0)
    p_values = conformal_p_values(cal_scores, tst_scores, mode=args.p_value_mode, rng=rng)
    probabilities = 1.0 - p_values

    min_p = 1.0 / (cal_scores.size + 1.0)
    if min_p > min(args.alphas):
        logging.warning(
            "Calibration set has only %d scores, so the smallest attainable p-value is "
            "%.4g -- no timepoint can ever be flagged at alpha < %.4g.",
            cal_scores.size, min_p, min_p,
        )

    result = {
        "num_calibration": int(cal_scores.size),
        "num_test": int(tst_scores.size),
        "min_attainable_p": float(min_p),
        "anomaly_rate": float(np.mean(tst_labels > 0)),
        "calibration_contamination": contaminated,
        "calibration_score_quantiles": {
            str(q): float(np.quantile(cal_scores, q)) for q in (0.5, 0.9, 0.95, 0.99, 1.0)
        },
        "val_elbo": _to_python(val_stats),
        "tst_elbo": _to_python(tst_stats),
        "per_alpha": [metrics_at_alpha(p_values, tst_labels, alpha) for alpha in args.alphas],
        "arrays": {
            "calibration_index": cal_index,
            "calibration_scores": cal_scores,
            "test_index": tst_index,
            "test_scores": tst_scores,
            "test_p_value": p_values,
            "test_anomaly_probability": probabilities,
            "test_labels": tst_labels,
        },
    }
    if args.ranking_metrics:
        result["ranking"] = _to_python(
            get_ts_eval(probabilities, tst_labels, window_length=args.data_window_length)
        )
    return result


# --------------------------------------------------------------------------- #
# Aggregation over seeds
# --------------------------------------------------------------------------- #
def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=0))


def aggregate_over_seeds(per_seed: list[dict], alphas: list[float]) -> dict:
    """Mean/std over the seeds of one trace, per alpha and for the ranking metrics."""
    out = {"num_seeds": len(per_seed), "per_alpha": [], "ranking": {}}

    for pos, alpha in enumerate(alphas):
        row = {"alpha": float(alpha)}
        for key in ALPHA_METRIC_COLS:
            if key == "alpha":
                continue
            values = [seed_result["per_alpha"][pos][key] for seed_result in per_seed
                      if key in seed_result["per_alpha"][pos]]
            if values:
                row[key], row[f"{key}_std"] = _mean_std(values)
        out["per_alpha"].append(row)

    ranking_keys = {key for seed_result in per_seed for key in seed_result.get("ranking", {})}
    for key in sorted(ranking_keys):
        values = [seed_result["ranking"][key] for seed_result in per_seed
                  if isinstance(seed_result.get("ranking", {}).get(key), (int, float))]
        if values:
            out["ranking"][key], out["ranking"][f"{key}_std"] = _mean_std(values)
    return out


def combine_seed_p_values(per_seed: list[dict], alphas: list[float], ranking_metrics: bool,
                          window_length: int) -> dict | None:
    """Vovk-Wang ensemble of the per-seed p-values on the shared timepoints.

    ``min(1, 2 * mean_i p_i)`` is a valid p-value for arbitrary dependence
    between the seeds, which is the right assumption here: the seeds share the
    data and differ only in initialisation and splits. Restricted to the
    timepoints all seeds scored, so every seed contributes to every ensemble
    p-value.
    """
    if len(per_seed) < 2:
        return None

    shared = per_seed[0]["arrays"]["test_index"]
    for seed_result in per_seed[1:]:
        shared = np.intersect1d(shared, seed_result["arrays"]["test_index"], assume_unique=True)
    if shared.size == 0:
        logging.warning("Seeds of this trace share no test timepoint; skipping the seed ensemble.")
        return None

    p_stack = []
    for seed_result in per_seed:
        positions = np.searchsorted(seed_result["arrays"]["test_index"], shared)
        p_stack.append(seed_result["arrays"]["test_p_value"][positions])
    first = per_seed[0]["arrays"]
    labels = first["test_labels"][np.searchsorted(first["test_index"], shared)]

    combined = np.minimum(1.0, 2.0 * np.mean(np.vstack(p_stack), axis=0))
    out = {
        "num_seeds": len(per_seed),
        "num_test": int(shared.size),
        "per_alpha": [metrics_at_alpha(combined, labels, alpha) for alpha in alphas],
        "arrays": {
            "test_index": shared,
            "test_p_value": combined,
            "test_anomaly_probability": 1.0 - combined,
            "test_labels": labels,
        },
    }
    if ranking_metrics:
        out["ranking"] = _to_python(get_ts_eval(1.0 - combined, labels, window_length=window_length))
    return out


def pool_over_traces(results: list[dict], alphas: list[float]) -> dict | None:
    """Evaluate all traces at once, each calibrated against its own trace.

    The per-trace means ("macro") weight a 2k-timepoint trace like a 200k one,
    and they say nothing about using *one* operating point across the whole
    benchmark. Pooling does: because every trace's scores were turned into
    p-values against that trace's own nominal calibration set, the numbers live
    on the same [0, 1] scale no matter how differently the traces are scaled or
    how well the model fits each one, so concatenating them is meaningful --
    which is exactly what concatenating raw anomaly scores across traces is
    not. One alpha then applies benchmark-wide, and ``empirical_fpr`` says
    whether it holds there.

    Done per seed (so every trace contributes once), then averaged over seeds.
    Range-based metrics are left out, see ``metrics_at_alpha``.
    """
    by_seed: dict[int, list[dict]] = defaultdict(list)
    for result in results:
        by_seed[result["seed"]].append(result)
    if not by_seed:
        return None

    per_seed_rows = []
    for seed in sorted(by_seed):
        group = sorted(by_seed[seed], key=lambda res: res["trace_key"])
        p_values = np.concatenate([res["arrays"]["test_p_value"] for res in group])
        labels = np.concatenate([res["arrays"]["test_labels"] for res in group])
        per_seed_rows.append({
            "seed": int(seed),
            "num_traces": len(group),
            "traces": [res["trace_key"] for res in group],
            "num_test": int(p_values.size),
            "per_alpha": [metrics_at_alpha(p_values, labels, alpha, include_ts=False)
                          for alpha in alphas],
        })

    pooled = aggregate_over_seeds(per_seed_rows, alphas)
    pooled.pop("ranking", None)
    pooled["seeds"] = [row["seed"] for row in per_seed_rows]
    pooled["num_traces_per_seed"] = [row["num_traces"] for row in per_seed_rows]
    pooled["num_test_per_seed"] = [row["num_test"] for row in per_seed_rows]
    pooled["per_seed"] = per_seed_rows
    return pooled


def _alpha_frame(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    frame = frame[[col for col in ALPHA_METRIC_COLS if col in frame.columns]]
    return frame.set_index("alpha")


def _strip_arrays(result: dict) -> dict:
    return {key: value for key, value in result.items() if key != "arrays"}


def _shrink_arrays(result: dict, keep: tuple[str, ...]):
    """Drop the score arrays that are no longer needed, in place.

    One run can hold a checkpoint per (trace, seed); keeping every score array
    of a 100-trace benchmark alive until the end is pointless once the arrays
    have been written to disk and the trace's own aggregation is done.
    """
    result["arrays"] = {key: value for key, value in result["arrays"].items() if key in keep}


def _save_arrays(path: Path, arrays: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


# --------------------------------------------------------------------------- #
# Checkpoint metadata / per-trace processing
# --------------------------------------------------------------------------- #
def _resolve_seed(path: Path, ckpt: dict, ckpt_args: Namespace, requested: int | None,
                  cli_args: Namespace, explicit: set[str]) -> int:
    """Seed to run this checkpoint under.

    Precedence: a seed given for this checkpoint (``--seeds``/``PATH=SEED``, or
    a plain ``--seed`` standing in for all of them) beats the seed stored in the
    checkpoint, which beats the parser default.
    """
    # Checkpoints written before the 'seed' key existed (and ones that stored it
    # as None) fall back to the seed inside the stored args.
    raw_stored = ckpt.get("seed")
    if raw_stored is None:
        raw_stored = getattr(ckpt_args, "seed", None)
    stored = None if raw_stored is None else int(raw_stored)
    # A plain `--seed` acts like `--seeds <seed>`: one seed for every checkpoint.
    if requested is None and "seed" in explicit:
        requested = int(cli_args.seed)

    if requested is None:
        if stored is not None:
            return stored
        logging.warning(
            "%s stores no seed and none was given; using --seed %s. The validation "
            "split is only the checkpoint's own if this matches its training seed.",
            path.name, cli_args.seed,
        )
        return int(cli_args.seed)

    if stored is not None and stored != requested:
        logging.warning(
            "Seed %d given for %s, but the checkpoint was trained with seed %s. "
            "The validation split will differ from the one used during training.",
            requested, path.name, stored,
        )
    return int(requested)


def collect_candidates(
    specs: list[tuple[Path, int | None]], cli_args: Namespace, explicit: set[str]
) -> tuple[list[dict], list[dict]]:
    """Read trace id, seed and effective args off every checkpoint.

    Only metadata is kept; the weights are loaded again when the checkpoint is
    actually calibrated. That is what allows the run to be ordered by trace
    (one provider alive at a time) instead of by the order the checkpoints were
    listed in.
    """
    candidates, failures = [], []
    for path, requested_seed in specs:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        ckpt_args = ckpt.get("args")
        if ckpt_args is None:
            logging.warning("Skipping %s: no stored args.", path.name)
            failures.append({"checkpoint": str(path), "seed": requested_seed, "error": "no stored args"})
            continue
        if isinstance(ckpt_args, dict):
            ckpt_args = Namespace(**ckpt_args)
        if getattr(ckpt_args, "dataset", None) != cli_args.dataset:
            logging.warning(
                "Skipping %s: checkpoint dataset %s != --dataset %s.",
                path.name, getattr(ckpt_args, "dataset", None), cli_args.dataset,
            )
            continue

        trace_id = _trace_from_checkpoint(ckpt, ckpt_args, path)
        if cli_args.trace_ids is not None and trace_id not in cli_args.trace_ids:
            logging.debug("Skipping %s: trace %s not in --trace-ids.", path.name, trace_id)
            continue

        args = resolve_effective_args(ckpt_args, cli_args, explicit)
        args.device = _select_device(args.device)
        args.trace_ids = [trace_id] if trace_id is not None else None
        args.seed = _resolve_seed(path, ckpt, ckpt_args, requested_seed, cli_args, explicit)
        # Knobs of this script only; never taken from the checkpoint.
        for key in ("alphas", "p_value_mode", "ranking_metrics", "combine_seeds",
                    "pool_traces", "save_scores", "regenerate_processed_data"):
            setattr(args, key, getattr(cli_args, key))

        candidates.append({
            "path": path,
            "trace_id": trace_id,
            "trace_key": str(trace_id) if trace_id is not None else SINGLE_TRACE_KEY,
            "seed": args.seed,
            "args": args,
        })
        del ckpt
    return candidates, failures


def process_trace(
    trace: str, candidates: list[dict], cli_args: Namespace, out_dir: Path, run_id: str
) -> tuple[list[dict], list[dict]]:
    """Calibrate every checkpoint of one trace, then release its providers.

    Providers are cached within the trace (several seeds of a trace usually
    share one data configuration) and dropped afterwards, so a benchmark with
    many traces does not keep every trace's windows in memory.
    """
    provider_cache: dict[tuple, object] = {}
    results, failures = [], []
    try:
        for cand in sorted(candidates, key=lambda c: c["seed"]):
            path, args = cand["path"], cand["args"]
            if args.seed > 0:
                set_seed(args.seed)

            cache_key = tuple(getattr(args, key, None) for key in DATA_ARG_KEYS) + (trace,)
            provider = provider_cache.get(cache_key)
            if provider is None:
                if cli_args.regenerate_processed_data:
                    # Drawn after set_seed above, so the train/val permutation is
                    # the one this seed produced during training.
                    delete_processed_data(args.dataset, data_dir=getattr(args, "data_dir", "data_dir"))
                provider = build_provider(args)
                provider_cache[cache_key] = provider
            ds_idx = _resolve_trace_index(provider, cand["trace_id"])

            logging.info("Calibrating %s (trace=%s, idx=%d, seed=%d)", path.name, trace, ds_idx, args.seed)
            logging.debug("Effective args: %s", vars(args))
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
            try:
                result = calibrate_and_score(args, ckpt, provider, ds_idx)
            except (RuntimeError, ValueError) as exc:
                logging.error("Skipping %s: %s", path.name, exc)
                failures.append({"checkpoint": str(path), "seed": args.seed,
                                 "trace_id": cand["trace_id"], "error": str(exc)})
                continue
            finally:
                del ckpt

            result.update({
                "checkpoint": str(path),
                "seed": int(args.seed),
                "trace_id": cand["trace_id"],
                "trace_key": cand["trace_key"],
                "dataset": args.dataset,
                "score_aggregation": args.score_aggregation,
                "score_smoothing_window": args.score_smoothing_window,
                "normalize_score": args.normalize_score,
                "mc_eval_samples": args.mc_eval_samples,
            })
            logging.info(
                "[%s/seed=%d] %d calibration scores, %d test timepoints (%.2f%% anomalous):\n%s",
                trace, args.seed, result["num_calibration"], result["num_test"],
                100 * result["anomaly_rate"],
                _alpha_frame(result["per_alpha"]).to_string(float_format=lambda v: f"{v:.4f}"),
            )
            if cli_args.save_scores:
                npz_path = (out_dir / "scores" /
                            f"CONFORMAL_{cli_args.dataset}_{run_id}__{path.stem}__seed{args.seed}.npz")
                _save_arrays(npz_path, result["arrays"])
                result["scores_file"] = str(npz_path)
                logging.debug("Stored score arrays at %s", npz_path)
            results.append(result)
    finally:
        for provider in provider_cache.values():
            if hasattr(provider, "cleanup"):
                try:
                    provider.cleanup()
                except Exception as err:
                    logging.warning("Provider cleanup failed: %s", err)
        provider_cache.clear()
    return results, failures


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

    if cli_args.checkpoints and cli_args.checkpoint_dir:
        parser.error("Use either --checkpoints or --checkpoint-dir, not both.")
    if not cli_args.checkpoints and not cli_args.checkpoint_dir:
        parser.error("Provide the checkpoints to calibrate via --checkpoints or --checkpoint-dir.")
    if any(not 0 < alpha < 1 for alpha in cli_args.alphas):
        parser.error("--alphas must all lie strictly between 0 and 1.")
    cli_args.alphas = sorted(dict.fromkeys(float(alpha) for alpha in cli_args.alphas))
    if cli_args.score_aggregation not in AGG_STRATEGIES:
        parser.error(f"Unknown --score-aggregation {cli_args.score_aggregation!r}.")

    if cli_args.checkpoints:
        specs = parse_checkpoint_specs(cli_args.checkpoints, cli_args.seeds, parser)
    else:
        checkpoint_dir = Path(cli_args.checkpoint_dir)
        if not checkpoint_dir.is_dir():
            parser.error(f"--checkpoint-dir {checkpoint_dir} is not a directory")
        paths = discover_checkpoints(checkpoint_dir, cli_args.checkpoint_pattern)
        seeds = cli_args.seeds
        if seeds is not None and len(seeds) not in (1, len(paths)):
            parser.error(
                f"--seeds must provide one seed per discovered checkpoint ({len(paths)}) "
                f"or a single seed, got {len(seeds)}."
            )
        if seeds is None:
            specs = [(path, None) for path in paths]
        elif len(seeds) == 1:
            specs = [(path, seeds[0]) for path in paths]
        else:
            specs = list(zip(paths, seeds))

    run_id = datetime.datetime.now().strftime("%y%m%d-%H:%M:%S")
    out_dir = Path(cli_args.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_up_logging(
        console_log_level=cli_args.loglevel,
        console_log_color=True,
        logfile_file=str(out_dir / f"CONFORMAL_{cli_args.dataset}_{run_id}.txt"),
        logfile_log_level=cli_args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s",
    )
    logging.info(
        "Calibrating %d checkpoint(s) for %s at alphas=%s (p-value mode=%s)",
        len(specs), cli_args.dataset, cli_args.alphas, cli_args.p_value_mode,
    )
    for path, seed in specs:
        logging.info("  %s (seed=%s)", path, "from checkpoint" if seed is None else seed)

    # ---- metadata pass: group the checkpoints by trace ----------------------
    candidates, failures = collect_candidates(specs, cli_args, explicit)
    if not candidates:
        raise SystemExit(f"None of the {len(specs)} checkpoint(s) matched {cli_args.dataset}; see log above.")

    by_trace_candidates: dict[str, list[dict]] = defaultdict(list)
    for cand in candidates:
        by_trace_candidates[cand["trace_key"]].append(cand)
    traces = sorted(by_trace_candidates, key=lambda t: (len(t), t))
    logging.info(
        "%d checkpoint(s) over %d trace(s); seeds per trace: %s",
        len(candidates), len(traces),
        {trace: sorted(cand["seed"] for cand in by_trace_candidates[trace]) for trace in traces},
    )
    seed_counts = {len(by_trace_candidates[trace]) for trace in traces}
    if len(seed_counts) > 1:
        logging.warning(
            "Traces do not all have the same number of checkpoints (%s). Per-trace means "
            "and the macro average are then over different numbers of seeds.",
            sorted(seed_counts),
        )

    # ---- per trace: calibrate every seed, aggregate, release the provider ---
    results, per_trace, ensembles = [], {}, {}
    for trace in traces:
        trace_results, trace_failures = process_trace(
            trace, by_trace_candidates[trace], cli_args, out_dir, run_id,
        )
        failures.extend(trace_failures)
        if not trace_results:
            logging.warning("Trace %s: no checkpoint could be calibrated.", trace)
            continue

        per_trace[trace] = aggregate_over_seeds(trace_results, cli_args.alphas)
        per_trace[trace]["seeds"] = [res["seed"] for res in trace_results]
        logging.info(
            "Trace %s -- mean over %d seed(s) %s:\n%s",
            trace, len(trace_results), per_trace[trace]["seeds"],
            _alpha_frame(per_trace[trace]["per_alpha"]).to_string(float_format=lambda v: f"{v:.4f}"),
        )
        if cli_args.combine_seeds:
            ensemble = combine_seed_p_values(
                trace_results, cli_args.alphas, cli_args.ranking_metrics, cli_args.data_window_length,
            )
            if ensemble is not None:
                logging.info(
                    "Trace %s -- seed ensemble (Vovk-Wang over %d seeds, %d shared timepoints):\n%s",
                    trace, ensemble["num_seeds"], ensemble["num_test"],
                    _alpha_frame(ensemble["per_alpha"]).to_string(float_format=lambda v: f"{v:.4f}"),
                )
                if cli_args.save_scores:
                    npz_path = (out_dir / "scores" /
                                f"CONFORMAL_{cli_args.dataset}_{run_id}__trace-{trace}__ensemble.npz")
                    _save_arrays(npz_path, ensemble["arrays"])
                    ensemble["scores_file"] = str(npz_path)
                ensembles[trace] = _strip_arrays(ensemble)

        # Everything per-trace is computed; only what the pooled view needs survives.
        for res in trace_results:
            _shrink_arrays(res, keep=("test_p_value", "test_labels") if cli_args.pool_traces else ())
        results.extend(trace_results)

    if not results:
        raise SystemExit(f"None of the {len(candidates)} checkpoint(s) could be calibrated; see log above.")

    # ---- pooled over traces (one operating point for the whole benchmark) --
    # With a single trace the pooled view is the per-trace view, so skip it.
    pooled = (pool_over_traces(results, cli_args.alphas)
              if cli_args.pool_traces and len(per_trace) > 1 else None)
    if pooled is not None:
        logging.info(
            "Pooled over all traces, per-trace calibration (mean over %d seed(s) %s; "
            "%s traces / %s timepoints per seed):\n%s",
            pooled["num_seeds"], pooled["seeds"], pooled["num_traces_per_seed"],
            pooled["num_test_per_seed"],
            _alpha_frame(pooled["per_alpha"]).to_string(float_format=lambda v: f"{v:.4f}"),
        )

    # ---- macro over traces --------------------------------------------------
    macro_rows = []
    for pos, alpha in enumerate(cli_args.alphas):
        row = {"alpha": float(alpha)}
        for key in ALPHA_METRIC_COLS:
            if key == "alpha":
                continue
            values = [per_trace[trace]["per_alpha"][pos][key] for trace in per_trace
                      if key in per_trace[trace]["per_alpha"][pos]]
            if values:
                row[key] = float(np.mean(values))
        macro_rows.append(row)
    macro_ranking = {}
    ranking_keys = {key for trace in per_trace for key in per_trace[trace]["ranking"]
                    if not key.endswith("_std")}
    for key in sorted(ranking_keys):
        values = [per_trace[trace]["ranking"][key] for trace in per_trace
                  if key in per_trace[trace]["ranking"]]
        if values:
            macro_ranking[f"macro_{key}"] = float(np.mean(values))

    logging.info(
        "Macro over %d trace(s):\n%s",
        len(per_trace), _alpha_frame(macro_rows).to_string(float_format=lambda v: f"{v:.4f}"),
    )
    if macro_ranking:
        logging.info("Macro ranking metrics (on the anomaly probabilities): %s", macro_ranking)
    for row in macro_rows:
        gap = row.get("calibration_gap")
        if gap is not None and gap > 0.5 * row["alpha"]:
            logging.warning(
                "At alpha=%.3g the observed false-positive rate is %.4f (nominal %.3g). The "
                "validation split is apparently not exchangeable with the nominal part of the "
                "test split, so the level is not attained.",
                row["alpha"], row["empirical_fpr"], row["alpha"],
            )

    # ---- report -------------------------------------------------------------
    report = {
        "run_id": run_id,
        "dataset": cli_args.dataset,
        "p_value_mode": cli_args.p_value_mode,
        "alphas": cli_args.alphas,
        "checkpoints": [{"checkpoint": str(path), "seed": seed} for path, seed in specs],
        "cli_overrides": {key: getattr(cli_args, key) for key in sorted(explicit)},
        "per_checkpoint": [_strip_arrays(result) for result in results],
        "failed_checkpoints": failures,
        "per_trace": per_trace,
        "seed_ensemble": ensembles,
        "pooled_over_traces": pooled,
        "macro_per_alpha": macro_rows,
        **macro_ranking,
    }
    report_path = out_dir / f"CONFORMAL_{cli_args.dataset}_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    if failures:
        logging.warning("%d checkpoint(s) failed: %s", len(failures), [f["checkpoint"] for f in failures])
    logging.info("Report written to %s", report_path)

    if cli_args.final_metrics_csv:
        for trace, stats in per_trace.items():
            for row in stats["per_alpha"]:
                append_final_metrics_csv(
                    csv_path=cli_args.final_metrics_csv,
                    benchmark=f"{cli_args.dataset}:{trace}:alpha={row['alpha']:g}",
                    run_datetime=run_id,
                    metrics={**row, "num_seeds": stats["num_seeds"]},
                )
        for row in macro_rows:
            append_final_metrics_csv(
                csv_path=cli_args.final_metrics_csv,
                benchmark=f"{cli_args.dataset}:macro:alpha={row['alpha']:g}",
                run_datetime=run_id,
                metrics={**row, "num_traces": len(per_trace)},
            )
        if pooled is not None:
            for row in pooled["per_alpha"]:
                append_final_metrics_csv(
                    csv_path=cli_args.final_metrics_csv,
                    benchmark=f"{cli_args.dataset}:pooled:alpha={row['alpha']:g}",
                    run_datetime=run_id,
                    metrics={**row, "num_seeds": pooled["num_seeds"], "num_traces": len(per_trace)},
                )
        logging.info("Appended per-trace, macro%s rows to %s",
                     " and pooled" if pooled is not None else "", cli_args.final_metrics_csv)


if __name__ == "__main__":
    main()