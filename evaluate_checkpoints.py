"""Evaluate stored ``best`` checkpoints without retraining.

Point the script at a checkpoint directory (``--checkpoint-dir``) and it

1. collects every ``checkpoint_*_best.h5`` file in there (``--checkpoint-pattern``
   changes the glob),
2. restores the training arguments stored inside each checkpoint (model
   architecture, data settings, trace id),
3. rebuilds the data provider and the model exactly like ``anomaly_detection.py``
   does for training,
4. runs the very same test/validation evaluation (score normalisation,
   feature weighting, aggregation, smoothing, ...) and
5. reports per-checkpoint, per-trace and macro (across traces) metrics.

Evaluation settings can be overridden on the command line with the same flags
``anomaly_detection.py`` accepts, e.g. ``--score-aggregation weighted-mse`` or
``--mc-eval-samples 10``. Model-architecture flags are always taken from the
checkpoint. Example::

    python evaluate_checkpoints.py --checkpoint-dir checkpoints --dataset QAD \
        --final-metrics-csv logs/eval/qad_best_checkpoints.csv
"""

from __future__ import annotations

import argparse
import copy
import datetime
import json
import logging
import os
import re
import sys
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from anomaly_detection import (
    DATASET_CHOICES,
    DatasetSlice,
    _load_dataset_config,
    _normalize_trace_ids,
    _validate_config_keys,
    build_modules_and_optim,
    build_provider,
    calculate_feature_reconstruction_weights,
    calculate_z_normalization_values,
    evaluate,
    extend_argparse,
)
from utils.logger import set_up_logging
from utils.misc import append_final_metrics_csv, set_seed
from utils.parser import generic_parser


# Arguments that define the network. They must match the stored weights and are
# therefore always taken from the checkpoint, never from the command line.
MODEL_ARG_KEYS = (
    "z_dim", "h_dim", "n_deg", "dec_hidden_dim", "n_dec_layers", "non_linear_decoder",
    "sphere_embedding", "sde", "use_atanh", "initial_sigma", "learnable_prior", "freeze_sigma",
    "dataset",
)

# Arguments that decide how the data is windowed/subsampled. They are the cache
# key for providers so several checkpoints of one trace share one provider.
DATA_ARG_KEYS = (
    "dataset", "data_dir", "data_window_length", "data_window_overlap", "subsample",
    "fixed_subsample_mask", "seed", "data_normalization_strategy", "data_decimation_factor",
    "debug",
)

CHECKPOINT_NAME_RE = re.compile(
    r"^checkpoint_(?P<exp>(?:DEBUG|AD_(?P<dataset>[^_]+))_\d{6}-\d{2}:\d{2}:\d{2})"
    r"(?:_trace-(?P<trace>.+?)|_(?P<legacy_trace>[^_]+))?"
    r"_(?P<epoch>best|\d+)\.h5$"
)


# --------------------------------------------------------------------------- #
# Argument handling
# --------------------------------------------------------------------------- #
def _bootstrap_dataset(argv) -> tuple[str, str | None]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", choices=DATASET_CHOICES, default="QAD")
    parser.add_argument("--config-file", type=str, default=None)
    known, _ = parser.parse_known_args(argv)
    return known.dataset, known.config_file


def build_parser(dataset: str, config_file: str | None) -> argparse.ArgumentParser:
    parser = extend_argparse(generic_parser)
    parser.description = "Evaluate stored best checkpoints with the anomaly_detection.py evaluation pipeline."
    dataset_cfg = _load_dataset_config(dataset, config_file)
    _validate_config_keys(parser, dataset_cfg, dataset)
    parser.set_defaults(**dataset_cfg)
    parser.set_defaults(dataset=dataset, final_metrics_csv=None, loglevel="info")

    group = parser.add_argument_group("Checkpoint evaluation arguments")
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
        "--skip-val",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the validation-set pass (only test metrics are computed).",
    )
    return parser


def _explicit_cli_dests(parser: argparse.ArgumentParser, argv) -> set[str]:
    """Return the ``dest`` names of all options that were given on the command line."""
    given = set()
    for token in argv:
        if token.startswith("--"):
            given.add(token.split("=", 1)[0])
    dests = set()
    for action in parser._actions:
        if any(opt in given for opt in action.option_strings):
            dests.add(action.dest)
    return dests


def resolve_effective_args(
    ckpt_args: Namespace, cli_args: Namespace, explicit: set[str]
) -> Namespace:
    """defaults (parser + dataset config) < checkpoint args < explicit CLI overrides."""
    args = copy.deepcopy(cli_args)
    for key, value in vars(ckpt_args).items():
        setattr(args, key, value)
    for key in explicit:
        if key in MODEL_ARG_KEYS:
            if getattr(cli_args, key) != getattr(args, key):
                logging.warning(
                    "Ignoring --%s=%s: model argument is fixed by the checkpoint (%s).",
                    key.replace("_", "-"), getattr(cli_args, key), getattr(args, key),
                )
            continue
        setattr(args, key, getattr(cli_args, key))
    # Evaluation-only script: never persist anything from the training path.
    args.enable_checkpointing = False
    args.checkpoint_at = []
    args.runs = 1
    return args


# --------------------------------------------------------------------------- #
# Checkpoint discovery
# --------------------------------------------------------------------------- #
def _parse_checkpoint_name(path: Path) -> dict:
    match = CHECKPOINT_NAME_RE.match(path.name)
    if match is None:
        return {"exp": path.stem, "dataset": None, "trace": None, "epoch": None}
    info = match.groupdict()
    info["trace"] = info["trace"] or info["legacy_trace"]
    return info


def _trace_from_checkpoint(ckpt: dict, ckpt_args: Namespace, path: Path) -> str | None:
    if ckpt.get("trace_id") is not None:
        return str(ckpt["trace_id"])
    trace_ids = getattr(ckpt_args, "trace_ids", None)
    if trace_ids and len(trace_ids) == 1:
        return str(trace_ids[0])
    return _parse_checkpoint_name(path)["trace"]


def discover_checkpoints(checkpoint_dir: Path, pattern: str) -> list[Path]:
    paths = sorted(checkpoint_dir.glob(pattern), key=lambda p: (p.stat().st_mtime, p.name))
    if not paths:
        raise FileNotFoundError(f"No checkpoints matching {pattern!r} in {checkpoint_dir}")
    return paths


# --------------------------------------------------------------------------- #
# Data / model helpers
# --------------------------------------------------------------------------- #
def _resolve_trace_index(provider, trace_id: str | None) -> int:
    """Map a trace id (dataset_id or numeric index) to the provider's sub-dataset index."""
    if trace_id is None:
        if provider.num_datasets != 1:
            raise ValueError(
                f"Checkpoint carries no trace id but provider exposes {provider.num_datasets} traces."
            )
        return 0

    id_to_idx = {}
    for ds_idx in range(provider.num_datasets):
        ds = provider._ds_trn.get_dataset(ds_idx)
        id_to_idx[str(ds.get("dataset_id", str(ds_idx)))] = ds_idx
    if trace_id in id_to_idx:
        return id_to_idx[trace_id]
    if provider.num_datasets == 1:
        # Providers such as QAD/TSB-AD-M were already restricted to this trace.
        return 0
    try:
        idx = int(trace_id)
    except ValueError as exc:
        raise ValueError(
            f"Unknown trace {trace_id!r}; available dataset_ids={sorted(id_to_idx)}"
        ) from exc
    if idx < 0 or idx >= provider.num_datasets:
        raise ValueError(f"Trace index {idx} out of range [0, {provider.num_datasets - 1}]")
    return idx


def _make_loaders(provider, ds_idx: int, args):
    has_hybrid_layout = all(
        hasattr(provider, attr) for attr in ["num_datasets", "input_dims", "num_timepoints_list"]
    ) and all(hasattr(provider, attr) for attr in ["_ds_trn", "_ds_tst", "_ds_val"])
    num_workers = min(8, max(0, int(getattr(args, "num_max_cpu_worker", 8))))
    common = dict(batch_size=args.batch_size, collate_fn=None, num_workers=num_workers, pin_memory=True)

    if has_hybrid_layout:
        trn = DatasetSlice(provider._ds_trn, ds_idx)
        tst = DatasetSlice(provider._ds_tst, ds_idx)
        val = DatasetSlice(provider._ds_val, ds_idx)
        dl_trn = DataLoader(trn, shuffle=False, drop_last=False, **common)
        dl_tst = DataLoader(tst, shuffle=False, **common)
        dl_val = DataLoader(val, shuffle=False, **common)
        return dl_trn, dl_tst, dl_val, provider.input_dims[ds_idx], provider.num_timepoints_list[ds_idx]

    dl_trn = provider.get_train_loader(shuffle=False, drop_last=False, **common)
    dl_tst = provider.get_test_loader(shuffle=False, **common)
    dl_val = provider.get_val_loader(shuffle=False, **common)
    return dl_trn, dl_tst, dl_val, provider.input_dim, provider.num_timepoints


def _restore_modules(args, ckpt: dict, input_dim: int, num_timepoints: int, device: str):
    desired_t = ckpt.get("desired_t")
    if desired_t is None:
        desired_t = torch.linspace(0, 1.00, num_timepoints, device=device).float()
    else:
        desired_t = desired_t.to(device).float()
    modules, _, _, elbo_loss = build_modules_and_optim(args, input_dim, desired_t)
    try:
        modules.load_state_dict(ckpt["modules"], strict=True)
    except RuntimeError as exc:
        ckpt_input_dim = ckpt["modules"].get("recon_net.map.2.bias", torch.empty(0)).numel() or "?"
        raise RuntimeError(
            f"Checkpoint weights do not fit the model built from its own args "
            f"(checkpoint output dim={ckpt_input_dim}, provider input_dim={input_dim}). "
            "The data pipeline has most likely changed since the checkpoint was written."
        ) from exc
    modules.to(device)
    modules.eval()
    return modules, elbo_loss, desired_t


def evaluate_checkpoint(args, ckpt: dict, provider, ds_idx: int, skip_val: bool) -> dict:
    """Mirror of the evaluation block inside ``train_one_dataset`` (no training)."""
    device = args.device
    dl_trn, dl_tst, dl_val, input_dim, num_timepoints = _make_loaders(provider, ds_idx, args)
    modules, elbo_loss, desired_t = _restore_modules(args, ckpt, input_dim, num_timepoints, device)

    normalization_scores = None
    if args.normalize_score:
        normalization_scores = calculate_z_normalization_values(args, dl_trn, modules, desired_t, device)

    feature_weights = None
    if args.score_aggregation in ("weighted-mse", "weighted-mse-exp"):
        weighting = "exp-inverse" if args.score_aggregation == "weighted-mse-exp" else "inverse"
        weight_stats = calculate_feature_reconstruction_weights(
            args, dl_trn, modules, desired_t, device, weighting=weighting)
        feature_weights = weight_stats["feature_weights"]

    tst_stats = evaluate(
        args, dl_tst, modules, elbo_loss, desired_t, device,
        normalization_stats=normalization_scores, feature_weights=feature_weights,
    )
    val_stats = None
    if not skip_val:
        val_stats = evaluate(
            args, dl_val, modules, elbo_loss, desired_t, device,
            normalization_stats=normalization_scores, test=False, feature_weights=feature_weights,
        )
    return {"tst": _to_python(tst_stats), "val": _to_python(val_stats) if val_stats is not None else None}


def _to_python(stats):
    if stats is None:
        return None
    out = {}
    for key, value in stats.items():
        if isinstance(value, (np.floating, np.integer)):
            out[key] = value.item()
        elif isinstance(value, torch.Tensor):
            out[key] = value.item() if value.numel() == 1 else value.tolist()
        elif isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _mean_of_stats(stats_list: list[dict]) -> dict:
    values = defaultdict(list)
    for stats in stats_list:
        for key, value in stats.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                values[key].append(float(value))
    return {key: float(np.mean(vals)) for key, vals in values.items()}


def _macro_metrics(per_trace: dict) -> dict:
    """Macro average (unweighted mean over traces) of every numeric test metric.

    Same idea as ``anomaly_detection.compute_macro_metrics`` but covers all
    metric keys produced by ``eval_scores`` instead of a fixed subset.
    """
    macro = {f"macro_{key}": value for key, value in _mean_of_stats(list(per_trace.values())).items()
             if key != "num_checkpoints"}
    return macro


def _select_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested (%s) but not available; falling back to CPU.", requested)
        return "cpu"
    return requested


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

    run_id = datetime.datetime.now().strftime("%y%m%d-%H:%M:%S")
    out_dir = Path(cli_args.eval_out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_up_logging(
        console_log_level=cli_args.loglevel,
        console_log_color=True,
        logfile_file=str(out_dir / f"EVAL_{cli_args.dataset}_{run_id}.txt"),
        logfile_log_level=cli_args.loglevel,
        logfile_log_color=False,
        log_line_template="%(color_on)s[%(asctime)s] [%(levelname)-8s] %(message)s%(color_off)s",
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
        "Evaluating %d checkpoint(s) for %s from %s (traces=%s)",
        len(candidates), cli_args.dataset, checkpoint_dir,
        sorted({c["trace_id"] for c in candidates}, key=lambda t: (len(str(t)), str(t))),
    )

    # ---- evaluate -----------------------------------------------------------
    provider_cache: dict[tuple, object] = {}
    results = []
    failures = []
    for cand in candidates:
        path, ckpt, trace_id = cand["path"], cand["ckpt"], cand["trace_id"]
        args = resolve_effective_args(cand["ckpt_args"], cli_args, explicit)
        args.device = _select_device(args.device)
        args.trace_ids = [trace_id] if trace_id is not None else None

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
            stats = evaluate_checkpoint(args, ckpt, provider, ds_idx, skip_val=cli_args.skip_val)
        except RuntimeError as exc:
            logging.error("Skipping %s: %s", path.name, exc)
            failures.append({"checkpoint": str(path), "trace_id": trace_id, "error": str(exc)})
            continue
        logging.info("[%s] %s: %s", trace_id, path.name, stats["tst"])
        results.append({
            "checkpoint": str(path),
            "trace_id": trace_id,
            "dataset": args.dataset,
            "score_aggregation": args.score_aggregation,
            "score_smoothing_window": args.score_smoothing_window,
            "normalize_score": args.normalize_score,
            "mc_eval_samples": args.mc_eval_samples,
            **stats,
        })

    # ---- aggregate ----------------------------------------------------------
    by_trace = defaultdict(list)
    for res in results:
        by_trace[res["trace_id"]].append(res["tst"])
    if not results:
        raise SystemExit(f"All {len(failures)} checkpoint(s) failed to evaluate; see log above.")
    per_trace = {str(trace): _mean_of_stats(stats_list) for trace, stats_list in by_trace.items()}
    for trace, stats_list in by_trace.items():
        per_trace[str(trace)]["num_checkpoints"] = len(stats_list)
    macro = _macro_metrics(per_trace)

    report = {
        "run_id": run_id,
        "dataset": cli_args.dataset,
        "checkpoint_dir": str(checkpoint_dir),
        "checkpoint_pattern": cli_args.checkpoint_pattern,
        "cli_overrides": {key: getattr(cli_args, key) for key in sorted(explicit)},
        "per_checkpoint": results,
        "failed_checkpoints": failures,
        "per_trace": per_trace,
        **macro,
    }
    report_path = out_dir / f"EVAL_{cli_args.dataset}_{run_id}.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    table_cols = ["f1", "precision", "recall", "auc_roc", "auc_pr", "vus_roc", "vus_pr", "loss", "num_checkpoints"]
    per_trace_df = pd.DataFrame.from_dict(per_trace, orient="index")
    per_trace_df = per_trace_df[[c for c in table_cols if c in per_trace_df.columns]]
    per_trace_df.index.name = "trace"
    logging.info("Per-trace test metrics:\n%s", per_trace_df.to_string(float_format=lambda v: f"{v:.4f}"))
    logging.info("Macro metrics over %d trace(s): %s", len(per_trace), macro)
    if failures:
        logging.warning("%d checkpoint(s) could not be evaluated: %s", len(failures), [f["checkpoint"] for f in failures])
    logging.info("Report written to %s", report_path)

    if cli_args.final_metrics_csv:
        for trace, stats in per_trace.items():
            append_final_metrics_csv(
                csv_path=cli_args.final_metrics_csv,
                benchmark=f"{cli_args.dataset}:{trace}",
                run_datetime=run_id,
                metrics=stats,
            )
        append_final_metrics_csv(
            csv_path=cli_args.final_metrics_csv,
            benchmark=f"{cli_args.dataset}:macro",
            run_datetime=run_id,
            metrics={**macro, "num_traces": len(per_trace)},
        )
        logging.info("Appended per-trace and macro rows to %s", cli_args.final_metrics_csv)


if __name__ == "__main__":
    main()
