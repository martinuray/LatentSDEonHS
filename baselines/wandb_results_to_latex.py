#!/usr/bin/env python3
r"""Pull baseline.py results straight from W&B and format them as LaTeX tables.

Each W&B run logged by ``baselines/baseline.py`` covers exactly one
(benchmark, classifier, seed) combination and stores its dataset-averaged
metrics under summary keys ``summary/macro/<benchmark>/<clf_name>/<metric>``
(see ``_wandb_summary_from_dataframe`` / ``_wandb_log_final_outputs`` in
baseline.py). This script fetches all runs of a project, groups those
per-run macro values by (benchmark, classifier), and aggregates them as
``mean +/- std`` (x100) across runs/seeds -- the same aggregation
baseline.py itself performs in ``aggregate_mean_std``.

Before aggregating, runs are filtered per (benchmark, classifier) to the most
recent DEFAULT_RUN_LIMIT (5) runs -- or just the single most recent run for
deterministic classifiers (COPOD, OCSVM, KNN, PCA, LOF), which produce
identical results across seeds so repeats add no information.

Three per-benchmark-group LaTeX tables are produced:
  - "water/server" table: SWaT, WaDi, PSM
  - "NASA/SMD" table: SMAP, MSL, SMD
  - "QAD" table: QAD alone (shown as "QAPPD (10)"), a single-benchmark version
    of the same layout
Each benchmark contributes three columns, in order: AUC, AUPRC, F1. Unlike
the other benchmarks, QAD's runs are never filtered by --after (they're
comparatively rare) -- the "5 most recent runs" recency cap still applies.

A third row group, "NeuralODE", is fetched from the separate W&B project used
by anomaly_detection.py / anomaly_detection_ode.py (default:
https://wandb.ai/martin-uray-salzburg-university-of-applied-sciences/latent-sde-on-hs-anomaly-detection-ode).
Those runs log a single model (no `clf_name`/`selection.benchmarks` config),
so its results are read differently: from `run.config.run_context.benchmark_name`
and, per run, either the top-level `summary/auc_roc|auc_pr|f1` (single-dataset
benchmarks) or the mean of `summary/per_dataset/<id>/auc_roc|auc_pr|f1`
across sub-datasets (multi-dataset benchmarks: SMAP, MSL, SMD) -- that
per-run mean plays the same role as baseline.py's `summary/macro/...` value.

A fourth row group, "our contribution", is fetched the same way but from
https://wandb.ai/martin-uray-salzburg-university-of-applied-sciences/latent-sde-on-hs-anomaly-detection
(the main anomaly_detection.py project). It splits into two rows by
`run_context.model_variant`: "LSD on $\mathbb{R}^n$ (ours)" (Rn, no sphere
embedding) and "LSD on $\mathbb{S}^n$ (ours)" (Sn, sphere embedding). Runs
restricted to a subset of traces via --trace-ids (benchmark_name containing
":") are excluded -- only full-benchmark runs, whose logged metrics are
already the mean over every trace, are used. Runs with
`args.fixed_subsample_mask` set are excluded as well: those belong to the QAD
sparsity sweep (eval_sparsity_data.py forces that flag on) and are a different
experiment, reported only in the sparsity table below. The same filter is
applied to the NeuralODE project, so both sides of every comparison are
restricted to the resampled-mask setting.

Cells are rendered as ``mean \std{std}`` (the ``\std`` macro must be defined
in the LaTeX preamble, e.g. ``\newcommand{\std}[1]{$\pm$#1}``). Within each
column, the 1st/2nd/3rd best values are wrapped in
``\cellcolor{first|second|third}{...}``; these three color names must also be
defined in the preamble, e.g. ``\colorlet{first}{yellow!60}``.

Each table also gets a leading "Avg. Rank" column: the mean, over that
table's columns, of each classifier's per-column rank (lower is better).
Cells read ``\rankbox{3.99}``, or ``\rankbox[first|second|third]{1.20}`` for
the best three average ranks; ``\rankbox`` must be defined in the preamble,
e.g. ``\newcommand{\rankbox}[2][]{\ifstrempty{#1}{#2}{\cellcolor{#1}{#2}}}``.

A fifth table compares the two benchmark groups' "Avg. Rank" columns
directly: one row per classifier with Overall/Single Trace/Multi Trace
columns (Overall is the plain mean of the other two), sorted ascending by
Overall, with each column's top 3 values highlighted via bare
``\cellcolor{first|second|third}{...}`` (no ``\rankbox`` needed there).

A sixth, appendix-only table covers QAD runs with a decimation factor of 1
(vs. the main QAD table's 10) -- an independent ablation, not mixed into any
other table's aggregation. It has no "Avg. Rank" column (so no rank-based
row sorting: rows keep CLASSIFIER_ORDER's fixed grouping) but still
highlights each column's top 3 the same way as the other tables.

A seventh table isolates the NeuralODE-vs-LSD comparison and transposes the
layout: one row per benchmark (all of them in a single table) and the models
as column groups, with NeuralODE on the left and our two LSD variants to its
right. Only the best model per (benchmark, metric) is highlighted, via a bare
``\cellcolor{first}{...}``; there is no "Avg. Rank" column.

An eighth table covers the sparsity sweep, fetched from yet another project
(default:
https://wandb.ai/martin-uray-salzburg-university-of-applied-sciences/latent-sde-on-hs-sparsity-baselines,
logged by baselines/eval_sparsity_baselines.py). It is QAD-only: each column
group is one subsample level (1% / 5% of the original training data kept) and
each row is one (classifier, interpolation method) pair -- "COPOD (linear)",
"COPOD (spline)", ... -- since every classifier was run under both a linear
and a spline interpolation of the burst-masked gaps. Our two LSD rows come
from the "ours" project instead (eval_sparsity_data.py logs there), and are
taken from exactly the runs the other tables exclude: QAD runs with
`args.fixed_subsample_mask` set and `args.subsample` at one of the sweep's
levels. They carry no interpolation variant -- the model consumes the sparse
series directly -- so they get one row each.

A console-only table reports how many (post-filtering) W&B runs were
found per (benchmark, classifier) configuration.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path
from typing import TypeVar

import numpy as np
import pandas as pd

try:
    import wandb
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The 'wandb' package is required to fetch results (pip install wandb)."
    ) from exc

ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT_DIR / "out" / "doc"

METRICS = ["auc_roc", "auc_pr", "f1"]
METRIC_LABELS = {"auc_roc": "AUC", "auc_pr": "AUPRC", "f1": "F1"}

TABLE_GROUPS = [
    ("SWaT_WaDi_PSM", ["SWaT", "WaDi", "PSM"]),
    ("SMAP_MSL_SMD", ["SMAP", "MSL", "SMD"]),
]
QAD_BENCHMARK = "QAD"
# QAD gets its own single-benchmark table (see main()); unlike the other
# benchmarks it's never filtered by --after (its runs are comparatively rare,
# so we always want all of them -- see select_recent_runs for the "5 most
# recent" cap that still applies).
QAD_TABLE_GROUP = ("QAD", [QAD_BENCHMARK])
# Only runs whose QAD decimation factor matches this are included in the main
# QAD table (baseline.py logs config.benchmark_window_settings.QAD.decimation;
# anomaly_detection.py/anomaly_detection_ode.py log config.args.data_decimation_factor
# instead); other decimations aren't comparable and would skew the mean.
QAD_REQUIRED_DECIMATION = 10
# A second, independent decimation=1 (no decimation / raw 100 Hz) QAD ablation
# goes into a separate appendix-only table (see build_appendix_qad_table);
# bucketed under this synthetic key so it never mixes with QAD_BENCHMARK's
# decimation=10 runs in `values`. It's not a real benchmark name, so it's
# deliberately not added to KNOWN_BENCHMARKS.
QAD_APPENDIX_DECIMATION = 1
QAD_DECIMATION1_BENCHMARK = "QAD (decimation=1)"
KNOWN_BENCHMARKS = {benchmark for _, benchmarks in TABLE_GROUPS for benchmark in benchmarks} | {QAD_BENCHMARK}

# These classifiers are deterministic (no randomness across seeds), so
# repeating them doesn't add information: only the most recent run is used.
DETERMINISTIC_CLASSIFIERS = {"COPOD", "OCSVM", "KNN", "PCA", "LOF"}
DEFAULT_RUN_LIMIT = 5
DETERMINISTIC_RUN_LIMIT = 1
# Non-deterministic classifiers are expected to have one run per seed in this
# range (5 seeds total); see report_missing_runs.
EXPECTED_SEEDS = list(range(42, 47))

# Classifier row groups, each separated by a booktabs \midrule: "shallow"
# (classical, non-deep) methods, the deep-learning methods, the NeuralODE
# (latent-SDE/ODE) model, and finally our own contribution.
CLASSIFIER_GROUP_SHALLOW = ["KNN", "PCA", "OCSVM", "COPOD", "LOF", "IForest"]
CLASSIFIER_GROUP_DEEP = ["TcnED", "USAD", "DeepIF", "TranAD", "TimesNet", "COUTA", "DeepSVDD", "AnomalyTransformer"]
NEURALODE_CLASSIFIER = "NeuralODE"
CLASSIFIER_GROUP_NEURALODE = [NEURALODE_CLASSIFIER]
LSD_RN_LABEL = r"LSD on $\mathbb{R}^n$ (ours)"
LSD_SN_LABEL = r"LSD on $\mathbb{S}^n$ (ours)"
CLASSIFIER_GROUP_OURS = [LSD_RN_LABEL, LSD_SN_LABEL]
NAMED_CLASSIFIER_GROUPS = [CLASSIFIER_GROUP_SHALLOW, CLASSIFIER_GROUP_DEEP, CLASSIFIER_GROUP_NEURALODE, CLASSIFIER_GROUP_OURS]
CLASSIFIER_ORDER = [clf for group in NAMED_CLASSIFIER_GROUPS for clf in group]

# The NeuralODE-vs-LSD table (see build_ode_vs_lsd_table) transposes the other
# tables' layout: one row per benchmark, and the models as column groups --
# NeuralODE first, our two LSD variants to its right. Row groups mirror
# TABLE_GROUPS (single-trace / multi-trace) plus QAD, so the \midrule
# structure matches the per-group tables. QAD_DECIMATION1_BENCHMARK stays out:
# it's a decimation ablation, not a benchmark of its own.
ODE_VS_LSD_MODELS = [NEURALODE_CLASSIFIER, LSD_RN_LABEL, LSD_SN_LABEL]
ODE_VS_LSD_BENCHMARK_GROUPS = [benchmarks for _, benchmarks in TABLE_GROUPS] + [[QAD_BENCHMARK]]

# clf_name -> shorter display label for the LaTeX "Method"/"Model" column;
# classifiers not listed here are shown under their plain name. Internal
# matching (wandb clf_name, DETERMINISTIC_CLASSIFIERS, etc.) always keeps
# using the full name -- only table rendering uses the abbreviation.
MODEL_DISPLAY_LABELS = {"AnomalyTransformer": "AnomalyTrans."}


def display_label(clf_name: str) -> str:
    return MODEL_DISPLAY_LABELS.get(clf_name, clf_name)

# Separate W&B projects the NeuralODE and "ours" (anomaly_detection.py /
# anomaly_detection_ode.py) runs are logged to -- distinct from the PYOD
# baselines project above.
NEURALODE_DEFAULT_PROJECT = "latent-sde-on-hs-anomaly-detection-ode"
NEURALODE_DEFAULT_ENTITY = "martin-uray-salzburg-university-of-applied-sciences"
OURS_DEFAULT_PROJECT = "latent-sde-on-hs-anomaly-detection"
OURS_DEFAULT_ENTITY = "martin-uray-salzburg-university-of-applied-sciences"
# run_context.model_variant -> row label, for the "ours" project.
OURS_VARIANT_LABELS = {"Rn": LSD_RN_LABEL, "Sn": LSD_SN_LABEL}

# ---------------------------------------------------------------------------
# Sparsity sweep (baselines/eval_sparsity_baselines.py) -- its own W&B project.
# ---------------------------------------------------------------------------
SPARSITY_DEFAULT_PROJECT = "latent-sde-on-hs-sparsity-baselines"
SPARSITY_DEFAULT_ENTITY = "martin-uray-salzburg-university-of-applied-sciences"
# Fraction of the original training data still available; one column group each.
SPARSITY_SUBSAMPLES = [0.01, 0.05]
# Every classifier was run under both interpolation strategies for the
# burst-masked gaps, giving two rows ("<clf> (linear)" / "<clf> (spline)").
SPARSITY_INTERP_METHODS = ["linear", "spline"]
# Row groups, mirroring the main tables' shallow / deep / ours split. This is a
# deliberate subset of CLASSIFIER_ORDER -- only these were run under sparsity.
SPARSITY_GROUP_SHALLOW = ["COPOD", "IForest", "KNN", "LOF", "OCSVM"]
SPARSITY_GROUP_DEEP = ["DeepIF", "COUTA", "USAD", "DeepSVDD"]
SPARSITY_GROUP_OURS = [LSD_SN_LABEL, LSD_RN_LABEL]
SPARSITY_CLASSIFIER_GROUPS = [SPARSITY_GROUP_SHALLOW, SPARSITY_GROUP_DEEP, SPARSITY_GROUP_OURS]
# Our own LSD sparsity runs come from the "ours" project (eval_sparsity_data.py)
# and have no interpolation variant -- the model handles the sparse series
# natively -- so their rows use this sentinel as the interp part of the key.
SPARSITY_OURS_ROWS = set(SPARSITY_GROUP_OURS)
OURS_SPARSITY_INTERP = None
# Unlike the main tables, DETERMINISTIC_CLASSIFIERS does *not* apply here: the
# burst mask is drawn from the run seed, so even COPOD/KNN/LOF/OCSVM vary
# across seeds and all DEFAULT_RUN_LIMIT runs carry information.

# Matches both "summary/per_dataset/<id>/<metric>" (from _wandb_summary_from_dataframe)
# and "summary/per_dataset.<id>.<metric>" (from _flatten_numeric_metrics), the two
# equivalent key spellings anomaly_detection.py's _wandb_log_final_outputs logs.
PER_DATASET_KEY_RE = re.compile(
    r"^summary/per_dataset[./](?P<dataset_id>.+?)[./](?P<metric>" + "|".join(METRICS) + r")$"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch baseline.py W&B runs and render mean+-std LaTeX result tables."
    )
    parser.add_argument("--project", type=str, default="latent-sde-on-hs-baselines", help="W&B project name (PYOD baselines).")
    parser.add_argument("--entity", type=str, default=None, help="Optional W&B entity/team for --project (defaults to your W&B default entity).")
    parser.add_argument(
        "--ode-project", type=str, default=NEURALODE_DEFAULT_PROJECT,
        help="W&B project name for the NeuralODE model runs.",
    )
    parser.add_argument(
        "--ode-entity", type=str, default=NEURALODE_DEFAULT_ENTITY,
        help="W&B entity/team for --ode-project.",
    )
    parser.add_argument(
        "--skip-neuralode", action="store_true",
        help="Don't fetch/include the NeuralODE row group.",
    )
    parser.add_argument(
        "--ours-project", type=str, default=OURS_DEFAULT_PROJECT,
        help="W&B project name for our contribution's (LSD) model runs.",
    )
    parser.add_argument(
        "--ours-entity", type=str, default=OURS_DEFAULT_ENTITY,
        help="W&B entity/team for --ours-project.",
    )
    parser.add_argument(
        "--skip-ours", action="store_true",
        help="Don't fetch/include the 'ours' (LSD Rn/Sn) row group.",
    )
    parser.add_argument(
        "--sparsity-project", type=str, default=SPARSITY_DEFAULT_PROJECT,
        help="W&B project name for the eval_sparsity_baselines.py sweep.",
    )
    parser.add_argument(
        "--sparsity-entity", type=str, default=SPARSITY_DEFAULT_ENTITY,
        help="W&B entity/team for --sparsity-project.",
    )
    parser.add_argument(
        "--skip-sparsity", action="store_true",
        help="Don't fetch/render the QAD sparsity table.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory to write the .tex files into.")
    return parser.parse_args()


RunEntry = tuple[pd.Timestamp, "int | None", dict[str, float]]
# Most tables bucket runs by (benchmark, classifier); the sparsity table uses a
# (subsample, classifier, interp_method) key instead. dedupe_by_seed treats the
# key as opaque, so it works for either.
RecordKey = TypeVar("RecordKey", bound=tuple)


def fetch_run_records(
    project: str, entity: str | None, min_created_at: pd.Timestamp | None = None
) -> dict[tuple[str, str], list[RunEntry]]:
    """Return {(benchmark, clf_name): [(created_at, seed, {metric: value}), ...]}, one entry per W&B run.

    Runs created before ``min_created_at`` (if given) are omitted entirely.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, order="-created_at")

    records: dict[tuple[str, str], list[RunEntry]] = defaultdict(list)

    for run in runs:
        config = run.config or {}
        run_context = config.get("run_context", {})
        clf_name = run_context.get("clf_name")
        benchmarks = config.get("selection", {}).get("benchmarks", [])
        if not clf_name or not benchmarks:
            continue
        benchmark = benchmarks[0]
        if benchmark not in KNOWN_BENCHMARKS:
            continue

        # QAD runs split into two independent buckets by decimation factor:
        # QAD_BENCHMARK (decimation=10, the main table) and
        # QAD_DECIMATION1_BENCHMARK (decimation=1, appendix-only ablation).
        # Any other decimation value is dropped as not comparable to either.
        record_benchmark = benchmark
        if benchmark == QAD_BENCHMARK:
            qad_decimation = config.get("benchmark_window_settings", {}).get(QAD_BENCHMARK, {}).get("decimation")
            if qad_decimation == QAD_REQUIRED_DECIMATION:
                record_benchmark = QAD_BENCHMARK
            elif qad_decimation == QAD_APPENDIX_DECIMATION:
                record_benchmark = QAD_DECIMATION1_BENCHMARK
            else:
                continue

        created_at = pd.to_datetime(run.created_at, utc=True)
        # QAD runs are comparatively rare, so --after never filters them out.
        if benchmark != QAD_BENCHMARK and min_created_at is not None and created_at < min_created_at:
            continue

        seed = run_context.get("run_seed")

        summary = run.summary
        prefix = f"summary/macro/{benchmark}/{clf_name}/"
        metrics = {}
        for metric in METRICS:
            value = summary.get(f"{prefix}{metric}")
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[metric] = float(value)
        if not metrics:
            continue  # run failed or never produced a successful macro row

        records[(record_benchmark, clf_name)].append((created_at, seed, metrics))

    return records


def _extract_ad_summary_metrics(summary: dict) -> dict[str, float]:
    """Read auc_roc/auc_pr/f1 from an anomaly_detection.py-style run summary.

    Single-dataset benchmarks (SWaT, WaDi, PSM) log them directly under
    ``summary/<metric>``. Multi-dataset benchmarks (SMAP, MSL, SMD) never log
    a ready-made macro auc_roc/auc_pr (only macro_f1 -- see
    compute_macro_metrics), so those are instead averaged from the
    per-trace ``summary/per_dataset/<id>/<metric>`` entries.
    """
    metrics = {}
    for metric in METRICS:
        value = summary.get(f"summary/{metric}")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[metric] = float(value)

    if not metrics:
        per_dataset_values: dict[str, list[float]] = defaultdict(list)
        for key, value in summary.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            match = PER_DATASET_KEY_RE.match(key)
            if match:
                per_dataset_values[match.group("metric")].append(float(value))
        metrics = {metric: float(np.mean(vals)) for metric, vals in per_dataset_values.items() if vals}

    return metrics


def uses_fixed_subsample_mask(config: dict) -> bool:
    """Whether an anomaly_detection.py-style run had --fixed-subsample-mask set.

    Runs predating the flag have no such entry; the argparse default is False,
    so a missing key means "resampled every iteration". Only the QAD sparsity
    sweep (eval_sparsity_data.py) turns it on, which is what separates those
    runs from the ones every other table reports on.
    """
    return bool((config.get("args") or {}).get("fixed_subsample_mask", False))


def fetch_anomaly_detection_run_records(
    project: str,
    entity: str | None,
    clf_name_fn,
    min_created_at: pd.Timestamp | None = None,
    require_full_trace: bool = False,
) -> dict[tuple[str, str], list[RunEntry]]:
    """Fetch runs logged via anomaly_detection.py's shared W&B helpers.

    Used for both the NeuralODE project and the "ours" (LSD) project: both
    log a single model per run (no `clf_name`/benchmark-list config) and
    identify themselves via ``run_context.benchmark_name`` /
    ``run_context.model_variant``. `clf_name_fn(run_context)` maps a run's
    context to the row/classifier label to bucket it under (return a falsy
    value to skip the run). If `require_full_trace` is set, runs restricted
    to a subset of traces via --trace-ids (benchmark_name containing ":")
    are skipped -- their logged metrics wouldn't be a mean over every trace.

    Sparsity-sweep runs (``args.fixed_subsample_mask`` set) are always
    skipped: they train on a fixed, heavily subsampled mask and belong to the
    sparsity table only -- see fetch_ours_sparsity_run_records.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, order="-created_at")

    records: dict[tuple[str, str], list[RunEntry]] = defaultdict(list)

    for run in runs:
        config = run.config or {}
        run_context = config.get("run_context", {})
        raw_benchmark = run_context.get("benchmark_name")
        if not raw_benchmark:
            continue

        # Sparsity-sweep runs are reported by the sparsity table alone.
        if uses_fixed_subsample_mask(config):
            continue

        # Runs restricted to specific --trace-ids are tagged "DATASET:ids".
        if require_full_trace and ":" in raw_benchmark:
            continue
        benchmark = raw_benchmark.split(":", 1)[0]
        if benchmark not in KNOWN_BENCHMARKS:
            continue

        # QAD runs split into two independent buckets by decimation factor
        # (see the matching comment in fetch_run_records). anomaly_detection.py/
        # anomaly_detection_ode.py log this flat under args (only the QAD
        # provider honours it) rather than nested like baseline.py does.
        record_benchmark = benchmark
        if benchmark == QAD_BENCHMARK:
            qad_decimation = config.get("args", {}).get("data_decimation_factor")
            if qad_decimation == QAD_REQUIRED_DECIMATION:
                record_benchmark = QAD_BENCHMARK
            elif qad_decimation == QAD_APPENDIX_DECIMATION:
                record_benchmark = QAD_DECIMATION1_BENCHMARK
            else:
                continue

        clf_name = clf_name_fn(run_context)
        if not clf_name:
            continue

        created_at = pd.to_datetime(run.created_at, utc=True)
        # QAD runs are comparatively rare, so --after never filters them out.
        if benchmark != QAD_BENCHMARK and min_created_at is not None and created_at < min_created_at:
            continue

        seed = run_context.get("run_seed")
        metrics = _extract_ad_summary_metrics(dict(run.summary))
        if not metrics:
            continue  # run failed or never produced usable metrics

        records[(record_benchmark, clf_name)].append((created_at, seed, metrics))

    return records


def fetch_neuralode_run_records(
    project: str, entity: str | None, min_created_at: pd.Timestamp | None = None
) -> dict[tuple[str, str], list[RunEntry]]:
    """Fetch NeuralODE runs; all of them are bucketed under one row/classifier."""
    return fetch_anomaly_detection_run_records(
        project, entity,
        clf_name_fn=lambda run_context: NEURALODE_CLASSIFIER,
        min_created_at=min_created_at,
        require_full_trace=False,
    )


def fetch_ours_run_records(
    project: str, entity: str | None, min_created_at: pd.Timestamp | None = None
) -> dict[tuple[str, str], list[RunEntry]]:
    """Fetch our own LSD runs, split into Rn/Sn rows by run_context.model_variant.

    Only full-benchmark runs are used (require_full_trace=True): a run
    restricted to a subset of traces via --trace-ids wouldn't have metrics
    averaged over every trace, and would silently skew the aggregate.
    """
    return fetch_anomaly_detection_run_records(
        project, entity,
        clf_name_fn=lambda run_context: OURS_VARIANT_LABELS.get(run_context.get("model_variant")),
        min_created_at=min_created_at,
        require_full_trace=True,
    )


# (subsample, clf_name, interp_method); interp is OURS_SPARSITY_INTERP (None)
# for our own LSD rows, which have no interpolation variant.
SparsityKey = tuple[float, str, "str | None"]


def fetch_sparsity_run_records(
    project: str, entity: str | None
) -> dict[SparsityKey, list[RunEntry]]:
    """Fetch eval_sparsity_baselines.py runs, keyed by (subsample, clf, interp).

    One W&B run there covers a single (seed, subsample) task but evaluates
    *several* classifiers at once (config.selection.classifiers), so each run
    contributes one entry per classifier. Its per-classifier, dataset-averaged
    metrics live under ``summary/<clf_name>/<metric>`` (written by
    ``_wandb_log_task_summary``) -- the same role baseline.py's
    ``summary/macro/<benchmark>/<clf_name>/<metric>`` plays elsewhere.

    Only QAD runs at QAD_REQUIRED_DECIMATION are kept, matching the main QAD
    table. Runs that are still in flight simply have no ``summary/...`` keys
    yet and drop out on their own.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, order="-created_at")

    records: dict[SparsityKey, list[RunEntry]] = defaultdict(list)

    for run in runs:
        config = run.config or {}
        selection = config.get("selection", {}) or {}
        task = config.get("task", {}) or {}
        if selection.get("benchmark") != QAD_BENCHMARK:
            continue
        if (config.get("window_settings") or {}).get("decimation") != QAD_REQUIRED_DECIMATION:
            continue

        interp = config.get("interp_method")
        subsample = task.get("subsample")
        if interp not in SPARSITY_INTERP_METHODS or subsample not in SPARSITY_SUBSAMPLES:
            continue

        created_at = pd.to_datetime(run.created_at, utc=True)
        seed = task.get("run_seed")
        summary = run.summary

        for clf_name in selection.get("classifiers", []):
            metrics = {}
            for metric in METRICS:
                value = summary.get(f"summary/{clf_name}/{metric}")
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    metrics[metric] = float(value)
            if not metrics:
                continue  # classifier failed, or the run hasn't finished yet
            records[(float(subsample), clf_name, interp)].append((created_at, seed, metrics))

    return records


def fetch_ours_sparsity_run_records(
    project: str, entity: str | None
) -> dict[SparsityKey, list[RunEntry]]:
    """Fetch our own LSD sparsity runs (eval_sparsity_data.py) for the sparsity table.

    These live in the same project as the main "ours" runs, not in the
    baselines' sparsity project, and are exactly the runs
    fetch_ours_run_records skips: QAD at QAD_REQUIRED_DECIMATION with
    ``args.fixed_subsample_mask`` set -- a mask drawn once at load time, which
    is what makes a run a sparsity experiment rather than a main-table one.
    The subsample level comes from ``args.subsample``; levels outside
    SPARSITY_SUBSAMPLES (the table's column groups) are dropped.

    Keys use OURS_SPARSITY_INTERP for the interpolation slot: our model takes
    the subsampled series as-is, with no gap interpolation to vary.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, order="-created_at")

    records: dict[SparsityKey, list[RunEntry]] = defaultdict(list)

    for run in runs:
        config = run.config or {}
        run_context = config.get("run_context", {}) or {}
        run_args = config.get("args", {}) or {}

        if not uses_fixed_subsample_mask(config):
            continue

        raw_benchmark = run_context.get("benchmark_name")
        # As in fetch_ours_run_records, --trace-ids subsets ("DATASET:ids") are
        # not averaged over every trace and must not be mixed in.
        if not raw_benchmark or ":" in raw_benchmark:
            continue
        if raw_benchmark != QAD_BENCHMARK:
            continue
        if run_args.get("data_decimation_factor") != QAD_REQUIRED_DECIMATION:
            continue

        clf_name = OURS_VARIANT_LABELS.get(run_context.get("model_variant"))
        if not clf_name:
            continue

        subsample = run_args.get("subsample")
        if not isinstance(subsample, (int, float)) or isinstance(subsample, bool):
            continue
        if float(subsample) not in SPARSITY_SUBSAMPLES:
            continue

        metrics = _extract_ad_summary_metrics(dict(run.summary))
        if not metrics:
            continue  # run failed or never produced usable metrics

        created_at = pd.to_datetime(run.created_at, utc=True)
        seed = run_context.get("run_seed")
        records[(float(subsample), clf_name, OURS_SPARSITY_INTERP)].append((created_at, seed, metrics))

    return records


def select_recent_sparsity_runs(
    records: dict[SparsityKey, list[RunEntry]],
) -> dict[SparsityKey, dict[str, list[float]]]:
    """Keep the DEFAULT_RUN_LIMIT most recent runs per (subsample, clf, interp).

    DETERMINISTIC_CLASSIFIERS deliberately gets no special treatment here
    (unlike select_recent_runs): the burst mask is drawn from the run seed, so
    even COPOD/KNN/LOF/OCSVM differ from seed to seed under sparsity.
    """
    values: dict[SparsityKey, dict[str, list[float]]] = {}
    for key, entries in records.items():
        most_recent = sorted(entries, key=lambda item: item[0], reverse=True)[:DEFAULT_RUN_LIMIT]
        metric_values = {m: [] for m in METRICS}
        for _, _, metrics in most_recent:
            for metric, value in metrics.items():
                metric_values[metric].append(value)
        values[key] = metric_values
    return values


def dedupe_by_seed(records: dict[RecordKey, list[RunEntry]]) -> dict[RecordKey, list[RunEntry]]:
    """Keep at most one entry per seed within each (benchmark, classifier) group.

    If the same seed was run more than once, keep only the most recently
    created run. Entries without a resolvable seed are all kept (nothing to
    dedupe them against).
    """
    deduped: dict[tuple[str, str], list[RunEntry]] = {}
    for key, entries in records.items():
        best_by_seed: dict[object, RunEntry] = {}
        unseeded: list[RunEntry] = []
        for entry in entries:
            created_at, seed, _ = entry
            if seed is None:
                unseeded.append(entry)
                continue
            existing = best_by_seed.get(seed)
            if existing is None or created_at > existing[0]:
                best_by_seed[seed] = entry
        deduped[key] = list(best_by_seed.values()) + unseeded
    return deduped


def select_recent_runs(
    records: dict[tuple[str, str], list[RunEntry]],
) -> dict[tuple[str, str], dict[str, list[float]]]:
    """Keep only the N most recent runs per (benchmark, classifier).

    Deterministic classifiers (no randomness across seeds) keep only their
    single most recent run; all others keep the DEFAULT_RUN_LIMIT most
    recent runs.
    """
    values: dict[tuple[str, str], dict[str, list[float]]] = {}
    for (benchmark, clf_name), entries in records.items():
        limit = DETERMINISTIC_RUN_LIMIT if clf_name in DETERMINISTIC_CLASSIFIERS else DEFAULT_RUN_LIMIT
        most_recent = sorted(entries, key=lambda item: item[0], reverse=True)[:limit]

        metric_values = {m: [] for m in METRICS}
        for _, _, metrics in most_recent:
            for metric, value in metrics.items():
                metric_values[metric].append(value)
        values[(benchmark, clf_name)] = metric_values

    return values


def build_run_count_table(values: dict[tuple[str, str], dict[str, list[float]]]) -> pd.DataFrame:
    counts = {}
    for (benchmark, clf_name), metric_values in values.items():
        # A run counts if it contributed at least one metric value.
        num_runs = max((len(v) for v in metric_values.values()), default=0)
        counts.setdefault(benchmark, {})[clf_name] = num_runs

    count_df = pd.DataFrame(counts).fillna(0).astype(int)
    count_df = count_df.reindex(index=[c for c in CLASSIFIER_ORDER if c in count_df.index]
                                 + sorted(set(count_df.index) - set(CLASSIFIER_ORDER)))
    count_df.index.name = "classifier"
    return count_df


def _missing_sparsity_run_lines(
    sparsity_records: dict[SparsityKey, list[RunEntry]],
) -> list[str]:
    """Report lines for the QAD sparsity sweep's still-missing seeds.

    One entry per (subsample, classifier, interpolation) cell of the sparsity
    table. DETERMINISTIC_CLASSIFIERS gets no exemption here (unlike the
    benchmark report above): the burst mask is drawn from the run seed, so
    every cell needs all EXPECTED_SEEDS regardless of classifier.

    Our own LSD rows are reported only as present/absent: eval_sparsity_data.py
    seeds its runs by task index (0, 1, 2, ...), not from EXPECTED_SEEDS, so
    naming specific missing seeds would be meaningless. The per-cell run counts
    in build_sparsity_run_count_table cover those rows instead.
    """
    lines = []
    for subsample in SPARSITY_SUBSAMPLES:
        for group in sparsity_row_plan():
            for clf_name, interp, label in group:
                entries = sparsity_records.get((subsample, clf_name, interp), [])
                if clf_name in SPARSITY_OURS_ROWS:
                    if not entries:
                        lines.append(f"  QAD sparsity {subsample * 100:g}% / {label}: no run found")
                    continue
                seeds_present = {seed for _, seed, _ in entries if seed is not None}
                missing_seeds = [s for s in EXPECTED_SEEDS if s not in seeds_present]
                if missing_seeds:
                    lines.append(
                        f"  QAD sparsity {subsample * 100:g}% / {label}: missing seed(s) {missing_seeds}"
                    )
    return lines


def report_missing_runs(
    records: dict[tuple[str, str], list[RunEntry]],
    benchmarks: list[str],
    sparsity_records: dict[SparsityKey, list[RunEntry]] | None = None,
) -> None:
    """Print, for every (benchmark, classifier) pair, which of the 5 expected
    seeds (42-46) are still missing.

    Deterministic classifiers (DETERMINISTIC_CLASSIFIERS) only ever need a
    single run regardless of seed, so they're reported as missing only when
    no run at all was found for that (benchmark, classifier) pair. `records`
    should be the seed-deduped records dict (post dedupe_by_seed), i.e.
    before select_recent_runs discards anything beyond the recency limit --
    this reports on every run W&B has, not just the ones a table will use.

    `sparsity_records` (same post-dedupe form, but keyed by
    (subsample, classifier, interp)) appends the QAD sparsity sweep's 1% / 5%
    configurations to the same list -- see _missing_sparsity_run_lines.
    """
    missing_lines = []
    for benchmark in benchmarks:
        for clf_name in CLASSIFIER_ORDER:
            entries = records.get((benchmark, clf_name), [])
            if clf_name in DETERMINISTIC_CLASSIFIERS:
                if not entries:
                    missing_lines.append(f"  {benchmark} / {clf_name}: no run found")
                continue

            seeds_present = {seed for _, seed, _ in entries if seed is not None}
            missing_seeds = [s for s in EXPECTED_SEEDS if s not in seeds_present]
            if missing_seeds:
                missing_lines.append(
                    f"  {benchmark} / {clf_name}: missing seed(s) {missing_seeds}"
                )

    if sparsity_records is not None:
        missing_lines += _missing_sparsity_run_lines(sparsity_records)

    print(f"\nRuns still left to do (expected seeds {EXPECTED_SEEDS[0]}-{EXPECTED_SEEDS[-1]}):")
    print("\n".join(missing_lines) if missing_lines else "  none -- all expected runs found.")


# cellcolor names for the 1st/2nd/3rd best value in a column; must be
# defined in the LaTeX preamble, e.g. \colorlet{first}{yellow!60}.
RANK_COLORS = {1: "first", 2: "second", 3: "third"}


def mean_std(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    arr = np.asarray(vals, dtype=float) * 100.0
    return float(arr.mean()), float(arr.std(ddof=0))


def format_cell(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "--"
    return f"{mean:.2f} \\std{{{std:.2f}}}"


def highlight_top3(mean_table: pd.DataFrame, text_table: pd.DataFrame) -> pd.DataFrame:
    """In-place: wrap each populated cell in text_table for its column's top 3
    (by mean_table, higher = better; ties share a rank via "min" method) in
    ``\\cellcolor{color}{...}``, everything else in bare ``{...}`` ("--"
    placeholders stay unwrapped). Returns the computed per-column ranks.
    """
    ranks = mean_table.rank(axis=0, method="min", ascending=False)
    for col in mean_table.columns:
        for idx in mean_table.index:
            cell_text = text_table.loc[idx, col]
            if cell_text == "--":
                continue
            rank = ranks.loc[idx, col]
            color = RANK_COLORS.get(rank) if pd.notna(rank) else None
            if color is not None:
                text_table.loc[idx, col] = f"\\cellcolor{{{color}}}{{{cell_text}}}"
            else:
                text_table.loc[idx, col] = f"{{{cell_text}}}"
    return ranks


def highlight_best_per_metric(
    mean_table: pd.DataFrame, text_table: pd.DataFrame, models: list[str], metric_labels: list[str]
) -> None:
    """In-place row-wise counterpart of highlight_top3, for tables whose columns
    are (model, metric) and whose rows are benchmarks.

    For every (row, metric) the best-scoring model's cell is wrapped in
    ``\\cellcolor{first}{...}`` and the remaining populated cells in bare
    ``{...}``; "--" placeholders stay unwrapped. Only the winner is coloured
    (not the top 3 as in highlight_top3): with just a handful of model columns
    a top-3 highlight would colour essentially every cell.
    """
    for idx in mean_table.index:
        for metric_label in metric_labels:
            cols = [(model, metric_label) for model in models]
            row_values = mean_table.loc[idx, cols]
            best_col = row_values.idxmax() if row_values.notna().any() else None
            for col in cols:
                cell_text = text_table.loc[idx, col]
                if cell_text == "--":
                    continue
                if col == best_col:
                    text_table.loc[idx, col] = f"\\cellcolor{{{RANK_COLORS[1]}}}{{{cell_text}}}"
                else:
                    text_table.loc[idx, col] = f"{{{cell_text}}}"


def build_latex_table(
    values: dict[tuple[str, str], dict[str, list[float]]], benchmarks: list[str]
) -> tuple[pd.DataFrame, pd.Series]:
    """Return (formatted table, avg_rank series indexed by classifier name).

    avg_rank is the unsorted, unformatted mean-per-column rank computed here
    (same numbers the table's "Avg. Rank" column renders) -- exposed so
    callers can combine it across multiple benchmark groups (see
    build_comparison_table).
    """
    # Always list every known classifier (missing ones just show "--"), so the
    # shallow/deep row grouping and its \midrule stay at a fixed position.
    clf_names = sorted({clf for (_, clf) in values.keys()})
    ordered_clfs = CLASSIFIER_ORDER + sorted(set(clf_names) - set(CLASSIFIER_ORDER))

    columns = pd.MultiIndex.from_product([benchmarks, [METRIC_LABELS[m] for m in METRICS]])
    mean_table = pd.DataFrame(index=ordered_clfs, columns=columns, dtype=float)
    text_table = pd.DataFrame(index=ordered_clfs, columns=columns, dtype=object)
    mean_table.index.name = "classifier"
    text_table.index.name = "classifier"

    for clf_name in ordered_clfs:
        for benchmark in benchmarks:
            metric_values = values.get((benchmark, clf_name), {m: [] for m in METRICS})
            for metric in METRICS:
                col = (benchmark, METRIC_LABELS[metric])
                mean, std = mean_std(metric_values[metric])
                mean_table.loc[clf_name, col] = mean
                text_table.loc[clf_name, col] = format_cell(mean, std)

    # Higher is always better for AUC/AUPRC/F1: highlight each column's top 3.
    ranks = highlight_top3(mean_table, text_table)

    # Mean average rank across this table's columns (lower = better), with
    # the best/2nd/3rd average rank highlighted the same way as the cells.
    avg_rank = ranks.mean(axis=1, skipna=True)
    avg_rank_position = avg_rank.rank(method="min", ascending=True)
    rank_texts = {}
    for clf_name in ordered_clfs:
        value = avg_rank.get(clf_name)
        if pd.isna(value):
            rank_texts[clf_name] = "--"
            continue
        position = avg_rank_position.get(clf_name)
        color = RANK_COLORS.get(position) if pd.notna(position) else None
        if color is not None:
            rank_texts[clf_name] = f"\\rankbox[{color}]{{{value:.2f}}}"
        else:
            rank_texts[clf_name] = f"\\rankbox{{{value:.2f}}}"

    # Rank column first, then the method name (classifier is otherwise only
    # the DataFrame index, so surface it as an explicit leading column too).
    text_table.insert(0, ("", "Avg. Rank"), pd.Series(rank_texts))
    text_table.insert(1, ("", "Method"), pd.Series({c: display_label(c) for c in ordered_clfs}))

    # Sort rows ascending by average rank (lower = better) within each fixed
    # row group, so the shallow/deep block membership never changes, only
    # the ordering inside it.
    def _sort_key(clf_name: str) -> float:
        value = avg_rank.get(clf_name)
        return value if pd.notna(value) else float("inf")

    extras = sorted(set(clf_names) - set(CLASSIFIER_ORDER))
    row_groups = NAMED_CLASSIFIER_GROUPS + [extras]
    final_order = [clf for group in row_groups for clf in sorted(group, key=_sort_key)]

    return text_table.loc[final_order].fillna("--"), avg_rank


# Benchmark header labels; benchmarks not listed here (SWaT, WaDi, PSM) are
# shown under their plain name. The counts are each benchmark's number of
# sub-traces/machines/channels.
BENCHMARK_LABELS = {
    "SMAP": "SMAP (55)",
    "MSL": "MSL (27)",
    "SMD": "SMD (28)",
    "QAD": "QAPPD (10)",
}


def _cmidrule_spans(num_groups: int, first_metric_col: int = 3) -> str:
    """`\\cmidrule(lr){...}` under each benchmark's 3 metric columns, starting at `first_metric_col`."""
    spans = []
    start = first_metric_col
    for _ in range(num_groups):
        end = start + 2
        spans.append(f"\\cmidrule(lr){{{start}-{end}}}")
        start = end + 1
    return "".join(spans)


def build_table_header(benchmarks: list[str]) -> str:
    group_cells = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\small \\textbf{{{BENCHMARK_LABELS.get(b, b)}}}}}" for b in benchmarks
    )
    top_row = f"&& {group_cells} \\\\"
    cmidrules = _cmidrule_spans(len(benchmarks))
    metric_cells = ["\\textbf{Rank}\\big\\downarrow", "\\textbf{Model}"]
    for _ in benchmarks:
        metric_cells += ["\\textbf{AUC} \\big\\uparrow", "\\textbf{AUPRC} \\big\\uparrow", "\\textbf{F1} \\big\\uparrow"]
    metric_row = " & ".join(metric_cells) + r" \\"
    return "\n".join([top_row, cmidrules, metric_row])


def render_data_rows(table: pd.DataFrame) -> str:
    """Render each row as a LaTeX table row, inserting a \\midrule after each
    named classifier row-group boundary (see NAMED_CLASSIFIER_GROUPS)."""
    boundaries = set()
    running_total = 0
    for group in NAMED_CLASSIFIER_GROUPS[:-1]:
        running_total += len(group)
        boundaries.add(running_total)

    lines = []
    for i, row in enumerate(table.itertuples(index=False), start=1):
        lines.append(" & ".join(str(value) for value in row) + r" \\")
        if i in boundaries:
            lines.append(r"\midrule")
    return "\n".join(lines)


def to_latex(table: pd.DataFrame, benchmarks: list[str]) -> str:
    # Columns are [Avg. Rank, Method, <benchmark metrics>...]; Method is a
    # plain data column now (not the DataFrame index). The header is hand-built
    # (blank Rank/Model cells, \small\textbf benchmark names spanning 3 columns
    # each, partial \cmidrule under each) rather than derived from pandas'
    # MultiIndex column rendering, to match the exact requested layout.
    column_format = "cl " + " ".join(["ccc"] * len(benchmarks))
    header = build_table_header(benchmarks)
    rows = render_data_rows(table)
    return (
        f"\\begin{{tabular}}{{{column_format}}}\n"
        f"\\toprule\n"
        f"{header}\n"
        f"\\midrule\n"
        f"{rows}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}"
    )


def build_appendix_qad_table(values: dict[tuple[str, str], dict[str, list[float]]]) -> pd.DataFrame:
    """QAD-only, decimation=1 ablation table for the appendix.

    Independent of the other tables: no "Avg. Rank" column (and so no
    rank-based row sorting -- rows just keep CLASSIFIER_ORDER's fixed shallow/
    deep/NeuralODE/ours grouping), but the top 3 per column are still
    highlighted the same way as the main tables.
    """
    columns = [METRIC_LABELS[m] for m in METRICS]
    mean_table = pd.DataFrame(index=CLASSIFIER_ORDER, columns=columns, dtype=float)
    text_table = pd.DataFrame(index=CLASSIFIER_ORDER, columns=columns, dtype=object)

    for clf_name in CLASSIFIER_ORDER:
        metric_values = values.get((QAD_DECIMATION1_BENCHMARK, clf_name), {m: [] for m in METRICS})
        for metric in METRICS:
            col = METRIC_LABELS[metric]
            mean, std = mean_std(metric_values[metric])
            mean_table.loc[clf_name, col] = mean
            text_table.loc[clf_name, col] = format_cell(mean, std)

    highlight_top3(mean_table, text_table)

    text_table.insert(0, "Model", pd.Series({c: display_label(c) for c in CLASSIFIER_ORDER}))
    return text_table


def to_appendix_qad_latex(table: pd.DataFrame) -> str:
    """Render the QAD decimation=1 appendix table: Model + AUC/AUPRC/F1, no rank column."""
    column_format = "l ccc"
    header = (
        f"& \\multicolumn{{3}}{{c}}{{\\small \\textbf{{{BENCHMARK_LABELS.get(QAD_BENCHMARK, QAD_BENCHMARK)}}}}} \\\\\n"
        f"\\cmidrule(lr){{2-4}}\n"
        f"\\textbf{{Model}} & \\textbf{{AUC}} \\big\\uparrow & \\textbf{{AUPRC}} \\big\\uparrow & \\textbf{{F1}} \\big\\uparrow \\\\"
    )
    rows = render_data_rows(table)
    return (
        f"\\begin{{tabular}}{{{column_format}}}\n"
        f"\\toprule\n"
        f"{header}\n"
        f"\\midrule\n"
        f"{rows}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}"
    )


def build_ode_vs_lsd_table(values: dict[tuple[str, str], dict[str, list[float]]]) -> pd.DataFrame:
    """NeuralODE vs. LSD table: one row per benchmark, models as column groups.

    Transposes the per-group tables' layout so that every benchmark appears in
    a single table and the NeuralODE baseline sits immediately left of our two
    LSD variants (see ODE_VS_LSD_MODELS), making the ablation readable row by
    row. Within each (benchmark, metric) the best of the model columns is
    highlighted; there is no "Avg. Rank" column, since ranking over three
    models carries little information.
    """
    benchmarks = [b for group in ODE_VS_LSD_BENCHMARK_GROUPS for b in group]
    metric_labels = [METRIC_LABELS[m] for m in METRICS]

    columns = pd.MultiIndex.from_product([ODE_VS_LSD_MODELS, metric_labels])
    mean_table = pd.DataFrame(index=benchmarks, columns=columns, dtype=float)
    text_table = pd.DataFrame(index=benchmarks, columns=columns, dtype=object)
    mean_table.index.name = "benchmark"
    text_table.index.name = "benchmark"

    for benchmark in benchmarks:
        for model in ODE_VS_LSD_MODELS:
            metric_values = values.get((benchmark, model), {m: [] for m in METRICS})
            for metric in METRICS:
                col = (model, METRIC_LABELS[metric])
                mean, std = mean_std(metric_values[metric])
                mean_table.loc[benchmark, col] = mean
                text_table.loc[benchmark, col] = format_cell(mean, std)

    highlight_best_per_metric(mean_table, text_table, ODE_VS_LSD_MODELS, metric_labels)

    text_table.insert(0, ("", "Benchmark"), pd.Series({b: BENCHMARK_LABELS.get(b, b) for b in benchmarks}))
    return text_table.fillna("--")


def to_ode_vs_lsd_latex(table: pd.DataFrame) -> str:
    """Render the NeuralODE-vs-LSD table: Benchmark + 3 metrics per model group.

    A \\midrule separates the benchmark row groups (single-trace, multi-trace,
    QAD), mirroring how the per-group tables are split.
    """
    column_format = "l " + " ".join(["ccc"] * len(ODE_VS_LSD_MODELS))
    group_cells = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\small \\textbf{{{display_label(model)}}}}}" for model in ODE_VS_LSD_MODELS
    )
    metric_cells = [r"\textbf{Benchmark}"]
    for _ in ODE_VS_LSD_MODELS:
        metric_cells += [r"\textbf{AUC} \big\uparrow", r"\textbf{AUPRC} \big\uparrow", r"\textbf{F1} \big\uparrow"]
    header = "\n".join([
        f"& {group_cells} \\\\",
        _cmidrule_spans(len(ODE_VS_LSD_MODELS), first_metric_col=2),
        " & ".join(metric_cells) + r" \\",
    ])

    boundaries = set()
    running_total = 0
    for group in ODE_VS_LSD_BENCHMARK_GROUPS[:-1]:
        running_total += len(group)
        boundaries.add(running_total)

    lines = []
    for i, row in enumerate(table.itertuples(index=False), start=1):
        lines.append(" & ".join(str(value) for value in row) + r" \\")
        if i in boundaries:
            lines.append(r"\midrule")
    rows = "\n".join(lines)

    return (
        f"\\begin{{tabular}}{{{column_format}}}\n"
        f"\\toprule\n"
        f"{header}\n"
        f"\\midrule\n"
        f"{rows}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}"
    )


def sparsity_column_label(subsample: float) -> str:
    """Column-group header for a subsample level, e.g. 0.01 -> ``QAPPD (1\\%)``."""
    return f"{BENCHMARK_LABELS.get(QAD_BENCHMARK, QAD_BENCHMARK).split(' (')[0]} ({subsample * 100:g}\\%)"


SparsityRow = tuple[str, "str | None", str]  # (clf_name, interp_method, label)


def sparsity_row_label(clf_name: str, interp: "str | None") -> str:
    """Row label for a sparsity cell: ``COPOD (linear)``, or the bare model name
    for our LSD rows, which have no interpolation variant."""
    label = display_label(clf_name)
    return f"{label} ({interp})" if interp else label


def sparsity_row_plan() -> list[list[SparsityRow]]:
    """Return the table's rows as ``(clf_name, interp, display label)``, grouped.

    One row per (classifier, interpolation method) pair -- "COPOD (linear)",
    "COPOD (spline)", ... -- except for SPARSITY_OURS_ROWS (our own LSD
    variants), which get a single row each (interp OURS_SPARSITY_INTERP), fed
    from the "ours" project's fixed-subsample-mask runs. The outer list is the
    \\midrule row grouping, mirroring SPARSITY_CLASSIFIER_GROUPS.
    """
    plan: list[list[SparsityRow]] = []
    for group in SPARSITY_CLASSIFIER_GROUPS:
        rows: list[SparsityRow] = []
        for clf_name in group:
            interps = [OURS_SPARSITY_INTERP] if clf_name in SPARSITY_OURS_ROWS else SPARSITY_INTERP_METHODS
            for interp in interps:
                rows.append((clf_name, interp, sparsity_row_label(clf_name, interp)))
        plan.append(rows)
    return plan


def build_sparsity_table(values: dict[SparsityKey, dict[str, list[float]]]) -> pd.DataFrame:
    """QAD sparsity table: one column group per subsample level, rows per
    (classifier, interpolation) pair.

    Layout mirrors the main per-group tables -- leading "Avg. Rank" + "Model"
    columns, AUC/AUPRC/F1 per group, top 3 highlighted per column -- but rows
    keep sparsity_row_plan()'s fixed order rather than being sorted by rank,
    so a classifier's linear/spline pair always stays adjacent.
    """
    rows = [row for group in sparsity_row_plan() for row in group]
    labels = [label for _, _, label in rows]

    columns = pd.MultiIndex.from_product(
        [[sparsity_column_label(s) for s in SPARSITY_SUBSAMPLES], [METRIC_LABELS[m] for m in METRICS]]
    )
    mean_table = pd.DataFrame(index=labels, columns=columns, dtype=float)
    text_table = pd.DataFrame(index=labels, columns=columns, dtype=object)

    empty = {m: [] for m in METRICS}
    for clf_name, interp, label in rows:
        for subsample in SPARSITY_SUBSAMPLES:
            metric_values = values.get((subsample, clf_name, interp), empty)
            for metric in METRICS:
                col = (sparsity_column_label(subsample), METRIC_LABELS[metric])
                mean, std = mean_std(metric_values[metric])
                mean_table.loc[label, col] = mean
                text_table.loc[label, col] = format_cell(mean, std)

    ranks = highlight_top3(mean_table, text_table)

    avg_rank = ranks.mean(axis=1, skipna=True)
    avg_rank_position = avg_rank.rank(method="min", ascending=True)
    rank_texts = {}
    for label in labels:
        value = avg_rank.get(label)
        if pd.isna(value):
            rank_texts[label] = "--"
            continue
        position = avg_rank_position.get(label)
        color = RANK_COLORS.get(position) if pd.notna(position) else None
        if color is not None:
            rank_texts[label] = f"\\rankbox[{color}]{{{value:.2f}}}"
        else:
            rank_texts[label] = f"\\rankbox{{{value:.2f}}}"

    text_table.insert(0, ("", "Avg. Rank"), pd.Series(rank_texts))
    text_table.insert(1, ("", "Model"), pd.Series({label: label for label in labels}))
    return text_table.fillna("--")


def to_sparsity_latex(table: pd.DataFrame) -> str:
    """Render the QAD sparsity table, with a \\midrule between row groups."""
    column_format = "cl " + " ".join(["ccc"] * len(SPARSITY_SUBSAMPLES))
    group_cells = " & ".join(
        f"\\multicolumn{{3}}{{c}}{{\\small \\textbf{{{sparsity_column_label(s)}}}}}" for s in SPARSITY_SUBSAMPLES
    )
    metric_cells = [r"\textbf{Rank}\big\downarrow", r"\textbf{Model}"]
    for _ in SPARSITY_SUBSAMPLES:
        metric_cells += [r"\textbf{AUC} \big\uparrow", r"\textbf{AUPRC} \big\uparrow", r"\textbf{F1} \big\uparrow"]
    header = "\n".join([
        f"&& {group_cells} \\\\",
        _cmidrule_spans(len(SPARSITY_SUBSAMPLES)),
        " & ".join(metric_cells) + r" \\",
    ])

    boundaries = set()
    running_total = 0
    for group in sparsity_row_plan()[:-1]:
        running_total += len(group)
        boundaries.add(running_total)

    lines = []
    for i, row in enumerate(table.itertuples(index=False), start=1):
        lines.append(" & ".join(str(value) for value in row) + r" \\")
        if i in boundaries:
            lines.append(r"\midrule")
    rows = "\n".join(lines)

    return (
        f"\\begin{{tabular}}{{{column_format}}}\n"
        f"\\toprule\n"
        f"{header}\n"
        f"\\midrule\n"
        f"{rows}\n"
        f"\\bottomrule\n"
        f"\\end{{tabular}}"
    )


def build_sparsity_run_count_table(values: dict[SparsityKey, dict[str, list[float]]]) -> pd.DataFrame:
    """Console-only: number of contributing runs per (clf, interp) x subsample.

    Every planned row is listed even when nothing was found for it (shown as
    0), so still-missing sparsity configurations are visible at a glance --
    this is the sparsity counterpart of report_missing_runs.
    """
    counts: dict[str, dict[str, int]] = {}
    for (subsample, clf_name, interp), metric_values in values.items():
        num_runs = max((len(v) for v in metric_values.values()), default=0)
        counts.setdefault(sparsity_column_label(subsample), {})[sparsity_row_label(clf_name, interp)] = num_runs

    count_df = pd.DataFrame(counts)
    expected = [label for group in sparsity_row_plan() for _, _, label in group]
    count_df = count_df.reindex(
        index=expected + sorted(set(count_df.index) - set(expected)),
        columns=[sparsity_column_label(s) for s in SPARSITY_SUBSAMPLES],
    ).fillna(0).astype(int)
    count_df.index.name = "model (interp)"
    return count_df


def build_comparison_table(avg_rank_single: pd.Series, avg_rank_multi: pd.Series) -> pd.DataFrame:
    """Combine two tables' avg-rank columns into Overall/Single Trace/Multi Trace.

    "Overall" is the plain mean of the two group averages (skipping a
    missing one) -- not a rank freshly computed over all 18 columns
    combined. Rows with no rank in either source table are dropped.
    """
    combined = pd.DataFrame({"Single Trace": avg_rank_single, "Multi Trace": avg_rank_multi})
    combined["Overall"] = combined[["Single Trace", "Multi Trace"]].mean(axis=1, skipna=True)
    combined = combined.dropna(subset=["Overall"])
    return combined[["Overall", "Single Trace", "Multi Trace"]]


def to_comparison_latex(comparison_df: pd.DataFrame) -> str:
    """Render the Overall/Single Trace/Multi Trace table, sorted by Overall ascending.

    Each column is independently ranked (lower = better) and its top 3
    values are wrapped in bare ``\\cellcolor{first|second|third}{...}``
    (no ``\\rankbox`` here, since every cell in this table already is a
    rank number). No caption/label is passed to to_latex(), so pandas emits
    only the tabular environment.
    """
    comparison_df = comparison_df.sort_values("Overall", ascending=True)
    ranks = comparison_df.rank(axis=0, method="min", ascending=True)

    display_df = pd.DataFrame(index=comparison_df.index)
    for col in comparison_df.columns:
        cells = []
        for clf_name in comparison_df.index:
            value = comparison_df.loc[clf_name, col]
            if pd.isna(value):
                cells.append("--")
                continue
            rank = ranks.loc[clf_name, col]
            color = RANK_COLORS.get(rank) if pd.notna(rank) else None
            text = f"{value:.2f}"
            cells.append(f"\\cellcolor{{{color}}}{{{text}}}" if color is not None else f"{{{text}}}")
        display_df[col] = cells

    display_df.insert(0, "Model", [display_label(clf_name) for clf_name in comparison_df.index])
    display_df = display_df.rename(columns={
        "Model": r"\textbf{Model}",
        "Overall": r"\textbf{Overall}",
        "Single Trace": r"\textbf{Single Trace}",
        "Multi Trace": r"\textbf{Multi Trace}",
    })

    return display_df.to_latex(
        index=False,
        escape=False,
        column_format="lrrr",
    )


def main() -> None:
    args = parse_args()

    records = fetch_run_records(args.project, args.entity)

    if not args.skip_neuralode:
        # --after only filters the PYOD baselines; NeuralODE runs are always
        # fetched in full regardless of --after.
        ode_records = fetch_neuralode_run_records(args.ode_project, args.ode_entity, min_created_at=None)
        for key, entries in ode_records.items():
            records[key].extend(entries)

    if not args.skip_ours:
        ours_records = fetch_ours_run_records(args.ours_project, args.ours_entity)
        for key, entries in ours_records.items():
            records[key].extend(entries)

    if not records:
        searched = f"project '{args.project}' (entity={args.entity})"
        if not args.skip_neuralode:
            searched += f" or project '{args.ode_project}' (entity={args.ode_entity})"
        if not args.skip_ours:
            searched += f" or project '{args.ours_project}' (entity={args.ours_entity})"
        raise SystemExit(f"No usable runs found in {searched}.")

    # A seed may only contribute once per (benchmark, classifier); if it was
    # run more than once, keep the most recently created run for that seed.
    records = dedupe_by_seed(records)

    values = select_recent_runs(records)

    print(f"\nRuns found per benchmark/classifier configuration (after filtering to the most recent "
          f"{DEFAULT_RUN_LIMIT} runs, or {DETERMINISTIC_RUN_LIMIT} for deterministic classifiers "
          f"{sorted(DETERMINISTIC_CLASSIFIERS)}):")
    count_df = build_run_count_table(values)
    print(count_df.to_string())

    args.output_dir.mkdir(parents=True, exist_ok=True)

    avg_ranks_by_slug: dict[str, pd.Series] = {}
    for slug, benchmarks in TABLE_GROUPS + [QAD_TABLE_GROUP]:
        table, avg_rank = build_latex_table(values, benchmarks)
        avg_ranks_by_slug[slug] = avg_rank
        latex = to_latex(table, benchmarks)

        print(f"\nLaTeX table ({', '.join(benchmarks)})")
        print(latex)

        out_path = args.output_dir / f"baseline_table_{slug}.tex"
        out_path.write_text(latex)
        print(f"Saved to {out_path}")

    single_trace_slug, multi_trace_slug = TABLE_GROUPS[0][0], TABLE_GROUPS[1][0]
    comparison_df = build_comparison_table(avg_ranks_by_slug[single_trace_slug], avg_ranks_by_slug[multi_trace_slug])
    comparison_latex = to_comparison_latex(comparison_df)

    print("\nLaTeX table (rank comparison: single-trace vs. multi-trace)")
    print(comparison_latex)

    comparison_out_path = args.output_dir / "baseline_table_rank_comparison.tex"
    comparison_out_path.write_text(comparison_latex)
    print(f"Saved to {comparison_out_path}")

    appendix_table = build_appendix_qad_table(values)
    appendix_latex = to_appendix_qad_latex(appendix_table)

    print("\nLaTeX table (appendix: QAD, decimation=1)")
    print(appendix_latex)

    appendix_out_path = args.output_dir / "baseline_table_QAD_decimation1_appendix.tex"
    appendix_out_path.write_text(appendix_latex)
    print(f"Saved to {appendix_out_path}")

    ode_vs_lsd_table = build_ode_vs_lsd_table(values)
    ode_vs_lsd_latex = to_ode_vs_lsd_latex(ode_vs_lsd_table)

    print("\nLaTeX table (NeuralODE vs. LSD, all benchmarks)")
    print(ode_vs_lsd_latex)

    ode_vs_lsd_out_path = args.output_dir / "baseline_table_ode_vs_lsd.tex"
    ode_vs_lsd_out_path.write_text(ode_vs_lsd_latex)
    print(f"Saved to {ode_vs_lsd_out_path}")

    sparsity_records = None
    if not args.skip_sparsity:
        # Independent of everything above: its own project, its own
        # (subsample, classifier, interp) bucketing. Our LSD rows are the one
        # exception -- they come from the "ours" project, from precisely the
        # fixed-subsample-mask runs the tables above filter out.
        raw_sparsity_records = fetch_sparsity_run_records(args.sparsity_project, args.sparsity_entity)
        if not args.skip_ours:
            ours_sparsity_records = fetch_ours_sparsity_run_records(args.ours_project, args.ours_entity)
            for key, entries in ours_sparsity_records.items():
                raw_sparsity_records[key].extend(entries)
        sparsity_records = dedupe_by_seed(raw_sparsity_records)
        if sparsity_records:
            sparsity_values = select_recent_sparsity_runs(sparsity_records)

            print(f"\nSparsity runs found per subsample/model configuration (most recent "
                  f"{DEFAULT_RUN_LIMIT} per cell):")
            print(build_sparsity_run_count_table(sparsity_values).to_string())

            sparsity_latex = to_sparsity_latex(build_sparsity_table(sparsity_values))

            print("\nLaTeX table (QAD sparsity sweep)")
            print(sparsity_latex)

            sparsity_out_path = args.output_dir / "baseline_table_QAD_sparsity.tex"
            sparsity_out_path.write_text(sparsity_latex)
            print(f"Saved to {sparsity_out_path}")
        else:
            print(f"\nNo usable sparsity runs found in project '{args.sparsity_project}' "
                  f"(entity={args.sparsity_entity}), nor any fixed-subsample-mask QAD runs in "
                  f"'{args.ours_project}'; skipping the sparsity table.")

    all_benchmarks = [b for _, benchmarks in TABLE_GROUPS for b in benchmarks] + [
        QAD_BENCHMARK, QAD_DECIMATION1_BENCHMARK
    ]
    report_missing_runs(records, all_benchmarks, sparsity_records=sparsity_records)


if __name__ == "__main__":
    main()