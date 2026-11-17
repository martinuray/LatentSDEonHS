#!/usr/bin/env python3
r"""Per-trace QAPPD (QAD) sparsity trends of our two LSD variants (matplotlib).

Companion to ``wandb_results_to_latex.py``: same W&B project and run selection
rules, but it pulls the *full epoch history* of every selected run and plots,
per QAD trace, LSD on R^n vs LSD on S^n at 100%, 5% and 1% training data.

A "mode" is one (benchmark, data level) pair -- one row block of the big table
and one row of the agreement table. --datasets picks the benchmarks and each
gets the --levels it has runs for, so adding a benchmark needs no other change.
QAD (decimation QAD_REQUIRED_DECIMATION, full-benchmark runs only; --trace-ids
subsets are skipped) has all three levels:
  - 100%: the main-table runs (``args.fixed_subsample_mask`` unset; the model
    sees the whole training series, resampling its subsample mask each
    iteration).
  - 5% / 1%: the sparsity sweep (``args.fixed_subsample_mask`` set, level from
    ``args.subsample``) -- exactly the runs the sparsity table reports.
PSM is a single-dataset benchmark (one "trace", named after the benchmark) run
undecimated and only without a fixed subsample mask, so it contributes the 100%
mode alone -- it appears in both tables, but not in the trend/curve figures,
which compare data levels and would show a single point.
Per (benchmark, level, variant), runs are deduplicated by seed (most recent
wins) and cut to the --run-limit most recent ones, as in the LaTeX tables.

What is fetched: for every selected run, every history row anomaly_detection.py
logs per trace and evaluation epoch, i.e. all ``epoch/<trace>/<split>/<metric>``
keys (``val/loss``, ``val/log_pxz``, ``tst/auc_roc``, ``tst/auc_pr``, ``tst/f1``,
``trn/loss``, ...) against the trace's own step ``<trace>/epoch``. They are
stored as one long-format table (one row per run x trace x epoch, one column per
``<split>/<metric>``) in ``qad_epoch_history.csv.gz``, which doubles as the
cache -- pass --refresh to re-fetch from W&B. A derived column ``val/ll`` =
``-val/log_pxz`` is added: anomaly_detection.py logs the reconstruction
*negative* log-likelihood, so ``val/ll`` is the validation log-likelihood
(higher is better).

Selected epoch: per (run, trace), the epoch train_one_dataset kept as best --
first strict improvement of ``val/loss`` by more than the run's
early_stopping_min_delta. Its test metrics are what the run reports in its
summary (and what the LaTeX tables aggregate); the script cross-checks its
``tst/auc_pr`` against ``summary/per_dataset/<trace>/auc_pr`` (or, for
single-dataset benchmarks, ``summary/auc_pr``) and warns on a mismatch.

Figures (into --output-dir, one file per --formats entry):
  - ``trend_trace<id>``: metrics at the selected epoch vs. training-data
    fraction (one panel per --trend-metrics entry), mean +/- std over runs.
  - ``trends_all``: the same for all traces in one grid.
  - ``curves_trace<id>``: metric curves over epochs, rows = --curve-metrics,
    columns = data levels; mean over runs with a +/- std band. Runs stop early
    at different epochs, so each epoch averages only the runs that reached it
    (the band ends where fewer than --min-runs-per-epoch remain).
Tables (into --output-dir/tables):
Both .tex files hold a bare ``tabular`` and nothing else -- no float, no
\centering, no caption, no label, no \resizebox: wrap them at the use site
(\input inside your own table/table*). They are styled like the tables of
wandb_results_to_latex.py -- bold headers with a \big arrow for the better
direction, "AUC"/"AUPRC" as metric names, ``\small\textbf`` column groups,
``mean \std{std}`` to two decimals -- but no cell is coloured or otherwise
marked. They need booktabs, multirow, graphicx and the ``\std`` macro in the
preamble. val_status_all is wide enough to need a \resizebox or a
full-width float of its own.
  - ``val_status_all.tex``: every mode as row blocks (QAPPD at 100%, 5%, 1%
    training data, then PSM) separated by vertical space and a rule, each block
    labelled by a rotated cell in the leftmost column.
    Each row is one trace of that block's benchmark (by id; the anomaly types
    are deliberately not shown anywhere here) and shows the state of the
    early-stopping criterion at the selected (lowest val loss) epoch -- the
    selected epoch, ``val/loss`` (the criterion itself), ``val/log_pxz`` (the
    reconstruction NLL) and the test AUC and AUPRC (x100) the run reports
    for that epoch, ``mean \std{std}`` over runs. As in the per-trace table
    of wandb_results_to_latex.py, each geometry is one column group holding
    every metric. No cell is highlighted.
  - ``val_status.md``: the same table in Markdown, for a quick look; here the
    better geometry per metric (val loss, NLL: lower; AUC, AUPRC: higher) is
    in bold.
  - ``geometry_agreement.tex``: how often model selection by validation loss
    also picks the better-scoring geometry -- one row per mode (the same blocks
    as the big table), one column per score in --agreement-metrics, each score
    evaluated independently. Every (trace, seed) pair votes once -- for that
    seed, the geometry with the lower val loss against the geometry with the
    better test score -- and the cells pool all of that mode's pairs:
    "agreeing pairs / all pairs (percent)", i.e. 10 traces x 5 seeds = 50 per
    QAPPD mode. A pair where a geometry is missing for that seed or the two
    values are exactly tied is counted in neither the numerator nor the
    denominator. Plus
    ``geometry_agreement.md`` (the same table) and ``geometry_agreement.csv``
    (per trace: the seed votes behind the pooled counts, plus the pick each
    criterion makes on that trace's run means).
Plus ``selected_epoch_scores.csv``: mean/std/n at the selected epoch for every
(benchmark, trace, level, variant, metric).
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

try:
    import wandb
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "The 'wandb' package is required to fetch results (pip install wandb)."
    ) from exc

from wandb_results_to_latex import (  # noqa: E402
    BENCHMARK_LABELS,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_LIMIT,
    LSD_RN_LABEL,
    LSD_SN_LABEL,
    MISSING_CELL,
    OURS_DEFAULT_ENTITY,
    OURS_DEFAULT_PROJECT,
    PSM_BENCHMARK,
    QAD_BENCHMARK,
    QAD_REQUIRED_DECIMATION,
    dedupe_by_seed,
    uses_fixed_subsample_mask,
)

# Fraction of training data available -> plotted left to right in this order.
# 1.0 stands for the main-table runs (no fixed subsample mask), see module doc.
DEFAULT_LEVELS = [1.0, 0.05, 0.01]
FULL_DATA_LEVEL = 1.0

# Benchmarks covered, in table order. Only QAD has a sparsity sweep; PSM (a
# single-dataset benchmark) exists at full training data only, so it
# contributes one block/row instead of three -- which is handled generically:
# every benchmark gets the levels it actually has runs for.
DEFAULT_DATASETS = [QAD_BENCHMARK, PSM_BENCHMARK]
# Decimation factor a run must have been trained with to be comparable, per
# benchmark (as in the LaTeX tables; everything but QAD runs undecimated).
REQUIRED_DECIMATION = {QAD_BENCHMARK: QAD_REQUIRED_DECIMATION}
DEFAULT_REQUIRED_DECIMATION = 1

VARIANTS = ["Rn", "Sn"]
VARIANT_LABEL = {"Rn": r"LSD on $\mathbb{R}^n$", "Sn": r"LSD on $\mathbb{S}^n$"}
VARIANT_COLOR = {"Rn": "#1F77B4", "Sn": "#D62728"}
VARIANT_MARKER = {"Rn": "s", "Sn": "o"}
# Horizontal offset (in category units) so the two variants' error bars don't overlap.
VARIANT_X_SHIFT = {"Rn": -0.06, "Sn": 0.06}

META_COLS = ["run_id", "run_name", "benchmark", "level", "variant", "seed", "created_at", "min_delta",
             "trace", "epoch"]
HISTORY_KEY_RE = re.compile(r"^epoch/(?P<trace>[^/]+)/(?P<split>trn|val|tst|oth)/(?P<metric>.+)$")
VAL_LOSS = "val/loss"
VAL_LL = "val/ll"
TST_AUPRC = "tst/auc_pr"

DEFAULT_TREND_METRICS = [TST_AUPRC, VAL_LL]
DEFAULT_CURVE_METRICS = [VAL_LOSS, VAL_LL, TST_AUPRC, "tst/auc_roc"]
# Metrics shown x100, like the LaTeX tables.
PERCENT_METRICS = {"tst/auc_pr", "tst/auc_roc", "tst/f1", "tst/vus_pr", "tst/vus_roc", "tst/r_auc_pr", "tst/r_auc_roc"}
METRIC_LABEL = {
    VAL_LOSS: "Val. loss (neg. ELBO)",
    VAL_LL: "Val. log-likelihood",
    "val/log_pxz": "Val. NLL",
    TST_AUPRC: "AUPRC (x100)",
    "tst/auc_roc": "AUC (x100)",
    "tst/f1": "F1 (x100)",
    "trn/loss": "Train loss",
}
# Loss-like curves have huge early-epoch excursions; their y-range is clipped
# to the bulk of the curve so the later, relevant epochs stay readable.
ROBUST_YLIM_METRICS = {VAL_LOSS, VAL_LL, "val/log_pxz", "trn/loss", "tst/loss"}
# Of those, the ones that start very low and rise (the rest start high and fall);
# decides which side of the range gets clipped.
ROBUST_YLIM_RISING = {VAL_LL}
# Share of the curve's (early, extreme) values cut off on the clipped side.
ROBUST_YLIM_CLIP_PCT = 15

# Columns of the per-trace early-stopping tables: the selected epoch, the val
# loss (the criterion) and reconstruction NLL, and the test scores at that epoch.
# The KL terms (val/kl0, val/klp) are left out by default; add them back via
# --table-metrics.
DEFAULT_TABLE_METRICS = ["epoch", VAL_LOSS, "val/log_pxz", "tst/auc_roc", TST_AUPRC]
# Header cell (tex, markdown) per table column. Spelled like the tables of
# wandb_results_to_latex.py: bold name plus a \big arrow for the better
# direction, "AUC" (not "AUC-ROC") as in its METRIC_LABELS.
TABLE_HEADER = {
    "epoch": (r"\textbf{Epoch}", "Epoch"),
    VAL_LOSS: (r"\textbf{Val.\ loss} \big\downarrow", "Val. loss ↓"),
    "val/log_pxz": (r"\textbf{NLL} \big\downarrow", "NLL ↓"),
    "val/kl0": (r"\textbf{KL$_0$}", "KL0"),
    "val/klp": (r"\textbf{KL$_{\mathrm{path}}$}", "KL path"),
    VAL_LL: (r"\textbf{Val.\ log-lik.} \big\uparrow", "Val. log-lik. ↑"),
    TST_AUPRC: (r"\textbf{AUPRC} \big\uparrow", "AUPRC ↑"),
    "tst/auc_roc": (r"\textbf{AUC} \big\uparrow", "AUC ↑"),
}
# Metrics with a "better" direction get the better geometry of each level
# set in bold in the Markdown table (+1: higher is better, -1: lower is better). The KL terms and the epoch
# have none -- a smaller KL is not per se better -- so they are never highlighted.
TABLE_BETTER = {VAL_LOSS: -1, "val/log_pxz": -1, "val/elbo": -1, VAL_LL: +1, TST_AUPRC: +1, "tst/auc_roc": +1}
# Column-group header per geometry, the row labels wandb_results_to_latex.py uses for them.
VARIANT_TEX = {"Rn": LSD_RN_LABEL, "Sn": LSD_SN_LABEL}

# Test scores the geometry-agreement tables check the validation-loss pick
# against -- one independent column group each.
DEFAULT_AGREEMENT_METRICS = ["tst/auc_roc", TST_AUPRC]
# Short (tex, markdown) group headers for those scores; unlisted ones use their key.
SCORE_SHORT = {
    "tst/auc_roc": (r"\textbf{AUC}", "AUC"),
    TST_AUPRC: (r"\textbf{AUPRC}", "AUPRC"),
    "tst/f1": (r"\textbf{F1}", "F1"),
    "tst/vus_pr": (r"\textbf{VUS-PR}", "VUS-PR"),
    "tst/vus_roc": (r"\textbf{VUS-ROC}", "VUS-ROC"),
}

HISTORY_CACHE = "qad_epoch_history.csv.gz"
SUMMARY_CACHE = "qad_summary_auprc.csv"
CACHE_META = "qad_epoch_history.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--project", default=OURS_DEFAULT_PROJECT, help="W&B project of our LSD runs.")
    parser.add_argument("--entity", default=OURS_DEFAULT_ENTITY, help="W&B entity for --project.")
    parser.add_argument(
        "--datasets", nargs="+", default=list(DEFAULT_DATASETS), metavar="BENCHMARK",
        help="Benchmarks to cover, in table order; each gets the --levels it has runs for. "
             f"Default: {' '.join(DEFAULT_DATASETS)}.",
    )
    parser.add_argument(
        "--levels", type=float, nargs="+", default=list(DEFAULT_LEVELS), metavar="FRACTION",
        help="Training-data fractions to plot, left to right (1 = main-table runs). "
             f"Default: {' '.join(f'{lvl:g}' for lvl in DEFAULT_LEVELS)}.",
    )
    parser.add_argument(
        "--run-limit", type=int, default=DEFAULT_RUN_LIMIT,
        help=f"Most recent runs (after seed dedupe) per level and variant. Default: {DEFAULT_RUN_LIMIT}.",
    )
    parser.add_argument(
        "--trend-metrics", nargs="+", default=list(DEFAULT_TREND_METRICS), metavar="SPLIT/METRIC",
        help=f"Panels of the trend figures. Default: {' '.join(DEFAULT_TREND_METRICS)}.",
    )
    parser.add_argument(
        "--curve-metrics", nargs="+", default=list(DEFAULT_CURVE_METRICS), metavar="SPLIT/METRIC",
        help=f"Rows of the per-epoch curve figures. Default: {' '.join(DEFAULT_CURVE_METRICS)}.",
    )
    parser.add_argument(
        "--min-runs-per-epoch", type=int, default=2,
        help="Only plot an epoch of a curve if at least this many runs reached it. Default: 2.",
    )
    parser.add_argument(
        "--table-metrics", nargs="+", default=list(DEFAULT_TABLE_METRICS), metavar="SPLIT/METRIC",
        help=f"Columns of the per-trace early-stopping tables. Default: {' '.join(DEFAULT_TABLE_METRICS)}.",
    )
    parser.add_argument(
        "--agreement-metrics", nargs="+", default=list(DEFAULT_AGREEMENT_METRICS), metavar="SPLIT/METRIC",
        help="Test scores the geometry-agreement tables compare the validation-loss pick against, each "
             f"independently. Default: {' '.join(DEFAULT_AGREEMENT_METRICS)}.",
    )
    parser.add_argument("--formats", nargs="+", default=["pdf", "png"], help="Figure file formats.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR / "qad_sparsity_trends")
    parser.add_argument("--refresh", action="store_true", help="Ignore the cached history and re-fetch from W&B.")
    args = parser.parse_args()
    if len(set(args.levels)) != len(args.levels):
        parser.error(f"--levels has duplicates: {args.levels}")
    if any(not 0 < lvl <= 1 for lvl in args.levels):
        parser.error(f"--levels takes fractions in (0, 1], got {args.levels} (1% is 0.01)")
    return args


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def _is_number(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def run_mode(config: dict, datasets: list[str]) -> tuple[str, float] | None:
    """(benchmark, training-data fraction) a run belongs to, or None if it's no trend run."""
    run_context = config.get("run_context") or {}
    run_args = config.get("args") or {}
    # Exact match: "QAD:1,2" (--trace-ids subsets) never averages every trace.
    benchmark = run_context.get("benchmark_name")
    if benchmark not in datasets:
        return None
    if run_args.get("data_decimation_factor") != REQUIRED_DECIMATION.get(benchmark, DEFAULT_REQUIRED_DECIMATION):
        return None
    if not uses_fixed_subsample_mask(config):
        return benchmark, FULL_DATA_LEVEL
    subsample = run_args.get("subsample")
    return (benchmark, float(subsample)) if _is_number(subsample) else None


def per_trace_summary_auprc(summary: dict, benchmark: str) -> dict[str, float]:
    """``{trace_id: auc_pr}`` from ``summary/per_dataset/<id>/auc_pr``.

    Single-dataset benchmarks (PSM, SWaT, ...) have nothing to break down, so
    anomaly_detection.py writes no per_dataset keys for them; their one score
    is ``summary/auc_pr``, keyed here by the benchmark name -- which is also the
    dataset id their history keys use (``epoch/PSM/tst/auc_pr``).
    """
    out = {}
    prefix, suffix = "summary/per_dataset/", "/auc_pr"
    for key, value in summary.items():
        if key.startswith(prefix) and key.endswith(suffix) and _is_number(value):
            out[key[len(prefix):-len(suffix)]] = float(value)
    if not out and _is_number(summary.get("summary/auc_pr")):
        out[benchmark] = float(summary["summary/auc_pr"])
    return out


def history_rows(history: list[dict]) -> list[dict]:
    """Split W&B history rows into one ``{trace, epoch, <split>/<metric>: value}`` per trace and epoch.

    A history row may carry metrics of a single trace (one _wandb_log_epoch
    call); grouping by (trace, epoch) also merges rows should W&B ever split
    one call's payload.
    """
    merged: dict[tuple[str, float], dict] = defaultdict(dict)
    for row in history:
        per_trace: dict[str, dict] = defaultdict(dict)
        for key, value in row.items():
            if not _is_number(value):
                continue
            match = HISTORY_KEY_RE.match(key)
            if match:
                per_trace[match.group("trace")][f"{match.group('split')}/{match.group('metric')}"] = float(value)
        for trace, metrics in per_trace.items():
            epoch = row.get(f"{trace}/epoch")
            if _is_number(epoch):
                merged[(trace, float(epoch))].update(metrics)
    return [{"trace": trace, "epoch": int(epoch), **metrics} for (trace, epoch), metrics in merged.items()]


def fetch_history(
    project: str, entity: str | None, datasets: list[str], levels: list[float], run_limit: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (epoch history, summary AUPRC) of the selected runs, both long-format.

    Run selection uses only the cheap run metadata/summary; the (slow) epoch
    history is scanned afterwards, for the selected runs only.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"config.run_context.benchmark_name": {"$in": datasets}}, order="-created_at")

    candidates: dict[tuple[str, float, str], list] = defaultdict(list)
    run_by_id = {}
    for run in runs:
        config = run.config or {}
        mode = run_mode(config, datasets)
        if mode is None or mode[1] not in levels:
            continue
        benchmark, level = mode
        variant = (config.get("run_context") or {}).get("model_variant")
        if variant not in VARIANTS:
            continue
        summary_auprc = per_trace_summary_auprc(dict(run.summary), benchmark)
        if not summary_auprc:
            continue  # crashed/killed before the final summary was written
        created_at = pd.to_datetime(run.created_at, utc=True)
        seed = (config.get("run_context") or {}).get("run_seed")
        run_by_id[run.id] = (run, summary_auprc)
        # dedupe_by_seed expects (created_at, seed, payload); the payload is the run id.
        candidates[(benchmark, level, variant)].append((created_at, seed, run.id))

    # Parallel array jobs can give two runs of the same seed the same W&B
    # created_at second, and dedupe_by_seed keeps whichever it sees first --
    # which would leave it to the (unstable) W&B listing order which of the two
    # a fetch uses, and that moves the reported means. Ordering every bucket by
    # (created_at, run name) descending up front pins it down: the newest run
    # wins, a tie going to the later-started one, whose launcher timestamp is
    # part of its name.
    def recency(entry):
        return entry[0], run_by_id[entry[2]][0].name

    candidates = {key: sorted(entries, key=recency, reverse=True) for key, entries in candidates.items()}

    history_frames, summary_rows = [], []
    for (benchmark, level, variant), entries in sorted(dedupe_by_seed(candidates).items()):
        selected = sorted(entries, key=recency, reverse=True)[:run_limit]
        for created_at, seed, run_id in selected:
            run, summary_auprc = run_by_id[run_id]
            print(f"  scanning history: {benchmark} {level:g} {variant} seed={seed} ({run.name})")
            meta = {
                "run_id": run_id, "run_name": run.name, "benchmark": benchmark, "level": level,
                "variant": variant, "seed": seed, "created_at": created_at.isoformat(),
                "min_delta": float((run.config.get("args") or {}).get("early_stopping_min_delta") or 0.0),
            }
            rows = history_rows(run.scan_history(page_size=10000))
            history_frames.append(pd.DataFrame([{**meta, **row} for row in rows]))
            summary_rows += [{**meta, "trace": trace, "auc_pr": value} for trace, value in summary_auprc.items()]

    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame(columns=META_COLS)
    metric_cols = sorted(col for col in history.columns if col not in META_COLS)
    history = history[META_COLS + metric_cols].sort_values(
        ["benchmark", "level", "variant", "run_id", "trace", "epoch"])
    return history.reset_index(drop=True), pd.DataFrame(summary_rows)


def load_or_fetch(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = args.output_dir
    cache_key = {"project": args.project, "entity": args.entity, "datasets": args.datasets,
                 "levels": args.levels, "run_limit": args.run_limit}
    meta_path, hist_path, summ_path = out / CACHE_META, out / HISTORY_CACHE, out / SUMMARY_CACHE
    if not args.refresh and meta_path.exists() and hist_path.exists() and summ_path.exists():
        if json.loads(meta_path.read_text()) == cache_key:
            print(f"Using cached history from {hist_path} (pass --refresh to re-fetch).")
            dtypes = {"trace": str, "run_id": str, "benchmark": str}
            return pd.read_csv(hist_path, dtype=dtypes), pd.read_csv(summ_path, dtype=dtypes)

    print(f"Fetching {'/'.join(args.datasets)} runs from W&B project {args.entity}/{args.project} ...")
    history, summary = fetch_history(args.project, args.entity, args.datasets, args.levels, args.run_limit)
    history.to_csv(hist_path, index=False)
    summary.to_csv(summ_path, index=False)
    meta_path.write_text(json.dumps(cache_key))
    return history, summary


# ---------------------------------------------------------------------------
# Derived quantities
# ---------------------------------------------------------------------------

def add_derived_columns(history: pd.DataFrame) -> pd.DataFrame:
    history = history.copy()
    if "val/log_pxz" in history:
        history[VAL_LL] = -history["val/log_pxz"]
    for col in PERCENT_METRICS & set(history.columns):
        history[col] = history[col] * 100.0
    return history


def selected_epochs(history: pd.DataFrame) -> pd.DataFrame:
    """One history row per (run, trace): the epoch train_one_dataset kept as best.

    Mirrors its rule: walk the logged epochs in order and keep the first one
    whose val loss improves on the best so far by more than min_delta. If no
    epoch ever improves (val loss NaN throughout), train_one_dataset falls back
    to the last tst_stats, so the last logged epoch is used.
    """
    picked = []
    for _, group in history.groupby(["run_id", "trace"], sort=False):
        group = group.sort_values("epoch")
        min_delta = group["min_delta"].iloc[0]
        best_loss, best_idx = np.inf, None
        for idx, loss in group[VAL_LOSS].items():
            if np.isfinite(loss) and loss < best_loss - min_delta:
                best_loss, best_idx = loss, idx
        picked.append(group.index[-1] if best_idx is None else best_idx)
    return history.loc[picked].reset_index(drop=True)


def check_against_summary(selected: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Warn if the recomputed selected-epoch AUPRC disagrees with the run's own summary."""
    if TST_AUPRC not in selected or summary.empty:
        return
    merged = selected[["run_id", "trace", "run_name", TST_AUPRC]].merge(
        summary[["run_id", "trace", "auc_pr"]], on=["run_id", "trace"], how="inner"
    )
    bad = merged[~np.isclose(merged[TST_AUPRC], merged["auc_pr"] * 100.0, atol=1e-6)]
    if bad.empty:
        print(f"Selected-epoch check: tst/auc_pr matches the run summary for all {len(merged)} (run, trace) pairs.")
    else:
        print(f"  warning: selected-epoch tst/auc_pr differs from the run summary for {len(bad)} "
              f"of {len(merged)} (run, trace) pairs, e.g.:")
        print(bad.head(5).to_string(index=False))


def aggregate_selected(selected: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    metrics = [m for m in metrics if m in selected]
    long = selected.melt(id_vars=["benchmark", "trace", "level", "variant"], value_vars=metrics, var_name="metric")
    long = long.dropna(subset=["value"])
    grouped = long.groupby(["benchmark", "trace", "level", "variant", "metric"])["value"]
    # Population std (ddof=0), like the LaTeX tables.
    return grouped.agg(mean="mean", std=lambda v: float(np.std(v, ddof=0)), n="count").reset_index()


def trace_sort_key(trace: str):
    return (0, int(trace), "") if str(trace).isdigit() else (1, 0, str(trace))


def modes(history: pd.DataFrame, datasets: list[str], levels: list[float]) -> list[tuple[str, float]]:
    """(benchmark, level) pairs that have runs, in --datasets x --levels order.

    One mode is one row block of the big table and one row of the agreement
    table. A benchmark without a sparsity sweep (PSM) simply yields a single
    mode at FULL_DATA_LEVEL.
    """
    present = set(map(tuple, history[["benchmark", "level"]].drop_duplicates().itertuples(index=False)))
    return [(bench, lvl) for bench in datasets for lvl in levels if (bench, lvl) in present]


def traces_by_benchmark(history: pd.DataFrame, datasets: list[str]) -> dict[str, list[str]]:
    """Rows of each benchmark's block: its trace/dataset ids, QAD's numerically sorted."""
    grouped = history.groupby("benchmark")["trace"].unique()
    return {bench: sorted(grouped[bench], key=trace_sort_key) for bench in datasets if bench in grouped.index}


def report_run_counts(history: pd.DataFrame, run_modes: list[tuple[str, float]], run_limit: int) -> None:
    counts = (
        history.groupby(["benchmark", "level", "variant"])["run_id"].nunique()
        .unstack("variant").reindex(index=pd.MultiIndex.from_tuples(run_modes, names=["benchmark", "level"]),
                                    columns=VARIANTS).fillna(0).astype(int)
    )
    print("\nRuns per benchmark/level/variant:")
    print(counts.to_string())
    short = [(bench, lvl, var) for bench, lvl in run_modes for var in VARIANTS
             if counts.loc[(bench, lvl), var] < run_limit]
    if short:
        # Runs are deduplicated by seed, so a short bucket means W&B holds fewer
        # distinct usable seeds than --run-limit; name the ones that were found.
        print(f"  note: fewer than {run_limit} runs (distinct seeds) for:")
        for bench, lvl, var in short:
            cell = history[(history["benchmark"] == bench) & (history["level"] == lvl)
                           & (history["variant"] == var)]
            seeds = sorted(cell.drop_duplicates("run_id")["seed"].dropna().astype(int).tolist())
            print(f"    {bench} {lvl:g}/{var}: {len(seeds)} run(s), seeds {seeds}")


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def level_label(level: float) -> str:
    return f"{level * 100:g}%"


def dataset_label(benchmark: str) -> str:
    """Display name of a benchmark without the trace-count suffix ("QAD" -> "QAPPD")."""
    return BENCHMARK_LABELS.get(benchmark, benchmark).split(" (")[0]


def metric_label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric)


def trace_title(trace: str) -> str:
    # Deliberately just the id: the anomaly type of a trace is not shown anywhere
    # in these figures/tables (it is part of the dataset description, not a result).
    return f"Trace {trace}"


def save(fig, out_dir: Path, stem: str, formats: list[str]) -> None:
    for fmt in formats:
        fig.savefig(out_dir / f"{stem}.{fmt}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def add_header(fig, axes, title: str | None, top: float) -> None:
    """Title at the very top, the shared legend right below it, the axes below `top`."""
    fig.tight_layout(rect=(0, 0, 1, top))
    if title:
        fig.suptitle(title, y=0.995, va="top", fontsize=11)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend_y = top + (1 - top) * (0.45 if title else 0.5)
    fig.legend(handles, labels, loc="center", bbox_to_anchor=(0.5, legend_y), ncol=len(VARIANTS), frameon=False)


def plot_trend_axis(ax, scores: pd.DataFrame, trace: str, metric: str, levels: list[float]) -> None:
    sub = scores[(scores["trace"] == trace) & (scores["metric"] == metric)]
    for variant in VARIANTS:
        rows = sub[sub["variant"] == variant].set_index("level").reindex(levels)
        x = np.arange(len(levels)) + VARIANT_X_SHIFT[variant]
        ax.errorbar(
            x, rows["mean"], yerr=rows["std"], label=VARIANT_LABEL[variant], color=VARIANT_COLOR[variant],
            marker=VARIANT_MARKER[variant], markersize=4.5, linewidth=1.6, capsize=3, elinewidth=1,
        )
    ax.set_xticks(range(len(levels)), [level_label(lvl) for lvl in levels])
    ax.set_xlim(-0.35, len(levels) - 0.65)
    ax.set_ylabel(metric_label(metric))
    ax.grid(axis="y", alpha=0.3)


def plot_trend_figures(scores: pd.DataFrame, traces: list[str], metrics: list[str], levels, out_dir, formats):
    for trace in traces:
        fig, axes = plt.subplots(1, len(metrics), figsize=(3.4 * len(metrics), 2.8), squeeze=False)
        for ax, metric in zip(axes[0], metrics):
            plot_trend_axis(ax, scores, trace, metric, levels)
            ax.set_xlabel("Training data")
        add_header(fig, axes, trace_title(trace), top=0.78)
        save(fig, out_dir, f"trend_trace{trace}", formats)

    # Overview: two traces per grid row, each with its metric panels side by side.
    per_row = 2
    n_rows = int(np.ceil(len(traces) / per_row))
    n_cols = per_row * len(metrics)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.6 * n_cols, 2.3 * n_rows), squeeze=False)
    for i, trace in enumerate(traces):
        for j, metric in enumerate(metrics):
            ax = axes[i // per_row, (i % per_row) * len(metrics) + j]
            plot_trend_axis(ax, scores, trace, metric, levels)
            ax.set_title(f"{trace_title(trace)}", fontsize=9)
    for k in range(len(traces), n_rows * per_row):
        for j in range(len(metrics)):
            axes[k // per_row, (k % per_row) * len(metrics) + j].set_visible(False)
    add_header(fig, axes, None, top=0.95)
    save(fig, out_dir, "trends_all", formats)


def robust_ylim(lows: np.ndarray, highs: np.ndarray, rising: bool) -> tuple[float, float] | None:
    """y-range covering the converged part of a curve, clipping its early excursion.

    Only the side the curve comes *from* is clipped (the bottom for a rising
    curve such as the log-likelihood, the top for a falling loss), so the
    region around the selected epoch is always fully visible.
    """
    lows, highs = lows[np.isfinite(lows)], highs[np.isfinite(highs)]
    if lows.size == 0 or highs.size == 0:
        return None
    if rising:
        lo, hi = np.percentile(lows, ROBUST_YLIM_CLIP_PCT), highs.max()
    else:
        lo, hi = lows.min(), np.percentile(highs, 100 - ROBUST_YLIM_CLIP_PCT)
    pad = 0.05 * (hi - lo) if hi > lo else 1.0
    return lo - pad, hi + pad


def plot_curve_figures(history, selected, traces, metrics, levels, min_runs, out_dir, formats):
    metrics = [m for m in metrics if m in history]
    for trace in traces:
        sub = history[history["trace"] == trace]
        fig, axes = plt.subplots(
            len(metrics), len(levels), figsize=(3.3 * len(levels), 2.2 * len(metrics)),
            squeeze=False, sharex="col",
        )
        for r, metric in enumerate(metrics):
            for c, level in enumerate(levels):
                ax = axes[r, c]
                cell_lows, cell_highs = [], []
                for variant in VARIANTS:
                    cell = sub[(sub["level"] == level) & (sub["variant"] == variant)]
                    stats = cell.groupby("epoch")[metric].agg(["mean", "std", "count"])
                    stats = stats[stats["count"] >= min_runs]
                    if stats.empty:
                        continue
                    std = stats["std"].fillna(0.0)
                    color = VARIANT_COLOR[variant]
                    ax.plot(stats.index, stats["mean"], color=color, linewidth=1.2, label=VARIANT_LABEL[variant])
                    ax.fill_between(stats.index, stats["mean"] - std, stats["mean"] + std, color=color, alpha=0.18,
                                    linewidth=0)
                    cell_lows.append((stats["mean"] - std).to_numpy())
                    cell_highs.append((stats["mean"] + std).to_numpy())
                    # Ticks along the bottom: each run's selected (best-val-loss) epoch.
                    picks = selected[(selected["trace"] == trace) & (selected["level"] == level)
                                     & (selected["variant"] == variant)]["epoch"]
                    for epoch in picks:
                        ax.axvline(epoch, color=color, alpha=0.35, linewidth=0.8, linestyle=":")
                ax.grid(alpha=0.3)
                if r == 0:
                    ax.set_title(f"{level_label(level)} training data", fontsize=10)
                if r == len(metrics) - 1:
                    ax.set_xlabel("Epoch")
                if c == 0:
                    ax.set_ylabel(metric_label(metric))
                # No shared y-axis: the 100% runs span a far wider range than the
                # sparse ones, which would flatten their curves.
                if metric in ROBUST_YLIM_METRICS and cell_lows:
                    ylim = robust_ylim(np.concatenate(cell_lows), np.concatenate(cell_highs),
                                       rising=metric in ROBUST_YLIM_RISING)
                    if ylim:
                        ax.set_ylim(*ylim)
        add_header(fig, axes, f"{trace_title(trace)}: mean $\\pm$ std over runs (dotted: each run's selected epoch)",
                   top=0.91)
        save(fig, out_dir, f"curves_trace{trace}", formats)


# ---------------------------------------------------------------------------
# Early-stopping state tables
# ---------------------------------------------------------------------------

def _row_cells(scores: pd.DataFrame, mode: tuple[str, float], trace: str, metrics: list[str]):
    """``{(metric, variant): (mean, std, best)}`` for one table row (one trace in one mode).

    `best` marks the better geometry of the pair for metrics in TABLE_BETTER.
    """
    benchmark, level = mode
    sub = scores[(scores["benchmark"] == benchmark) & (scores["trace"] == trace)
                 & (scores["level"] == level)].set_index(["variant", "metric"])
    cells = {}
    for metric in metrics:
        present = {v: sub.loc[(v, metric)] for v in VARIANTS if (v, metric) in sub.index}
        best = None
        direction = TABLE_BETTER.get(metric)
        finite = {v: r["mean"] for v, r in present.items() if np.isfinite(r["mean"])}
        if direction and len(finite) == len(VARIANTS):
            best = (max if direction > 0 else min)(finite, key=finite.get)
        for variant, row in present.items():
            cells[(metric, variant)] = (row["mean"], row["std"], variant == best)
    return cells


def _fmt_cell(mean: float, std: float, metric: str) -> tuple[str, str]:
    """Two decimals like wandb_results_to_latex.format_cell; the epoch is an integer."""
    dec = 0 if metric == "epoch" else 2
    return f"{mean:.{dec}f}", f"{std:.{dec}f}"


def _tex_value_cell(cells: dict, metric: str, variant: str) -> str:
    """``mean \\std{std}``, unmarked -- the better geometry is not highlighted in the tex table."""
    if (metric, variant) not in cells:
        return MISSING_CELL
    mean, std, _best = cells[(metric, variant)]
    m, sd = _fmt_cell(mean, std, metric)
    return f"{m} \\std{{{sd}}}"


def _md_value_cell(cells: dict, metric: str, variant: str) -> str:
    if (metric, variant) not in cells:
        return "–"
    mean, std, best = cells[(metric, variant)]
    m, sd = _fmt_cell(mean, std, metric)
    return f"**{m}** ± {sd}" if best else f"{m} ± {sd}"


def mode_label(mode: tuple[str, float], levels_per_benchmark: dict[str, int], tex: bool = True) -> str:
    """Block label of one mode, e.g. ``QAPPD (5\\%)`` or just ``PSM``.

    Spelled like wandb_results_to_latex.sparsity_column_label. The
    training-data fraction is named only for benchmarks that have more than
    one level -- PSM has no sparsity sweep, so "PSM (100%)" would suggest a
    comparison the table does not make.
    """
    benchmark, level = mode
    if levels_per_benchmark.get(benchmark, 1) < 2:
        return dataset_label(benchmark)
    pct = "\\%" if tex else "%"
    return f"{dataset_label(benchmark)} ({level * 100:g}{pct})"


def render_val_status_tex(scores: pd.DataFrame, run_modes: list[tuple[str, float]],
                          traces: dict[str, list[str]], metrics: list[str]) -> str:
    """All modes in one tabular: one row block per (benchmark, level).

    Laid out like wandb_results_to_latex's per-trace table: one column group
    per geometry (``\\multicolumn{..}{c}{\\small \\textbf{LSD on ...}}`` with a
    ``\\cmidrule`` under it), the metrics inside each group. Every block is
    labelled by a rotated cell in the leftmost column, and blocks are set off by
    ``\\addlinespace[2.5ex]`` and a rule.

    Just the ``tabular`` -- float, caption, label and any \\resizebox are the
    caller's. Needs booktabs, multirow, graphicx and, as for the other
    tables, the ``\\std`` macro in the preamble.
    """
    n_met = len(metrics)
    levels_per_benchmark = Counter(bench for bench, _ in run_modes)
    group_cells = " & ".join(
        f"\\multicolumn{{{n_met}}}{{c}}{{\\small \\textbf{{{VARIANT_TEX[v]}}}}}" for v in VARIANTS
    )
    cmidrules = "".join(
        f"\\cmidrule(lr){{{3 + i * n_met}-{2 + (i + 1) * n_met}}}" for i in range(len(VARIANTS))
    )
    metric_row = " & ".join(
        [r"\textbf{Data}", r"\textbf{Trace}"]
        + [TABLE_HEADER.get(m, (m, m))[0] for _ in VARIANTS for m in metrics]
    ) + r" \\"
    column_format = "cl " + " ".join(["r" * n_met] * len(VARIANTS))
    lines = [
        f"\\begin{{tabular}}{{{column_format}}}",
        r"\toprule",
        f"&& {group_cells} \\\\",
        cmidrules,
        metric_row,
        r"\midrule",
    ]
    for k, mode in enumerate(run_modes):
        if k > 0:
            lines += [r"\addlinespace[2.5ex]", r"\midrule"]
        block = traces[mode[0]]
        label_tex = mode_label(mode, levels_per_benchmark)
        for i, trace in enumerate(block):
            # A rotated label needs the height of several rows; a one-row block
            # (PSM) gets it upright instead, or it would stick out of its row.
            if i > 0:
                label = ""
            elif len(block) == 1:
                label = f"\\small {label_tex}"
            else:
                label = f"\\multirow{{{len(block)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{\\small {label_tex}}}}}"
            cells = _row_cells(scores, mode, trace, metrics)
            row = [label, trace] + [_tex_value_cell(cells, m, v) for v in VARIANTS for m in metrics]
            lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_val_status_md(scores: pd.DataFrame, run_modes: list[tuple[str, float]],
                         traces: dict[str, list[str]], metrics: list[str]) -> str:
    """The same table in Markdown: columns grouped by geometry, best of each pair in bold."""
    short = {"Rn": "Rn", "Sn": "Sn"}
    header = ["Data", "Trace"] + [f"{short[v]}: {TABLE_HEADER.get(m, (m, m))[1]}" for v in VARIANTS for m in metrics]
    levels_per_benchmark = Counter(bench for bench, _ in run_modes)
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 2 + ["---:"] * (len(VARIANTS) * len(metrics))) + "|"]
    for k, mode in enumerate(run_modes):
        if k > 0:  # visual gap between the blocks
            lines.append("| " + " | ".join([""] * len(header)) + " |")
        for i, trace in enumerate(traces[mode[0]]):
            cells = _row_cells(scores, mode, trace, metrics)
            row = ([f"**{mode_label(mode, levels_per_benchmark, tex=False)}**" if i == 0 else "", trace]
                   + [_md_value_cell(cells, m, v) for v in VARIANTS for m in metrics])
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines + [""])


def write_val_status_tables(scores, traces, run_modes, metrics, out_dir: Path) -> None:
    metrics = [m for m in metrics if m in set(scores["metric"])]
    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    (table_dir / "val_status_all.tex").write_text(render_val_status_tex(scores, run_modes, traces, metrics))
    md = ["# Early-stopping criterion at the selected (lowest val loss) epoch", "",
          "mean ± std over runs; bold = better geometry (loss/NLL: lower, AUC/AUPRC: higher); "
          "AUC and AUPRC x100.", "", render_val_status_md(scores, run_modes, traces, metrics)]
    (table_dir / "val_status.md").write_text("\n".join(md))


# ---------------------------------------------------------------------------
# Does the validation loss pick the better geometry?
# ---------------------------------------------------------------------------

def _winner(values: dict[str, float], direction: int) -> str | None:
    """Variant with the best value (direction +1: largest, -1: smallest), or None if undecided.

    Undecided means a variant is missing/NaN or the two are exactly equal: such
    a (trace, metric) pair says nothing about agreement and is therefore left
    out of both the numerator and the denominator of the match counts.
    """
    finite = {variant: value for variant, value in values.items() if np.isfinite(value)}
    if len(finite) < len(VARIANTS):
        return None
    ranked = sorted(finite.items(), key=lambda item: -direction * item[1])
    return None if ranked[0][1] == ranked[1][1] else ranked[0][0]


def _seed_agreement(selected: pd.DataFrame, mode: tuple[str, float], trace: str,
                    metric: str) -> tuple[int, int]:
    """(agreeing seeds, usable seeds) for one (mode, trace, score).

    The comparison is paired per seed rather than made once on the run means:
    runs are deduplicated by seed, so a seed identifies at most one run per
    geometry, and each such pair votes on its own -- which geometry had the
    lower validation loss, and which had the better test score. Seeds a
    geometry is missing are skipped, keeping every vote paired.
    """
    benchmark, level = mode
    sub = selected[(selected["benchmark"] == benchmark) & (selected["level"] == level)
                   & (selected["trace"] == trace)]
    if sub.empty or metric not in sub or VAL_LOSS not in sub:
        return 0, 0
    matches = pairs = 0
    for _, group in sub.groupby("seed", dropna=True):
        by_variant = group.drop_duplicates("variant").set_index("variant")
        if not set(VARIANTS) <= set(by_variant.index):
            continue
        val_pick = _winner({v: by_variant.at[v, VAL_LOSS] for v in VARIANTS}, TABLE_BETTER[VAL_LOSS])
        score_pick = _winner({v: by_variant.at[v, metric] for v in VARIANTS}, TABLE_BETTER.get(metric, +1))
        if val_pick is None or score_pick is None:
            continue
        pairs += 1
        matches += int(val_pick == score_pick)
    return matches, pairs


def agreement_rows(scores: pd.DataFrame, selected: pd.DataFrame, run_modes: list[tuple[str, float]],
                   traces: dict[str, list[str]], metrics: list[str]) -> pd.DataFrame:
    """One row per (mode, trace, score): which geometry each criterion picks, and whether they agree.

    Two views of the same question, both kept so the CSV explains the table:
    `match` compares the geometries on the run means of that trace (one vote
    per trace), `seed_matches`/`seed_pairs` compare them seed by seed (one vote
    per trace and seed). The table reports the pooled seed votes.
    """
    rows = []
    for mode in run_modes:
        benchmark, level = mode
        for trace in traces[benchmark]:
            mean = scores[(scores["benchmark"] == benchmark) & (scores["trace"] == trace)
                          & (scores["level"] == level)].set_index(["metric", "variant"])["mean"]
            val_pick = _winner({v: mean.get((VAL_LOSS, v), np.nan) for v in VARIANTS}, TABLE_BETTER[VAL_LOSS])
            for metric in metrics:
                score_pick = _winner({v: mean.get((metric, v), np.nan) for v in VARIANTS},
                                     TABLE_BETTER.get(metric, +1))
                seed_matches, seed_pairs = _seed_agreement(selected, mode, trace, metric)
                rows.append({
                    "benchmark": benchmark, "level": level, "trace": trace, "metric": metric,
                    "val_pick": val_pick, "score_pick": score_pick,
                    "match": None if val_pick is None or score_pick is None else val_pick == score_pick,
                    "seed_matches": seed_matches, "seed_pairs": seed_pairs,
                })
    return pd.DataFrame(rows)


def _agreement_totals(rows: pd.DataFrame, mode: tuple[str, float], metric: str) -> tuple[int, int]:
    """(agreeing seeds, usable seeds) pooled over every trace and seed of one (mode, score).

    For QAD that pools 10 traces x 5 seeds = 50 paired votes per mode; a
    single-dataset benchmark pools its one trace's seeds.
    """
    benchmark, level = mode
    sub = rows[(rows["benchmark"] == benchmark) & (rows["level"] == level) & (rows["metric"] == metric)]
    return int(sub["seed_matches"].sum()), int(sub["seed_pairs"].sum())


def score_short(metric: str, tex: bool) -> str:
    return SCORE_SHORT.get(metric, (metric, metric))[0 if tex else 1]


def _share(matches: int, total: int, pct: str, missing: str) -> str:
    return missing if total == 0 else f"{matches}/{total} ({100 * matches / total:.0f}{pct})"


def _agreement_body(rows: pd.DataFrame, run_modes: list[tuple[str, float]], metrics: list[str],
                    pct: str, missing: str) -> list[list[str]]:
    """Table body: one row per mode, one cell per score, all of that mode's (trace, seed) pairs pooled."""
    levels_per_benchmark = Counter(bench for bench, _ in run_modes)
    body = []
    for mode in run_modes:
        row = [mode_label(mode, levels_per_benchmark, tex=pct != "%")]
        for metric in metrics:
            row.append(_share(*_agreement_totals(rows, mode, metric), pct=pct, missing=missing))
        body.append(row)
    return body


def render_agreement_tex(rows: pd.DataFrame, run_modes: list[tuple[str, float]], metrics: list[str]) -> str:
    """One small ``tabular``: per mode, the share of (trace, seed) pairs where both criteria agree.

    Just the ``tabular`` (needs booktabs); float, caption and label are the
    caller's, as for `render_val_status_tex`.
    """
    body = _agreement_body(rows, run_modes, metrics, pct=r"\%", missing=MISSING_CELL)
    lines = [
        f"\\begin{{tabular}}{{l{'c' * len(metrics)}}}",
        r"\toprule",
        " & ".join([r"\textbf{Data}"] + [score_short(m, tex=True) for m in metrics]) + r" \\",
        r"\midrule",
    ]
    lines += [" & ".join(row) + r" \\" for row in body]
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def render_agreement_md(rows: pd.DataFrame, run_modes: list[tuple[str, float]], metrics: list[str]) -> str:
    header = ["Data"] + [score_short(m, tex=False) for m in metrics]
    body = _agreement_body(rows, run_modes, metrics, pct="%", missing="–")
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] + ["---:"] * len(metrics)) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines + [""])


def write_agreement_tables(scores, selected, traces, run_modes, metrics, out_dir: Path) -> None:
    metrics = [m for m in metrics if m in set(scores["metric"])]
    if not metrics or VAL_LOSS not in set(scores["metric"]):
        print("  warning: no val loss / score metrics for the geometry-agreement table -- skipping it.")
        return
    rows = agreement_rows(scores, selected, run_modes, traces, metrics)
    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    # The pooled counts only; the per-trace picks and seed votes stay in the CSV.
    rows.to_csv(table_dir / "geometry_agreement.csv", index=False)
    (table_dir / "geometry_agreement.tex").write_text(render_agreement_tex(rows, run_modes, metrics))
    (table_dir / "geometry_agreement.md").write_text("\n".join([
        "# Does the validation loss pick the better geometry?", "",
        "Every (trace, seed) pair votes once: for that seed, the geometry with the lower validation loss at "
        "its selected epoch vs. the geometry with the higher test score. The cells pool all pairs of that "
        "benchmark and amount of training data (agreeing pairs / all pairs; 10 traces x 5 seeds = 50 for "
        "QAPPD). Pairs where a geometry is missing for that seed or the two values are exactly tied are left "
        "out of both counts. Each score is evaluated independently. `geometry_agreement.csv` also keeps the "
        "per-trace picks taken on the run means.", "",
        render_agreement_md(rows, run_modes, metrics),
    ]))

    print("\nValidation-loss pick vs. best-scoring geometry (agreeing (trace, seed) pairs / usable pairs):")
    levels_per_benchmark = Counter(bench for bench, _ in run_modes)
    width = max(len(mode_label(m, levels_per_benchmark, tex=False)) for m in run_modes)
    for mode in run_modes:
        parts = [f"{score_short(m, tex=False)} {_share(*_agreement_totals(rows, mode, m), pct='%', missing='n/a')}"
                 for m in metrics]
        print(f"  {mode_label(mode, levels_per_benchmark, tex=False):>{width}}: " + " | ".join(parts))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history, summary = load_or_fetch(args)
    if history.empty:
        raise SystemExit("No usable runs found.")
    history["trace"] = history["trace"].astype(str)
    run_modes = modes(history, args.datasets, args.levels)
    traces = traces_by_benchmark(history, args.datasets)
    report_run_counts(history, run_modes, args.run_limit)

    history = add_derived_columns(history)
    for metric in set(args.trend_metrics) | set(args.curve_metrics):
        if metric not in history:
            print(f"  warning: metric {metric!r} not in the history; available: "
                  + ", ".join(c for c in history.columns if c not in META_COLS))

    selected = selected_epochs(history)
    check_against_summary(selected, summary)

    # "epoch" is aggregated too, so the tables can report the selected epoch.
    metric_cols = [c for c in history.columns if c not in META_COLS] + ["epoch"]
    scores = aggregate_selected(selected, metric_cols)
    scores.sort_values(["benchmark", "trace", "metric", "variant", "level"],
                       key=lambda col: col.map(trace_sort_key) if col.name == "trace" else col) \
        .to_csv(args.output_dir / "selected_epoch_scores.csv", index=False)

    # The figures are about the sparsity trend, so they only cover benchmarks
    # that actually have more than one data level (PSM has no sparsity sweep).
    trend_metrics = [m for m in args.trend_metrics if m in history]
    for benchmark in args.datasets:
        levels = [lvl for bench, lvl in run_modes if bench == benchmark]
        if len(levels) < 2:
            if levels:
                print(f"  note: {benchmark} has runs at {level_label(levels[0])} training data only -- "
                      f"it appears in the tables, but not in the trend/curve figures.")
            continue
        plot_trend_figures(scores[scores["benchmark"] == benchmark], traces[benchmark], trend_metrics,
                           levels, args.output_dir, args.formats)
        plot_curve_figures(history[history["benchmark"] == benchmark], selected[selected["benchmark"] == benchmark],
                           traces[benchmark], args.curve_metrics, levels, args.min_runs_per_epoch,
                           args.output_dir, args.formats)

    write_val_status_tables(scores, traces, run_modes, args.table_metrics, args.output_dir)
    write_agreement_tables(scores, selected, traces, run_modes, args.agreement_metrics, args.output_dir)
    print(f"\nWrote trend + curve figures and early-stopping tables for "
          f"{sum(len(t) for t in traces.values())} traces to {args.output_dir}")


if __name__ == "__main__":
    main()
