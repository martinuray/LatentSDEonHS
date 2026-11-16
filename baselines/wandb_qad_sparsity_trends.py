#!/usr/bin/env python3
r"""Per-trace QAPPD (QAD) sparsity trends of our two LSD variants (matplotlib).

Companion to ``wandb_results_to_latex.py``: same W&B project and run selection
rules, but it pulls the *full epoch history* of every selected run and plots,
per QAD trace, LSD on R^n vs LSD on S^n at 100%, 5% and 1% training data.

Data levels (all QAD, decimation QAD_REQUIRED_DECIMATION, full-benchmark runs
only -- --trace-ids subsets are skipped):
  - 100%: the main-table runs (``args.fixed_subsample_mask`` unset; the model
    sees the whole training series, resampling its subsample mask each
    iteration).
  - 5% / 1%: the sparsity sweep (``args.fixed_subsample_mask`` set, level from
    ``args.subsample``) -- exactly the runs the sparsity table reports.
Per (level, variant), runs are deduplicated by seed (most recent wins) and cut
to the --run-limit most recent ones, as in the LaTeX tables.

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
``tst/auc_pr`` against ``summary/per_dataset/<trace>/auc_pr`` and warns on a
mismatch.

Figures (into --output-dir, one file per --formats entry):
  - ``trend_trace<id>``: metrics at the selected epoch vs. training-data
    fraction (one panel per --trend-metrics entry), mean +/- std over runs.
  - ``trends_all``: the same for all traces in one grid.
  - ``curves_trace<id>``: metric curves over epochs, rows = --curve-metrics,
    columns = data levels; mean over runs with a +/- std band. Runs stop early
    at different epochs, so each epoch averages only the runs that reached it
    (the band ends where fewer than --min-runs-per-epoch remain).
Tables (into --output-dir/tables):
  - ``val_status_all.tex``: one ``table*`` covering every data level, as
    row blocks (100%, 5%, 1% training data) separated by vertical space and a
    rule, each block labelled by a rotated cell in the leftmost column. Each
    row is one trace (with its anomaly type) and shows the state of the
    early-stopping criterion at the selected (lowest val loss) epoch -- the
    selected epoch, ``val/loss`` (the criterion itself), ``val/log_pxz`` (the
    reconstruction NLL) and the test AUC-ROC and AUPRC (x100) the run reports
    for that epoch, ``mean \std{std}`` over runs. Each metric is a column
    group with Rn and Sn side by side; the better value of each pair (val
    loss, NLL: lower; AUC-ROC, AUPRC: higher) is set in bold. Needs booktabs,
    multirow and graphicx; ``\std`` must be defined in the preamble as for the
    other tables (e.g. ``\newcommand{\std}[1]{\,{\scriptsize$\pm$#1}}``).
  - ``val_status.md``: the same table in Markdown, for a quick look.
Plus ``selected_epoch_scores.csv``: mean/std/n at the selected epoch for every
(trace, level, variant, metric).
"""

import argparse
import json
import re
from collections import defaultdict
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
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_LIMIT,
    OURS_DEFAULT_ENTITY,
    OURS_DEFAULT_PROJECT,
    QAD_BENCHMARK,
    QAD_REQUIRED_DECIMATION,
    anomaly_type,
    dedupe_by_seed,
    uses_fixed_subsample_mask,
)

# Fraction of training data available -> plotted left to right in this order.
# 1.0 stands for the main-table runs (no fixed subsample mask), see module doc.
DEFAULT_LEVELS = [1.0, 0.05, 0.01]
FULL_DATA_LEVEL = 1.0

VARIANTS = ["Rn", "Sn"]
VARIANT_LABEL = {"Rn": r"LSD on $\mathbb{R}^n$", "Sn": r"LSD on $\mathbb{S}^n$"}
VARIANT_COLOR = {"Rn": "#1F77B4", "Sn": "#D62728"}
VARIANT_MARKER = {"Rn": "s", "Sn": "o"}
# Horizontal offset (in category units) so the two variants' error bars don't overlap.
VARIANT_X_SHIFT = {"Rn": -0.06, "Sn": 0.06}

META_COLS = ["run_id", "run_name", "level", "variant", "seed", "created_at", "min_delta", "trace", "epoch"]
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
    "tst/auc_roc": "AUC-ROC (x100)",
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
TABLE_HEADER = {
    "epoch": ("Epoch", "Epoch"),
    VAL_LOSS: (r"Val.\ loss $\downarrow$", "Val. loss (criterion)"),
    "val/log_pxz": (r"$-\log p(x|z)$ $\downarrow$", "-log p(x|z)"),
    "val/kl0": (r"$\mathrm{KL}_0$", "KL0"),
    "val/klp": (r"$\mathrm{KL}_{\mathrm{path}}$", "KL path"),
    VAL_LL: ("Val.\\ log-lik.\\ $\\uparrow$", "Val. log-lik."),
    TST_AUPRC: (r"AUPRC $\uparrow$", "AUPRC (x100)"),
    "tst/auc_roc": (r"AUC-ROC $\uparrow$", "AUC-ROC (x100)"),
}
# Metrics with a "better" direction get the better geometry of each level in
# bold (+1: higher is better, -1: lower is better). The KL terms and the epoch
# have none -- a smaller KL is not per se better -- so they are never highlighted.
TABLE_BETTER = {VAL_LOSS: -1, "val/log_pxz": -1, "val/elbo": -1, VAL_LL: +1, TST_AUPRC: +1, "tst/auc_roc": +1}
VARIANT_TEX = {"Rn": r"$\mathbb{R}^n$", "Sn": r"$\mathbb{S}^n$"}

HISTORY_CACHE = "qad_epoch_history.csv.gz"
SUMMARY_CACHE = "qad_summary_auprc.csv"
CACHE_META = "qad_epoch_history.meta.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--project", default=OURS_DEFAULT_PROJECT, help="W&B project of our LSD runs.")
    parser.add_argument("--entity", default=OURS_DEFAULT_ENTITY, help="W&B entity for --project.")
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


def run_level(config: dict) -> float | None:
    """Training-data fraction a run belongs to, or None if it's not a QAD trend run."""
    run_context = config.get("run_context") or {}
    run_args = config.get("args") or {}
    # Exact match: "QAD:1,2" (--trace-ids subsets) never averages every trace.
    if run_context.get("benchmark_name") != QAD_BENCHMARK:
        return None
    if run_args.get("data_decimation_factor") != QAD_REQUIRED_DECIMATION:
        return None
    if not uses_fixed_subsample_mask(config):
        return FULL_DATA_LEVEL
    subsample = run_args.get("subsample")
    return float(subsample) if _is_number(subsample) else None


def per_trace_summary_auprc(summary: dict) -> dict[str, float]:
    """``{trace_id: auc_pr}`` from ``summary/per_dataset/<id>/auc_pr``."""
    out = {}
    prefix, suffix = "summary/per_dataset/", "/auc_pr"
    for key, value in summary.items():
        if key.startswith(prefix) and key.endswith(suffix) and _is_number(value):
            out[key[len(prefix):-len(suffix)]] = float(value)
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
    project: str, entity: str | None, levels: list[float], run_limit: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (epoch history, summary AUPRC) of the selected runs, both long-format.

    Run selection uses only the cheap run metadata/summary; the (slow) epoch
    history is scanned afterwards, for the selected runs only.
    """
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = api.runs(path, filters={"config.run_context.benchmark_name": QAD_BENCHMARK}, order="-created_at")

    candidates: dict[tuple[float, str], list] = defaultdict(list)
    run_by_id = {}
    for run in runs:
        config = run.config or {}
        level = run_level(config)
        if level is None or level not in levels:
            continue
        variant = (config.get("run_context") or {}).get("model_variant")
        if variant not in VARIANTS:
            continue
        summary_auprc = per_trace_summary_auprc(dict(run.summary))
        if not summary_auprc:
            continue  # crashed/killed before the final per-trace summary was written
        created_at = pd.to_datetime(run.created_at, utc=True)
        seed = (config.get("run_context") or {}).get("run_seed")
        run_by_id[run.id] = (run, summary_auprc)
        # dedupe_by_seed expects (created_at, seed, payload); the payload is the run id.
        candidates[(level, variant)].append((created_at, seed, run.id))

    history_frames, summary_rows = [], []
    for (level, variant), entries in sorted(dedupe_by_seed(candidates).items()):
        selected = sorted(entries, key=lambda entry: entry[0], reverse=True)[:run_limit]
        for created_at, seed, run_id in selected:
            run, summary_auprc = run_by_id[run_id]
            print(f"  scanning history: {level:g} {variant} seed={seed} ({run.name})")
            meta = {
                "run_id": run_id, "run_name": run.name, "level": level, "variant": variant, "seed": seed,
                "created_at": created_at.isoformat(),
                "min_delta": float((run.config.get("args") or {}).get("early_stopping_min_delta") or 0.0),
            }
            rows = history_rows(run.scan_history(page_size=10000))
            history_frames.append(pd.DataFrame([{**meta, **row} for row in rows]))
            summary_rows += [{**meta, "trace": trace, "auc_pr": value} for trace, value in summary_auprc.items()]

    history = pd.concat(history_frames, ignore_index=True) if history_frames else pd.DataFrame(columns=META_COLS)
    metric_cols = sorted(col for col in history.columns if col not in META_COLS)
    history = history[META_COLS + metric_cols].sort_values(["level", "variant", "run_id", "trace", "epoch"])
    return history.reset_index(drop=True), pd.DataFrame(summary_rows)


def load_or_fetch(args) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = args.output_dir
    cache_key = {"project": args.project, "entity": args.entity, "levels": args.levels, "run_limit": args.run_limit}
    meta_path, hist_path, summ_path = out / CACHE_META, out / HISTORY_CACHE, out / SUMMARY_CACHE
    if not args.refresh and meta_path.exists() and hist_path.exists() and summ_path.exists():
        if json.loads(meta_path.read_text()) == cache_key:
            print(f"Using cached history from {hist_path} (pass --refresh to re-fetch).")
            dtypes = {"trace": str, "run_id": str}
            return pd.read_csv(hist_path, dtype=dtypes), pd.read_csv(summ_path, dtype=dtypes)

    print(f"Fetching QAD runs from W&B project {args.entity}/{args.project} ...")
    history, summary = fetch_history(args.project, args.entity, args.levels, args.run_limit)
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
    long = selected.melt(id_vars=["trace", "level", "variant"], value_vars=metrics, var_name="metric")
    long = long.dropna(subset=["value"])
    grouped = long.groupby(["trace", "level", "variant", "metric"])["value"]
    # Population std (ddof=0), like the LaTeX tables.
    return grouped.agg(mean="mean", std=lambda v: float(np.std(v, ddof=0)), n="count").reset_index()


def trace_sort_key(trace: str):
    return (0, int(trace), "") if str(trace).isdigit() else (1, 0, str(trace))


def report_run_counts(history: pd.DataFrame, levels: list[float], run_limit: int) -> None:
    counts = (
        history.groupby(["level", "variant"])["run_id"].nunique()
        .unstack("variant").reindex(index=levels, columns=VARIANTS).fillna(0).astype(int)
    )
    print("\nRuns per level/variant:")
    print(counts.to_string())
    short = [(lvl, var) for lvl in levels for var in VARIANTS if counts.loc[lvl, var] < run_limit]
    if short:
        print(f"  note: fewer than {run_limit} runs for: " + ", ".join(f"{lvl:g}/{var}" for lvl, var in short))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def level_label(level: float) -> str:
    return f"{level * 100:g}%"


def metric_label(metric: str) -> str:
    return METRIC_LABEL.get(metric, metric)


def trace_title(trace: str) -> str:
    kind = anomaly_type(trace)
    return f"Trace {trace}" + (f" ({kind})" if kind else "")


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

def _decimals(value: float) -> int:
    """Decimals for ~3 significant digits.

    Chosen per cell, not per column: within one column the geometries can
    differ by orders of magnitude (KL0 is ~2000 on R^n but ~10 on S^n), and a
    shared precision would round the small side to nothing.
    """
    magnitude = abs(value)
    if magnitude >= 100:
        return 0
    if magnitude >= 10:
        return 1
    if magnitude >= 1:
        return 2
    return 3


def _row_cells(scores: pd.DataFrame, level: float, trace: str, metrics: list[str]):
    """``{(metric, variant): (mean, std, bold)}`` for one table row (one trace at one level).

    `bold` marks the better geometry of the pair for metrics in TABLE_BETTER.
    """
    sub = scores[(scores["trace"] == trace) & (scores["level"] == level)].set_index(["variant", "metric"])
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
    dec = 0 if metric == "epoch" else _decimals(mean)
    return f"{mean:.{dec}f}", f"{std:.{dec}f}"


def _value_cells(cells: dict, metrics: list[str], bold_fmt, pm: str, missing: str) -> list[str]:
    """Rn/Sn value cells of one row, in column order."""
    row = []
    for metric in metrics:
        for variant in VARIANTS:
            if (metric, variant) not in cells:
                row.append(missing)
                continue
            mean, std, bold = cells[(metric, variant)]
            m, sd = _fmt_cell(mean, std, metric)
            row.append(f"{bold_fmt(m) if bold else m}{pm(sd)}")
    return row


def render_val_status_tex(scores: pd.DataFrame, levels: list[float], traces: list[str], metrics: list[str]) -> str:
    """All data levels in one table: one row block per level, separated by space and a rule.

    The leftmost column names each block's training-data fraction (rotated,
    spanning the block); rows within a block are traces, column groups are
    metrics, each split into Rn | Sn.
    """
    n_var = len(VARIANTS)
    n_cols = 3 + n_var * len(metrics)
    group_header = [r"\multirow{2}{*}{Data}", r"\multirow{2}{*}{Trace}", r"\multirow{2}{*}{Type}"] + [
        f"\\multicolumn{{{n_var}}}{{c}}{{{TABLE_HEADER.get(m, (m, m))[0]}}}" for m in metrics
    ]
    rules = " ".join(
        f"\\cmidrule(lr){{{4 + i * n_var}-{3 + (i + 1) * n_var}}}" for i in range(len(metrics))
    )
    variant_header = ["", "", ""] + [VARIANT_TEX[v] for _ in metrics for v in VARIANTS]
    lines = [
        "% QAPPD: early-stopping criterion at the selected (lowest val loss) epoch, per trace and data level.",
        "% Generated by baselines/wandb_qad_sparsity_trends.py -- do not edit by hand.",
        "% Needs booktabs, multirow, graphicx and the \\std macro.",
        r"\begin{table*}[ht]",
        r"\centering",
        r"\small",
        r"\caption{QAPPD per trace and amount of training data: validation and test metrics at the epoch of "
        r"lowest validation loss, LSD on $\mathbb{R}^n$ vs.\ $\mathbb{S}^n$ (mean $\pm$ std over runs; "
        r"AUC-ROC and AUPRC $\times 100$; bold: better geometry).}",
        r"\label{tab:qad_val_status}",
        r"\resizebox{\linewidth}{!}{%",
        f"\\begin{{tabular}}{{clc{'r' * (n_var * len(metrics))}}}",
        r"\toprule",
        " & ".join(group_header) + r" \\",
        rules,
        " & ".join(variant_header) + r" \\",
        r"\midrule",
    ]
    for k, level in enumerate(levels):
        if k > 0:
            lines += [r"\addlinespace[0.8em]", r"\midrule"]
        level_tex = level_label(level).replace("%", r"\%")
        for i, trace in enumerate(traces):
            label = (f"\\multirow{{{len(traces)}}}{{*}}{{\\rotatebox[origin=c]{{90}}{{{level_tex} training data}}}}"
                     if i == 0 else "")
            cells = _row_cells(scores, level, trace, metrics)
            row = [label, trace, anomaly_type(trace)] + _value_cells(
                cells, metrics, bold_fmt=lambda m: "\\textbf{" + m + "}",
                pm=lambda sd: f" \\std{{{sd}}}", missing="--",
            )
            lines.append(" & ".join(row) + r" \\")
    assert n_cols == len(row)
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    return "\n".join(lines)


def render_val_status_md(scores: pd.DataFrame, levels: list[float], traces: list[str], metrics: list[str]) -> str:
    header = ["Data", "Trace", "Type"] + [f"{TABLE_HEADER.get(m, (m, m))[1]} {v}" for m in metrics for v in VARIANTS]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join(["---"] * 3 + ["---:"] * (len(VARIANTS) * len(metrics))) + "|"]
    for k, level in enumerate(levels):
        if k > 0:  # visual gap between the level blocks
            lines.append("| " + " | ".join([""] * len(header)) + " |")
        for i, trace in enumerate(traces):
            cells = _row_cells(scores, level, trace, metrics)
            row = [f"**{level_label(level)}**" if i == 0 else "", trace, anomaly_type(trace)] + _value_cells(
                cells, metrics, bold_fmt=lambda m: f"**{m}**", pm=lambda sd: f" ± {sd}", missing="–",
            )
            lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines + [""])


def write_val_status_tables(scores, traces, levels, metrics, out_dir: Path) -> None:
    metrics = [m for m in metrics if m in set(scores["metric"])]
    table_dir = out_dir / "tables"
    table_dir.mkdir(exist_ok=True)
    (table_dir / "val_status_all.tex").write_text(render_val_status_tex(scores, levels, traces, metrics))
    md = ["# QAPPD: early-stopping criterion at the selected (lowest val loss) epoch", "",
          "mean ± std over runs; bold = better geometry (loss/NLL: lower, AUC-ROC/AUPRC: higher); "
          "AUC-ROC and AUPRC x100.", "", render_val_status_md(scores, levels, traces, metrics)]
    (table_dir / "val_status.md").write_text("\n".join(md))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    history, summary = load_or_fetch(args)
    if history.empty:
        raise SystemExit("No usable runs found.")
    history["trace"] = history["trace"].astype(str)
    report_run_counts(history, args.levels, args.run_limit)

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
    scores.sort_values(["trace", "metric", "variant", "level"],
                       key=lambda col: col.map(trace_sort_key) if col.name == "trace" else col) \
        .to_csv(args.output_dir / "selected_epoch_scores.csv", index=False)

    traces = sorted(history["trace"].unique(), key=trace_sort_key)
    trend_metrics = [m for m in args.trend_metrics if m in history]
    plot_trend_figures(scores, traces, trend_metrics, args.levels, args.output_dir, args.formats)
    plot_curve_figures(history, selected, traces, args.curve_metrics, args.levels, args.min_runs_per_epoch,
                       args.output_dir, args.formats)
    write_val_status_tables(scores, traces, args.levels, args.table_metrics, args.output_dir)
    print(f"\nWrote trend + curve figures and early-stopping tables for {len(traces)} traces to {args.output_dir}")


if __name__ == "__main__":
    main()
