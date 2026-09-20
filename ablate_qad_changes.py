#!/usr/bin/env python
"""End-to-end ablation of the 2026-09-16 QAD pipeline changes. Single script, no SLURM.

Background (see done.md): between the previous QAD configuration
(cfg/anomaly_detection/QAD.json at bec6e6c~1) and the current one, six groups
of settings changed at once and macro F1 went from ~0.13 to ~0.5. This script
tells the groups apart by training the latent SDE with every change group
switched between its OLD and NEW value, then writes a report:

  new_all            all changes (= current cfg/anomaly_detection/QAD.json)
  old_all            none of the changes (= the previous configuration)
  new_minus_<group>  leave-one-out: everything new, one group reverted -> "is it necessary?"
  old_plus_<group>   add-one-in:   everything old, one group applied  -> "is it sufficient?"
                     (only with --design add|both)

Change groups (OLD -> NEW):
  windowing  100 Hz, window 280, stride 280   -> 10 Hz decimation, window 200, overlap 0.9
  kl         kl0 1e-5, klp 1e-5              -> kl0 1e-3, klp 1e-2
  subsample  observation subsample 0.5        -> 0.1
  capacity   z 5, h 12, n_deg 10              -> z 4, h 20, n_deg 5
  sigma      initial sigma 0.05               -> 0.22
  scoring    max aggregation, raw scores, smoothing 5 -> weighted-mse, normalised, smoothing 10

Not ablated (baked into the code, active in every arm): the Current0 column
fix, honoured window overlap and the rounded stride.

Each (arm, seed, trace) is one training run of anomaly_detection.py on a
single trace, executed sequentially with the Python interpreter that runs this
script. Every run gets a private data directory (raw QAD data symlinked,
processed windows per run), so runs never wipe each other's processed data.
Runs are ordered trace-major (all arms of trace 1, then trace 2, ...), so a
run that is stopped early still gives a balanced comparison. Finished runs are
skipped when the script is restarted with the same --name.

Usage (from the repository root, in the environment that runs anomaly_detection.py):
  python ablate_qad_changes.py --plan-only                # arms, run count, time estimate
  python ablate_qad_changes.py --name abl1                # run everything, then report
  python ablate_qad_changes.py --name abl1 --traces 1 2 3 4 5 6 7 8 9 10 --seeds 1 2   # full version
  python ablate_qad_changes.py --name abl1 --max-hours 6  # stop launching new runs after 6 h
  python ablate_qad_changes.py --name abl1 --collect-only # only rebuild the report

Results: out/ablation_qad/<name>/{report.md, results_summary.csv, results_per_trace.csv}
Extra flags for anomaly_detection.py go after "--", e.g. `-- --no-sphere-embedding`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_ROOT = PROJECT_DIR / "out" / "ablation_qad"
DEFAULT_RAW_DIR = PROJECT_DIR / "data_dir" / "QAD" / "raw"
ALL_TRACES = list(range(1, 11))
# range anomalies (1, 5), a hard trace (3) and dynamics anomalies (4, 9); ~40 runs with 8 arms
DEFAULT_TRACES = [1, 3, 4, 5, 9]
SEC_PER_EPOCH_GUESS = 3.0  # rtx2080ti, 10 Hz config incl. eval every 10 epochs (2026-09-16 logs)
RUN_OVERHEAD_S = 60.0

METRICS = ["f1", "auc_roc", "auc_pr", "vus_roc", "vus_pr", "precision", "recall"]

# ---------------------------------------------------------------------------
# Change groups
# ---------------------------------------------------------------------------
GROUPS: "OrderedDict[str, dict]" = OrderedDict(
    windowing=dict(
        old=dict(data_decimation_factor=1, data_window_length=280, data_window_overlap=0.0),
        new=dict(data_decimation_factor=10, data_window_length=200, data_window_overlap=0.9),
        why="decimate to 10 Hz + strided 20 s windows (anomalies were longer than the old 2.8 s windows)",
    ),
    kl=dict(
        old=dict(kl0_weight=1e-5, klp_weight=1e-5),
        new=dict(kl0_weight=1e-3, klp_weight=1e-2),
        why="KL terms were effectively off",
    ),
    subsample=dict(
        old=dict(subsample=0.5),
        new=dict(subsample=0.1),
        why="observation subsample at 10 Hz",
    ),
    capacity=dict(
        old=dict(z_dim=5, h_dim=12, n_deg=10),
        new=dict(z_dim=4, h_dim=20, n_deg=5),
        why="smaller latent / path degree, wider hidden",
    ),
    scoring=dict(
        old=dict(normalize_score=False, score_aggregation="max", score_smoothing_window=5),
        new=dict(normalize_score=True, score_aggregation="weighted-mse", score_smoothing_window=10),
        why="score post-processing only (no effect on training)",
    ),
)


def build_arms(design: str, only: list[str] | None):
    """Ordered dict arm_name -> (state per group, reference arm)."""
    arms = OrderedDict()
    all_new = {g: "new" for g in GROUPS}
    all_old = {g: "old" for g in GROUPS}
    arms["new_all"] = (dict(all_new), None)
    arms["old_all"] = (dict(all_old), None)
    if design in ("loo", "both"):
        for g in GROUPS:
            st = dict(all_new)
            st[g] = "old"
            arms[f"new_minus_{g}"] = (st, "new_all")
    if design in ("add", "both"):
        for g in GROUPS:
            st = dict(all_old)
            st[g] = "new"
            arms[f"old_plus_{g}"] = (st, "old_all")
    if only:
        unknown = sorted(set(only) - set(arms))
        if unknown:
            raise SystemExit(f"Unknown arm(s) {unknown}. Available: {list(arms)}")
        arms = OrderedDict((k, v) for k, v in arms.items() if k in only)
    return arms


def arm_params(state: dict) -> dict:
    params = {}
    for g, which in state.items():
        params.update(GROUPS[g][which])
    return params


def params_to_cli(params: dict) -> list[str]:
    out = []
    for key, value in params.items():
        flag = key.replace("_", "-")
        if isinstance(value, bool):
            out.append(f"--{flag}" if value else f"--no-{flag}")
        else:
            out.extend([f"--{flag}", str(value)])
    return out


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------
def make_runs(args, arms):
    """Trace-major order: all arms x seeds for trace 1, then trace 2, ..."""
    runs = []
    for trace in args.traces:
        for seed in args.seeds:
            for arm, (state, ref) in arms.items():
                runs.append(dict(
                    run_name=f"{arm}__s{seed}__t{trace}",
                    arm=arm, seed=seed, trace=trace, reference=ref,
                    params=arm_params(state),
                ))
    return runs


def run_command(args, run: dict, run_dir: Path) -> list[str]:
    cmd = [
        sys.executable, "anomaly_detection.py",
        "--dataset", "QAD",
        "--trace-ids", str(run["trace"]),
        "--runs", "1",
        "--seed", str(run["seed"]),
        "--n-epochs", str(args.n_epochs),
        "--batch-size", str(args.batch_size),
        "--data-dir", str(run_dir / "data"),
        "--enable-file-logging",
        "--log-dir", str(run_dir / "logs"),
        "--no-enable-checkpointing",
        "--final-metrics-csv", str(run_dir / "final_metrics.csv"),
        "--device", args.device,
        "--delete-processed-data",
        "--wandb-mode", args.wandb_mode,
        "--wandb-group", f"qad-ablation-{args.name}",
        "--wandb-name", run["run_name"],
        "--wandb-tags", "qad-ablation", args.name, run["arm"],
    ]
    cmd += params_to_cli(run["params"])
    cmd += list(args.extra_args)
    return cmd


def prepare_run_dir(args, run: dict, out_dir: Path) -> Path:
    run_dir = out_dir / "runs" / run["run_name"]
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    qad_dir = run_dir / "data" / "QAD"
    qad_dir.mkdir(parents=True, exist_ok=True)
    raw_link = qad_dir / "raw"
    if raw_link.is_symlink() and not raw_link.exists():
        raw_link.unlink()  # dangling (e.g. copied from another machine)
    if not raw_link.exists():
        raw_link.symlink_to(args.raw_dir.resolve(), target_is_directory=True)
    return run_dir


def run_state(run_dir: Path) -> str:
    if (run_dir / "DONE").exists() and (run_dir / "final_metrics.csv").exists():
        return "done"
    if (run_dir / "FAILED").exists():
        return "failed"
    return "pending"


def fmt_hours(seconds: float) -> str:
    return f"{seconds / 3600:.1f} h" if seconds >= 3600 else f"{seconds / 60:.0f} min"


# ---------------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------------
def md_table(rows: list[list], header: list[str]) -> str:
    widths = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
              for i, h in enumerate(header)]
    fmt = "| " + " | ".join("{:<" + str(w) + "}" for w in widths) + " |"
    lines = [fmt.format(*header), "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    lines += [fmt.format(*[str(c) for c in r]) for r in rows]
    return "\n".join(lines)


def _mean(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and math.isnan(x))]
    if len(xs) < 2:
        return float("nan")
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def _fmt(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


_LOG_RE = re.compile(
    r"\[(?P<trace>\d+)\] (?P<epoch>\d+) \|.*?val_loss=(?P<val>[-+.\deEna]+).*?"
    r"tst_auc_roc=(?P<auc>[-+.\deEna]+).*?tst_f1=(?P<f1>[-+.\deEna]+)")


def parse_history(run_dir: Path) -> dict:
    """Per trace: epochs run, last/best test F1 and AUC over the evaluated epochs."""
    hist = defaultdict(list)
    for txt in sorted((run_dir / "logs").glob("*.txt")):
        for line in open(txt, errors="replace"):
            m = _LOG_RE.search(line)
            if m:
                try:
                    hist[int(m["trace"])].append((int(m["epoch"]), float(m["f1"]), float(m["auc"])))
                except ValueError:
                    continue
    out = {}
    for trace, h in hist.items():
        out[trace] = dict(epochs=h[-1][0], f1_last=h[-1][1], f1_oracle=max(x[1] for x in h),
                          auc_roc_oracle=max(x[2] for x in h))
    return out


def read_run_metrics(run: dict, run_dir: Path) -> dict | None:
    csv_path = run_dir / "final_metrics.csv"
    if not csv_path.exists():
        return None
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    row = rows[-1]

    def _get(col):
        try:
            return float(row.get(col, ""))
        except (TypeError, ValueError):
            return float("nan")

    rec = dict(arm=run["arm"], seed=run["seed"], trace=run["trace"], reference=run.get("reference"))
    rec.update({m: _get(f"{m}_mean") for m in METRICS})
    rec.update(parse_history(run_dir).get(int(run["trace"]), {}))
    return rec


def collect(args, runs: list[dict], out_dir: Path, quiet=False) -> str | None:
    arm_order = list(OrderedDict.fromkeys(r["arm"] for r in runs))
    ref_of = {r["arm"]: r.get("reference") for r in runs}
    n_traces_expected = len(args.traces)

    records = []
    for run in runs:
        rec = read_run_metrics(run, out_dir / "runs" / run["run_name"])
        if rec:
            records.append(rec)
    if not records:
        if not quiet:
            print("No results yet.")
        return None

    keys = ["arm", "seed", "trace", "reference"] + METRICS + ["epochs", "f1_last", "f1_oracle", "auc_roc_oracle"]
    with open(out_dir / "results_per_trace.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in sorted(records, key=lambda r: (arm_order.index(r["arm"]), r["seed"], r["trace"])):
            w.writerow(r)

    # macro per (arm, seed) over the traces finished so far
    by_arm_seed = defaultdict(list)
    for r in records:
        by_arm_seed[(r["arm"], r["seed"])].append(r)
    macro_keys = ["f1", "auc_roc", "auc_pr", "vus_roc", "vus_pr", "f1_oracle"]
    macro = {}
    for key, rs in by_arm_seed.items():
        macro[key] = {k: _mean([r.get(k, float("nan")) for r in rs]) for k in macro_keys}
        macro[key]["n_traces"] = len({r["trace"] for r in rs})

    summary = OrderedDict()
    for arm in arm_order:
        seeds = sorted(s for (a, s) in macro if a == arm)
        if not seeds:
            continue
        e = {"arm": arm, "reference": ref_of.get(arm), "n_seeds": len(seeds),
             "min_traces": min(macro[(arm, s)]["n_traces"] for s in seeds)}
        for k in macro_keys:
            vals = [macro[(arm, s)][k] for s in seeds]
            e[f"{k}_mean"], e[f"{k}_std"] = _mean(vals), _std(vals)
        summary[arm] = e
    for arm, e in summary.items():
        ref = e["reference"]
        for k in macro_keys:
            e[f"{k}_delta"] = (e[f"{k}_mean"] - summary[ref][f"{k}_mean"]) if ref in summary else float("nan")

    with open(out_dir / "results_summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(next(iter(summary.values())).keys()))
        w.writeheader()
        for e in summary.values():
            w.writerow(e)

    out = [f"# QAD change ablation: {args.name}\n",
           f"Generated {dt.datetime.now():%Y-%m-%d %H:%M}. {len(records)} trace results from "
           f"{len(by_arm_seed)} (arm, seed) combinations; {n_traces_expected} traces expected each "
           f"(traces {args.traces}), {args.n_epochs} epochs, seeds {args.seeds}, batch {args.batch_size}, "
           f"device {args.device}. Values are macro means over the finished traces, then mean +/- std "
           f"over seeds. Delta is against the arm's reference (new_all for new_minus_*, old_all for old_plus_*).\n",
           "## Macro metrics per arm\n"]
    header = ["arm", "seeds", "traces", "F1", "dF1", "AUC-ROC", "dAUC", "AUC-PR", "dAUPR", "VUS-PR", "F1 oracle*"]
    rows = []
    for arm, e in summary.items():
        has_ref = bool(e["reference"])
        rows.append([
            arm, e["n_seeds"], f"{e['min_traces']}/{n_traces_expected}",
            f"{_fmt(e['f1_mean'])} +/- {_fmt(e['f1_std'])}", _fmt(e["f1_delta"]) if has_ref else "-",
            _fmt(e["auc_roc_mean"]), _fmt(e["auc_roc_delta"]) if has_ref else "-",
            _fmt(e["auc_pr_mean"]), _fmt(e["auc_pr_delta"]) if has_ref else "-",
            _fmt(e["vus_pr_mean"]), _fmt(e["f1_oracle_mean"]),
        ])
    out.append(md_table(rows, header))
    out.append("\n*F1 oracle = best test F1 over the evaluated epochs (ignores val-loss model selection); "
               "shows whether a change helps the model or only the selection.\n")

    loo = {g: summary.get(f"new_minus_{g}") for g in GROUPS}
    add = {g: summary.get(f"old_plus_{g}") for g in GROUPS}
    if any(loo.values()) or any(add.values()):
        out.append("## Per change group (macro F1 / AUC-ROC)\n")
        rows = []
        for g in GROUPS:
            l, a = loo[g], add[g]
            rows.append([
                g,
                _fmt(-l["f1_delta"]) if l else "-",
                _fmt(a["f1_delta"]) if a else "-",
                _fmt(-l["auc_roc_delta"]) if l else "-",
                _fmt(a["auc_roc_delta"]) if a else "-",
                GROUPS[g]["why"],
            ])
        out.append(md_table(rows, ["group", "F1 lost if reverted", "F1 gained alone",
                                   "AUC lost if reverted", "AUC gained alone", "what it is"]))
        out.append("\nPositive 'lost if reverted' = the change is needed on top of the others; "
                   "positive 'gained alone' = the change helps by itself on the old setup.\n")

    out.append("## Per-trace F1 (mean over seeds)\n")
    traces = sorted({r["trace"] for r in records})
    by_arm_trace = defaultdict(list)
    for r in records:
        by_arm_trace[(r["arm"], r["trace"])].append(r["f1"])
    rows = [[arm] + [_fmt(_mean(by_arm_trace.get((arm, t), [])), 2) for t in traces] for arm in summary]
    out.append(md_table(rows, ["arm"] + [f"t{t}" for t in traces]))

    out.append("\n## Per-trace AUC-ROC (mean over seeds)\n")
    by_arm_trace = defaultdict(list)
    for r in records:
        by_arm_trace[(r["arm"], r["trace"])].append(r["auc_roc"])
    rows = [[arm] + [_fmt(_mean(by_arm_trace.get((arm, t), [])), 2) for t in traces] for arm in summary]
    out.append(md_table(rows, ["arm"] + [f"t{t}" for t in traces]))

    if len(args.seeds) > 1:
        out.append("\n## Macro F1 per (arm, seed)\n")
        seeds = sorted({s for (_, s) in macro})
        rows = [[arm] + [_fmt(macro[(arm, s)]["f1"]) if (arm, s) in macro else "-" for s in seeds]
                + [f"{summary[arm]['min_traces']}/{n_traces_expected}"] for arm in summary]
        out.append(md_table(rows, ["arm"] + [f"seed {s}" for s in seeds] + ["traces"]))

    report = "\n".join(out) + "\n"
    (out_dir / "report.md").write_text(report)
    if not quiet:
        print(report)
        print(f"Wrote {out_dir / 'report.md'}, results_summary.csv, results_per_trace.csv")
    return report


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------
def print_plan(args, arms, runs):
    header = ["arm", "reference"] + list(GROUPS)
    rows = [[name, ref or "-"] + [state[g] for g in GROUPS] for name, (state, ref) in arms.items()]
    print("Arms (which value each change group takes):")
    print(md_table(rows, header))
    print("\nGroup values:")
    for g, spec in GROUPS.items():
        print(f"  {g:10s} old={spec['old']}\n  {'':10s} new={spec['new']}   # {spec['why']}")
    est = len(runs) * (args.n_epochs * SEC_PER_EPOCH_GUESS + RUN_OVERHEAD_S)
    print(f"\nRuns: {len(runs)} ({len(arms)} arms x {len(args.seeds)} seeds x {len(args.traces)} traces), "
          f"{args.n_epochs} epochs each, sequential on {args.device}.")
    print(f"Rough estimate: {fmt_hours(est)} total at ~{SEC_PER_EPOCH_GUESS:.0f} s/epoch (rtx2080ti); "
          f"the script re-estimates from measured run times as it goes.")
    print(f"Example command ({runs[0]['run_name']}):")
    print("  " + " ".join(shlex.quote(c) for c in run_command(args, runs[0], Path('<run_dir>'))))


def check_environment(args):
    if not (PROJECT_DIR / "anomaly_detection.py").exists():
        raise SystemExit(f"anomaly_detection.py not found next to this script in {PROJECT_DIR}")
    if not args.raw_dir.exists() or not any(args.raw_dir.glob("train_*.pkl")):
        raise SystemExit(f"QAD raw data (train_*.pkl) not found in {args.raw_dir}; use --raw-dir")
    missing = [t for t in args.traces if not (args.raw_dir / f"train_{t}.pkl").exists()]
    if missing:
        raise SystemExit(f"Raw files missing for traces {missing} in {args.raw_dir}")
    if args.device.startswith("cuda"):
        try:
            import torch
            if not torch.cuda.is_available():
                raise SystemExit("CUDA not available in this interpreter; pass --device cpu or fix the environment")
        except ImportError:
            raise SystemExit("torch is not importable with this interpreter; run inside the project environment")


def main():
    args = build_parser().parse_args()
    if args.extra_args and args.extra_args[0] == "--":
        args.extra_args = args.extra_args[1:]
    args.raw_dir = Path(args.raw_dir)
    args.out_root = Path(args.out_root)

    arms = build_arms(args.design, args.arms)
    runs = make_runs(args, arms)
    out_dir = args.out_root / args.name

    if args.plan_only:
        print_plan(args, arms, runs)
        return
    if args.collect_only:
        collect(args, runs, out_dir)
        return

    check_environment(args)
    out_dir.mkdir(parents=True, exist_ok=True)
    print_plan(args, arms, runs)
    manifest = {
        "name": args.name,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "groups": GROUPS,
        "runs": [{k: v for k, v in r.items()} for r in runs],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    t_start = time.time()
    durations = []
    n_done = n_failed = n_skipped = 0
    pending = [r for r in runs if run_state(out_dir / "runs" / r["run_name"]) != "done" or args.force]
    n_skipped = len(runs) - len(pending)
    print(f"\n{len(runs)} runs, {n_skipped} already done, {len(pending)} to run. Output: {out_dir}\n")

    for i, run in enumerate(pending, 1):
        elapsed = time.time() - t_start
        if args.max_hours and elapsed > args.max_hours * 3600:
            print(f"Time budget of {args.max_hours} h exhausted; {len(pending) - i + 1} runs left "
                  f"(restart with the same --name to continue).")
            break
        run_dir = prepare_run_dir(args, run, out_dir)
        cmd = run_command(args, run, run_dir)
        (run_dir / "command.txt").write_text(" ".join(shlex.quote(c) for c in cmd) + "\n")
        for marker in ("DONE", "FAILED"):
            (run_dir / marker).unlink(missing_ok=True)

        per_run = _mean(durations) if durations else args.n_epochs * SEC_PER_EPOCH_GUESS + RUN_OVERHEAD_S
        eta = per_run * (len(pending) - i + 1)
        print(f"[{i}/{len(pending)}] {run['run_name']}  (elapsed {fmt_hours(elapsed)}, "
              f"ETA {fmt_hours(eta)}) ...", flush=True)
        t0 = time.time()
        with open(run_dir / "run.log", "w") as log:
            res = subprocess.run(cmd, cwd=PROJECT_DIR, stdout=log, stderr=subprocess.STDOUT)
        took = time.time() - t0
        if res.returncode == 0 and (run_dir / "final_metrics.csv").exists():
            (run_dir / "DONE").touch()
            durations.append(took)
            n_done += 1
            rec = read_run_metrics(run, run_dir) or {}
            print(f"    done in {took / 60:.1f} min: F1 {_fmt(rec.get('f1'))}, AUC-ROC {_fmt(rec.get('auc_roc'))}, "
                  f"epochs {rec.get('epochs', '?')}", flush=True)
        else:
            (run_dir / "FAILED").touch()
            n_failed += 1
            print(f"    FAILED (exit {res.returncode}) after {took / 60:.1f} min, see {run_dir / 'run.log'}", flush=True)
            if args.stop_on_failure:
                break
        collect(args, runs, out_dir, quiet=True)  # keep report.md current after every run

    print(f"\nFinished: {n_done} done, {n_failed} failed, {n_skipped} skipped, "
          f"total {fmt_hours(time.time() - t_start)}.\n")
    collect(args, runs, out_dir)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="qad_ablation", help="run name; results go to <out-root>/<name>")
    p.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    p.add_argument("--design", choices=["loo", "add", "both", "endpoints"], default="loo",
                   help="loo: new_all/old_all + leave-one-out (default, 8 arms); add: + add-one-in; "
                        "both: all 14 arms; endpoints: only new_all and old_all")
    p.add_argument("--arms", nargs="*", default=None, help="restrict to these arm names")
    p.add_argument("--seeds", type=int, nargs="+", default=[1],
                   help="seeds; the same config varied by ~0.05 macro F1 across seeds, so use 2+ if time allows")
    p.add_argument("--traces", type=int, nargs="+", default=DEFAULT_TRACES,
                   help=f"traces to use (default {DEFAULT_TRACES}; all: 1 ... 10)")
    p.add_argument("--n-epochs", type=int, default=300,
                   help="epoch budget per run; with the new config test F1 is far from final at 150 "
                        "and mostly there at 300")
    p.add_argument("--batch-size", type=int, default=128,
                   help="passed explicitly (the 2026-09-16 runs that produced the improvement used 128)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR, help="folder with QAD train_*.pkl etc.")
    p.add_argument("--max-hours", type=float, default=None,
                   help="stop starting new runs after this many hours; rerun with the same --name to continue")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default="disabled")
    p.add_argument("--force", action="store_true", help="re-run runs that are already done")
    p.add_argument("--stop-on-failure", action="store_true")
    p.add_argument("--plan-only", action="store_true", help="print arms and estimate, run nothing")
    p.add_argument("--collect-only", action="store_true", help="only rebuild report from finished runs")
    p.add_argument("extra_args", nargs=argparse.REMAINDER,
                   help="extra flags for anomaly_detection.py after '--' (e.g. -- --no-sphere-embedding)")
    return p


if __name__ == "__main__":
    main()
