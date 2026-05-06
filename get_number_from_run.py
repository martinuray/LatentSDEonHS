#!/usr/bin/env python3
"""Extract the dictionary from the last row of a log file."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any


def _try_parse_dict(text: str) -> dict | None:
    """Parse dict from text using JSON first, then Python literal syntax."""
    candidate = text.strip()
    if not candidate:
        return None

    try:
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    try:
        parsed = ast.literal_eval(candidate)
        if isinstance(parsed, dict):
            return parsed
    except (ValueError, SyntaxError):
        pass

    return None


def _extract_dict_from_line(line: str) -> dict | None:
    """Try full-line parse, then parse the last {...} span in the line."""
    parsed = _try_parse_dict(line)
    if parsed is not None:
        return parsed

    start = line.find("{")
    end = line.rfind("}")
    if start != -1 and end != -1 and end > start:
        return _try_parse_dict(line[start : end + 1])

    return None


def extract_last_dict(log_path: Path) -> dict:
    """Find and return the last parseable dictionary in the file."""
    lines = log_path.read_text(errors="replace").splitlines()

    for line in reversed(lines):
        parsed = _extract_dict_from_line(line)
        if parsed is not None:
            return parsed

    raise ValueError(f"No parseable dictionary found in: {log_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract dict from the last row of a log file.")
    parser.add_argument("log_path", type=Path, help="Path to the log file")
    parser.add_argument("--indent", type=int, default=2, help="JSON indentation for output")
    return parser.parse_args()

def format_float(value: float) -> float:
    return round(float(value*100), 2)


def _resolve_metric_stat_key(results: dict[str, Any], metric: str, stat: str) -> str:
    """Resolve metric key with fallback to macro_* and wildcard-like variants."""
    exact_candidates = [
        f"{metric}_{stat}",
        f"macro_{metric}_{stat}",
    ]

    for key in exact_candidates:
        if key in results:
            return key

    pattern_candidates = []
    for key in results:
        if key.startswith(metric) and key.endswith(f"_{stat}"):
            pattern_candidates.append(key)
    for key in results:
        if key.startswith(f"macro_{metric}") and key.endswith(f"_{stat}"):
            pattern_candidates.append(key)

    if pattern_candidates:
        return sorted(pattern_candidates)[0]

    raise ValueError(
        f"Missing value for metric='{metric}', stat='{stat}'. "
        f"Expected one of {exact_candidates} or {metric}*/macro_{metric}* variants. "
        f"Available keys: {sorted(results.keys())}"
    )


def build_latex_table_string(results: dict[str, Any]) -> str:
    auc_mean_key = _resolve_metric_stat_key(results, "auc", "mean")
    auc_std_key = _resolve_metric_stat_key(results, "auc", "std")
    auprc_mean_key = _resolve_metric_stat_key(results, "auprc", "mean")
    auprc_std_key = _resolve_metric_stat_key(results, "auprc", "std")
    f1_mean_key = _resolve_metric_stat_key(results, "f1", "mean")
    f1_std_key = _resolve_metric_stat_key(results, "f1", "std")

    return (
        f"{format_float(results[auc_mean_key])}$\\pm${format_float(results[auc_std_key])} & "
        f"{format_float(results[auprc_mean_key])}$\\pm${format_float(results[auprc_std_key])} &"
        f"{format_float(results[f1_mean_key])}$\\pm${format_float(results[f1_std_key])}"
    )


def main() -> None:
    args = parse_args()
    result = extract_last_dict(args.log_path)
    print(build_latex_table_string(result))


if __name__ == "__main__":
    main()

