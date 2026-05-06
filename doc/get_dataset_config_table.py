#!/usr/bin/env python3
"""Build a parameter x dataset table from config files under cfg/."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# Explicit row order for the output table (edit this list as needed).
DEFAULT_ROW_ORDER = [
    "batch_size",
    "lr",
    "z_dim",
    "h_dim",
    "n_deg",

    "dec_hidden_dim",

    "kl0_weight",
    "klp_weight",
    "pxz_weight",

    "initial_sigma",
    "subsample",

    "normalize_score",
]

# Human-readable labels for config parameters.
PARAMETER_LABELS = {
    "batch_size": "Batch Size",
    "lr": "Learning Rate",
    "z_dim": "Latent Dim. (z)",
    "h_dim": "Hidden Dim. (h)",
    "n_deg": "Degree (interpolation)",
    "dec_hidden_dim": "Decoder Hidden Dim.",
    "n_dec_layers": "\\# Decoder Layers",
    "non_linear_decoder": "Nonlinear Decoder",
    "kl0_weight": "KL$_0$ Weight",
    "klp_weight": "KL$_p$ Weight",
    "pxz_weight": "Likelihood Weight",
    "initial_sigma": "Initial $\\sigma$",
    "subsample": "Subsample Ratio",
    "normalize_score": "Normalize Score",
    "data_normalization_strategy": "Data Norm Strategy",
    "sphere_embedding": "Sphere Embedding",
    "use_atanh": "Use Atanh",
    "learnable_prior": "Learnable Prior",
    "freeze_sigma": "Freeze \\sigma",
    "mc_train_samples": "MC Train Samples",
    "mc_eval_samples": "MC Eval Samples",
    "n_epochs": "Max. Epochs",
    "restart": "LR Restart",
    "data_window_length": "Window Length",
    "data_window_overlap": "Window Overlap",
    "seed": "Random Seed",
    "runs": "\\# Runs",
    "debug": "Debug Mode",
    "loglevel": "Log Level",
    "enable_file_logging": "File Logging",
    "log_dir": "Log Dir",
    "enable_checkpointing": "Checkpointing",
    "checkpoint_dir": "Checkpoint Dir",
    "checkpoint_at": "Checkpoint At",
    "final_metrics_csv": "Metrics CSV",
    "delete_processed_data": "Delete Proc. Data",
    "fixed_subsample_mask": "Fixed Subsample Mask",
    "eval_every_n_epochs": "Eval Every N",
    "num_max_cpu_worker": "CPU Workers",
    "device": "Device",
    "data_dir": "Data Dir",
    "dataset": "Dataset",
    "early_stopping_min_delta": "Early Stop Δ",
}


def _load_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix == ".json":
        return json.loads(path.read_text())

    if suffix in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "YAML file found but PyYAML is not installed. Install pyyaml or use JSON configs."
            ) from exc
        return yaml.safe_load(path.read_text()) or {}

    raise ValueError(f"Unsupported config format: {path}")


def _column_name(cfg_root: Path, path: Path) -> str:
    # Use file stem as dataset name; prefix with parent if duplicates exist.
    parent = path.parent.relative_to(cfg_root)
    if str(parent) in {"", "."}:
        return path.stem
    return f"{path.stem}"


def build_dataset_config_table(
    cfg_root: Path,
    row_order: list[str] | None = None,
    include_unlisted_rows: bool = False,
) -> pd.DataFrame:
    cfg_files = sorted(
        [*cfg_root.rglob("*.json"), *cfg_root.rglob("*.yaml"), *cfg_root.rglob("*.yml")]
    )
    if not cfg_files:
        raise FileNotFoundError(f"No config files found under: {cfg_root}")

    table_dict: dict[str, dict[str, Any]] = {}
    for path in cfg_files:
        col = _column_name(cfg_root, path)
        cfg = _load_config(path)
        if not isinstance(cfg, dict):
            raise ValueError(f"Config must be a key/value object: {path}")
        table_dict[col] = cfg

    # Rows: parameters, Columns: benchmark datasets
    df = pd.DataFrame(table_dict)
    df.index.name = "parameter"
    df = df.sort_index()

    if row_order is None:
        return df

    ordered_existing = [row for row in row_order if row in df.index]
    if include_unlisted_rows:
        remaining = [row for row in df.index.tolist() if row not in ordered_existing]
        ordered_existing.extend(remaining)
    return df.reindex(ordered_existing)


def _format_value(value: Any, float_format: str, use_fontawesome: bool = False) -> Any:
    """Format a single value for display/export.

    Parameters
    ----------
    value
        Value to format.
    float_format
        Python format spec for floats (e.g., '.3f').
    use_fontawesome
        If True, use fontawesome symbols for booleans. Otherwise use checkmark/times.
    """
    if isinstance(value, bool):
        if use_fontawesome:
            return r"$\faCheck$" if value else r"$\faTimes$"
        return r"$\checkmark$" if value else r"$\times$"
    if isinstance(value, (float, np.floating)):
        if pd.isna(value):
            return value
        return format(float(value), float_format)
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    return value


def format_dataframe_numbers(df: pd.DataFrame, float_format: str, use_fontawesome: bool = False) -> pd.DataFrame:
    """Apply numeric and boolean formatting to all cells in the DataFrame."""
    return df.apply(
        lambda col: col.map(lambda x: _format_value(x, float_format, use_fontawesome=use_fontawesome))
    )


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Read all config files from cfg/ and print a parameter x dataset table."
    )
    parser.add_argument(
        "--cfg-root",
        type=Path,
        default=repo_root / "cfg",
        help="Config root directory (default: <repo>/cfg)",
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=None,
        help="Optional path to save the table as CSV.",
    )
    parser.add_argument(
        "--out-tex",
        type=Path,
        default=None,
        help="Optional path to save the table as LaTeX (.tex).",
    )
    parser.add_argument(
        "--rows",
        type=str,
        default=",".join(DEFAULT_ROW_ORDER),
        help="Comma-separated property names to include and order as rows.",
    )
    parser.add_argument(
        "--include-unlisted-rows",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Append rows not listed in --rows at the end.",
    )
    parser.add_argument(
        "--float-format",
        type=str,
        default=".4g",
        help="Python format spec for floating-point values (default: .4g).",
    )
    parser.add_argument(
        "--format-csv",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write CSV with formatted numeric strings instead of raw numeric values.",
    )
    parser.add_argument(
        "--latex-column-format",
        type=str,
        default="l|ccccccc",
        help="LaTeX column format passed to pandas.to_latex.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    selected_rows = [row.strip() for row in args.rows.split(",") if row.strip()]
    df = build_dataset_config_table(
        args.cfg_root,
        row_order=selected_rows,
        include_unlisted_rows=args.include_unlisted_rows,
    )
    df_formatted = format_dataframe_numbers(df, args.float_format)

    # Apply human-readable labels to the index
    df_formatted.index = df_formatted.index.map(
        lambda x: PARAMETER_LABELS.get(x, x)
    )

    with pd.option_context("display.max_rows", None, "display.max_columns", None, "display.width", 200):
        print(df_formatted)

    if args.out_csv is not None:
        args.out_csv.parent.mkdir(parents=True, exist_ok=True)
        (df_formatted if args.format_csv else df).to_csv(args.out_csv)
        print(f"\nSaved CSV to: {args.out_csv}")

    if args.out_tex is not None:
        args.out_tex.parent.mkdir(parents=True, exist_ok=True)
        latex_str = df_formatted.to_latex(
            index=True,
            index_names=True,
            escape=False,
            column_format=args.latex_column_format,
        )
        args.out_tex.write_text(latex_str)
        print(latex_str)
        print(f"Saved LaTeX to: {args.out_tex}")


if __name__ == "__main__":
    main()

