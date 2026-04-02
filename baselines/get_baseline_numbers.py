#!/usr/bin/env python3
"""Format baseline macro mean/std metrics into report-ready strings."""

import argparse
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "out" / "baselines_macro_mean_std.csv"
DEFAULT_OUTPUT = ROOT_DIR / "out" / "baseline_numbers.csv"
LATEX_BENCHMARK_ORDER = ["SWaT", "WaDi", "SMAP", "MSL", "PSM", "SMD", "QAD"]
LATEX_METHOD_ORDER = ["KNN", "OCSVM", "LOF", "IForest", "COPOD", "PCA"]

AUROC_MEAN_CANDIDATES = ["auroc_mean", "auc_mean", "aucroc_mean", "macro_auroc_mean", "macro_auc_mean"]
AUROC_STD_CANDIDATES = ["auroc_std", "auc_std", "aucroc_std", "macro_auroc_std", "macro_auc_std"]


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Read baselines_macro_mean_std.csv and format auc/auprc/f1 as mean(+/- std)."
	)
	parser.add_argument(
		"--input",
		type=Path,
		default=DEFAULT_INPUT,
		help="Path to the macro mean/std CSV.",
	)
	parser.add_argument(
		"--output",
		type=Path,
		default=DEFAULT_OUTPUT,
		help="Path to save the formatted DataFrame as CSV.",
	)
	return parser.parse_args()


def _resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
	for candidate in candidates:
		if candidate in df.columns:
			return candidate
	raise ValueError(f"Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)}")


def _load_latest_rows(csv_path: Path) -> pd.DataFrame:
	if not csv_path.exists():
		raise FileNotFoundError(f"Input CSV not found: {csv_path}")

	df = pd.read_csv(csv_path)
	if df.empty:
		raise ValueError(f"Input CSV is empty: {csv_path}")

	required_cols = ["benchmark", "clf_name"]
	missing = [col for col in required_cols if col not in df.columns]
	if missing:
		raise ValueError(f"Missing required columns {missing} in {csv_path}")

	df = df.dropna(how="all").copy()
	df["_row_order"] = range(len(df))
	latest_df = (
		df.sort_values("_row_order")
		.drop_duplicates(subset=["benchmark", "clf_name"], keep="last")
		.drop(columns=["_row_order"])
		.sort_values(["benchmark", "clf_name"])
		.reset_index(drop=True)
	)
	return latest_df


def _format_metric(mean_series: pd.Series, std_series: pd.Series) -> pd.Series:
	return (mean_series * 100).map(lambda x: f"{x:.2f}") + "$\\pm$" + (std_series * 100).map(lambda x: f"{x:.2f}")


def build_formatted_df(df: pd.DataFrame) -> pd.DataFrame:
	auc_mean_col = _resolve_column(df, AUROC_MEAN_CANDIDATES, "AUROC mean")
	auc_std_col = _resolve_column(df, AUROC_STD_CANDIDATES, "AUROC std")
	auprc_mean_col = _resolve_column(df, ["auprc_mean"], "AUPRC mean")
	auprc_std_col = _resolve_column(df, ["auprc_std"], "AUPRC std")
	f1_mean_col = _resolve_column(df, ["f1_mean"], "F1 mean")
	f1_std_col = _resolve_column(df, ["f1_std"], "F1 std")

	numeric_cols = [auc_mean_col, auc_std_col, auprc_mean_col, auprc_std_col, f1_mean_col, f1_std_col]
	formatted_df = df.copy()
	for col in numeric_cols:
		formatted_df[col] = pd.to_numeric(formatted_df[col], errors="coerce")

	if formatted_df[numeric_cols].isna().any().any():
		raise ValueError("Some metric mean/std values could not be parsed as numeric.")

	result_df = pd.DataFrame(
		{
			"benchmark": formatted_df["benchmark"],
			"method": formatted_df["clf_name"],
			"auc": _format_metric(formatted_df[auc_mean_col], formatted_df[auc_std_col]),
			"auprc": _format_metric(formatted_df[auprc_mean_col], formatted_df[auprc_std_col]),
			"f1": _format_metric(formatted_df[f1_mean_col], formatted_df[f1_std_col]),
		}
	)
	return result_df


def build_latex_wide_df(result_df: pd.DataFrame) -> pd.DataFrame:
	# One row per method; each benchmark contributes 3 cells: auc, auprc, f1.
	metric_order = ["auc", "auprc", "f1"]
	melted = result_df.melt(
		id_vars=["method", "benchmark"],
		value_vars=metric_order,
		var_name="metric",
		value_name="value",
	)
	wide_df = melted.pivot_table(
		index="method",
		columns=["benchmark", "metric"],
		values="value",
		aggfunc="first",
	)

	# Reorder methods for LaTeX output (case-insensitive matching).
	index_map = {str(idx).lower(): idx for idx in wide_df.index}
	preferred_index = [index_map[name.lower()] for name in LATEX_METHOD_ORDER if name.lower() in index_map]
	remaining_index = sorted([idx for idx in wide_df.index if idx not in preferred_index])
	wide_df = wide_df.reindex(preferred_index + remaining_index)

	expected_cols = pd.MultiIndex.from_product([LATEX_BENCHMARK_ORDER, metric_order])
	wide_df = wide_df.reindex(columns=expected_cols).fillna("--")
	wide_df.index.name = "method"
	return wide_df


def main() -> None:
	args = parse_args()
	latest_df = _load_latest_rows(args.input)
	result_df = build_formatted_df(latest_df)
	latex_df = build_latex_wide_df(result_df)

	args.output.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(args.output, index=False)

	print(result_df.to_string(index=False))
	print(latex_df.to_latex(escape=False))
	print(f"\nSaved formatted baseline numbers to {args.output}")


if __name__ == "__main__":
	main()

