#!/usr/bin/env python3
"""Format baseline macro mean/std metrics into report-ready strings."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import studentized_range


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT_DIR / "out" / "baselines_macro_mean_std.csv"
DEFAULT_OUTPUT = ROOT_DIR / "out" / "baseline_numbers.csv"
DEFAULT_USED_RESULTS_OUTPUT = ROOT_DIR / "out" / "baseline_used_results.csv"
DEFAULT_MEAN_TABLE_PREFIX = ROOT_DIR / "out" / "baseline_mean"
DEFAULT_CD_PLOT_PREFIX = ROOT_DIR / "out" / "baseline_cd"
LATEX_BENCHMARK_ORDER = ["SWaT", "WaDi", "SMAP", "MSL", "PSM", "SMD"]
LATEX_METHOD_ORDER = ["KNN", "OCSVM", "LOF", "IForest", "COPOD", "PCA", "DeepSVDD", "USAD", "TcnED", "TranAD", "AnomalyTransformer", "DeepIF", "TimesNet", "COUTA"]

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
        parser.add_argument(
                "--used-results-output",
                type=Path,
                default=DEFAULT_USED_RESULTS_OUTPUT,
                help="Path to save all cleaned input rows (e.g. each run/trace) in formatted metric columns.",
        )
        parser.add_argument(
                "--mean-table-prefix",
                type=Path,
                default=DEFAULT_MEAN_TABLE_PREFIX,
                help="Prefix for per-metric mean-only tables. Files are saved as <prefix>_{metric}.csv.",
        )
        parser.add_argument(
                "--cd-plot-prefix",
                type=Path,
                default=DEFAULT_CD_PLOT_PREFIX,
                help="Prefix for critical difference plot files. Files are saved as <prefix>_{metric}.png.",
        )
        parser.add_argument(
                "--cd-alpha",
                type=float,
                default=0.05,
                help="Significance level for Nemenyi critical difference (default: 0.05).",
        )
        return parser.parse_args()


def _resolve_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
        for candidate in candidates:
                if candidate in df.columns:
                        return candidate
        raise ValueError(f"Could not find {label} column. Tried: {candidates}. Available: {list(df.columns)}")


def _load_all_rows(csv_path: Path) -> pd.DataFrame:
        if not csv_path.exists():
                raise FileNotFoundError(f"Input CSV not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if df.empty:
                raise ValueError(f"Input CSV is empty: {csv_path}")

        required_cols = ["benchmark", "clf_name"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
                raise ValueError(f"Missing required columns {missing} in {csv_path}")

        cleaned_df = df.dropna(how="all").copy()
        cleaned_df = cleaned_df[cleaned_df["benchmark"].astype(str).str.strip().str.upper() != "QAD"]
        if cleaned_df.empty:
                raise ValueError(f"No non-QAD rows available after filtering in {csv_path}")
        return cleaned_df


def _load_latest_rows(csv_path: Path) -> pd.DataFrame:
        df = _load_all_rows(csv_path)
        df["_row_order"] = range(len(df))
        latest_df = (
                df.sort_values("_row_order")
                .drop_duplicates(subset=["benchmark", "clf_name"], keep="last")
                .drop(columns=["_row_order"])
                .sort_values(["benchmark", "clf_name"])
                .reset_index(drop=True)
        )
        return latest_df


def _to_percent(mean_series: pd.Series, std_series: pd.Series) -> tuple[pd.Series, pd.Series]:
        return (mean_series * 100), (std_series * 100)


def _format_metric(mean_series: pd.Series, std_series: pd.Series) -> pd.Series:
        return mean_series.map(lambda x: f"{x:.2f}") + "$\\pm$" + std_series.map(lambda x: f"{x:.2f}")


def build_formatted_df(df: pd.DataFrame, include_extra_cols: bool = False) -> pd.DataFrame:
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

        auc_mean_pct, auc_std_pct = _to_percent(formatted_df[auc_mean_col], formatted_df[auc_std_col])
        auprc_mean_pct, auprc_std_pct = _to_percent(formatted_df[auprc_mean_col], formatted_df[auprc_std_col])
        f1_mean_pct, f1_std_pct = _to_percent(formatted_df[f1_mean_col], formatted_df[f1_std_col])

        result_df = pd.DataFrame(
                {
                        "benchmark": formatted_df["benchmark"],
                        "method": formatted_df["clf_name"],
                        "auc_mean": auc_mean_pct,
                        "auc_std": auc_std_pct,
                        "auprc_mean": auprc_mean_pct,
                        "auprc_std": auprc_std_pct,
                        "f1_mean": f1_mean_pct,
                        "f1_std": f1_std_pct,
                }
        )

        if include_extra_cols:
                excluded = {
                        "clf_name",
                        auc_mean_col,
                        auc_std_col,
                        auprc_mean_col,
                        auprc_std_col,
                        f1_mean_col,
                        f1_std_col,
                }
                extra_cols = [c for c in formatted_df.columns if c not in excluded and c not in result_df.columns]
                for col in extra_cols:
                        result_df[col] = formatted_df[col].values

        return result_df


def build_latex_wide_df(result_df: pd.DataFrame) -> pd.DataFrame:
        # One row per method; each benchmark contributes 3 cells: auc, auprc, f1.
        display_df = pd.DataFrame(
                {
                        "method": result_df["method"],
                        "benchmark": result_df["benchmark"],
                        "auc": _format_metric(result_df["auc_mean"], result_df["auc_std"]),
                        "auprc": _format_metric(result_df["auprc_mean"], result_df["auprc_std"]),
                        "f1": _format_metric(result_df["f1_mean"], result_df["f1_std"]),
                }
        )

        metric_order = ["auc", "auprc", "f1"]
        melted = display_df.melt(
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


def build_metric_mean_table(result_df: pd.DataFrame, metric: str) -> pd.DataFrame:
        mean_col = f"{metric}_mean"
        if mean_col not in result_df.columns:
                raise ValueError(f"Missing required column: {mean_col}")

        table = result_df.pivot_table(
                index="method",
                columns="benchmark",
                values=mean_col,
                aggfunc="first",
        )

        index_map = {str(idx).lower(): idx for idx in table.index}
        preferred_index = [index_map[name.lower()] for name in LATEX_METHOD_ORDER if name.lower() in index_map]
        remaining_index = sorted([idx for idx in table.index if idx not in preferred_index])
        table = table.reindex(preferred_index + remaining_index)

        table = table.reindex(columns=LATEX_BENCHMARK_ORDER)
        table.index.name = "method"
        return table


def rank_table_by_column(table: pd.DataFrame) -> pd.DataFrame:
        # Rank each benchmark column independently: highest value -> 1.
        ranked = table.rank(axis=0, method="min", ascending=False)
        return ranked.astype("Int64")


def _prepare_cd_matrix(ranked_table: pd.DataFrame) -> pd.DataFrame:
        # Keep only benchmarks with at least two methods, then methods fully observed on those benchmarks.
        table = ranked_table.copy()
        valid_columns = [col for col in table.columns if table[col].notna().sum() >= 2]
        table = table[valid_columns]
        table = table.dropna(axis=0, how="any")
        return table


def _compute_critical_difference(k_methods: int, n_benchmarks: int, alpha: float) -> float:
        q_alpha = studentized_range.isf(alpha, k_methods, np.inf) / np.sqrt(2.0)
        return q_alpha * np.sqrt((k_methods * (k_methods + 1)) / (6.0 * n_benchmarks))


def save_cd_plot(ranked_table: pd.DataFrame, metric: str, out_path: Path, alpha: float = 0.05) -> bool:
        cd_matrix = _prepare_cd_matrix(ranked_table)
        if cd_matrix.shape[0] < 2 or cd_matrix.shape[1] < 2:
            print(
                    f"Skipping CD plot for {metric.upper()}: need >=2 methods and >=2 benchmarks after NaN filtering, "
                    f"got methods={cd_matrix.shape[0]}, benchmarks={cd_matrix.shape[1]}."
            )
            return False

        avg_ranks = cd_matrix.mean(axis=1).sort_values()
        k_methods = avg_ranks.shape[0]
        n_benchmarks = cd_matrix.shape[1]
        cd = _compute_critical_difference(k_methods, n_benchmarks, alpha)

        fig_height = max(3.2, 1.0 + 0.35 * k_methods)
        fig, ax = plt.subplots(figsize=(10, fig_height))

        y_positions = np.arange(k_methods)
        ax.scatter(avg_ranks.values, y_positions, s=30, color="tab:blue", zorder=3)
        for y, (method, rank_val) in enumerate(avg_ranks.items()):
            ax.text(rank_val + 0.03, y, method, va="center", fontsize=9)

        # Draw a CD bar above the ranked methods.
        x_start = 1.0
        x_end = min(k_methods, x_start + cd)
        y_cd = k_methods - 0.2
        ax.plot([x_start, x_end], [y_cd, y_cd], color="black", linewidth=2)
        ax.plot([x_start, x_start], [y_cd - 0.1, y_cd + 0.1], color="black", linewidth=2)
        ax.plot([x_end, x_end], [y_cd - 0.1, y_cd + 0.1], color="black", linewidth=2)
        ax.text((x_start + x_end) / 2.0, y_cd + 0.12, f"CD={cd:.2f}", ha="center", va="bottom", fontsize=9)

        ax.set_title(f"Critical Difference Plot ({metric.upper()})")
        ax.set_xlabel("Average Rank (lower is better)")
        ax.set_xlim(0.8, k_methods + 0.4)
        ax.set_ylim(-0.6, k_methods + 0.6)
        ax.set_yticks([])
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        ax.invert_yaxis()

        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
        return True


def format_col(s):
    unique_sorted = sorted(s.unique(), reverse=True)
    max_val = unique_sorted[0]
    second_val = unique_sorted[1] if len(unique_sorted) > 1 else None

    out = []
    for v in s:
        val = v
        if v == max_val:
            out.append(f"\\textbf{{{val}}}")
        elif second_val is not None and v == second_val:
            out.append(f"\\underline{{{val}}}")
        else:
            out.append(val)
    return out



def main() -> None:
        args = parse_args()
        all_rows_df = _load_all_rows(args.input)
        used_results_df = build_formatted_df(all_rows_df, include_extra_cols=True)
        latest_df = _load_latest_rows(args.input)
        result_df = build_formatted_df(latest_df)
        latex_df = build_latex_wide_df(result_df)
        metric_tables = {
                "auc": build_metric_mean_table(result_df, "auc"),
                "auprc": build_metric_mean_table(result_df, "auprc"),
                "f1": build_metric_mean_table(result_df, "f1"),
        }
        ranked_metric_tables = {
                metric: rank_table_by_column(table)
                for metric, table in metric_tables.items()
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        result_df.to_csv(args.output, index=False)
        args.used_results_output.parent.mkdir(parents=True, exist_ok=True)
        used_results_df.to_csv(args.used_results_output, index=False)
        latex_csv_path = ROOT_DIR / "out" / "latex-df.csv"
        latex_csv_path.parent.mkdir(parents=True, exist_ok=True)
        latex_df.to_csv(latex_csv_path)

        args.mean_table_prefix.parent.mkdir(parents=True, exist_ok=True)
        args.cd_plot_prefix.parent.mkdir(parents=True, exist_ok=True)
        saved_cd_plots = []
        for metric, table in ranked_metric_tables.items():
                out_path = Path(f"{args.mean_table_prefix}_{metric}.csv")
                table.to_csv(out_path)
                print(f"\n{metric.upper()} rank table (best=1)")
                print(table.astype("string").fillna("--"))

                cd_plot_path = Path(f"{args.cd_plot_prefix}_{metric}.png")
                if save_cd_plot(table, metric, cd_plot_path, alpha=args.cd_alpha):
                    saved_cd_plots.append(cd_plot_path)

        column_format = "l|" + "|".join(["ccc"] * len(LATEX_BENCHMARK_ORDER))
        latex = latex_df.apply(format_col).to_latex(
                index=True,
                escape=False,
                column_format=column_format,
        )
        print("\nLaTeX table")
        print(latex)

        print(f"\nSaved formatted baseline numbers to {args.output}")
        print(f"Saved all used formatted rows to {args.used_results_output}")
        print(f"Saved ranked metric tables with prefix {args.mean_table_prefix}_<metric>.csv")
        print(f"Saved LaTeX wide table data to {latex_csv_path}")
        if saved_cd_plots:
                print("Saved CD plots:")
                for plot_path in saved_cd_plots:
                        print(f"- {plot_path}")


if __name__ == "__main__":
        main()

