from pathlib import Path

import pandas as pd

from get_number_from_run import extract_last_dict, parse_args


def _iter_log_files(log_path: Path):
    if log_path.is_file():
        yield log_path
        return

    if not log_path.exists():
        raise FileNotFoundError(f"Path does not exist: {log_path}")

    for file_path in sorted(path for path in log_path.rglob("*.log") if path.is_file()):
        yield file_path


def _extract_file_metadata(file_path: Path) -> dict[str, str]:
    parts = file_path.stem.split("_")
    benchmark = parts[0] if parts else file_path.stem
    clf_name = parts[1] if len(parts) > 1 else ""
    return {
        "file_path": str(file_path),
        "benchmark": benchmark,
        "clf_name": clf_name,
    }


def _iter_per_dataset_results(result: dict, *, base_row: dict[str, object]):
    per_dataset_rows: dict[str, dict[str, object]] = {}

    for key, value in result.items():
        if not key.startswith("per_dataset."):
            continue

        _, dataset_id, metric_name = key.split(".", 2)
        row = per_dataset_rows.setdefault(
            dataset_id,
            {
                **base_row,
                "dataset_id": dataset_id,
            },
        )
        row[metric_name] = value

    for dataset_id in sorted(per_dataset_rows):
        yield per_dataset_rows[dataset_id]

    if per_dataset_rows:
        return

    yield {
        **base_row,
        "dataset_id": str(base_row["benchmark"]),
        "auc_mean": result.get("auc_mean"),
        "auc_std": result.get("auc_std"),
        "auprc_mean": result.get("auprc_mean"),
        "auprc_std": result.get("auprc_std"),
        "f1_mean": result.get("f1_mean"),
        "f1_std": result.get("f1_std"),
    }


def main() -> None:
    args = parse_args()
    rows = []

    for file_path in _iter_log_files(args.log_path):
        result = extract_last_dict(file_path)
        base_row = {
            **_extract_file_metadata(file_path),
            "num_runs": result.get("num_runs"),
        }

        for dataset_result in _iter_per_dataset_results(result, base_row=base_row):
            rows.append(dataset_result)

    results_df = pd.DataFrame(rows)
    if results_df.empty:
        results_df = pd.DataFrame(
            {
                "benchmark": pd.Series(dtype="object"),
                "clf_name": pd.Series(dtype="object"),
                "dataset_id": pd.Series(dtype="object"),
                "auroc_mean": pd.Series(dtype="float64"),
                "auroc_std": pd.Series(dtype="float64"),
                "auprc_mean": pd.Series(dtype="float64"),
                "auprc_std": pd.Series(dtype="float64"),
                "f1_mean": pd.Series(dtype="float64"),
                "f1_std": pd.Series(dtype="float64"),
            }
        )
    else:
        results_df = results_df.rename(
            columns={
                "auc_mean": "auroc_mean",
                "auc_std": "auroc_std",
            }
        )
        for column in ["benchmark", "clf_name", "dataset_id", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std", "f1_mean", "f1_std"]:
            if column not in results_df.columns:
                results_df[column] = pd.NA
        results_df = results_df[["benchmark", "clf_name", "dataset_id", "auroc_mean", "auroc_std", "auprc_mean", "auprc_std", "f1_mean", "f1_std"]]

    print(results_df.to_string(index=False))
    results_df.to_csv(args.log_path / "results_our_methods.csv", index=False)


if __name__ == "__main__":
    main()


