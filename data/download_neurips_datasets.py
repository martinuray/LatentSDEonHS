"""Download and preprocess NeurIPS AD benchmark helper datasets.

Adapted from:
    https://github.com/datamllab/tods/tree/benchmark/benchmark/realworld_data/data
"""

import argparse
from pathlib import Path
from typing import Callable

import pandas as pd
import requests


DATASETS = {
    "creditcard": {
        "url": "https://www.openml.org/data/get_csv/1673544/phpKo8OWT",
        "raw_name": "openml_creditcard.csv",
        "output_name": "creditcard.csv",
    },
    "gecco": {
        "url": "https://zenodo.org/record/3884398/files/1_gecco2018_water_quality.csv?download=1",
        "raw_name": "gecco.csv",
        "output_name": "water_quality.csv",
    },
}


def download_file(url: str, dest: Path, timeout: int = 60, retries: int = 3, force: bool = False) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        return dest

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as response:
                response.raise_for_status()
                tmp_path = dest.with_suffix(dest.suffix + ".part")
                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp_path.replace(dest)
                return dest
        except requests.RequestException as exc:
            last_err = exc
            print(f"Download failed (attempt {attempt}/{retries}) for {url}: {exc}")

    raise RuntimeError(f"Could not download {url}") from last_err


def preprocess_creditcard(raw_path: Path, output_path: Path) -> None:
    df = pd.read_csv(raw_path)
    df = df.dropna()

    # Keep legacy behavior: move last column to front, then sort by Time.
    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]

    if "Time" in df.columns:
        df = df.sort_values(by=["Time"])
        df = df.drop(columns=["Time"])

    # OpenML often stores target as quoted strings.
    df["Class"] = df["Class"].astype(str).str.replace("'", "", regex=False).astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def preprocess_gecco(raw_path: Path, output_path: Path) -> None:
    df = pd.read_csv(raw_path)
    df = df.dropna()

    drop_cols = []
    if "Time" in df.columns:
        drop_cols.append("Time")
    if len(df.columns) > 0:
        drop_cols.append(df.columns[0])
    # Preserve old semantics while being resilient when columns differ.
    df = df.drop(columns=list(dict.fromkeys(drop_cols)), errors="ignore")

    cols = df.columns.tolist()
    df = df[cols[-1:] + cols[:-1]]

    if "EVENT" in df.columns:
        df["EVENT"] = df["EVENT"].map({False: "0", True: "1", "False": "0", "True": "1", 0: "0", 1: "1"}).fillna(df["EVENT"])
        df = df.rename(columns={"EVENT": "label"})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8")


def run_dataset(name: str, output_root: Path, preprocess_fn: Callable[[Path, Path], None],
                timeout: int, retries: int, force: bool) -> None:
    spec = DATASETS[name]
    raw_path = output_root / spec["raw_name"]
    out_path = output_root / spec["output_name"]

    print(f"[{name}] downloading -> {raw_path}")
    download_file(spec["url"], raw_path, timeout=timeout, retries=retries, force=force)
    print(f"[{name}] preprocessing -> {out_path}")
    preprocess_fn(raw_path, out_path)
    print(f"[{name}] done")
    # remove out_path
    if raw_path.exists():
        raw_path.unlink()


def parse_args() -> argparse.Namespace:
    default_output_root = Path("data_dir") / "NeurIPS" / "raw"

    parser = argparse.ArgumentParser(description="Download and preprocess NeurIPS AD benchmark helper datasets.")
    parser.add_argument("--dataset", choices=["creditcard", "gecco", "all"], default="all")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=default_output_root,
        help="Root output directory. Files are written to output_root/{dataset}/...",
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--force", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    todo = [args.dataset] if args.dataset != "all" else ["gecco", "creditcard"]

    preprocessors = {
        "creditcard": preprocess_creditcard,
        "gecco": preprocess_gecco,
    }

    for name in todo:
        run_dataset(
            name=name,
            output_root=args.output_root,
            preprocess_fn=preprocessors[name],
            timeout=args.timeout,
            retries=args.retries,
            force=args.force,
        )


if __name__ == "__main__":
    main()
