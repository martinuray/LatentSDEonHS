#!/usr/bin/env python3
"""Convert QAD pickle files into TXT exports.

Reference behavior:
- Uses pickle loading compatible with `data/qad_provider.py::load_pickl`.
- Converts `pandas.Series` payloads into a one-column DataFrame named `labels`.

Examples:
    python convert_qad_pickles.py \
        --input-root /path/to/QAD/raw/qad_clean_pkl_100Hz \
        --delimiter '\t'

    # Optional: custom destination instead of provider default.
    python convert_qad_pickles.py \
        --input-root /path/to/QAD/raw/qad_clean_pkl_100Hz \
        --output-root /path/to/custom/output
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def _validate_numpy_version() -> None:
    major_str = np.__version__.split(".", 1)[0]
    try:
        major = int(major_str)
    except ValueError as exc:
        raise RuntimeError(f"Could not parse numpy version: {np.__version__}") from exc

    if major < 2:
        raise RuntimeError(
            "This converter requires numpy>=2.0.0 to load the QAD pickle files. "
            f"Detected numpy=={np.__version__}."
        )


def _load_pickl(dataset_path: Path):
    with dataset_path.open("rb") as f:
        data = pickle.load(f)
    if isinstance(data, pd.Series):
        data = data.to_frame(name="labels")
    return data


def _to_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, pd.Series):
        return data.to_frame(name="labels")

    if isinstance(data, np.ndarray):
        if data.ndim == 0:
            return pd.DataFrame({"value": [data.item()]})
        if data.ndim == 1:
            return pd.DataFrame({"value": data})
        if data.ndim == 2:
            return pd.DataFrame(data)

        # Keep first axis as samples and flatten the remaining axes.
        flattened = data.reshape(data.shape[0], -1)
        return pd.DataFrame(flattened)

    if isinstance(data, (list, tuple)):
        arr = np.asarray(data)
        return _to_dataframe(arr)

    # Final fallback for dicts/scalars/custom objects.
    return pd.DataFrame([data])


def _iter_pickles(input_root: Path, glob_pattern: str) -> Iterable[Path]:
    return sorted(p for p in input_root.rglob(glob_pattern) if p.is_file())


def _target_path(src_path: Path, input_root: Path, output_root: Path, out_ext: str) -> Path:
    rel = src_path.relative_to(input_root)
    return output_root / rel.with_suffix(out_ext)


def convert_all(
    input_root: Path,
    output_root: Path,
    delimiter: str,
    glob_pattern: str,
    overwrite: bool,
    dry_run: bool,
) -> int:
    files = list(_iter_pickles(input_root, glob_pattern))
    if len(files) == 0:
        print(f"No pickle files found in {input_root} (pattern: {glob_pattern})")
        return 0

    out_ext = ".txt"
    converted = 0
    skipped = 0

    for src_path in files:
        dst_path = _target_path(src_path, input_root, output_root, out_ext)

        if dst_path.exists() and not overwrite:
            print(f"[skip] exists: {dst_path}")
            skipped += 1
            continue

        print(f"[convert] {src_path} -> {dst_path}")
        if dry_run:
            converted += 1
            continue

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        data = _load_pickl(src_path)
        df = _to_dataframe(data)

        df.to_csv(dst_path, index=False, sep=delimiter)

        converted += 1

    print(
        "Done. "
        f"Total: {len(files)}, converted: {converted}, skipped: {skipped}, "
        f"format: txt"
    )
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Convert all QAD pickle files to TXT.")
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("data/QAD/raw/qad_clean_pkl_100Hz"),
        help="Folder containing QAD *.pkl files."
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/QAD/raw/qad_clean_txt_100Hz"),
        help="Folder where converted files are written (default matches qad_provider.py raw_subdir)."
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Delimiter for TXT output."
    )
    parser.add_argument(
        "--glob",
        default="*.pkl",
        help="Glob pattern used recursively under input-root."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing converted files."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned conversions without writing files."
    )
    return parser


def main() -> int:
    _validate_numpy_version()

    parser = _build_parser()
    args = parser.parse_args()

    input_root = args.input_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()

    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist or is not a directory: {input_root}")

    return convert_all(
        input_root=input_root,
        output_root=output_root,
        delimiter=args.delimiter,
        glob_pattern=args.glob,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())

