#!/usr/bin/env python3
"""Backup dataset raw folders into a single zip archive.

This script scans a data root (default: data_dir), finds dataset folders that
contain a direct `raw/` subfolder, and stores all raw files in one zip archive.
"""

import argparse
import datetime as dt
import os
from pathlib import Path
import zipfile


def find_raw_dirs(data_dir: Path, datasets: list[str] | None = None) -> list[Path]:
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data dir does not exist: {data_dir}")

    candidates = []
    for child in sorted(data_dir.iterdir()):
        if not child.is_dir():
            continue
        if datasets is not None and child.name not in datasets:
            continue
        raw = child / "raw"
        if raw.is_dir():
            candidates.append(raw)

    if datasets is not None:
        found = {p.parent.name for p in candidates}
        missing = [d for d in datasets if d not in found]
        if missing:
            raise ValueError(f"Requested datasets missing raw/ folder: {missing}")

    return candidates


def resolve_output_zip(output: Path) -> Path:
    if output.suffix.lower() == ".zip":
        output.parent.mkdir(parents=True, exist_ok=True)
        return output

    output.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return output / f"datasets_raw_backup_{stamp}.zip"


def collect_files(raw_dirs: list[Path], include_hidden: bool) -> list[tuple[Path, Path]]:
    """Return list of (src_file, arcname) pairs."""
    files: list[tuple[Path, Path]] = []
    for raw_dir in raw_dirs:
        dataset_name = raw_dir.parent.name
        for root, _, fnames in os.walk(raw_dir):
            root_p = Path(root)
            for fname in fnames:
                if not include_hidden and fname.startswith("."):
                    continue
                src = root_p / fname
                rel_in_raw = src.relative_to(raw_dir)
                arcname = Path(dataset_name) / "raw" / rel_in_raw
                files.append((src, arcname))
    return files


def create_zip(zip_path: Path, files: list[tuple[Path, Path]]) -> int:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arcname in files:
            zf.write(src, arcname.as_posix())
    return len(files)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backup all dataset raw/ folders into one zip archive.")
    parser.add_argument("--data-dir", default="data_dir", help="Root directory containing dataset folders.")
    parser.add_argument(
        "--output",
        required=True,
        help="Output .zip file path or destination directory where the archive should be created.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Optional dataset folder names to include (e.g. SMD QAD SWaT).",
    )
    parser.add_argument(
        "--include-hidden",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Include hidden files (dotfiles) from raw/ folders.",
    )
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print what would be archived without creating a zip.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    data_dir = Path(args.data_dir).resolve()
    output = Path(args.output).resolve()

    raw_dirs = find_raw_dirs(data_dir, args.datasets)
    if not raw_dirs:
        raise RuntimeError(f"No raw/ folders found under: {data_dir}")

    files = collect_files(raw_dirs, include_hidden=args.include_hidden)
    if not files:
        raise RuntimeError("No files found in raw/ folders.")

    print("Raw folders to back up:")
    for raw in raw_dirs:
        print(f" - {raw}")
    print(f"Total files: {len(files)}")

    zip_path = resolve_output_zip(output)
    if args.dry_run:
        print(f"[DRY RUN] Archive path: {zip_path}")
        return 0

    count = create_zip(zip_path, files)
    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Created archive: {zip_path}")
    print(f"Archived files: {count}")
    print(f"Archive size: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

