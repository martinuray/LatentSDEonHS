# Dataset Analysis Script

Use `analyze_dataset_stats.py` to summarize anomaly-detection datasets.

## What it reports

- number of sub-datasets per benchmark
- per-dataset train/test length
- per-dataset feature count
- per-dataset anomaly ratio
- benchmark-level mean and weighted anomaly ratio

## Usage

```bash
python analyze_dataset_stats.py --benchmark SMD
python analyze_dataset_stats.py --benchmark QAD --qad-subdir qad_clean_pkl_100Hz
python analyze_dataset_stats.py --benchmark SMAP --limit 10
python analyze_dataset_stats.py --benchmark all --json-out logs/dataset_stats.json
```

## Benchmarks supported

- `SMD`
- `QAD`
- `SWaT`
- `WaDi`
- `SMAP`
- `MSL`
- `all`

## Notes

- The script reads from raw data folders under `data_dir`.
- For QAD, the raw subfolder can be selected via `--qad-subdir`.
- SWaT/WaDi are reported as one dataset each.

