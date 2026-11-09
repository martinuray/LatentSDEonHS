# AGENTS.md

## What this repo is
- Reference implementation for "Latent SDEs on Homogeneous Spaces" with experiment scripts at repo root (`anomaly_detection.py`, `rotating_mnist.py`, `pendulum_*.py`, `activity_classification.py`).
- Shared model/training code lives in `core/`; dataset-specific preprocessing/loading lives in `data/`; reusable utilities are in `utils/`.

## Core architecture (read these first)
- `core/models.py`: defines recognition nets, path-distribution encoder (`default_SOnPathDistributionEncoder`), likelihood decoders, and ELBO (`ELBO`).
- `core/pathdistribution.py` + `core/sde_solvers.py`: manifold/path-distribution mechanics used by all experiments.
- `core/training.py`: `generic_train(...)` is the common train step used by most scripts.
- `utils/parser.py`: global CLI defaults (logging/checkpointing/device/model/training args).

## Data flow pattern used across experiments
- Batch schema is dictionary-based and consistent: `inp_obs/inp_msk/inp_tps` -> recognition net -> `qzx_net` -> sampled latent paths -> `pxz_net`.
- ELBO consumes `evd_obs/evd_tid/evd_msk` (see `ELBO.forward` in `core/models.py`).
- For anomaly detection, scoring uses `-pxz.log_prob(...)` aggregated per original time index (`inp_indcs`) in `anomaly_detection.py:evaluate`.

## Anomaly-detection specifics (main benchmark path)
- Main entry point: `anomaly_detection.py` (supports `SWaT`, `WaDi`, `SMD`, `QAD`, `MSL`, `SMAP`, `PSM`).
- Providers with many sub-datasets (`SMD`, `QAD`, `MSL/SMAP`) expose hybrid layout (`num_datasets`, `input_dims`, `num_timepoints_list`); script trains each slice separately and macro-averages metrics.
- Processed windows are materialized by providers under `data_dir/.../processed` or temporary dirs like `LatentSDEonHS_<dataset>_processed_*`.
- `--delete-processed-data` (or internal cleanup between repeated runs) is important to avoid stale processed artifacts.

## Practical workflows
- Environment baseline from `README.md`: PyTorch + `einops scipy scikit-learn gdown`; this repo also pins broader deps in `requirements.txt`.
- Quick smoke check:
  - `python rotating_mnist.py --help`
- Typical benchmark run:
  - `python anomaly_detection.py --dataset SWaT --runs 1 --enable-file-logging --log-dir logs --data-dir data_dir`
- Multi-job SLURM launcher: `scripts/slurm/submit_all_benchmarks.sh` (wraps `anomaly_detection.py`, one job per dataset).

## Logging, outputs, and artifacts
- Metrics/log text: `logs/*.json` and `logs/*.txt` (via `utils/logger.py` + `utils/misc.save_stats`).
- TensorBoard: `runs/<experiment_id>/`.
- Checkpoints: `checkpoints/checkpoint_<experiment>_<epoch>.h5` when checkpointing is enabled.
- Aggregated final metrics CSV: default `logs/final_metrics.csv` (`append_final_metrics_csv`).

## Repo conventions to preserve
- Extend CLIs via `extend_argparse(generic_parser)` in each experiment script.
- Keep module names in `nn.ModuleDict` stable (`recog_net`, `recon_net`, `pxz_net`, `qzx_net`, optional `aux_net`) because training/eval code expects these keys.
- Use `ProgressMessage` format for epoch logs and keep stats dict keys consistent (`trn`, `val`, `tst`, `oth`).
- Respect masked-observation contract: masked entries in `obs` must be zero (`utils/misc.scatter_obs_and_msk` asserts this).

## Known gotchas
- `core/power_spherical/` must exist (README documents vendoring from `nicola-decao/power_spherical`).
- `run_all.sh` references `basic_data_anomaly_detection.py`, which is not present; use `anomaly_detection.py` instead.
- `tests.py` mixes geometry unit tests and dataset-dependent tests; running the full suite requires local datasets in `data_dir/`.
