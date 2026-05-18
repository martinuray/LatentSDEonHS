# SLURM launchers

This folder contains the repository's SLURM submission scripts and job launchers.

Run them from the repository root, for example:

```bash
./scripts/slurm/submit_rotating_mnist.sh
./scripts/slurm/submit_all_benchmarks.sh
./scripts/slurm/submit_qad_traces.sh
./scripts/slurm/submit_test_wandb.sh
```

The launchers activate the `baseline-latent` environment and then submit jobs from the repository root, so the existing relative paths to `data_dir/`, `logs/`, `checkpoints/`, and `out/` continue to work.

