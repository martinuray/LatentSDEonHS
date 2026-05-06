#!/bin/bash
#SBATCH --partition=rtx2080ti
#SBATCH --time=48:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40GB
#SBATCH --job-name=run_all_anomaly
#SBATCH --output=slurm_logs/run_all_%j.log
#SBATCH --error=slurm_logs/run_all_%j.log

set -euo pipefail

# Array of datasets (default: SWaT only; uncomment full list when needed)
# DATASETS=(SMAP MSL QAD SWaT WaDi PSM SMD)
DATASETS=(WaDi)

# Track failed runs
FAILED_DATASETS=()

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
mkdir -p "${PROJECT_DIR}/slurm_logs"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent
cd "${PROJECT_DIR}"

# Run for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "Running with dataset: ${dataset}"
  if ! python anomaly_detection.py --dataset "${dataset}"; then
    echo "ERROR: Dataset ${dataset} failed!"
    FAILED_DATASETS+=("${dataset}")
  else
    echo "SUCCESS: Dataset ${dataset} completed"
  fi
  echo "---"
done

# Report results
echo ""
echo "========== SUMMARY =========="
if [ ${#FAILED_DATASETS[@]} -eq 0 ]; then
  echo "All datasets completed successfully!"
else
  echo "Failed datasets: ${FAILED_DATASETS[*]}"
  exit 1
fi
