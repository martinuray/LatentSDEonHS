#!/bin/bash

################################################################################
# SLURM Script: Submit fixed ablation jobs on PSM.
#
# Usage:
#   bash scripts/slurm/submit_ablation.sh
################################################################################

set -euo pipefail

# SLURM configuration
PARTITION="rtx2080ti,a6000"
TIMEOUT="48:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="40GB"
JOB_NAME_PREFIX="ablation"

# Experiment configuration
DATASET="PSM"
RUNS=1
SEEDS=(41 42 43 44 45)
LOG_DIR="slurm_logs_ablation"

# NOTE:
# The repository currently has no dedicated CLI flag to disable SDE dynamics only.
# Keep this variable as placeholder and set it once such a flag exists.
RN_NO_SDE_EXTRA_FLAGS="--no-sde"

# ---- Initialize conda ----
source $(conda info --base)/etc/profile.d/conda.sh

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting fixed PSM ablation jobs"
echo "Dataset        : ${DATASET}"
echo "Runs/job       : ${RUNS}"
echo "Seeds          : ${SEEDS[*]}"
echo "Partition      : ${PARTITION}"
echo "Timeout        : ${TIMEOUT}"
echo "Log directory  : ${LOG_DIR}"
echo "=================================="
echo ""

submit_variant_for_seed() {
  local variant_name="$1"
  local partition="$2"
  local extra_flags="$3"
  local seed="$4"

  sbatch \
    --partition="${partition}" \
    --time="${TIMEOUT}" \
    --gpus="${NUM_GPUS}" \
    --cpus-per-task="${NUM_CPUS}" \
    --mem="${MEMORY}" \
    --job-name="${JOB_NAME_PREFIX}_${variant_name}_${DATASET}_s${seed}" \
    --output="${LOG_DIR}/${DATASET}_${variant_name}_s${seed}_%j.log" \
    --error="${LOG_DIR}/${DATASET}_${variant_name}_s${seed}_%j.log" \
    --wrap="python anomaly_detection.py \
      --dataset ${DATASET} \
      --runs ${RUNS} \
      --seed ${seed} \
      ${extra_flags}"
}

# 1) Sn
for seed in "${SEEDS[@]}"; do
  submit_variant_for_seed "Sn" "rtx2080ti" "--sphere-embedding" "${seed}"
  sleep 0.2
done

# 2) Rn
for seed in "${SEEDS[@]}"; do
  submit_variant_for_seed "Rn" "rtx2080ti" "--no-sphere-embedding" "${seed}"
  sleep 0.2
done

# 3) Rn without SDE (placeholder flags: update RN_NO_SDE_EXTRA_FLAGS when available)
for seed in "${SEEDS[@]}"; do
  submit_variant_for_seed "RnNoSDE" "a6000" "--no-sphere-embedding ${RN_NO_SDE_EXTRA_FLAGS}" "${seed}"
  sleep 0.2
done

# 4) Stronger decoder
for seed in "${SEEDS[@]}"; do
  submit_variant_for_seed "StrongDec" "rtx2080ti" "--sphere-embedding --dec-hidden-dim 128 --n-dec-layers 4" "${seed}"
  sleep 0.2
done

echo ""
echo "=================================="
echo "All ablation jobs submitted."
echo "=================================="
echo "Monitor with: squeue -u \$USER"
echo "Logs in: ${LOG_DIR}"
