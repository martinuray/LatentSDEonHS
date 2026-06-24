#!/bin/bash

################################################################################
# SLURM Script: Submit Irregular Sine Interpolation training
#
# Usage: ./scripts/slurm/submit_irregular_sine_interpolation.sh
################################################################################

set -euo pipefail

# SLURM configuration
PARTITION="rtx2080ti"
TIMEOUT="48:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="40GB"
JOB_NAME="irregular_sine"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

# Log directory for SLURM output
LOG_DIR="slurm_logs_irregular_sine"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting Irregular Sine training job"
echo "=================================="
echo "Partition: ${PARTITION}"
echo "Timeout: ${TIMEOUT}"
echo "GPUs per job: ${NUM_GPUS}"
echo "CPUs per job: ${NUM_CPUS}"
echo "Memory per job: ${MEMORY}"
echo "Log directory: ${LOG_DIR}"
echo "=================================="
echo ""

sbatch \
    --partition="${PARTITION}" \
    --time="${TIMEOUT}" \
    --gpus="${NUM_GPUS}" \
    --cpus-per-task="${NUM_CPUS}" \
    --mem="${MEMORY}" \
    --job-name="${JOB_NAME}" \
    --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
    --error="${LOG_DIR}/${JOB_NAME}_%j.log" \
    --wrap="python irregular_sine_interpolation.py \
        --enable-file-logging \
        --log-dir logs \
        --enable-checkpointing \
        --checkpoint-dir checkpoints \
        --checkpoint-at 90 390 990 2190 3990 \
        --lr 0.001 \
        --n-epochs 3990 \
        --kl0-weight 0.001 \
        --klp-weight 0.01 \
        --pxz-weight 1.0 \
        --seed -1 \
        --restart 30 \
        --device cuda:0 \
        --z-dim 3 \
        --h-dim 3 \
        --n-deg 6 \
        --no-learnable-prior \
        --freeze-sigma \
        --mc-eval-samples 10 \
        --mc-train-samples 10 \
        --loglevel debug"

echo ""
echo "Submitted Irregular Sine training job."
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in: ${LOG_DIR}"
echo ""
