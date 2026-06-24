#!/bin/bash

################################################################################
# SLURM Script: Submit Rotating MNIST experiment
#
# Usage: ./scripts/slurm/submit_rotating_mnist.sh
################################################################################

set -euo pipefail

# SLURM Configuration
PARTITION="rtx2080ti"
TIMEOUT="48:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="40GB"
JOB_NAME="rot_mnist"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

# Log directory for SLURM output
LOG_DIR="slurm_logs_rotating_mnist"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting Rotating MNIST job"
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
    --wrap="python rotating_mnist.py \
        --data-dir data_dir \
        --enable-file-logging \
        --log-dir logs \
        --enable-checkpointing \
        --checkpoint-dir checkpoints \
        --checkpoint-at 30 330 660 990 \
        --batch-size 32 \
        --lr 0.001 \
        --n-epochs 990 \
        --kl0-weight 0.0001 \
        --klp-weight 0.0001 \
        --pxz-weight 1.0 \
        --seed -1 \
        --restart 30 \
        --device cuda:0 \
        --z-dim 16 \
        --h-dim 32 \
        --n-deg 6 \
        --no-learnable-prior \
        --no-freeze-sigma \
        --mc-eval-samples 1 \
        --mc-train-samples 1 \
        --loglevel debug \
        --n-filters 8"

echo ""
echo "Submitted Rotating MNIST job."
echo "Monitor with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in: ${LOG_DIR}"
echo ""
