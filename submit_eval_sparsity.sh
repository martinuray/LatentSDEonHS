#!/bin/bash

################################################################################
# SLURM Script: Run eval_sparsity_data.py in parallel
#
# Submits one SLURM array job with 55 tasks (5 seeds × 11 subsamples).
# Each task runs a single (seed, subsample) experiment and writes a JSON.
# A second job (depends on the array finishing) aggregates the results.
#
# Usage:
#   ./submit_eval_sparsity.sh [DATASET] [RESULTS_DIR]
# Example:
#   ./submit_eval_sparsity.sh SWaT out/sparsity_results_SWaT
################################################################################

set -euo pipefail

# ---- Configuration ----
PARTITION="rtx2080ti"
TIMEOUT="6:00:00"        # each single run is much shorter than the full sequential job
NUM_GPUS=1
NUM_CPUS=4
MEMORY="20GB"

DATASET="${1:-SWaT}"
RESULTS_DIR="${2:-out/sparsity_results_${DATASET}}"

# Number of tasks: NUM_SEEDS(5) × NUM_SUBSAMPLES(11) = 55
NUM_TASKS=54   # 0-based last index (0..54)

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
LOG_DIR="${PROJECT_DIR}/slurm_logs_sparsity"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent

cd "${PROJECT_DIR}"

# ---- Submit the array job ----
ARRAY_JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="${TIMEOUT}" \
    --gpus="${NUM_GPUS}" \
    --cpus-per-task="${NUM_CPUS}" \
    --mem="${MEMORY}" \
    --job-name="sparsity_${DATASET}" \
    --output="${LOG_DIR}/sparsity_${DATASET}_%A_%a.log" \
    --error="${LOG_DIR}/sparsity_${DATASET}_%A_%a.log" \
    --array="0-${NUM_TASKS}" \
    --wrap="python eval_sparsity_data.py \
        --mode single \
        --task-id \${SLURM_ARRAY_TASK_ID} \
        --results-dir ${RESULTS_DIR} \
        --dataset ${DATASET} \
        --runs 1 \
        --enable-file-logging \
        --log-dir logs \
        --data-dir data_dir \
        --device cuda \
        --delete-processed-data" \
    --parsable)

echo "Submitted array job: ${ARRAY_JOB_ID}  (tasks 0-${NUM_TASKS})"

# ---- Submit aggregation job (runs after all array tasks succeed) ----
AGG_JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="00:30:00" \
    --gpus=0 \
    --cpus-per-task=2 \
    --mem="8GB" \
    --job-name="sparsity_agg_${DATASET}" \
    --output="${LOG_DIR}/sparsity_agg_${DATASET}_%j.log" \
    --error="${LOG_DIR}/sparsity_agg_${DATASET}_%j.log" \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --wrap="python eval_sparsity_data.py \
        --mode aggregate \
        --results-dir ${RESULTS_DIR} \
        --dataset ${DATASET} \
        --data-dir data_dir" \
    --parsable)

echo "Submitted aggregation job: ${AGG_JOB_ID}  (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor array:       squeue -j ${ARRAY_JOB_ID}"
echo "Monitor aggregation: squeue -j ${AGG_JOB_ID}"
echo "Results will be in:  ${RESULTS_DIR}/"
