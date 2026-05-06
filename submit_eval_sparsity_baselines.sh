#!/bin/bash

################################################################################
# SLURM Script: Run baselines/eval_sparsity_baselines.py in parallel
#
# Submits one SLURM array job (NUM_SEEDS × NUM_SUBSAMPLES tasks).
# Each task evaluates one (seed, subsample) pair and writes a JSON.
# A second job (depends on the array finishing) aggregates the results.
#
# All tasks run on CPU only (no GPU requested).
#
# Usage:
#   ./submit_eval_sparsity_baselines.sh [BENCHMARK] [CLASSIFIERS] [RESULTS_DIR]
# Examples:
#   ./submit_eval_sparsity_baselines.sh
#       → default: QAD, KNN, out/sparsity_baselines_QAD_KNN
#   ./submit_eval_sparsity_baselines.sh SMD "KNN,IForest" out/sparsity_baselines_SMD
################################################################################

set -euo pipefail

# ---- Configuration ----
PARTITION="gtx1080ti"
TIMEOUT="4:00:00"       # each single task finishes in minutes for most classifiers
NUM_CPUS=4
NUM_GPUS=1
MEMORY="16GB"

# Defaults: QAD benchmark, KNN classifier, CPU only
BENCHMARK="${1:-QAD}"
CLASSIFIERS="${2:-KNN}"
RESULTS_DIR="${3:-out/sparsity_baselines_${BENCHMARK}_${CLASSIFIERS}}"

# Subsample grid (12 levels) and number of seeds → 60 total tasks (0-based: 0..59)
NUM_SUBSAMPLES=12
NUM_SEEDS=5
NUM_TASKS=$(( NUM_SEEDS * NUM_SUBSAMPLES - 1 ))   # 0-based upper bound

SUBSAMPLES="0.001,0.01,0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9"

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
LOG_DIR="${PROJECT_DIR}/slurm_logs_sparsity_baselines"
mkdir -p "${LOG_DIR}" "${RESULTS_DIR}"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent

cd "${PROJECT_DIR}"

# ---- Submit the array job ----
ARRAY_JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="${TIMEOUT}" \
    --cpus-per-task="${NUM_CPUS}" \
    --mem="${MEMORY}" \
    --gpus="${NUM_GPUS}" \
    --job-name="sparsity_bl_${BENCHMARK}" \
    --output="${LOG_DIR}/sparsity_bl_${BENCHMARK}_%A_%a.log" \
    --error="${LOG_DIR}/sparsity_bl_${BENCHMARK}_%A_%a.log" \
    --array="0-${NUM_TASKS}" \
    --wrap="python baselines/eval_sparsity_baselines.py \
        --mode single \
        --task-id \${SLURM_ARRAY_TASK_ID} \
        --benchmark ${BENCHMARK} \
        --classifiers ${CLASSIFIERS} \
        --subsamples '${SUBSAMPLES}' \
        --num-seeds ${NUM_SEEDS} \
        --results-dir ${RESULTS_DIR}" \
    --parsable)

echo "Submitted array job: ${ARRAY_JOB_ID}  (tasks 0-${NUM_TASKS})"

# ---- Submit aggregation job (runs after all array tasks succeed) ----
AGG_JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="00:30:00" \
    --cpus-per-task=2 \
    --mem="8GB" \
    --job-name="sparsity_bl_agg_${BENCHMARK}" \
    --output="${LOG_DIR}/sparsity_bl_agg_${BENCHMARK}_%j.log" \
    --error="${LOG_DIR}/sparsity_bl_agg_${BENCHMARK}_%j.log" \
    --dependency="afterok:${ARRAY_JOB_ID}" \
    --wrap="python baselines/eval_sparsity_baselines.py \
        --mode aggregate \
        --benchmark ${BENCHMARK} \
        --results-dir ${RESULTS_DIR}" \
    --parsable)

echo "Submitted aggregation job: ${AGG_JOB_ID}  (depends on ${ARRAY_JOB_ID})"
echo ""
echo "Monitor array:       squeue -j ${ARRAY_JOB_ID}"
echo "Monitor aggregation: squeue -j ${AGG_JOB_ID}"
echo "Results will be in:  ${RESULTS_DIR}/"

