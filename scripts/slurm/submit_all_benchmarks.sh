#!/bin/bash

################################################################################
# SLURM Script: Submit all benchmark experiments
#
# This script submits a separate SLURM job for each (benchmark, seed) pair,
# so each of the 5 seeds runs as its own independent job with automatic
# deletion of processed data. No cross-run aggregation is performed.
#
# Usage: ./scripts/slurm/submit_all_benchmarks.sh
################################################################################

set -euo pipefail

# SLURM Configuration - Modify as needed
PARTITION="rtx2080ti"              # Partition to submit to
TIMEOUT="64:00:00"           # Timeout per job (HH:MM:SS)
NUM_GPUS=1                   # Number of GPUs per job
NUM_CPUS=8                   # Number of CPUs per job
MEMORY="40GB"                # Memory per job
JOB_NAME_PREFIX="LSD"    # Prefix for job names

# Seeds to submit independently (one job per seed)
SEEDS=(42 43 44 45 46)

# Common anomaly_detection.py parameters (dataset and runs are set per benchmark below)

# Benchmarks to run (from anomaly_detection.py)
BENCHMARKS=("SWaT WaDi PSM MSL SMAP SMD")


# ---- Conda / project setup ----
CONDA_BASE="${CONDA_BASE:-/home2/muray/.miniconda3}"
CONDA_ENV="baseline-latent"
PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"

# ---- Initialize conda (for this submitter script) ----
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

# ---- Move to project directory ----
cd "${PROJECT_DIR}"

# Environment setup prepended to every sbatch --wrap so the job (which runs in a
# fresh non-login shell on the compute node) has the right python on PATH.
JOB_SETUP="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && cd ${PROJECT_DIR} &&"

# Log directory for SLURM output
LOG_DIR="slurm_logs_benchmark"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting all benchmark jobs"
echo "=================================="
echo "Total benchmarks: ${#BENCHMARKS[@]}"
echo "Seeds per benchmark: ${SEEDS[*]}"
echo "Partition: ${PARTITION}"
echo "Timeout: ${TIMEOUT}"
echo "GPUs per job: ${NUM_GPUS}"
echo "CPUs per job: ${NUM_CPUS}"
echo "Memory per job: ${MEMORY}"
echo "Log directory: ${LOG_DIR}"
echo "=================================="
echo ""

# Submit a job for each (benchmark, seed) pair
for BENCHMARK in "${BENCHMARKS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        echo "Submitting jobs for benchmark: ${BENCHMARK}, seed: ${SEED}"

        sbatch \
            --partition="${PARTITION}" \
            --time="${TIMEOUT}" \
            --gpus="${NUM_GPUS}" \
            --cpus-per-task="${NUM_CPUS}" \
            --mem="${MEMORY}" \
            --job-name="${JOB_NAME_PREFIX}_Sn_${BENCHMARK}_s${SEED}" \
            --output="${LOG_DIR}/${BENCHMARK}_Sn_s${SEED}_%j.log" \
            --error="${LOG_DIR}/${BENCHMARK}_Sn_s${SEED}_%j.log" \
            --wrap="python anomaly_detection.py \
                --dataset ${BENCHMARK} \
                --runs 1 \
                --seed ${SEED} \
                --sphere-embedding"

        sleep 0.5

        sbatch \
            --partition="${PARTITION}" \
            --time="${TIMEOUT}" \
            --gpus="${NUM_GPUS}" \
            --cpus-per-task="${NUM_CPUS}" \
            --mem="${MEMORY}" \
            --job-name="${JOB_NAME_PREFIX}_Rn_${BENCHMARK}_s${SEED}" \
            --output="${LOG_DIR}/${BENCHMARK}_Rn_s${SEED}_%j.log" \
            --error="${LOG_DIR}/${BENCHMARK}_Rn_s${SEED}_%j.log" \
            --wrap="python anomaly_detection.py \
                --dataset ${BENCHMARK} \
                --runs 1 \
                --seed ${SEED} \
                --no-sphere-embedding"

        # Small delay to avoid overwhelming the scheduler
        sleep 0.5
    done
done

echo ""
echo "=================================="
echo "All jobs submitted successfully!"
echo "=================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in: ${LOG_DIR}"
echo ""

