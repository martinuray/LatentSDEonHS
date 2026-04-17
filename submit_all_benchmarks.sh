#!/bin/bash

################################################################################
# SLURM Script: Submit all benchmark experiments
#
# This script submits a separate SLURM job for each benchmark dataset.
# Each job runs 5 independent runs with automatic deletion of processed data.
#
# Usage: ./submit_all_benchmarks.sh
################################################################################

# SLURM Configuration - Modify as needed
PARTITION="gpu"              # Partition to submit to
TIMEOUT="02:00:00"           # Timeout per job (HH:MM:SS)
NUM_GPUS=1                   # Number of GPUs per job
NUM_CPUS=8                   # Number of CPUs per job
MEMORY="40GB"                # Memory per job
JOB_NAME_PREFIX="anomaly"    # Prefix for job names

# Number of runs per benchmark
RUNS=5

# Benchmarks to run (from anomaly_detection.py)
BENCHMARKS=("SWaT" "WaDi" "SMD" "aero" "QAD" "MSL" "SMAP" "PSM")

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log directory for SLURM output
LOG_DIR="${SCRIPT_DIR}/slurm_logs"
mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting all benchmark jobs"
echo "=================================="
echo "Total benchmarks: ${#BENCHMARKS[@]}"
echo "Runs per benchmark: ${RUNS}"
echo "Partition: ${PARTITION}"
echo "Timeout: ${TIMEOUT}"
echo "GPUs per job: ${NUM_GPUS}"
echo "CPUs per job: ${NUM_CPUS}"
echo "Memory per job: ${MEMORY}"
echo "Log directory: ${LOG_DIR}"
echo "=================================="
echo ""

# Submit a job for each benchmark
for BENCHMARK in "${BENCHMARKS[@]}"; do
    LOG_FILE="${LOG_DIR}/${BENCHMARK}_%j.log"

    echo "Submitting job for benchmark: ${BENCHMARK}"

    sbatch \
        --partition="${PARTITION}" \
        --time="${TIMEOUT}" \
        --gpus="${NUM_GPUS}" \
        --cpus-per-task="${NUM_CPUS}" \
        --mem="${MEMORY}" \
        --job-name="${JOB_NAME_PREFIX}_${BENCHMARK}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --wrap="cd ${SCRIPT_DIR} && python anomaly_detection.py \
            --dataset ${BENCHMARK} \
            --runs ${RUNS} \
            --delete-processed-data \
            --seed 42"

    # Small delay to avoid overwhelming the scheduler
    sleep 0.5
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

