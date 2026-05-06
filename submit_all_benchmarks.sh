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
PARTITION="rtx2080ti,a6000"              # Partition to submit to
TIMEOUT="48:00:00"           # Timeout per job (HH:MM:SS)
NUM_GPUS=1                   # Number of GPUs per job
NUM_CPUS=8                   # Number of CPUs per job
MEMORY="40GB"                # Memory per job
JOB_NAME_PREFIX="ano"    # Prefix for job names

# Number of runs per benchmark
RUNS=5

# Common anomaly_detection.py parameters (dataset and runs are set per benchmark below)

# Benchmarks to run (from anomaly_detection.py)
BENCHMARKS=("SWaT")


# ---- Initialize conda ----
source $(conda info --base)/etc/profile.d/conda.sh

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

# Log directory for SLURM output
LOG_DIR="slurm_logs_benchmark"
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
    echo "Submitting jobs for benchmark: ${BENCHMARK}"

    sbatch \
        --partition="${PARTITION}" \
        --time="${TIMEOUT}" \
        --gpus="${NUM_GPUS}" \
        --cpus-per-task="${NUM_CPUS}" \
        --mem="${MEMORY}" \
        --job-name="${JOB_NAME_PREFIX}_Sn_${BENCHMARK}" \
        --output="${LOG_DIR}/${BENCHMARK}_Sn_%j.log" \
        --error="${LOG_DIR}/${BENCHMARK}_Sn_%j.log" \
        --wrap="python anomaly_detection.py \
            --dataset ${BENCHMARK} \
            --runs ${RUNS} \
            --sphere-embedding"

    sleep 0.5

#    sbatch \
#        --partition="${PARTITION}" \
#        --time="${TIMEOUT}" \
#        --gpus="${NUM_GPUS}" \
#        --cpus-per-task="${NUM_CPUS}" \
#        --mem="${MEMORY}" \
#        --job-name="${JOB_NAME_PREFIX}_Rn_${BENCHMARK}" \
#        --output="${LOG_DIR}/${BENCHMARK}_Rn_%j.log" \
#        --error="${LOG_DIR}/${BENCHMARK}_Rn_%j.log" \
#        --wrap="python anomaly_detection.py \
#            --dataset ${BENCHMARK} \
#	    --subsample 0.5 \
#            --runs ${RUNS} \
#            --no-sphere-embedding"

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

