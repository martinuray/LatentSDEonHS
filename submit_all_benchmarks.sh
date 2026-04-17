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
PARTITION="rtx2080ti"              # Partition to submit to
TIMEOUT="48:00:00"           # Timeout per job (HH:MM:SS)
NUM_GPUS=1                   # Number of GPUs per job
NUM_CPUS=8                   # Number of CPUs per job
MEMORY="40GB"                # Memory per job
JOB_NAME_PREFIX="anomaly"    # Prefix for job names

# Number of runs per benchmark
RUNS=5

# Common anomaly_detection.py parameters (dataset and runs are set per benchmark below)
COMMON_ARGS="\
--data-dir data_dir \
--enable-file-logging \
--log-dir logs \
--enable-checkpointing \
--checkpoint-dir checkpoints \
--checkpoint-at 1 2 3 4 5 6 90 150 190 210 390 590 990 1350 2100 \
--final-metrics-csv logs/final_metrics.csv \
--data-window-length 100 \
--data-window-overlap 0.0 \
--batch-size 512 \
--lr 0.001 \
--n-epochs 211 \
--kl0-weight 0.0001 \
--klp-weight 100.0 \
--pxz-weight 10.0 \
--seed -1 \
--restart 30 \
--device cuda \
--z-dim 16 \
--h-dim 512 \
--n-deg 12 \
--no-learnable-prior \
--freeze-sigma \
--initial-sigma 0.15 \
--mc-eval-samples 1 \
--mc-train-samples 1 \
--num-max-cpu-worker 10 \
--eval-every-n-epochs 30 \
--loglevel debug \
--no-use-atanh \
--no-debug \
--subsample 0.2 \
--no-normalize-score \
--data-normalization-strategy none \
--dec-hidden-dim 512 \
--n-dec-layers 2 \
--early-stopping-patience 10 \
--early-stopping-min-delta 0 \
--non-linear-decoder \
--delete-processed-data \
--fixed-subsample-mask"

# Benchmarks to run (from anomaly_detection.py)
BENCHMARKS=("SWaT" "WaDi" "SMD" "QAD" "MSL" "SMAP" "PSM")


# ---- Load modules ----
module load anaconda

# ---- Initialize conda ----
source $(conda info --base)/etc/profile.d/conda.sh

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

# Log directory for SLURM output
LOG_DIR="slurm_logs"
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
        --wrap="python anomaly_detection.py \
            --dataset ${BENCHMARK} \
            --runs ${RUNS} \
            ${COMMON_ARGS}"


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

