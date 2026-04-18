#!/bin/bash

################################################################################
# SLURM Script: Parameter grid search on SWaT benchmark
#
# This script submits a separate SLURM job for each combination of the
# parameter grid below, all evaluated on the SWaT benchmark dataset.
#
# Usage: ./submit_param_search_SWaT.sh
################################################################################

# SLURM Configuration - Modify as needed
PARTITION="gpu"              # Partition to submit to
TIMEOUT="02:00:00"           # Timeout per job (HH:MM:SS)
NUM_GPUS=1                   # Number of GPUs per job
NUM_CPUS=8                   # Number of CPUs per job
MEMORY="40GB"                # Memory per job
JOB_NAME_PREFIX="swat_gs"    # Prefix for job names

# Dataset
BENCHMARK="SWaT"

# Number of runs per configuration
RUNS=1

# Parameter grid (add / remove values as needed)
LR_VALUES=(0.01 0.001 0.0001)
KL0_WEIGHT_VALUES=(0.001 0.0001 0.00001)
KLP_WEIGHT_VALUES=(10.0 100.0 1000.0)
PXZ_WEIGHT_VALUES=(1.0 10.0 100.0)
Z_DIM_VALUES=(8 16 32)
H_DIM_VALUES=(256 512)
N_DEG_VALUES=(6 12)

# Common anomaly_detection.py parameters (grid-searched params are set in the loop)
BASE_ARGS="\
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
--n-epochs 211 \
--seed -1 \
--restart 30 \
--device cuda:2 \
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
--delete-processed-data"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Log directory for SLURM output
LOG_DIR="${SCRIPT_DIR}/slurm_logs/param_search_${BENCHMARK}"
mkdir -p "${LOG_DIR}"

# Count total configurations
TOTAL=$(( ${#LR_VALUES[@]} * ${#KL0_WEIGHT_VALUES[@]} * ${#KLP_WEIGHT_VALUES[@]} * \
          ${#PXZ_WEIGHT_VALUES[@]} * ${#Z_DIM_VALUES[@]} * ${#H_DIM_VALUES[@]} * \
          ${#N_DEG_VALUES[@]} ))

echo "=================================="
echo "Parameter grid search on ${BENCHMARK}"
echo "=================================="
echo "Total configurations: ${TOTAL}"
echo "Runs per configuration: ${RUNS}"
echo "Partition: ${PARTITION}"
echo "Timeout: ${TIMEOUT}"
echo "GPUs per job: ${NUM_GPUS}"
echo "CPUs per job: ${NUM_CPUS}"
echo "Memory per job: ${MEMORY}"
echo "Log directory: ${LOG_DIR}"
echo "=================================="
echo ""

JOB_COUNT=0

for LR in "${LR_VALUES[@]}"; do
for KL0 in "${KL0_WEIGHT_VALUES[@]}"; do
for KLP in "${KLP_WEIGHT_VALUES[@]}"; do
for PXZ in "${PXZ_WEIGHT_VALUES[@]}"; do
for ZDIM in "${Z_DIM_VALUES[@]}"; do
for HDIM in "${H_DIM_VALUES[@]}"; do
for NDEG in "${N_DEG_VALUES[@]}"; do

    JOB_COUNT=$(( JOB_COUNT + 1 ))
    JOB_TAG="lr${LR}_kl0${KL0}_klp${KLP}_pxz${PXZ}_z${ZDIM}_h${HDIM}_deg${NDEG}"
    LOG_FILE="${LOG_DIR}/${JOB_TAG}_%j.log"

    echo "[${JOB_COUNT}/${TOTAL}] Submitting: ${JOB_TAG}"

    sbatch \
        --partition="${PARTITION}" \
        --time="${TIMEOUT}" \
        --gpus="${NUM_GPUS}" \
        --cpus-per-task="${NUM_CPUS}" \
        --mem="${MEMORY}" \
        --job-name="${JOB_NAME_PREFIX}_${JOB_COUNT}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --wrap="cd ${SCRIPT_DIR} && python anomaly_detection.py \
            --dataset ${BENCHMARK} \
            --runs ${RUNS} \
            --lr ${LR} \
            --kl0-weight ${KL0} \
            --klp-weight ${KLP} \
            --pxz-weight ${PXZ} \
            --z-dim ${ZDIM} \
            --h-dim ${HDIM} \
            --n-deg ${NDEG} \
            ${BASE_ARGS}"

    # Small delay to avoid overwhelming the scheduler
    sleep 0.5

done
done
done
done
done
done
done

echo ""
echo "=================================="
echo "All ${JOB_COUNT} jobs submitted!"
echo "=================================="
echo ""
echo "Monitor jobs with:"
echo "  squeue -u \$USER"
echo ""
echo "View logs in: ${LOG_DIR}"
echo ""
