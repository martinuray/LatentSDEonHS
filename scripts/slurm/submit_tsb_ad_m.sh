#!/bin/bash

################################################################################
# SLURM Script: Submit TSB-AD-M experiments — one job per (dataset, seed)
#
# Submits 200 datasets × 5 seeds = 1000 independent SLURM jobs.
# Each job trains and evaluates on a single TSB-AD-M sub-dataset with one seed.
#
# Usage: ./scripts/slurm/submit_tsb_ad_m.sh [dataset_ids...]
#   dataset_ids  optional subset of 1-based file indices (default: 1..200)
#
# Examples:
#   ./scripts/slurm/submit_tsb_ad_m.sh          # all 200 datasets
#   ./scripts/slurm/submit_tsb_ad_m.sh 1 2 3    # only datasets 1, 2, 3
################################################################################

set -euo pipefail

# ---- SLURM configuration ----
PARTITION="rtx2080ti"
TIMEOUT="48:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="40GB"
JOB_NAME_PREFIX="tsb"

# ---- Experiment configuration ----
SEEDS=(1 2 3 4 5)
DATA_DIR="data_dir"
PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
LOG_DIR="${PROJECT_DIR}/slurm_logs_tsb_ad_m"

# ---- Dataset indices ----
if [[ $# -gt 0 ]]; then
    DATASET_IDS=("$@")
else
    DATASET_IDS=($(seq 1 200))
fi

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent
cd "${PROJECT_DIR}"

mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting TSB-AD-M jobs"
echo "=================================="
echo "Datasets  : ${#DATASET_IDS[@]}"
echo "Seeds     : ${SEEDS[*]}"
echo "Variants  : Sn (sphere) + Rn (no-sphere)"
echo "Total jobs: $(( ${#DATASET_IDS[@]} * ${#SEEDS[@]} * 2 ))"
echo "Partition : ${PARTITION}"
echo "Timeout   : ${TIMEOUT}"
echo "GPUs/job  : ${NUM_GPUS}"
echo "CPUs/job  : ${NUM_CPUS}"
echo "Memory    : ${MEMORY}"
echo "Log dir   : ${LOG_DIR}"
echo "=================================="
echo ""

JOB_IDS=()

for DS_ID in "${DATASET_IDS[@]}"; do
    for SEED in "${SEEDS[@]}"; do
        for VARIANT in "Sn" "Rn"; do
            if [[ "${VARIANT}" == "Sn" ]]; then
                SPHERE_FLAG="--sphere-embedding"
            else
                SPHERE_FLAG="--no-sphere-embedding"
            fi

            JOB_NAME="${JOB_NAME_PREFIX}_${VARIANT}_ds${DS_ID}_seed${SEED}"

            JOB_ID=$(sbatch \
                --partition="${PARTITION}" \
                --time="${TIMEOUT}" \
                --gres="gpu:${NUM_GPUS}" \
                --cpus-per-task="${NUM_CPUS}" \
                --mem="${MEMORY}" \
                --job-name="${JOB_NAME}" \
                --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
                --error="${LOG_DIR}/${JOB_NAME}_%j.log" \
                --wrap="CUDA_LAUNCH_BLOCKING=1 python anomaly_detection.py \
                    --dataset TSB-AD-M \
                    --config-file ${PROJECT_DIR}/cfg/anomaly_detection/TSB-AD-M.json \
                    --trace-ids ${DS_ID} \
                    --seed ${SEED} \
                    --data-dir ${DATA_DIR} \
                    ${SPHERE_FLAG} \
                    --delete-processed-data" \
                --parsable)

            if [[ -z "${JOB_ID}" ]]; then
                echo "Error: failed to submit job for dataset ${DS_ID} seed ${SEED} variant ${VARIANT}" >&2
                exit 1
            fi

            JOB_IDS+=("${JOB_ID}")
            echo "  dataset=${DS_ID}  seed=${SEED}  variant=${VARIANT}  → job ${JOB_ID}"
            sleep 0.1
            exit 1
        done
    done
done

echo ""
echo "=================================="
echo "Submitted ${#JOB_IDS[@]} jobs."
echo "=================================="
echo ""
echo "Monitor:  squeue -u \$USER"
echo "Logs:     ${LOG_DIR}"
echo ""
