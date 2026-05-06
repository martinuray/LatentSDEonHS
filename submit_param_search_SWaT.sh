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
PARTITION="rtx2080ti"              # Partition to submit to
TIMEOUT="48:00:00"           # Timeout per job (HH:MM:SS)
NUM_GPUS=1                   # Number of GPUs per job
NUM_CPUS=8                   # Number of CPUs per job
MEMORY="40GB"                # Memory per job
JOB_NAME_PREFIX="swat_gs_sig_norm"    # Prefix for job names

# Dataset
BENCHMARK="SWaT"

# Number of runs per configuration
RUNS=1

# lr0.01_kl00.001_klp100.0_pxz10.0_z8_h256_deg6_64622
# Parameter grid (add / remove values as needed)
LR_VALUES=(0.001)
KL0_WEIGHT_VALUES=(0.001)
KLP_WEIGHT_VALUES=(1.0)
PXZ_WEIGHT_VALUES=(1.0)
Z_DIM_VALUES=(16 8)
Z_DIM_VALUES=(16)
H_DIM_VALUES=(32, 64)
H_DIM_VALUES=(32)
N_DEG_VALUES=(8 12 16)
N_DEG_VALUES=(16)
SIG_VALUES=(0.1)
EVAL_SAMPLES=(1 25)

# Common anomaly_detection.py parameters (grid-searched params are set in the loop)
BASE_ARGS="\
--data-dir data_dir \
--enable-file-logging \
--log-dir logs \
--final-metrics-csv logs/final_metrics.csv \
--data-window-length 100 \
--data-window-overlap 0.0 \
--batch-size 128 \
--n-epochs 2000 \
--seed 1 \
--restart 30 \
--device cuda \
--no-learnable-prior \
--freeze-sigma \
--mc-train-samples 1 \
--loglevel debug \
--no-use-atanh \
--no-debug \
--subsample 0.2 \
--no-normalize-score \
--data-normalization-strategy min-max \
--dec-hidden-dim 32 \
--n-dec-layers 1 \
--early-stopping-min-delta 0 \
--non-linear-decoder \
--delete-processed-data \
--no-fixed-subsample-mask"


# ---- Initialize conda ----
source $(conda info --base)/etc/profile.d/conda.sh

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd /home2/muray/Code/LatentSDEonHS

# Log directory for SLURM output
LOG_DIR="slurm_logs"
mkdir -p "${LOG_DIR}"

# Count total configurations
TOTAL=$(( ${#LR_VALUES[@]} * ${#KL0_WEIGHT_VALUES[@]} * ${#KLP_WEIGHT_VALUES[@]} * \
          ${#PXZ_WEIGHT_VALUES[@]} * ${#Z_DIM_VALUES[@]} * ${#H_DIM_VALUES[@]} * \
          ${#N_DEG_VALUES[@]} * ${#SIG_VALUES[@]} * ${#EVAL_SAMPLES[@]}))

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
for SIG in "${SIG_VALUES[@]}"; do
for EVAL_SAMPLE in "${EVAL_SAMPLES[@]}"; do

    JOB_COUNT=$(( JOB_COUNT + 1 ))
    JOB_TAG="sig_${SIG}_lr${LR}_kl0${KL0}_klp${KLP}_pxz${PXZ}_z${ZDIM}_h${HDIM}_deg${NDEG}"
    LOG_FILE="${LOG_DIR}/${JOB_TAG}_%j.log"

    # Skip if any log file already exists
    if ls "${LOG_DIR}/${JOB_TAG}_"*.log 1> /dev/null 2>&1; then
        echo "[${JOB_COUNT}/${TOTAL}] Skipping: ${JOB_TAG} (log exists)"
        #continue
    fi

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
        --wrap="python anomaly_detection.py \
            --dataset ${BENCHMARK} \
            --runs ${RUNS} \
            --lr ${LR} \
            --mc-eval-samples ${EVAL_SAMPLE} \
	          --initial-sigma ${SIG} \
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
