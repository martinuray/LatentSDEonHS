#!/bin/bash

################################################################################
# SLURM Script: Submit USAD (SWaT, WaDi, PSM) and TcnED (SWaT, WaDi) baseline
# jobs, 5 seeds each - one SLURM job per (classifier, benchmark, seed).
#
# Each submitted job runs scripts/slurm/submitt_baselines.slurm with an
# explicit seed and run count (--runs 1) forwarded to baselines/baseline.py.
#
# Usage:
#   bash scripts/slurm/submit_usad_tcned_matrix.sh
################################################################################

set -euo pipefail

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
RUN_SLURM_SCRIPT="${PROJECT_DIR}/scripts/slurm/submitt_baselines.slurm"
LOG_ROOT="${PROJECT_DIR}/slurm_logs_baselines"

PARTITION="rtx2080ti"
NUM_GPUS=1

# classifier:benchmark pairs to submit
JOB_SPECS=(
  "USAD:SWaT"
  "USAD:WaDi"
  "USAD:PSM"
  "TcnED:SWaT"
  "TcnED:WaDi"
)

SEEDS=(42 43 44 45 46)
RUNS=1   # one seed per job

if [[ ! -f "${RUN_SLURM_SCRIPT}" ]]; then
  echo "Error: missing run script at ${RUN_SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

TOTAL_JOBS=$(( ${#JOB_SPECS[@]} * ${#SEEDS[@]} ))

echo "=================================="
echo "Submitting USAD/TcnED baseline matrix jobs"
echo "Run script       : ${RUN_SLURM_SCRIPT}"
echo "Classifier/benchmark pairs:"
for spec in "${JOB_SPECS[@]}"; do
  echo "  - ${spec}"
done
echo "Seeds            : ${SEEDS[*]}"
echo "Total jobs        : ${TOTAL_JOBS}"
echo "Partition        : ${PARTITION}"
echo "GPUs per job     : ${NUM_GPUS}"
echo "Log root         : ${LOG_ROOT}"
echo "=================================="

JOB_IDS=()

submit_job() {
  local classifier="$1"
  local benchmark="$2"
  local seed="$3"
  local safe_classifier
  safe_classifier="$(echo "${classifier}" | tr '[:upper:]' '[:lower:]')"
  local job_name="bl_${benchmark}_${safe_classifier}_seed${seed}"
  local cfg_log_dir="${LOG_ROOT}/${benchmark}"

  mkdir -p "${cfg_log_dir}"

  echo "Submitting ${job_name}"
  local job_id
  job_id=$(sbatch \
    --job-name="${job_name}" \
    --partition="${PARTITION}" \
    --gres="gpu:${NUM_GPUS}" \
    --output="${cfg_log_dir}/${safe_classifier}_seed${seed}_%j.out" \
    --error="${cfg_log_dir}/${safe_classifier}_seed${seed}_%j.err" \
    "${RUN_SLURM_SCRIPT}" "${benchmark}" "${classifier}" "${seed}" "${RUNS}")

  JOB_IDS+=("${job_id}")
  echo "  -> ${job_id}"
}

for spec in "${JOB_SPECS[@]}"; do
  IFS=':' read -r classifier benchmark <<< "${spec}"
  for seed in "${SEEDS[@]}"; do
    submit_job "${classifier}" "${benchmark}" "${seed}"
    sleep 0.1
  done
done

echo ""
echo "=================================="
echo "Submitted ${#JOB_IDS[@]} jobs"
echo "=================================="
echo "Monitor with: squeue -u \$USER"
