#!/bin/bash

################################################################################
# Submit only the requested OCSVM baseline jobs.
#
# Each submitted job runs scripts/slurm/run.slurm with an explicit seed and run
# count forwarded to baselines/baseline.py.
#
# Usage:
#   bash scripts/slurm/submit_all_baselines_matrix.sh
################################################################################

set -euo pipefail

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
RUN_SLURM_SCRIPT="${PROJECT_DIR}/scripts/slurm/run.slurm"
LOG_ROOT="${PROJECT_DIR}/slurm_logs_baselines"

CLASSIFIER="OCSVM"
JOB_BENCHMARKS=("SWaT" "WaDi")

if [[ ! -f "${RUN_SLURM_SCRIPT}" ]]; then
  echo "Error: missing run script at ${RUN_SLURM_SCRIPT}" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}"
cd "${PROJECT_DIR}"

echo "=================================="
echo "Submitting baseline matrix jobs"
echo "Run script      : ${RUN_SLURM_SCRIPT}"
echo "Benchmarks      : ${JOB_BENCHMARKS[*]}"
echo "Classifier      : ${CLASSIFIER}"
echo "Total jobs      : 5"
echo "SWaT seeds      : 46"
echo "WaDi seeds      : 43 44 45 46"
echo "Log root        : ${LOG_ROOT}"
echo "=================================="

JOB_IDS=()

SAFE_CLASSIFIER="$(echo "${CLASSIFIER}" | tr '[:upper:]' '[:lower:]')"

submit_job() {
  local benchmark="$1"
  local seed="$2"
  local runs="$3"
  local seed_label="$4"
  local job_name="bl_${benchmark}_${SAFE_CLASSIFIER}_${seed_label}"
  local cfg_log_dir="${LOG_ROOT}/${benchmark}"

  mkdir -p "${cfg_log_dir}"

  echo "Submitting ${job_name}"
  local job_id
  job_id=$(sbatch \
    --job-name="${job_name}" \
    --output="${cfg_log_dir}/${SAFE_CLASSIFIER}_${seed_label}_%j.out" \
    --error="${cfg_log_dir}/${SAFE_CLASSIFIER}_${seed_label}_%j.err" \
    "${RUN_SLURM_SCRIPT}" "${benchmark}" "${CLASSIFIER}" "${seed}" "${runs}")

  JOB_IDS+=("${job_id}")
  echo "  -> ${job_id}"
}

submit_job "SWaT" 46 1 "seed46"
sleep 0.1
submit_job "WaDi" 43 1 "seed43"
sleep 0.1
submit_job "WaDi" 44 1 "seed44"
sleep 0.1
submit_job "WaDi" 45 1 "seed45"
sleep 0.1
submit_job "WaDi" 46 1 "seed46"

echo ""
echo "=================================="
echo "Submitted ${#JOB_IDS[@]} jobs"
echo "=================================="
echo "Monitor with: squeue -u $USER"

