#!/bin/bash

################################################################################
# SLURM Script: Submit the NeuralODE counterpart of the LSD sparsity sweep
#
# Runs anomaly_detection_ode.py at the sparsity levels the LSD sweep reports
# (1% and 5% of the training data kept), each with a *fixed* subsampling mask
# (--fixed-subsample-mask: the mask is drawn once at dataset load time instead
# of being resampled every iteration), one job per (subsample, seed).
#
# Unlike submit_eval_sparsity_lsd.sh there is no driver script and no
# aggregation job: eval_sparsity_data.py builds on anomaly_detection.py (the
# SDE model) and has no ODE mode, so each level/seed is a plain
# anomaly_detection_ode.py run. Results are read back from W&B -- project
# latent-sde-on-hs-anomaly-detection-ode, filtered on
# args.fixed_subsample_mask and args.subsample, the same way
# baselines/wandb_results_to_latex.py picks up our LSD sparsity runs -- and
# from logs/final_metrics.csv.
#
# NOTE (QAD): anomaly_detection_ode.py has no --data-decimation-factor flag and
# cfg/anomaly_detection_ode/ does not exist, so these runs use the parser
# defaults: raw 100 Hz with 100-step non-overlapping windows. The LSD sparsity
# runs use cfg/anomaly_detection/QAD.json (decimation 10, window 200,
# overlap 0.9), so the two are not sample-rate-aligned. Put any settings needed
# to align them in EXTRA_FLAGS once the ODE side supports them.
#
# Usage:
#   ./scripts/slurm/submit_eval_sparsity_ode.sh [DATASET]
# Examples:
#   ./scripts/slurm/submit_eval_sparsity_ode.sh                 # QAD
#   ./scripts/slurm/submit_eval_sparsity_ode.sh PSM
#   SUBSAMPLES="0.01" SEEDS="42 43" ./scripts/slurm/submit_eval_sparsity_ode.sh
################################################################################

set -euo pipefail

# ---- SLURM configuration ----
PARTITION="${PARTITION:-rtx2080ti}"
TIMEOUT="${TIMEOUT:-24:00:00}"   # one (seed, subsample) run only; far less data than a full run
NUM_GPUS=1
NUM_CPUS=8
MEMORY="40GB"
JOB_NAME_PREFIX="sparsity_ode"

# ---- Experiment configuration ----
DATASET="${1:-QAD}"
# The levels reported in the sparsity table (SPARSITY_SUBSAMPLES in
# baselines/wandb_results_to_latex.py), as fractions of the training data kept.
read -r -a SUBSAMPLES <<< "${SUBSAMPLES:-0.01 0.05}"
# Five seeds, matching the baselines' sparsity sweep and the per-cell run
# budget of the sparsity table (DEFAULT_RUN_LIMIT).
read -r -a SEEDS <<< "${SEEDS:-42 43 44 45 46}"
RUNS=1
# Appended verbatim to every job's command line (e.g. "--n-epochs 500").
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

PROJECT_DIR="/home2/muray/Code/LatentSDEonHS"
LOG_DIR="${PROJECT_DIR}/slurm_logs_sparsity_ode"

# ---- Initialize conda ----
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate baseline-latent

cd "${PROJECT_DIR}"

mkdir -p "${LOG_DIR}"

echo "=================================="
echo "Submitting NeuralODE sparsity jobs"
echo "Dataset        : ${DATASET}"
echo "Subsamples     : ${SUBSAMPLES[*]}"
echo "Seeds          : ${SEEDS[*]}"
echo "Runs/job       : ${RUNS}"
echo "Jobs           : $(( ${#SUBSAMPLES[@]} * ${#SEEDS[@]} ))"
echo "Partition      : ${PARTITION}"
echo "Timeout        : ${TIMEOUT}"
echo "Extra flags    : ${EXTRA_FLAGS:-<none>}"
echo "Log directory  : ${LOG_DIR}"
echo "=================================="
echo ""

submit_sparsity_run() {
  local subsample="$1"
  local seed="$2"
  # 0.01 -> "1pct", 0.05 -> "5pct"; job/log names can't carry a dot sensibly.
  local level
  level="$(awk -v s="${subsample}" 'BEGIN { printf "%gpct", s * 100 }')"

  sbatch \
    --partition="${PARTITION}" \
    --time="${TIMEOUT}" \
    --gpus="${NUM_GPUS}" \
    --cpus-per-task="${NUM_CPUS}" \
    --mem="${MEMORY}" \
    --job-name="${JOB_NAME_PREFIX}_${DATASET}_${level}_s${seed}" \
    --output="${LOG_DIR}/${DATASET}_${level}_s${seed}_%j.log" \
    --error="${LOG_DIR}/${DATASET}_${level}_s${seed}_%j.log" \
    --wrap="python anomaly_detection_ode.py \
      --dataset ${DATASET} \
      --runs ${RUNS} \
      --seed ${seed} \
      --subsample ${subsample} \
      --fixed-subsample-mask \
      --enable-file-logging \
      --log-dir logs \
      --data-dir data_dir \
      --device cuda \
      --delete-processed-data \
      ${EXTRA_FLAGS} \
      --wandb-tags sparsity fixed-mask ${level}"

  sleep 0.2
}

for subsample in "${SUBSAMPLES[@]}"; do
  for seed in "${SEEDS[@]}"; do
    submit_sparsity_run "${subsample}" "${seed}"
  done
done

echo ""
echo "=================================="
echo "All NeuralODE sparsity jobs submitted."
echo "=================================="
echo "Monitor with: squeue -u \$USER"
echo "Logs in: ${LOG_DIR}"