#!/bin/bash
################################################################################
# Submit QAD traces 1-16 as 16 independent SLURM jobs (one trace per job)
# on the gtx1080ti partition.
#
# Usage:
#   chmod +x submit_qad_traces.sh
#   ./submit_qad_traces.sh
#
# After all jobs finish, aggregate results with:
#   python aggregate_qad_results.py
################################################################################

PARTITION="gtx1080ti"
TIMEOUT="48:00:00"
NUM_GPUS=1
NUM_CPUS=8
MEMORY="32GB"
JOB_NAME_PREFIX="qad"
LOG_DIR="slurm_logs_qad"
WORK_DIR="/scratch4/muray/LatentSDEonHS"

BATCH_SIZE=1024

# All 16 QAD traces — one job per trace
TRACES=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)

mkdir -p "${LOG_DIR}"

# ---- Initialize conda ----
source $(conda info --base)/etc/profile.d/conda.sh

# ---- Activate environment ----
conda activate baseline-latent

# ---- Move to project directory ----
cd "${WORK_DIR}"

echo "=================================="
echo "Submitting QAD trace jobs"
echo "Partition  : ${PARTITION}"
echo "Jobs       : ${#TRACES[@]}"
echo "Traces/job : 1"
echo "Batch size : ${BATCH_SIZE}"
echo "Log dir    : ${LOG_DIR}"
echo "=================================="
echo ""

JOB_IDS=()

for TRACE in "${TRACES[@]}"; do
    JOB_NAME="${JOB_NAME_PREFIX}_trace${TRACE}"

    echo "Submitting job for trace ${TRACE}"

    JOB_ID=$(sbatch \
        --partition="${PARTITION}" \
        --time="${TIMEOUT}" \
        --gres=gpu:"${NUM_GPUS}" \
        --cpus-per-task="${NUM_CPUS}" \
        --mem="${MEMORY}" \
        --job-name="${JOB_NAME}" \
        --output="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --error="${LOG_DIR}/${JOB_NAME}_%j.log" \
        --wrap="python anomaly_detection.py \
                --dataset QAD \
                --batch-size ${BATCH_SIZE} \
                --trace-ids ${TRACE}" \
        --parsable)

    JOB_IDS+=("${JOB_ID}")
    echo "  → SLURM job ID: ${JOB_ID}"
    sleep 0.5
done

echo ""
echo "=================================="
echo "All ${#TRACES[@]} jobs submitted!"
echo "Job IDs: ${JOB_IDS[*]}"
echo "=================================="
echo ""

# Build a colon-separated dependency string for the aggregation job
DEPS=$(IFS=:; echo "${JOB_IDS[*]}")

echo "Submitting aggregation job (depends on: ${DEPS})"

AGG_JOB_ID=$(sbatch \
    --partition="${PARTITION}" \
    --time="00:10:00" \
    --cpus-per-task=2 \
    --mem="4GB" \
    --job-name="${JOB_NAME_PREFIX}_aggregate" \
    --output="${LOG_DIR}/${JOB_NAME_PREFIX}_aggregate_%j.log" \
    --error="${LOG_DIR}/${JOB_NAME_PREFIX}_aggregate_%j.log" \
    --dependency="afterok:${DEPS}" \
    --wrap="python aggregate_qad_results.py \
              --csv logs/final_metrics.csv \
              --dataset QAD \
              --n-traces 16 \
              --out logs/qad_aggregated_results.csv" \
    --parsable)

echo "  → Aggregation job ID: ${AGG_JOB_ID}"
echo ""
echo "Monitor with:"
echo "  squeue -u \$USER"
echo "  tail -f ${LOG_DIR}/${JOB_NAME_PREFIX}_trace1_*.log"
echo ""
echo "Results will be at: logs/qad_aggregated_results.csv"
echo ""

