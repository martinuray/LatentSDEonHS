#!/bin/bash
################################################################################
# Run one shard of the baselines/baseline_param_search.py grid search.
#
# baseline_param_search.py has no built-in way to select a single
# hyperparameter combination, but --classifiers and --benchmarks already
# restrict a run to one classifier's full grid on one benchmark. Since
# classifiers and benchmarks are independent of each other, that cross
# product is embarrassingly parallel and is the natural sharding unit:
#   KNN   (CPU) x {SWaT, WaDi, PSM, SMD, QAD, MSL, SMAP}
#   TcnED (GPU) x {SWaT, WaDi, PSM, SMD, QAD, MSL, SMAP}
# -> 14 independent shards.
#
# Run this script manually on each machine you have available, one call per
# shard you assign to it (see the loop example at the bottom of the file).
# Each shard gets its own --output-dir: the local CSVs baseline_param_search.py
# writes are appended to with no file locking, so two shards writing into the
# same directory (e.g. over a shared NFS home) WILL corrupt each other's
# output. W&B logging is already one run per (classifier#param_id, benchmark),
# so that's the safe place to pull all shards' results back together
# afterwards, rather than merging the local CSVs by hand.
#
# Usage:
#   ./scripts/run_param_search_shard.sh <CLASSIFIER> <BENCHMARK> [GPU_ID]
#
# Examples:
#   ./scripts/run_param_search_shard.sh KNN SWaT            # CPU-only, no GPU needed
#   ./scripts/run_param_search_shard.sh TcnED QAD 0          # pins to GPU 0
#   nohup ./scripts/run_param_search_shard.sh TcnED QAD 0 &  # keep running after logout
################################################################################

set -euo pipefail

CLASSIFIER="${1:?Usage: $0 <CLASSIFIER> <BENCHMARK> [GPU_ID]}"
BENCHMARK="${2:?Usage: $0 <CLASSIFIER> <BENCHMARK> [GPU_ID]}"
GPU_ID="${3:-}"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

# ---- Activate environment ----
source .venv/bin/activate

OUTPUT_DIR="out/param_search_${BENCHMARK}_${CLASSIFIER}"
LOG_DIR="logs_param_search"
mkdir -p "${LOG_DIR}"

GPU_FLAG=""
if [[ -n "${GPU_ID}" ]]; then
    GPU_FLAG="--gpu-id ${GPU_ID}"
fi

LOG_FILE="${LOG_DIR}/${BENCHMARK}_${CLASSIFIER}_$(date +%Y%m%d-%H%M%S).log"
echo "Running ${CLASSIFIER} on ${BENCHMARK} -> ${OUTPUT_DIR} (log: ${LOG_FILE})"

python baselines/baseline_param_search.py \
    --classifiers "${CLASSIFIER}" \
    --benchmarks "${BENCHMARK}" \
    --output-dir "${OUTPUT_DIR}" \
    --wandb-group "param-search-matrix__${BENCHMARK}__${CLASSIFIER}" \
    --wandb-tags param-search-matrix "${BENCHMARK}" "${CLASSIFIER}" \
    ${GPU_FLAG} \
    2>&1 | tee "${LOG_FILE}"

# To run several shards sequentially on one machine (e.g. all KNN benchmarks
# on a single CPU box), loop over this script instead of calling it once:
#   for BENCHMARK in SWaT WaDi PSM SMD QAD MSL SMAP; do
#       ./scripts/run_param_search_shard.sh KNN "${BENCHMARK}"
#   done