#!/bin/bash

# Common parameters for all runs
COMMON_ARGS="--enable-file-logging --log-dir logs \
--enable-checkpointing --checkpoint-dir checkpoints \
--checkpoint-at 1 2 3 4 5 6 90 150 190 210 \
--lr 0.001 --n-epochs 211 \
--kl0-weight .0001 --klp-weight 100 --pxz-weight 10 \
--seed -1 --restart 30 --device cuda:2 \
--z-dim 16 --h-dim 512 --n-deg 12 --n-dec-layers 2 --dec-hidden-dim 512 \
--no-learnable-prior --freeze-sigma \
--mc-eval-samples 1 --subsample 0.2 --mc-train-samples 1 \
--loglevel debug --no-use-atanh --batch-size 512 \
--data-window-overlap 0.0 --data-window-length 100 \
--initial-sigma 0.15 --data-normalization-strategy none"

# Array of datasets
#DATASETS=(SMAP MSL QAD SWaT WaDi PSM SMD)
DATASETS=(SWaT WaDi PSM)

# Track failed runs
FAILED_DATASETS=()

# Run for each dataset
for dataset in "${DATASETS[@]}"; do
  echo "Running with dataset: $dataset"
  if ! python basic_data_anomaly_detection.py $COMMON_ARGS --dataset "$dataset"; then
    echo "ERROR: Dataset $dataset failed!"
    FAILED_DATASETS+=("$dataset")
  else
    echo "SUCCESS: Dataset $dataset completed"
  fi
  echo "---"
done

# Report results
echo ""
echo "========== SUMMARY =========="
if [ ${#FAILED_DATASETS[@]} -eq 0 ]; then
  echo "All datasets completed successfully!"
else
  echo "Failed datasets: ${FAILED_DATASETS[*]}"
  exit 1
fi
