#!/bin/bash
# Train the fusion head on the frozen encoders (learning rate 1e-4).
# Run classification/run_train.sh for the same task first.
#
#   bash fusion/run_fusion.sh wic
#   bash fusion/run_fusion.sh ps

set -e
cd "$(dirname "$0")/.."

TASK="${1:-wic}"

python -m fusion.train_fusion --task "${TASK}" --learning_rate 1e-4 \
    --rationale_checkpoint "checkpoints/${TASK}_deberta-v3-large_rationale/best_model" \
    --context_checkpoint "checkpoints/${TASK}_electra-large-discriminator_context/best_model"
