#!/bin/bash
# McNemar tests of the fusion model against each branch. Needs the prediction files
# written by classification/run_train.sh and fusion/run_fusion.sh.
#
#   bash evaluation/run_evaluation.sh wic
#   bash evaluation/run_evaluation.sh ps

set -e
cd "$(dirname "$0")/.."

TASK="${1:-wic}"
P="results/predictions/${TASK}"

python -m evaluation.mcnemar_test \
    --model_a_name context   --model_a_predictions "${P}_electra-large-discriminator_context.json" \
    --model_b_name fusion    --model_b_predictions "${P}_fusion.json"

python -m evaluation.mcnemar_test \
    --model_a_name rationale --model_a_predictions "${P}_deberta-v3-large_rationale.json" \
    --model_b_name fusion    --model_b_predictions "${P}_fusion.json"
