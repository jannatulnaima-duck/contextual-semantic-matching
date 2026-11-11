#!/bin/bash
# Train the encoders for one task. Settings follow the paper:
# DeBERTa-v3-large on rationales, ELECTRA-large on context.
#
#   bash classification/run_train.sh wic
#   bash classification/run_train.sh ps

set -e
cd "$(dirname "$0")/.."

TASK="${1:-wic}"

# rationale branch
python -m classification.train_rationale_only --task "${TASK}" \
    --model_name microsoft/deberta-v3-large \
    --max_length 128 --batch_size 16 --grad_accum 2 --learning_rate 1e-5

# context branch (ELECTRA's learning rate and warmup are set inside train_encoder.py)
python -m classification.train_encoder --task "${TASK}" \
    --model_names google/electra-large-discriminator \
    --input_mode context \
    --max_length 512 --batch_size 8 --grad_accum 4

# single-encoder baselines: context and rationale in one input
python -m classification.train_encoder --task "${TASK}" \
    --model_names microsoft/deberta-v3-large,google/electra-large-discriminator \
    --input_mode context_rationale
