#!/bin/bash
# Rank label-associated words on the train split, sweep thresholds, then lock the
# masking at |z| > 3.0 and write data/<task>/rationales_masked_locked.json.
#
#   bash leakage_control/run_leakage_control.sh

set -e
cd "$(dirname "$0")/.."

for TASK in wic ps; do
    python -m leakage_control.sweep_threshold --task "${TASK}"
    python -m leakage_control.lock_masking --task "${TASK}" --threshold 3.0
done
