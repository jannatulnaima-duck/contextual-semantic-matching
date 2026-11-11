#!/bin/bash
# Generate rationales for one task and parse them into data/<task>/rationales.json.
#
#   bash rationale_generation/run_generation.sh wic vertex   # Vertex AI batch jobs (default)
#   bash rationale_generation/run_generation.sh ps vertex
#   bash rationale_generation/run_generation.sh ps openai    # one request at a time
#
# vertex needs:  GOOGLE_CLOUD_PROJECT, GCS_BUCKET (and gcloud / gsutil logged in)
# openai needs:  LLM_API_KEY (and optionally LLM_BASE_URL)

set -e
cd "$(dirname "$0")/.."

TASK="${1:-wic}"
BACKEND="${2:-vertex}"

if [ "${BACKEND}" = "vertex" ]; then
    : "${GOOGLE_CLOUD_PROJECT:?set GOOGLE_CLOUD_PROJECT}"
    : "${GCS_BUCKET:?set GCS_BUCKET}"

    python -m rationale_generation.prepare_requests --task "${TASK}" --out_dir batch_requests
    gsutil -m cp batch_requests/${TASK}_*_batch_requests.jsonl "${GCS_BUCKET}/${TASK}_rationales/"
    python -m rationale_generation.submit_batch --task "${TASK}"

    mkdir -p "output/${TASK}"
    gsutil -m cp -r "${GCS_BUCKET}/${TASK}_rationales/output/*" "output/${TASK}/"
else
    python -m rationale_generation.generate_sync --task "${TASK}"
fi

python -m rationale_generation.parse_responses --task "${TASK}"
