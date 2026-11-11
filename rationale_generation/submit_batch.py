"""Submit the batch files to Gemini on Vertex AI and wait for the jobs.

Settings come from environment variables:
    GOOGLE_CLOUD_PROJECT   GCP project id
    GCS_BUCKET             e.g. gs://my-bucket
    GOOGLE_CLOUD_LOCATION  optional, default us-central1
    GEMINI_MODEL           optional, default from rationale_generation/config.py

Authentication uses application default credentials
(`gcloud auth application-default login`, or GOOGLE_APPLICATION_CREDENTIALS
pointing at a key file that is kept outside the repo).
"""

import argparse
import os
import time

from google import genai
from google.genai.types import CreateBatchJobConfig, HttpOptions, JobState

from common.tasks import SPLITS, TASKS
from rationale_generation import config

FINISHED = (
    JobState.JOB_STATE_SUCCEEDED,
    JobState.JOB_STATE_FAILED,
    JobState.JOB_STATE_CANCELLED,
)


def make_client():
    return genai.Client(
        vertexai=True,
        project=os.environ["GOOGLE_CLOUD_PROJECT"],
        location=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"),
        http_options=HttpOptions(api_version="v1"),
    )


def submit(client, task: str, split: str, bucket: str, model: str):
    src = f"{bucket}/{task}_rationales/{task}_{split}_batch_requests.jsonl"
    dest = f"{bucket}/{task}_rationales/output/{split}/"
    job = client.batches.create(model=model, src=src, config=CreateBatchJobConfig(dest=dest))
    print(f"{split}: submitted {job.name}\n  in:  {src}\n  out: {dest}")
    return job


def wait(client, job, poll_seconds: int = 60):
    while job.state not in FINISHED:
        time.sleep(poll_seconds)
        job = client.batches.get(name=job.name)
        print(f"  {job.name}: {job.state}")
    if job.state != JobState.JOB_STATE_SUCCEEDED:
        print(f"  {job.name} ended with {job.state}: {getattr(job, 'error', None)}")
    return job


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="wic")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--no_wait", action="store_true")
    args = parser.parse_args()

    bucket = os.environ["GCS_BUCKET"].rstrip("/")
    model = os.environ.get("GEMINI_MODEL", config.MODEL)
    client = make_client()

    jobs = [submit(client, args.task, s, bucket, model) for s in args.splits]
    if args.no_wait:
        return
    for job in jobs:
        wait(client, job)


if __name__ == "__main__":
    main()
