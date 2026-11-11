"""Generate rationales one request at a time through an OpenAI-compatible endpoint.

This is the second backend. We used it for PiC-PS, with Gemini 2.5 Flash served
through Lightning AI. Model, temperature, token limit and thinking budget come
from config.py, the same settings the Vertex backend uses. It writes the same JSONL layout as the Vertex batch
output, so parse_responses.py handles both.

    export LLM_API_KEY=<key>
    python -m rationale_generation.generate_sync --task ps --out_dir output/ps

Environment:
    LLM_API_KEY    API key (LIGHTNING_API_KEY is also read)
    LLM_BASE_URL   optional, default https://lightning.ai/api/v1

The output file doubles as a checkpoint: examples already in it are skipped, so
an interrupted run can simply be started again.
"""

import argparse
import json
import os
import time

from datasets import load_dataset
from openai import OpenAI
from tqdm import tqdm

from common.tasks import SPLITS, TASKS
from rationale_generation import config
from rationale_generation.items import build_item

DEFAULT_BASE_URL = "https://lightning.ai/api/v1"

# provider-specific setting sent as extra_body: thinking disabled
MODEL_EXTRA_BODY = {
    "google/gemini-2.5-flash": {"thinking_budget": config.THINKING_BUDGET},
}

MAX_RETRIES = 3
BACKOFF_BASE = 1.5      # seconds, doubled after every failed attempt
REQUEST_TIMEOUT = 120


def complete(client, model: str, item: dict, max_tokens: int):
    """Return the response text, or None if every retry failed."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": item["system"]},
            {"role": "user", "content": item["prompt"]},
        ],
        "max_tokens": max_tokens,
        "temperature": config.TEMPERATURE,
        "timeout": REQUEST_TIMEOUT,
    }
    if model in MODEL_EXTRA_BODY:
        kwargs["extra_body"] = MODEL_EXTRA_BODY[model]

    for attempt in range(MAX_RETRIES):
        try:
            return client.chat.completions.create(**kwargs).choices[0].message.content
        except Exception as err:
            wait = BACKOFF_BASE * (2 ** attempt)
            print(f"  attempt {attempt + 1}/{MAX_RETRIES} failed: {err}. Retrying in {wait:.1f}s")
            time.sleep(wait)
    return None


def done_ids(path: str) -> set:
    ids = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    ids.add(json.loads(line)["metadata"]["id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return ids


def to_record(item: dict, text: str) -> dict:
    # same shape as a Vertex batch output line
    return {
        "metadata": item["metadata"],
        "response": {"candidates": [{"content": {"parts": [{"text": text}]}}]},
    }


def run_split(client, task, dataset, split, model, out_dir, max_tokens, sleep, limit):
    path = os.path.join(out_dir, split, "responses.jsonl")
    os.makedirs(os.path.dirname(path), exist_ok=True)

    rows = dataset[split]
    if limit:
        rows = rows.select(range(min(limit, len(rows))))
    items = [build_item(task, dict(row), split, idx) for idx, row in enumerate(rows)]

    finished = done_ids(path)
    todo = [it for it in items if it["metadata"]["id"] not in finished]
    print(f"{split}: {len(items)} examples, {len(finished)} already done")

    with open(path, "a", encoding="utf-8") as out:
        for item in tqdm(todo, desc=split):
            text = complete(client, model, item, max_tokens)
            if text is None:
                print(f"  giving up on {item['metadata']['id']}, it will be retried next run")
                continue
            out.write(json.dumps(to_record(item, text), ensure_ascii=False) + "\n")
            out.flush()
            time.sleep(sleep)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="ps")
    parser.add_argument("--model", default=f"google/{config.MODEL}")
    parser.add_argument("--splits", nargs="+", default=list(SPLITS))
    parser.add_argument("--out_dir", default=None, help="default: output/<task>")
    parser.add_argument("--max_tokens", type=int, default=config.MAX_OUTPUT_TOKENS)
    parser.add_argument("--sleep", type=float, default=0.4, help="seconds between calls")
    parser.add_argument("--limit", type=int, default=None, help="only the first N rows per split")
    args = parser.parse_args()

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("LIGHTNING_API_KEY")
    if not api_key:
        raise SystemExit("set LLM_API_KEY (or LIGHTNING_API_KEY)")

    client = OpenAI(base_url=os.environ.get("LLM_BASE_URL", DEFAULT_BASE_URL), api_key=api_key)
    task = TASKS[args.task]
    out_dir = args.out_dir or os.path.join("output", task.name)
    dataset = load_dataset(task.dataset_name)

    for split in args.splits:
        run_split(client, task, dataset, split, args.model, out_dir,
                  args.max_tokens, args.sleep, args.limit)


if __name__ == "__main__":
    main()
