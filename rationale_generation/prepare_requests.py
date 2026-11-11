"""Build Gemini batch-request files (one JSONL per split) for the Vertex AI backend.

    python -m rationale_generation.prepare_requests --task wic --out_dir batch_requests
"""

import argparse
import json
import os

from datasets import load_dataset

from common.tasks import SPLITS, TASKS
from rationale_generation import config
from rationale_generation.items import build_item

GENERATION_CONFIG = {
    "temperature": config.TEMPERATURE,
    "max_output_tokens": config.MAX_OUTPUT_TOKENS,
    "thinking_config": {"thinking_budget": config.THINKING_BUDGET},
}


def to_request(item: dict) -> dict:
    return {
        "request": {
            "contents": [{"role": "user", "parts": [{"text": item["prompt"]}]}],
            "system_instruction": {"parts": [{"text": item["system"]}]},
            "generation_config": GENERATION_CONFIG,
        },
        # Vertex echoes this back in the output file, which is how
        # parse_responses.py matches rationales to examples.
        "metadata": item["metadata"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="wic")
    parser.add_argument("--out_dir", default="batch_requests")
    args = parser.parse_args()

    task = TASKS[args.task]
    os.makedirs(args.out_dir, exist_ok=True)
    dataset = load_dataset(task.dataset_name)

    for split in SPLITS:
        path = os.path.join(args.out_dir, f"{task.name}_{split}_batch_requests.jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for idx, example in enumerate(dataset[split]):
                item = build_item(task, dict(example), split, idx)
                f.write(json.dumps(to_request(item)) + "\n")
        print(f"{split}: {len(dataset[split])} requests -> {path}")


if __name__ == "__main__":
    main()
