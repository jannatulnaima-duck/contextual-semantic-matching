"""Turn a dataset row into a prompt plus the metadata that travels with it."""

from common.tasks import Task, check_alignment, example_id
from rationale_generation.prompts import (
    SYSTEM_PROMPTS, build_ps_prompt, build_wic_prompt,
)


def build_item(task: Task, example: dict, split: str, idx: int) -> dict:
    s1_marked, s2_marked = task.marked_inputs(example)

    metadata = {
        "id": example_id(task, split, idx),
        "split": split,
        "idx": idx,
        "sentence1": example["sentence1"],
        "sentence2": example["sentence2"],
        "sentence1_marked": s1_marked,
        "sentence2_marked": s2_marked,
        "label": int(example["label"]),
    }

    if task.name == "wic":
        prompt = build_wic_prompt(s1_marked, s2_marked, example["word"])
        metadata.update(word=example["word"], pos=example["pos"])
    else:
        prompt = build_ps_prompt(s1_marked, s2_marked, example["phrase1"], example["phrase2"])
        metadata.update(
            phrase1=example["phrase1"],
            phrase2=example["phrase2"],
            alignment_ok=check_alignment(s1_marked, s2_marked),
        )
        if "<TGT>" not in s1_marked or "<TGT>" not in s2_marked:
            print(f"  phrase not found in {metadata['id']}")

    return {"system": SYSTEM_PROMPTS[task.name], "prompt": prompt, "metadata": metadata}
