"""Task definitions shared by generation, filtering and classification.

Two tasks are supported:
    wic  Word-in-Context, the target is one word given by its position
    ps   PiC Phrase Similarity, the target is a phrase found by string search
"""

import json
import re
from dataclasses import dataclass
from typing import Callable, Tuple

SPLITS = ("train", "validation", "test")

TGT_OPEN = "<TGT>"
TGT_CLOSE = "</TGT>"


# --- WiC marking -------------------------------------------------------------

def mark_word(sentence: str, position: int) -> str:
    """Wrap the word at `position` (0-indexed, whitespace tokens) in <TGT> tags.

    WiC_actual stores word positions rather than character offsets, and the
    sentences use spaced punctuation, so str.split() lines up with them.
    """
    tokens = sentence.split()
    if position >= len(tokens):
        return sentence
    tokens[position] = TGT_OPEN + tokens[position] + TGT_CLOSE
    return " ".join(tokens)


# --- PS marking --------------------------------------------------------------

def find_phrase_span(sentence: str, phrase: str):
    """First whole-word match, then a prefix match; case-sensitive before insensitive."""
    whole = r"(?<!\w)" + re.escape(phrase) + r"(?!\w)"
    prefix = r"(?<!\w)" + re.escape(phrase)
    for pattern in (whole, prefix):
        for flags in (0, re.IGNORECASE):
            match = re.search(pattern, sentence, flags=flags)
            if match:
                return match.span()
    return None


def mark_phrase(sentence: str, phrase: str) -> str:
    span = find_phrase_span(sentence, phrase)
    if span is None:
        return sentence
    start, end = span
    return sentence[:start] + TGT_OPEN + sentence[start:end] + TGT_CLOSE + sentence[end:]


def check_alignment(s1_marked: str, s2_marked: str) -> bool:
    """True if each sentence has one marked phrase and the rest of the text is identical."""
    if s1_marked.count(TGT_OPEN) != 1 or s2_marked.count(TGT_OPEN) != 1:
        return False
    pattern = re.escape(TGT_OPEN) + ".*?" + re.escape(TGT_CLOSE)
    return re.sub(pattern, "<PHRASE>", s1_marked) == re.sub(pattern, "<PHRASE>", s2_marked)


# --- task registry -----------------------------------------------------------

def _wic_inputs(ex) -> Tuple[str, str]:
    return mark_word(ex["sentence1"], ex["start1"]), mark_word(ex["sentence2"], ex["start2"])


def _ps_inputs(ex) -> Tuple[str, str]:
    return mark_phrase(ex["sentence1"], ex["phrase1"]), mark_phrase(ex["sentence2"], ex["phrase2"])


@dataclass(frozen=True)
class Task:
    name: str
    dataset_name: str
    id_prefix: str
    marked_inputs: Callable
    metadata_fields: Tuple[str, ...]


TASKS = {
    "wic": Task(
        name="wic",
        dataset_name="Deehan1866/WiC_actual",
        id_prefix="wic",
        marked_inputs=_wic_inputs,
        metadata_fields=("word", "pos", "sentence1", "sentence2", "start1", "start2"),
    ),
    "ps": Task(
        name="ps",
        dataset_name="Deehan1866/processed_phrase_similarity",
        id_prefix="ps",
        marked_inputs=_ps_inputs,
        metadata_fields=("phrase1", "phrase2", "sentence1", "sentence2"),
    ),
}


def example_id(task: Task, split: str, idx: int) -> str:
    return f"{task.id_prefix}-{split}-{idx}"


def default_rationale_file(task_name: str, masked: bool = True) -> str:
    """Masked rationales (after leakage control) by default, the raw ones otherwise."""
    name = "rationales_masked_locked.json" if masked else "rationales.json"
    return f"data/{task_name}/{name}"


def load_rationales(path: str) -> dict:
    """Return {example_id: rationale} for every entry that has a rationale."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {k: v["rationale"] for k, v in raw.items() if v.get("rationale")}
