"""Turn the downloaded Gemini batch outputs into a single rationales file.

    python -m rationale_generation.parse_responses --task wic

The model's own Yes/No answer is only used for the accuracy printed in the
report; it is never written to the output file.
"""

import argparse
import glob
import json
import os
import re
from collections import Counter

from common.tasks import SPLITS, TASKS


def extract_rationale(text: str):
    """Return (rationale, predicted_answer). rationale is None if nothing usable."""
    rationale = None

    marked = re.search(r"^(.*?)\[END OF REASONING\]", text, re.DOTALL)
    if marked:
        rationale = marked.group(1)
    else:
        # no end marker: take whatever comes before the answer line
        parts = re.split(r"Answer\s*:", text, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2:
            rationale = parts[0]

    answer = None
    found = re.search(r"Answer:\s*(Yes|No)", text, re.IGNORECASE)
    if found:
        answer = found.group(1).capitalize()

    if rationale is None:
        return None, answer

    rationale = re.sub(r"^\s*Reasoning:\s*", "", rationale, flags=re.IGNORECASE)
    lines = [ln for ln in rationale.strip().splitlines()
             if not re.match(r"^\s*answer\s*:", ln, re.IGNORECASE)]
    rationale = " ".join(lines).strip()
    return (rationale or None), answer


def read_response_text(record: dict):
    try:
        parts = record["response"]["candidates"][0]["content"]["parts"]
        return " ".join(p.get("text", "") for p in parts).strip() or None
    except (KeyError, IndexError, TypeError):
        return None


def parse_metadata(raw) -> dict:
    """Vertex sometimes returns the metadata as a JSON string."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass
    return {}


def parse_file(path: str):
    examples = []
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as err:
                print(f"  bad JSON on line {line_no} of {path}: {err}")
                continue

            meta = parse_metadata(record.get("metadata", {}))
            text = read_response_text(record)
            rationale, answer = extract_rationale(text) if text else (None, None)
            examples.append({
                "id": meta.get("id"),
                "label": meta.get("label"),
                "rationale": rationale,
                "answer": answer,
                "marked": {k: meta.get(k) for k in
                           ("sentence1_marked", "sentence2_marked", "alignment_ok")},
            })
    return examples


def skewed_words(examples, ratio_threshold: float = 3.0, min_rate: float = 0.02):
    """Words whose per-rationale rate differs by more than ratio_threshold across labels."""
    counts = {0: Counter(), 1: Counter()}
    totals = {0: 0, 1: 0}
    for e in examples:
        if e["label"] not in (0, 1):
            continue
        totals[e["label"]] += 1
        for w in re.findall(r"[a-z]+", e["rationale"].lower()):
            counts[e["label"]][w] += 1
    if not totals[0] or not totals[1]:
        return []

    flagged = []
    for w in set(counts[0]) | set(counts[1]):
        f0 = counts[0][w] / totals[0]
        f1 = counts[1][w] / totals[1]
        if f0 < min_rate and f1 < min_rate:
            continue
        ratio = max(f0, f1) / (min(f0, f1) + 1e-9)
        if ratio > ratio_threshold:
            flagged.append((ratio, w, f0, f1))
    return sorted(flagged, reverse=True)


def report(examples, split):
    usable = [e for e in examples if e["rationale"]]
    print(f"\n[{split}] {len(usable)}/{len(examples)} usable rationales")

    graded = [e for e in examples if e["answer"] and e["label"] is not None]
    if graded:
        hits = sum(e["answer"] == ("Yes" if e["label"] == 1 else "No") for e in graded)
        print(f"  Gemini yes/no accuracy: {hits}/{len(graded)} ({100 * hits / len(graded):.1f}%)")
    if usable:
        lengths = [len(e["rationale"].split()) for e in usable]
        print(f"  rationale length: mean {sum(lengths) / len(lengths):.1f} words "
              f"(min {min(lengths)}, max {max(lengths)})")

    skew = skewed_words(usable)
    if skew:
        print(f"  {len(skew)} words are used much more often for one label; top 10:")
        for ratio, word, f0, f1 in skew[:10]:
            print(f"    {word:<18} {ratio:5.1f}x   label0={f0:.3f}  label1={f1:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, default="wic")
    parser.add_argument("--raw_dir", default=None,
                        help="folder with one sub-folder per split (default: output/<task>)")
    parser.add_argument("--out", default=None, help="default: data/<task>/rationales.json")
    args = parser.parse_args()
    raw_dir = args.raw_dir or os.path.join("output", args.task)
    out_path = args.out or os.path.join("data", args.task, "rationales.json")

    everything = []
    for split in SPLITS:
        files = glob.glob(os.path.join(raw_dir, split, "**", "*.jsonl"), recursive=True)
        if not files:
            print(f"no .jsonl files found in {raw_dir}/{split}")
            continue
        parsed = [e for path in files for e in parse_file(path)]
        report(parsed, split)
        everything.extend(parsed)

    if not everything:
        raise SystemExit("nothing to write")

    # The raw response is not saved: it contains the model's Yes/No answer.
    # For ps the marked sentences are kept too: leakage control uses them to
    # protect the target-phrase words from masking.
    output = {}
    for e in everything:
        entry = {
            "rationale": e["rationale"],
            "label": e["label"],
            "parse_error": e["rationale"] is None,
        }
        if args.task == "ps":
            entry.update(e["marked"])
        output[e["id"]] = entry
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    missing = sum(v["rationale"] is None for v in output.values())
    print(f"\nwrote {len(output)} entries to {out_path} ({missing} without a rationale)")


if __name__ == "__main__":
    main()
