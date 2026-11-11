"""Apply the chosen |z| threshold and write the final masked rationale file.

The word list comes from the train split (see sweep_threshold.py) and is applied
to train, validation and test alike. Nothing is re-ranked here.

    python -m leakage_control.lock_masking --task wic --threshold 3.0
    python -m leakage_control.lock_masking --task ps  --threshold 3.0

Writes data/<task>/rationales_masked_locked.json, and next to the sweep output a
masking_manifest.json listing the masked words and the word counts per entry.
"""

import argparse
import json
import os

from leakage_control.leakage_utils import build_mask_regex, mask_text_with_regex


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["wic", "ps"], required=True)
    parser.add_argument("--threshold", type=float, default=3.0)
    parser.add_argument("--input", default=None, help="default: data/<task>/rationales.json")
    parser.add_argument("--ranked_words", default=None,
                        help="default: data/<task>/leakage/ranked_words.json")
    parser.add_argument("--output", default=None,
                        help="default: data/<task>/rationales_masked_locked.json")
    parser.add_argument("--manifest", default=None,
                        help="default: data/<task>/leakage/masking_manifest.json")
    args = parser.parse_args()

    in_path = args.input or f"data/{args.task}/rationales.json"
    ranked_path = args.ranked_words or f"data/{args.task}/leakage/ranked_words.json"
    out_path = args.output or f"data/{args.task}/rationales_masked_locked.json"
    manifest_path = args.manifest or f"data/{args.task}/leakage/masking_manifest.json"

    with open(ranked_path) as f:
        ranked = json.load(f)
    words = [e["word"] for e in ranked if abs(e["z"]) > args.threshold]
    print(f"[{args.task}] |z| > {args.threshold}: masking {len(words)} words")
    print("  " + ", ".join(sorted(words)))
    regex = build_mask_regex(words)

    with open(in_path) as f:
        data = json.load(f)

    masked = {}
    manifest = {"task": args.task, "threshold": args.threshold,
                "n_words_masked": len(words), "words_masked": sorted(words), "per_entry": {}}
    words_before = words_after = 0
    emptied = []

    for key, entry in data.items():
        rationale = entry.get("rationale")
        if not rationale:
            masked[key] = entry
            continue
        new_rationale = mask_text_with_regex(rationale, regex)
        before, after = len(rationale.split()), len(new_rationale.split())
        words_before += before
        words_after += after
        masked[key] = {**entry, "rationale": new_rationale}
        manifest["per_entry"][key] = {"orig_words": before, "surviving_words": after}
        if after == 0:
            emptied.append(key)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(masked, f, indent=2)
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nkept {words_after}/{words_before} words ({100 * words_after / max(words_before, 1):.2f}%)")
    print(f"rationales emptied by masking: {len(emptied)}")
    print(f"saved {out_path} and {manifest_path}")


if __name__ == "__main__":
    main()
