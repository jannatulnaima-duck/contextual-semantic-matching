"""Rank rationale words by label association and sweep the masking threshold.

For one task this script
  1. takes the train split of the rationale file,
  2. ranks words by |z| (informative-Dirichlet log-odds), leaving out target
     words and words whose class-frequency ratio is below --min_freq_ratio,
  3. for each |z| threshold, masks the words above it and measures how well a
     TF-IDF logistic-regression probe predicts the label from the masked text,
  4. writes the ranked words and the sweep to data/<task>/leakage/.

    python -m leakage_control.sweep_threshold --task wic
    python -m leakage_control.sweep_threshold --task ps
"""

import argparse
import json
import os

from leakage_control.leakage_utils import (
    build_mask_regex, extract_protected_words, leakage_probe_accuracy,
    load_split, mask_text_with_regex, rank_biased_words,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=["wic", "ps"], required=True)
    parser.add_argument("--input", default=None, help="default: data/<task>/rationales.json")
    parser.add_argument("--thresholds", default="4.0,3.5,3.0,2.5,2.0,1.96,1.5,1.0",
                        help="comma-separated |z| thresholds, high to low")
    parser.add_argument("--min_freq_ratio", type=float, default=2.5,
                        help="a word is only a masking candidate if it is at least this many "
                             "times more frequent in one class than in the other")
    parser.add_argument("--out_dir", default=None, help="default: data/<task>/leakage")
    args = parser.parse_args()

    in_path = args.input or f"data/{args.task}/rationales.json"
    out_dir = args.out_dir or f"data/{args.task}/leakage"
    os.makedirs(out_dir, exist_ok=True)

    train = load_split(in_path, "train", args.task)
    print(f"[{args.task}] {len(train)} train rationales")

    protected = extract_protected_words(train, args.task)
    print(f"{len(protected)} target words are protected from masking")

    ranked = rank_biased_words(train, protected_words=protected,
                               min_freq_ratio=args.min_freq_ratio)
    ranked_path = os.path.join(out_dir, "ranked_words.json")
    with open(ranked_path, "w") as f:
        json.dump([{"word": w, **stats} for w, stats in ranked], f, indent=2)
    print(f"{len(ranked)} candidate words -> {ranked_path}")

    print("\nTop 25 words by |z|:")
    print(f"  {'word':<20} {'z':>8} {'count0':>8} {'count1':>8}")
    for w, stats in ranked[:25]:
        print(f"  {w:<20} {stats['z']:>8.2f} {stats['count0']:>8} {stats['count1']:>8}")

    texts = [v["rationale"] for v in train.values()]
    labels = [v["label"] for v in train.values()]

    baseline = leakage_probe_accuracy(texts, labels)
    print(f"\nunmasked probe accuracy {baseline['probe_accuracy']:.4f} "
          f"(majority baseline {baseline['majority_baseline']:.4f})")

    sweep = []
    print(f"\n{'|z| >':>8} {'masked':>8} {'probe':>8} {'gap':>8}")
    for threshold in (float(t) for t in args.thresholds.split(",")):
        words = [w for w, stats in ranked if abs(stats["z"]) > threshold]
        regex = build_mask_regex(words)
        probe = leakage_probe_accuracy([mask_text_with_regex(t, regex) for t in texts], labels)
        gap = probe["probe_accuracy"] - probe["majority_baseline"]
        print(f"{threshold:>8.2f} {len(words):>8} {probe['probe_accuracy']:>8.4f} {gap:>+8.4f}")
        sweep.append({"threshold": threshold, "n_words_masked": len(words),
                      "words_masked": words, **probe})

    sweep_path = os.path.join(out_dir, "threshold_sweep.json")
    with open(sweep_path, "w") as f:
        json.dump({"task": args.task, "unmasked_probe": baseline, "sweep": sweep}, f, indent=2)
    print(f"\nsweep -> {sweep_path}")


if __name__ == "__main__":
    main()
