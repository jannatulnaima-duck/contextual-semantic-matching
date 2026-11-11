"""McNemar's test and error overlap for two models on the same test set.

    python -m evaluation.mcnemar_test \
        --model_a_name rationale --model_a_predictions results/predictions/wic_deberta-v3-large_rationale.json \
        --model_b_name fusion    --model_b_predictions results/predictions/wic_fusion.json

Each predictions file is {example_id: {"pred": 0/1, "label": 0/1}}.
"""

import argparse
import json
import numpy as np


def mcnemar_exact_or_chi2(b01: int, b10: int):
    """
    b01 = # examples model A got WRONG and model B got RIGHT
    b10 = # examples model A got RIGHT and model B got WRONG
    Only the disagreement cells matter for McNemar's test.

    Exact binomial test when there are fewer than 25 discordant pairs,
    chi-square with continuity correction otherwise. Uses statsmodels when it
    is installed and scipy's exact test as a fallback.
    """
    n = b01 + b10
    if n == 0:
        return {"n_discordant": 0, "p_value": 1.0, "method": "no disagreement"}

    try:
        from statsmodels.stats.contingency_tables import mcnemar
        table = [[0, b01], [b10, 0]]  # only off-diagonal cells matter
        exact = n < 25
        result = mcnemar(table, exact=exact, correction=not exact)
        return {"n_discordant": n, "p_value": float(result.pvalue),
                "method": "exact binomial" if exact else "chi2 (continuity corrected)"}
    except ImportError:
        pass

    # Manual fallback: exact binomial test (scipy only, no statsmodels needed)
    from scipy.stats import binomtest
    result = binomtest(min(b01, b10), n, 0.5)
    return {"n_discordant": n, "p_value": float(result.pvalue), "method": "exact binomial (scipy fallback)"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a_name", default="Model A")
    parser.add_argument("--model_a_predictions", required=True)
    parser.add_argument("--model_b_name", default="Model B")
    parser.add_argument("--model_b_predictions", required=True)
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    with open(args.model_a_predictions) as f:
        preds_a = json.load(f)
    with open(args.model_b_predictions) as f:
        preds_b = json.load(f)

    shared_ids = sorted(set(preds_a.keys()) & set(preds_b.keys()))
    if len(shared_ids) < len(preds_a) or len(shared_ids) < len(preds_b):
        print(f"WARNING: prediction files don't fully overlap. "
              f"A has {len(preds_a)}, B has {len(preds_b)}, "
              f"comparing on {len(shared_ids)} shared examples.")

    both_right, both_wrong = 0, 0
    a_right_b_wrong, a_wrong_b_right = 0, 0  # b10, b01

    a_right_ids, b_right_ids = [], []

    for ex_id in shared_ids:
        a_correct = int(preds_a[ex_id]["pred"] == preds_a[ex_id]["label"])
        b_correct = int(preds_b[ex_id]["pred"] == preds_b[ex_id]["label"])

        if a_correct:
            a_right_ids.append(ex_id)
        if b_correct:
            b_right_ids.append(ex_id)

        if a_correct and b_correct:
            both_right += 1
        elif not a_correct and not b_correct:
            both_wrong += 1
        elif a_correct and not b_correct:
            a_right_b_wrong += 1
        else:
            a_wrong_b_right += 1

    n = len(shared_ids)
    acc_a = (both_right + a_right_b_wrong) / n
    acc_b = (both_right + a_wrong_b_right) / n

    print(f"N examples compared: {n}\n")
    print(f"{args.model_a_name} accuracy: {acc_a:.4f}")
    print(f"{args.model_b_name} accuracy: {acc_b:.4f}")
    print(f"Raw accuracy difference: {acc_b - acc_a:+.4f}\n")

    print("Contingency table (McNemar):")
    print(f"                          {args.model_b_name} right   {args.model_b_name} wrong")
    print(f"  {args.model_a_name} right   {both_right:>18}   {a_right_b_wrong:>18}")
    print(f"  {args.model_a_name} wrong   {a_wrong_b_right:>18}   {both_wrong:>18}\n")

    mc = mcnemar_exact_or_chi2(b01=a_wrong_b_right, b10=a_right_b_wrong)
    print(f"McNemar's test ({mc['method']}):")
    print(f"  Discordant pairs (n where models disagree): {mc['n_discordant']}")
    print(f"  p-value: {mc['p_value']:.4f}")
    if mc["p_value"] < args.alpha:
        winner = args.model_b_name if a_wrong_b_right > a_right_b_wrong else args.model_a_name
        print(f"  --> significant at alpha={args.alpha}: {winner} is better")
    else:
        print(f"  --> not significant at alpha={args.alpha}")

    print("\n--- Error-overlap analysis (matters regardless of the p-value above) ---")
    set_a, set_b = set(a_right_ids), set(b_right_ids)
    only_a = set_a - set_b
    only_b = set_b - set_a
    union = set_a | set_b
    jaccard = len(set_a & set_b) / max(len(union), 1)

    print(f"Examples only {args.model_a_name} gets right: {len(only_a)} "
          f"({100*len(only_a)/n:.1f}% of test set)")
    print(f"Examples only {args.model_b_name} gets right: {len(only_b)} "
          f"({100*len(only_b)/n:.1f}% of test set)")
    print(f"Jaccard overlap of correct-prediction sets: {jaccard:.4f} "
          f"(1.0 = identical errors, lower = more complementary)")
    print(f"Combined 'oracle' accuracy if you could pick whichever model is right "
          f"per example: {100*len(union)/n:.2f}%")



if __name__ == "__main__":
    main()
