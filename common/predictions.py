"""Save per-example test predictions, used by the evaluation script."""

import json
import os


def save_predictions(path: str, ids, preds, labels) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    out = {i: {"pred": int(p), "label": int(l)} for i, p, l in zip(ids, preds, labels)}
    with open(path, "w") as f:
        json.dump(out, f)
