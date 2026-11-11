"""Fine-tune one encoder on WiC or PiC-PS with the marked context, optionally
followed by the rationale in the same input (the single-encoder baselines).

    # context only
    python -m classification.train_encoder --task wic --input_mode context \
        --model_names google/electra-large-discriminator

    # context and rationale in one encoder
    python -m classification.train_encoder --task ps --input_mode context_rationale \
        --model_names microsoft/deberta-v3-large,google/electra-large-discriminator

The rationale-only encoder is trained by train_rationale_only.py.
"""

import argparse
import json
import os
from dataclasses import dataclass
from typing import Optional

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from classification.data import MatchingDataset
from common.predictions import save_predictions
from common.tasks import (
    SPLITS, TASKS, TGT_CLOSE, TGT_OPEN, default_rationale_file, load_rationales,
)

CONTEXT_MODES = ("context", "context_rationale")

# ELECTRA-large uses a lower learning rate and a longer warmup than DeBERTa.
MODEL_HP_OVERRIDES = {
    "google/electra-large-discriminator": {"learning_rate": 3e-6, "warmup_ratio": 0.2},
}


@dataclass
class Config:
    task: str = "wic"
    model_name: str = "microsoft/deberta-v3-large"
    input_mode: str = "context_rationale"
    rationale_file: Optional[str] = None   # default: data/<task>/rationales_masked_locked.json
    output_dir: str = "checkpoints"
    pred_dir: str = "results/predictions"
    max_length: int = 512
    batch_size: int = 8
    grad_accum: int = 4
    learning_rate: float = 1e-5
    num_epochs: int = 10
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    patience: int = 3
    seed: int = 42
    fp16: bool = True


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return accuracy.compute(predictions=np.argmax(logits, axis=-1), references=labels)


def run(cfg: Config):
    set_seed(cfg.seed)
    task = TASKS[cfg.task]
    model_tag = cfg.model_name.split("/")[-1]
    run_name = f"{cfg.task}_{model_tag}_{cfg.input_mode}"
    print(f"\n=== {run_name} ===")

    rationales = {}
    if cfg.input_mode != "context":
        rationale_file = cfg.rationale_file or default_rationale_file(cfg.task)
        rationales = load_rationales(rationale_file)
        print(f"loaded {len(rationales)} rationales from {rationale_file}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": [TGT_OPEN, TGT_CLOSE]})

    raw = load_dataset(task.dataset_name)
    train_ds, val_ds, test_ds = (
        MatchingDataset(task, raw[s], tokenizer, s, rationales, cfg.input_mode, cfg.max_length)
        for s in SPLITS
    )
    print(f"train {len(train_ds)} | validation {len(val_ds)} | test {len(test_ds)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name, num_labels=2, torch_dtype=torch.float32,
    )
    model.resize_token_embeddings(len(tokenizer))

    lr, warmup = cfg.learning_rate, cfg.warmup_ratio
    override = MODEL_HP_OVERRIDES.get(cfg.model_name)
    if override:
        lr = override.get("learning_rate", lr)
        warmup = override.get("warmup_ratio", warmup)
        print(f"using lr={lr}, warmup_ratio={warmup} for {cfg.model_name}")

    args = TrainingArguments(
        output_dir=os.path.join(cfg.output_dir, run_name),
        num_train_epochs=cfg.num_epochs,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=lr,
        weight_decay=cfg.weight_decay,
        warmup_ratio=warmup,
        max_grad_norm=1.0,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=cfg.fp16,
        seed=cfg.seed,
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=cfg.patience)],
    )
    trainer.train()

    best_path = os.path.join(cfg.output_dir, run_name, "best_model")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)

    val_acc = trainer.evaluate(val_ds)["eval_accuracy"]
    test_out = trainer.predict(test_ds)
    test_acc = float((np.argmax(test_out.predictions, axis=-1) == test_out.label_ids).mean())
    print(f"validation accuracy {val_acc:.4f} | test accuracy {test_acc:.4f}")
    print(f"best model saved to {best_path}")

    save_predictions(os.path.join(cfg.pred_dir, f"{run_name}.json"),
                     [ex["id"] for ex in test_ds.examples],
                     np.argmax(test_out.predictions, axis=-1), test_out.label_ids)

    return {"validation": val_acc, "test": test_acc}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="wic", choices=TASKS)
    p.add_argument("--model_names", default="microsoft/deberta-v3-large",
                   help="comma-separated Hub model names")
    p.add_argument("--input_mode", default="context_rationale", choices=CONTEXT_MODES)
    p.add_argument("--rationale_file", default=None,
                   help="default: data/<task>/rationales_masked_locked.json")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--pred_dir", default="results/predictions")
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no_fp16", action="store_true")
    args = p.parse_args()

    models = [m.strip() for m in args.model_names.split(",") if m.strip()]
    results = {}
    for name in models:
        cfg = Config(
            task=args.task, model_name=name, input_mode=args.input_mode,
            rationale_file=args.rationale_file, output_dir=args.output_dir,
            pred_dir=args.pred_dir, max_length=args.max_length, batch_size=args.batch_size,
            grad_accum=args.grad_accum, learning_rate=args.learning_rate,
            num_epochs=args.num_epochs, seed=args.seed, fp16=not args.no_fp16,
        )
        results[name] = run(cfg)

    print("\n" + "-" * 60)
    print(f"{'model':<42}{'val':>8}{'test':>8}   ({args.task}, {args.input_mode})")
    for name, r in results.items():
        print(f"{name:<42}{100 * r['validation']:>7.2f}%{100 * r['test']:>7.2f}%")

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"summary_{args.task}_{args.input_mode}.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
