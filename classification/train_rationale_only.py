"""Fine-tune an encoder on the (masked) rationale text alone.

No sentence pair and no target marking: the model only sees the rationale. This
is the rationale branch of the fusion model, and it is also a direct check of
how much a full transformer can get out of the rationale by itself.

    python -m classification.train_rationale_only --task wic
    python -m classification.train_rationale_only --task ps
"""

import argparse
import os

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
from common.tasks import SPLITS, TASKS, default_rationale_file, load_rationales

accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return accuracy.compute(predictions=np.argmax(logits, axis=-1), references=labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--model_name", default="microsoft/deberta-v3-large")
    p.add_argument("--rationale_file", default=None,
                   help="default: data/<task>/rationales_masked_locked.json")
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--pred_dir", default="results/predictions")
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--learning_rate", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--warmup_ratio", type=float, default=0.1)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    set_seed(args.seed)
    task = TASKS[args.task]
    run_name = f"{args.task}_{args.model_name.split('/')[-1]}_rationale"

    rationale_file = args.rationale_file or default_rationale_file(args.task)
    rationales = load_rationales(rationale_file)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    raw = load_dataset(task.dataset_name)
    train_ds, val_ds, test_ds = (
        MatchingDataset(task, raw[s], tokenizer, s, rationales, "rationale", args.max_length)
        for s in SPLITS
    )
    print(f"train {len(train_ds)} | validation {len(val_ds)} | test {len(test_ds)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=2, torch_dtype=torch.float32,
    )

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, run_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,
        seed=args.seed,
        report_to="none",
        logging_steps=50,
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, data_collator=default_data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    best_path = os.path.join(args.output_dir, run_name, "best_model")
    trainer.save_model(best_path)
    tokenizer.save_pretrained(best_path)

    val_acc = trainer.evaluate(val_ds)["eval_accuracy"]
    test_out = trainer.predict(test_ds)
    preds = np.argmax(test_out.predictions, axis=-1)
    test_acc = float((preds == test_out.label_ids).mean())
    print(f"[{args.task}] rationale-only {args.model_name}")
    print(f"validation accuracy {val_acc:.4f} | test accuracy {test_acc:.4f}")

    save_predictions(os.path.join(args.pred_dir, f"{run_name}.json"),
                     [ex["id"] for ex in test_ds.examples], preds, test_out.label_ids)


if __name__ == "__main__":
    main()
