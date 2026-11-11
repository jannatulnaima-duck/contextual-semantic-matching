"""Train the fusion head on top of the frozen rationale and context encoders.

The rationale tower is the DeBERTa from train_rationale_only.py and the context
tower is the ELECTRA from train_encoder.py (--input_mode context). Checkpoints
can be local folders or Hub ids.

    python -m fusion.train_fusion --task wic \
        --rationale_checkpoint checkpoints/wic_deberta-v3-large_rationale/best_model \
        --context_checkpoint checkpoints/wic_electra-large-discriminator_context/best_model
"""

import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse

import evaluate
import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer, EarlyStoppingCallback, Trainer, TrainingArguments, set_seed,
)

from common.predictions import save_predictions
from common.tasks import (
    SPLITS, TASKS, TGT_CLOSE, TGT_OPEN, default_rationale_file, example_id, load_rationales,
)
from fusion.fusion_model import ConcatFusionModel


class FusionDataset(Dataset):
    """Each example has a rationale encoding (rationale tower) and a context
    encoding (context tower). Examples without a rationale are skipped."""

    def __init__(self, task, hf_split, rat_tokenizer, ctx_tokenizer, rationales, split,
                 rat_max_length=128, ctx_max_length=512):
        self.rat_tokenizer = rat_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.rat_max_length = rat_max_length
        self.ctx_max_length = ctx_max_length
        self.examples = []

        for idx, example in enumerate(hf_split):
            key = example_id(task, split, idx)
            rationale = rationales.get(key)
            if not rationale:
                continue
            s1, s2 = task.marked_inputs(example)
            self.examples.append({"id": key, "rationale": rationale, "s1": s1, "s2": s2,
                                  "label": int(example["label"])})

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        rat = self.rat_tokenizer(ex["rationale"], max_length=self.rat_max_length,
                                 padding="max_length", truncation=True, return_tensors="pt")
        ctx = self.ctx_tokenizer(ex["s1"], ex["s2"], max_length=self.ctx_max_length,
                                 padding="max_length", truncation=True, return_tensors="pt")
        item = {
            "rat_input_ids": rat["input_ids"].squeeze(0),
            "rat_attention_mask": rat["attention_mask"].squeeze(0),
            "ctx_input_ids": ctx["input_ids"].squeeze(0),
            "ctx_attention_mask": ctx["attention_mask"].squeeze(0),
            "labels": torch.tensor(ex["label"], dtype=torch.long),
        }
        if "token_type_ids" in ctx:
            item["ctx_token_type_ids"] = ctx["token_type_ids"].squeeze(0)
        return item


class FusionCollator:
    def __call__(self, features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}


accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    return accuracy.compute(predictions=np.argmax(logits, axis=-1), references=labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=TASKS, required=True)
    p.add_argument("--rationale_checkpoint", required=True,
                   help="rationale-only DeBERTa (local folder or Hub id)")
    p.add_argument("--context_checkpoint", required=True,
                   help="context-only ELECTRA (local folder or Hub id)")
    p.add_argument("--rationale_file", default=None,
                   help="default: data/<task>/rationales_masked_locked.json")
    p.add_argument("--rat_max_length", type=int, default=128)
    p.add_argument("--ctx_max_length", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output_dir", default="checkpoints")
    p.add_argument("--pred_dir", default="results/predictions")
    args = p.parse_args()

    set_seed(args.seed)
    task = TASKS[args.task]
    run_name = f"{args.task}_fusion"

    rationales = load_rationales(args.rationale_file or default_rationale_file(args.task))

    rat_tokenizer = AutoTokenizer.from_pretrained(args.rationale_checkpoint)
    ctx_tokenizer = AutoTokenizer.from_pretrained(args.context_checkpoint)
    ctx_tokenizer.add_special_tokens({"additional_special_tokens": [TGT_OPEN, TGT_CLOSE]})

    raw = load_dataset(task.dataset_name)
    train_ds, val_ds, test_ds = (
        FusionDataset(task, raw[s], rat_tokenizer, ctx_tokenizer, rationales, s,
                      args.rat_max_length, args.ctx_max_length)
        for s in SPLITS
    )
    print(f"train {len(train_ds)} | validation {len(val_ds)} | test {len(test_ds)}")

    model = ConcatFusionModel(args.rationale_checkpoint, args.context_checkpoint, num_labels=2)
    model.context_encoder.resize_token_embeddings(len(ctx_tokenizer))

    training_args = TrainingArguments(
        output_dir=os.path.join(args.output_dir, run_name),
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        fp16=True,
        seed=args.seed,
        report_to="none",
        logging_steps=50,
        remove_unused_columns=False,   # the Trainer would drop our custom fields otherwise
    )
    trainer = Trainer(
        model=model, args=training_args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=compute_metrics, data_collator=FusionCollator(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()

    val_acc = trainer.evaluate(val_ds)["eval_accuracy"]
    test_out = trainer.predict(test_ds)
    preds = np.argmax(test_out.predictions, axis=-1)
    test_acc = float((preds == test_out.label_ids).mean())
    print(f"[{args.task}] fusion")
    print(f"validation accuracy {val_acc:.4f} | test accuracy {test_acc:.4f}")

    pred_path = os.path.join(args.pred_dir, f"{run_name}.json")
    save_predictions(pred_path, [ex["id"] for ex in test_ds.examples], preds, test_out.label_ids)
    print(f"test predictions -> {pred_path}")

    best_path = os.path.join(args.output_dir, run_name, "best_model")
    trainer.save_model(best_path)
    print(f"best model saved to {best_path}")


if __name__ == "__main__":
    main()
