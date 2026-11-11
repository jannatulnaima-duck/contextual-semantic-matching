"""Dataset that feeds the context (and optionally the rationale) to an encoder."""

import torch
from torch.utils.data import Dataset

from common.tasks import Task, example_id

INPUT_MODES = ("context", "rationale", "context_rationale")


class MatchingDataset(Dataset):
    """
    input_mode:
        context    [CLS] sentence1 [SEP] sentence2 [SEP]
        rationale  [CLS] rationale [SEP]
        context_rationale
                   [CLS] sentence1 [SEP] sentence2 [SEP] rationale [SEP]

    The target word or phrase is wrapped in <TGT> tags in both sentences.
    """

    def __init__(self, task: Task, hf_split, tokenizer, split: str, rationales: dict,
                 input_mode: str, max_length: int):
        assert input_mode in INPUT_MODES, input_mode
        self.tokenizer = tokenizer
        self.input_mode = input_mode
        self.max_length = max_length
        self.examples = []

        for idx, ex in enumerate(hf_split):
            s1, s2 = task.marked_inputs(ex)
            self.examples.append({
                "id": example_id(task, split, idx),
                "s1": s1,
                "s2": s2,
                "rationale": rationales.get(example_id(task, split, idx)),
                "label": int(ex["label"]),
            })

    def __len__(self):
        return len(self.examples)

    def _texts(self, ex):
        rationale = ex["rationale"]
        if self.input_mode == "context":
            return ex["s1"], ex["s2"]
        if self.input_mode == "rationale":
            return rationale or "", None
        # context_rationale: fall back to the context if an example has no rationale
        if rationale:
            return ex["s1"], f"{ex['s2']} {self.tokenizer.sep_token} {rationale}"
        return ex["s1"], ex["s2"]

    def __getitem__(self, idx):
        ex = self.examples[idx]
        text_a, text_b = self._texts(ex)
        enc = self.tokenizer(
            text_a, text_b,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item["labels"] = torch.tensor(ex["label"], dtype=torch.long)
        return item
