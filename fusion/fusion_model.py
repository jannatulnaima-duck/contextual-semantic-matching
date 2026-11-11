"""Feature-level fusion of the rationale encoder and the context encoder.

Both towers are loaded from their fine-tuned classification checkpoints with the
classification head dropped (a local folder or a Hub id) and stay frozen. The
final-layer [CLS] vectors are concatenated and passed through an MLP with 512
hidden units, ReLU and dropout, followed by a linear classification layer. Only
this head is trained.

The model returns a dict with "logits" (and "loss" when labels are given), so it
works with transformers.Trainer.
"""

import torch
import torch.nn as nn
from transformers import AutoModel


def load_encoder(checkpoint: str):
    return AutoModel.from_pretrained(checkpoint)


def pooled_cls(hidden_states: torch.Tensor) -> torch.Tensor:
    return hidden_states[:, 0, :]


class ConcatFusionModel(nn.Module):
    def __init__(self, rationale_checkpoint: str, context_checkpoint: str,
                 num_labels: int = 2, dropout: float = 0.1):
        super().__init__()
        self.rationale_encoder = load_encoder(rationale_checkpoint)
        self.context_encoder = load_encoder(context_checkpoint)

        for encoder in (self.rationale_encoder, self.context_encoder):
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

        h_rat = self.rationale_encoder.config.hidden_size
        h_ctx = self.context_encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(h_rat + h_ctx, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, num_labels),
        )

    def train(self, mode: bool = True):
        # Trainer calls model.train(); the frozen encoders must stay in eval mode
        # so that dropout is not active inside them.
        super().train(mode)
        self.rationale_encoder.eval()
        self.context_encoder.eval()
        return self

    def forward(self, rat_input_ids, rat_attention_mask,
                ctx_input_ids, ctx_attention_mask,
                ctx_token_type_ids=None, labels=None):
        # no autograd through the frozen towers, so no activations are kept
        with torch.no_grad():
            rat_out = self.rationale_encoder(input_ids=rat_input_ids,
                                             attention_mask=rat_attention_mask)
            ctx_kwargs = dict(input_ids=ctx_input_ids, attention_mask=ctx_attention_mask)
            if ctx_token_type_ids is not None:
                ctx_kwargs["token_type_ids"] = ctx_token_type_ids
            ctx_out = self.context_encoder(**ctx_kwargs)

        fused = torch.cat([pooled_cls(rat_out.last_hidden_state),
                           pooled_cls(ctx_out.last_hidden_state)], dim=-1)
        logits = self.classifier(fused)

        result = {"logits": logits}
        if labels is not None:
            result["loss"] = nn.functional.cross_entropy(logits, labels)
        return result
