"""Inference-only adapter for the unnormalized Beaver v1.0 LlamaForScore weights.

Checkpoint contract and pooling semantics verified against PKU-Alignment/safe-rlhf:
https://github.com/PKU-Alignment/safe-rlhf/tree/main/safe_rlhf/models/score_model
Uses the native Transformers Llama backbone and the checkpoint's score_head,
including its bias. Does not substitute a randomly initialized classifier head.
"""
from types import SimpleNamespace

import torch
from torch import nn
from transformers import LlamaModel, LlamaPreTrainedModel


class BeaverScoreModel(LlamaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        if config.model_type != "llama" or getattr(config, "score_dim", None) != 1:
            raise ValueError("Expected a scalar Beaver Llama score checkpoint")
        if getattr(config, "do_normalize", False):
            raise ValueError("This adapter supports only unnormalized Beaver checkpoints")
        self.model = LlamaModel(config)
        self.score_head = nn.Linear(config.hidden_size, 1, bias=getattr(config, "score_bias", True))
        self.post_init()

    def forward(self, input_ids, attention_mask, **kwargs):
        if (attention_mask.sum(1) == 0).any():
            raise ValueError("Cannot score an empty sequence")
        hidden = self.model(input_ids=input_ids, attention_mask=attention_mask,
                            use_cache=False, return_dict=True).last_hidden_state
        positions = torch.arange(input_ids.shape[1], device=input_ids.device)
        last = positions.expand_as(input_ids).masked_fill(~attention_mask.bool(), -1).max(1).values
        last_hidden = hidden[torch.arange(input_ids.shape[0], device=input_ids.device), last]
        return SimpleNamespace(end_scores=self.score_head(last_hidden).float())


def load_beaver(path, revision, dtype, device):
    model, info = BeaverScoreModel.from_pretrained(path, revision=revision,
        torch_dtype=dtype, output_loading_info=True)
    # Official checkpoints contain unused normalization statistics even though
    # do_normalize=false. Every backbone and scoring-head weight must load.
    allowed = {"normalizer.mean", "normalizer.var", "normalizer.count"}
    unexpected = set(info["unexpected_keys"]) - allowed
    if info["missing_keys"] or info.get("mismatched_keys") or unexpected or info.get("error_msgs"):
        raise ValueError(f"Incompatible Beaver weights: {info}")
    return model.to(device).eval().requires_grad_(False)


def beaver_text(prompt, response, eos_token):
    return f"BEGINNING OF CONVERSATION: USER: {prompt} ASSISTANT:{response}{eos_token or ''}"
