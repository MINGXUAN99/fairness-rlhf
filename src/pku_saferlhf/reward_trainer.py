"""Scalar reward training with the Bradley–Terry pairwise loss."""

import numpy as np
import torch
from transformers import Trainer


class PairCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, rows):
        # All chosen sequences first, then all rejected sequences.
        sequences = [{"input_ids": row[f"{side}_ids"]}
                     for side in ("chosen", "rejected") for row in rows]
        # Put pair-sized labels first so Trainer counts pairs, not 2B sequences.
        return {"labels": torch.zeros(len(rows), dtype=torch.long),
                **dict(self.tokenizer.pad(sequences, padding=True, return_tensors="pt"))}


def pairwise_accuracy(prediction):
    scores = prediction.predictions
    margin = scores[:, 0] - scores[:, 1]
    return {"accuracy": float(np.mean(margin > 0)),
            "tie_rate": float(np.mean(margin == 0)),
            "mean_margin": float(np.mean(margin))}


class PairwiseRewardTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        scores = model(**{k: v for k, v in inputs.items() if k != "labels"}).logits.float().reshape(-1)
        chosen, rejected = scores.chunk(2)
        loss = -torch.nn.functional.logsigmoid(chosen - rejected).mean()
        outputs = {"logits": torch.stack((chosen, rejected), dim=1)}
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
        if prediction_loss_only:
            return loss.detach(), None, None
        scores = outputs["logits"].detach()
        # Dummy targets enable Trainer's metric collection; ordering supplies labels.
        labels = torch.zeros(scores.shape[0], dtype=torch.long, device=scores.device)
        return loss.detach(), scores, labels
