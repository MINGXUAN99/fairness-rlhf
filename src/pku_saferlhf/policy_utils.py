"""PKU prompts and complete-model reward scoring for PPO."""
import json
from pathlib import Path

import torch
from torch import nn
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.helper.model_utils import configure_tokenizer
from src.pku_saferlhf.reward_utils import format_response


def prepare_prompts(rows, tokenizer, max_length, mode="auto", reject_pad_in_prompt=True):
    """Retain unique full prompts; skip overlength prompts instead of truncating."""
    records, lookup, seen = [], {}, set()
    skipped = 0
    for row in rows:
        prompt = row["prompt"]
        if prompt in seen:
            continue
        seen.add(prompt)
        chat = mode == "chat" or (mode == "auto" and bool(tokenizer.chat_template))
        if chat:
            if not tokenizer.chat_template:
                raise ValueError("Chat prompt formatting requires a chat template")
            ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                                 tokenize=True, add_generation_prompt=True)
        else:
            ids = tokenizer(f"User: {prompt}\nAssistant:", add_special_tokens=True)["input_ids"]
        if not ids or len(ids) > max_length:
            skipped += 1
            continue
        if reject_pad_in_prompt and tokenizer.pad_token_id in ids:
            raise ValueError("Prompt contains the reserved policy padding token")
        key = tuple(ids)
        if key in lookup and lookup[key] != prompt:
            raise ValueError("Different prompts tokenize identically; choose a suitable tokenizer")
        lookup[key] = prompt
        records.append({"input_ids": ids, "attention_mask": [1] * len(ids)})
    if not records:
        raise ValueError("No usable training prompts")
    return Dataset.from_list(records), lookup, {"unique_prompts": len(seen),
        "retained_prompts": len(records), "overlength_or_empty_prompts": skipped}


class RewardBundle(nn.Module):
    """Keep each frozen group's complete backbone, head, tokenizer, and format."""
    def __init__(self, paths, dtype, device, fallback_format="auto", fallback_max_length=1024):
        super().__init__()
        self.models = nn.ModuleList()
        self.tokenizers, self.specs = [], []
        self.names = list(paths)
        for name, path in paths.items():
            revision = None
            config_path = Path(path) / "run_config.json"
            if not Path(path).is_dir():
                from huggingface_hub import hf_hub_download, HfApi
                from huggingface_hub.errors import EntryNotFoundError
                revision = HfApi().model_info(path).sha
                try:
                    config_path = Path(hf_hub_download(path, "run_config.json", revision=revision))
                except EntryNotFoundError:
                    config_path = None
            spec = json.loads(config_path.read_text()) if config_path and config_path.exists() else {}
            if spec.get("group", name) != name:
                raise ValueError(f"Reward checkpoint {path} is for {spec['group']}, not {name}")
            tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
            configure_tokenizer(tokenizer, padding_side="left")
            tokenizer.truncation_side = "right"
            model = AutoModelForSequenceClassification.from_pretrained(path, revision=revision, torch_dtype=dtype)
            if model.config.num_labels != 1:
                raise ValueError(f"Reward checkpoint {path} must produce a scalar")
            model.config.pad_token_id = tokenizer.pad_token_id
            model.requires_grad_(False)
            model.eval().to(device)
            self.models.append(model)
            self.tokenizers.append(tokenizer)
            self.specs.append({"name": name, "path": path,
                "revision": revision,
                "format": spec.get("format", fallback_format),
                "max_length": spec.get("max_length", fallback_max_length)})
        self.config = self.models[0].config

    @torch.no_grad()
    def score(self, prompts, responses, batch_size):
        columns = []
        for model, tokenizer, spec in zip(self.models, self.tokenizers, self.specs):
            scores = []
            for start in range(0, len(prompts), batch_size):
                formatted = [format_response(tokenizer, p, r, spec["format"])
                             for p, r in zip(prompts[start:start+batch_size], responses[start:start+batch_size])]
                encoded = tokenizer([x[0] for x in formatted], padding=True, truncation=True,
                    max_length=spec["max_length"], add_special_tokens=formatted[0][1], return_tensors="pt")
                scores.append(model(**encoded.to(next(model.parameters()).device)).logits.float().flatten())
            columns.append(torch.cat(scores))
        result = torch.stack(columns, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Reward model produced nonfinite scores")
        return result


def select_rewards(scores, method):
    if method == "vanilla":
        return scores[:, 0], None
    if method == "maxmin":
        selected = scores.mean(0).argmin().item()
        return scores[:, selected], selected
    return scores.mean(1), None


def make_reward_function(bundle, tokenizer, prompt_lookup, method, shared,
                         scoring_batch_size, rollout_batch_size, log_path):
    def get_reward(model, query_responses, pad_token_id, context_length):
        if model is not bundle:
            # The value model must retain TRL's token-wise value extraction.
            return shared._original_get_reward(model, query_responses, pad_token_id, context_length)
        if query_responses.shape[0] != rollout_batch_size:
            raise ValueError("Reward selection requires a complete rollout batch")
        prompts, responses = [], []
        for row in query_responses:
            key = tuple(x for x in row[:context_length].tolist() if x != pad_token_id)
            prompts.append(prompt_lookup[key])
            responses.append(tokenizer.decode(row[context_length:], skip_special_tokens=True))
        scores = bundle.score(prompts, responses, scoring_batch_size).to(query_responses.device)
        rewards, selected = select_rewards(scores, method)
        if method in ("sumgrad", "pcgrad"):
            if shared._pcgrad_state["scores_r1_list"] is None:
                raise RuntimeError("Group PPO patch did not initialize its reward streams")
            shared._pcgrad_state["scores_r1_list"].append(scores[:, 0].detach())
            shared._pcgrad_state["scores_r2_list"].append(scores[:, 1].detach())
        with Path(log_path).open("a") as handle:
            handle.write(json.dumps({"group_mean_rewards": dict(zip(bundle.names, scores.mean(0).tolist())),
                "selected_group": bundle.names[selected] if selected is not None else None}) + "\n")
        lengths = shared.first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
        # Only the scalar score is consumed for the reward model; value extraction
        # above still returns the original token-wise logits.
        return scores[:, None, :], rewards, lengths
    return get_reward
