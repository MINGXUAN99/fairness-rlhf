import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
import importlib
import json
from torch import nn
from datasets import load_dataset
from accelerate import Accelerator
# Import trl.experimental.utils BEFORE PPOTrainer to be able to override get_reward
# We'll override it after defining our custom functions
try:
    trl_experimental_utils = importlib.import_module("trl.experimental.utils")
except ImportError:
    trl_experimental_utils = None

from trl.experimental.ppo import PPOConfig, PPOTrainer
from transformers import TrainerCallback
from transformers.modeling_outputs import SequenceClassifierOutputWithPast

# The actual PPOTrainer with get_reward lives in trl.experimental.ppo.ppo_trainer,
# even though `from trl import PPOTrainer` returns a thin wrapper in trl.trainer.ppo_trainer.
# We must monkey-patch get_reward in the EXPERIMENTAL module where it is actually used.
_experimental_ppo_module = importlib.import_module("trl.experimental.ppo.ppo_trainer")

# Save a reference to the ORIGINAL get_reward before any monkey-patching.
# This is critical: our custom reward functions need to fall back to the original
# for the value model, but after monkey-patching the module's get_reward points to
# our custom function, which would cause infinite recursion.
_original_get_reward = getattr(_experimental_ppo_module, 'get_reward', None)
if _original_get_reward is None and trl_experimental_utils is not None:
    _original_get_reward = trl_experimental_utils.get_reward

# ---------------------------------------------------------------------------
# Monkey-patch masked_whiten to handle the edge case where the mask is all zeros.
# This can happen when every response in a batch starts with EOS (= pad_token),
# causing the entire padding_mask to be True and ~padding_mask to be all False.
# In that case masked_var raises ValueError because it cannot compute variance
# over zero elements.  The safe behaviour is to return the values unchanged –
# they will immediately be masked to 0 anyway by the subsequent masked_fill.
# ---------------------------------------------------------------------------
_original_masked_whiten = _experimental_ppo_module.masked_whiten
_original_masked_var = _experimental_ppo_module.masked_var


def _safe_masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    """masked_var that returns 1.0 instead of raising when mask is all-zero."""
    if mask.sum() == 0:
        return torch.tensor(1.0, device=values.device, dtype=values.dtype)
    return _original_masked_var(values, mask, unbiased)


def _safe_masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """masked_whiten that gracefully handles all-zero masks."""
    if mask.sum() == 0:
        # Nothing to whiten – return values as-is (caller will mask them to 0).
        return values
    return _original_masked_whiten(values, mask, shift_mean)


_experimental_ppo_module.masked_var = _safe_masked_var
_experimental_ppo_module.masked_whiten = _safe_masked_whiten

# ---------------------------------------------------------------------------
# Monkey-patch PPOTrainer.train to use mean KL per token (KL / token length)
# instead of the sum of all tokens. This makes the KL penalty length-invariant.
# ---------------------------------------------------------------------------
import inspect
import textwrap


def _patch_ppo_kl_to_mean_per_token():
    """Patch PPOTrainer.train to use mean KL per token."""
    try:
        source = inspect.getsource(_experimental_ppo_module.PPOTrainer.train)
    except (TypeError, OSError):
        # Source unavailable (e.g. C extension or restricted)
        return

    # Replace: non_score_reward = -args.kl_coef * kl
    # with mean-per-token version (match indentation from getsource: 16 spaces at this block level)
    old_kl_block = "non_score_reward = -args.kl_coef * kl"
    new_kl_block = (
        "# Use mean KL per token (original KL / token length) instead of sum of all tokens\n"
        "                kl_mean_per_seq = kl.sum(1) / sequence_lengths_p1.clamp(min=1)\n"
        "                non_score_reward = -args.kl_coef * kl_mean_per_seq.unsqueeze(1).expand_as(kl)"
    )
    if old_kl_block in source and new_kl_block not in source:
        source = source.replace(old_kl_block, new_kl_block)
        # Replace: mean_kl = kl.sum(1).mean()
        source = source.replace("mean_kl = kl.sum(1).mean()", "mean_kl = kl_mean_per_seq.mean()")
        # Dedent so exec works at module level (getsource returns class-indented code)
        source = textwrap.dedent(source)
        # Compile in the module's namespace so all names resolve
        globals_dict = _experimental_ppo_module.__dict__
        exec(compile(source, "<patched_ppo_train>", "exec"), globals_dict)
        # exec creates a function named 'train' in the compiled block
        patched_train = globals_dict["train"]
        _experimental_ppo_module.PPOTrainer.train = patched_train


_patch_ppo_kl_to_mean_per_token()

# Shared state for tracking which group is selected as worst during MaxMin/SharedRep PPO.
# Written by custom get_reward functions, read by GroupSelectionCallback.
_group_selection_info = {
    "worst_group_idx": None,     # int: index of the worst group (0=minority, 1=majority for maxmin)
    "group_means": None,         # list of floats: mean reward per group for the batch
    "group_means_raw": None,     # list of floats: mean raw reward per group (when selection uses calibrated rewards)
    "group_names": None,         # list of str: human-readable group names
}

# Z-score normalization (from calibration/ JSON produced by normalize_reward_model.py).
# When enabled, rewards are transformed: (r - mean) / max(std, 1e-8). Lookup by reward model hub_id.
_reward_normalization = {
    "enabled": False,
    "vanilla": {"mean": None, "std": None},
    "maxmin": {"minority": {"mean": None, "std": None}, "majority": {"mean": None, "std": None}},
}


def load_reward_normalization_json(path: str, reward_model_hub_id_or_ids) -> None:
    """
    Load z-score normalization params from JSON produced by normalize_reward_model.py.
    File is keyed by preference_dataset_hub_id; reward_model_params[hub_id] = {mean, std, rm_type, group?}.
    reward_model_hub_id_or_ids: for vanilla a single str; for maxmin a dict {"minority": hub_id, "majority": hub_id}.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    params = payload.get("reward_model_params") or {}
    if not params:
        raise ValueError(f"Calibration JSON has no reward_model_params: {path}")

    if isinstance(reward_model_hub_id_or_ids, str):
        # Vanilla: one hub id
        hub_id = reward_model_hub_id_or_ids
        entry = params.get(hub_id)
        if entry is None and "-seed" in hub_id:
            # Fallback: try key without -seedNNN (e.g. old calibration files)
            entry = params.get(hub_id.rsplit("-seed", 1)[0])
        if entry is None:
            raise ValueError(
                f"Reward model hub_id not found in calibration JSON: {hub_id}. "
                f"Keys: {list(params.keys())}"
            )
        mean, std = entry.get("mean"), entry.get("std")
        if mean is None or std is None:
            raise ValueError(f"Calibration entry missing mean/std: {entry}")
        _reward_normalization["enabled"] = True
        _reward_normalization["vanilla"] = {"mean": float(mean), "std": float(std)}
    else:
        # Maxmin: two hub ids
        minority_id = reward_model_hub_id_or_ids.get("minority")
        majority_id = reward_model_hub_id_or_ids.get("majority")
        if not minority_id or not majority_id:
            raise ValueError("maxmin reward_model_hub_id_or_ids must have 'minority' and 'majority' keys")
        min_entry = params.get(minority_id)
        if min_entry is None and "-seed" in minority_id:
            min_entry = params.get(minority_id.rsplit("-seed", 1)[0])
        maj_entry = params.get(majority_id)
        if maj_entry is None and "-seed" in majority_id:
            maj_entry = params.get(majority_id.rsplit("-seed", 1)[0])
        if min_entry is None:
            raise ValueError(f"Minority reward model hub_id not in calibration: {minority_id}. Keys: {list(params.keys())}")
        if maj_entry is None:
            raise ValueError(f"Majority reward model hub_id not in calibration: {majority_id}. Keys: {list(params.keys())}")
        for name, ent in [("minority", min_entry), ("majority", maj_entry)]:
            if ent.get("mean") is None or ent.get("std") is None:
                raise ValueError(f"Calibration entry for {name} missing mean/std: {ent}")
        _reward_normalization["enabled"] = True
        _reward_normalization["maxmin"] = {
            "minority": {"mean": float(min_entry["mean"]), "std": float(min_entry["std"])},
            "majority": {"mean": float(maj_entry["mean"]), "std": float(maj_entry["std"])},
        }


def _apply_reward_normalization(rewards: torch.Tensor) -> torch.Tensor:
    """Apply z-score normalization to a 1d reward tensor (vanilla)."""
    if not _reward_normalization["enabled"]:
        return rewards
    v = _reward_normalization["vanilla"]
    mean, std = v["mean"], v["std"]
    if mean is None or std is None:
        return rewards
    std_safe = max(float(std), 1e-8)
    return (rewards.float() - mean) / std_safe


def _apply_reward_normalization_maxmin(all_group_rewards: torch.Tensor) -> torch.Tensor:
    """Apply z-score per group to (batch, 2) tensor [minority, majority]."""
    if not _reward_normalization["enabled"]:
        return all_group_rewards
    m = _reward_normalization["maxmin"]
    min_m, min_s = m["minority"]["mean"], max(m["minority"]["std"], 1e-8)
    maj_m, maj_s = m["majority"]["mean"], max(m["majority"]["std"], 1e-8)
    out = all_group_rewards.clone().float()
    out[:, 0] = (out[:, 0] - min_m) / min_s
    out[:, 1] = (out[:, 1] - maj_m) / maj_s
    return out


class GroupSelectionCallback(TrainerCallback):
    """Injects worst-group selection stats into PPOTrainer's logged metrics."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or _group_selection_info["worst_group_idx"] is None:
            return
        idx = _group_selection_info["worst_group_idx"]
        group_means = _group_selection_info["group_means"]
        group_means_raw = _group_selection_info.get("group_means_raw")
        group_names = _group_selection_info["group_names"]

        logs["group_selection/worst_group_idx"] = idx
        if group_names is not None:
            logs["group_selection/worst_group_name"] = group_names[idx]
        if group_means is not None:
            for i, mean_val in enumerate(group_means):
                name = group_names[i] if group_names else str(i)
                logs[f"group_selection/mean_reward_{name}"] = mean_val
        if group_means_raw is not None and group_names is not None:
            for i, mean_val in enumerate(group_means_raw):
                name = group_names[i] if group_names else str(i)
                logs[f"group_selection/mean_reward_raw_{name}"] = mean_val



def first_true_indices(mask: torch.Tensor) -> torch.Tensor:
    """
    Find the index of the first True value in each row of a boolean mask.
    Returns the length of the row if no True value is found.
    
    Args:
        mask: Boolean tensor of shape (batch_size, sequence_length)
    
    Returns:
        Tensor of shape (batch_size,) with indices of first True value
    """
    batch_size, sequence_length = mask.shape
    # Create indices tensor: [0, 1, 2, ..., sequence_length-1]
    indices_tensor = torch.arange(sequence_length, device=mask.device, dtype=torch.long).unsqueeze(0).expand(batch_size, -1)
    # Set indices to a large value where mask is False
    indices_tensor = torch.where(mask, indices_tensor, torch.tensor(sequence_length, device=mask.device, dtype=torch.long))
    # Find minimum index (first True), or sequence_length if all False
    result = indices_tensor.min(dim=-1)[0]
    return result
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from src.helper.imdb_utils import pre_processing_imdb


class MaxMinRewardWrapper(nn.Module):
    def __init__(self, reward_model, fc_maj, fc_min):
        super().__init__()
        self.reward_model = reward_model
        self.fc_maj = fc_maj
        self.fc_min = fc_min
        self.config = reward_model.config
        # Resolve the backbone (transformer) once so forward() can call it directly
        # without running the full GPT2ForSequenceClassification.forward() twice.
        if hasattr(reward_model, 'base_model_prefix'):
            self.base_model_prefix = reward_model.base_model_prefix
        else:
            self.base_model_prefix = "transformer"  # GPT2 default
        self.lm_backbone = getattr(reward_model, self.base_model_prefix)

    def forward(self, input_ids, **kwargs):
        # Run the backbone ONCE with use_cache=False (same as original get_reward
        # in trl.experimental.utils) to avoid allocating KV caches (~1.4 GB each).
        output = self.lm_backbone(
            input_ids=input_ids,
            return_dict=True,
            output_hidden_states=True,
            use_cache=False,
            **kwargs,
        )
        hidden_states = output.hidden_states[-1]

        # Apply BOTH score heads to the same hidden states
        min_logits = self.fc_min(hidden_states)  # (batch, seq, 1)
        maj_logits = self.fc_maj(hidden_states)  # (batch, seq, 1)

        # Stack as (batch, seq, 2): [minority, majority]
        stacked = torch.cat([min_logits, maj_logits], dim=-1)

        return SequenceClassifierOutputWithPast(
            loss=None,
            logits=stacked,
            past_key_values=None,
            hidden_states=output.hidden_states,
            attentions=None,
        )


def get_sharedrep_reward(model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward using batch-level MaxMin group selection for SharedRep models.
    
    Per Algorithm 1 of SharedRep-RLHF / MaxMin-RLHF, this function:
    1. Computes per-sample rewards for ALL heads (groups)
    2. Computes the mean reward per head across the batch
    3. Selects the worst-off head (lowest mean reward)
    4. Returns that head's per-sample rewards for PPO update

    Args:
        model (`torch.nn.Module`):
            The SharedRepGPT2RM model in maxmin mode (or value model for fallback).
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for all heads, shape (batch, n_heads).
            - `final_rewards` (`torch.Tensor`):
                The per-sample rewards from the worst-off head, shape (batch,).
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    if isinstance(model, SharedRepGPT2RM):
        assert model.maxmin, "The model must be in maxmin mode"

        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
        # IMPORTANT: keep original token ids (including pad_token_id) because
        # SharedRepGPT2RM.forward() computes last_non_pad_token from input_ids.
        # Zero-filling pads here would make pad detection fail and would shift
        # reward extraction to padded positions.
        input_ids = query_responses

        # Compute rewards for ALL heads (groups)
        with torch.no_grad():
            rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=False,  # avoid allocating KV caches (~1.4 GB) since we don't need them
            )
    
        sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
        
        # rewards.logits shape: (batch, n_heads)
        all_head_rewards = rewards.logits  # (batch, n_heads)

        # Batch-level group selection (Algorithm 1): pick the head with lowest mean reward
        head_means_raw = all_head_rewards.mean(dim=0)
        head_means = all_head_rewards.mean(dim=0)  # (n_heads,)
        worst_head_idx = head_means.argmin().item()
        selected_rewards = all_head_rewards[:, worst_head_idx]
        
        # Record group selection stats for logging
        n_heads = all_head_rewards.shape[1]
        _group_selection_info["worst_group_idx"] = worst_head_idx
        _group_selection_info["group_means"] = [head_means[i].item() for i in range(n_heads)]
        _group_selection_info["group_means_raw"] = [head_means_raw[i].item() for i in range(n_heads)]
        _group_selection_info["group_names"] = [f"head_{i}" for i in range(n_heads)]
        
        return (
            rewards.logits,
            selected_rewards,
            sequence_lengths,
        )
    else:
        # Value model or other model: use the ORIGINAL get_reward (not the overridden one)
        return _original_get_reward(model, query_responses, pad_token_id, context_length)


def get_vanilla_reward(model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vanilla (single RM) reward: use TRL's get_reward then apply z-score normalization if enabled.
    """
    reward_logits, final_rewards, sequence_lengths = _original_get_reward(model, query_responses, pad_token_id, context_length)
    if _reward_normalization["enabled"] and _reward_normalization["vanilla"]["mean"] is not None:
        final_rewards = _apply_reward_normalization(final_rewards)
    return reward_logits, final_rewards, sequence_lengths


def get_maxmin_reward(model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Computes the reward using batch-level MaxMin group selection (Algorithm 1 of MaxMin-RLHF).
    
    Instead of taking element-wise min across groups per sample, this function:
    1. Computes per-sample rewards for ALL groups
    2. Computes the mean reward per group across the batch
    3. Selects the worst-off group (lowest mean reward)
    4. Returns that group's per-sample rewards for PPO update

    Args:
        model (`torch.nn.Module`):
            The MaxMinRewardWrapper model (or value model for fallback).
        query_responses (`torch.Tensor`):
            The tensor containing the query responses.
        pad_token_id (`int`):
            The token ID representing the pad token.
        context_length (`int`):
            The length of the context in the query responses.

    Returns:
        tuple:
            - `reward_logits` (`torch.Tensor`):
                The logits for all groups, shape (batch, seq, n_groups).
            - `final_rewards` (`torch.Tensor`):
                The per-sample rewards from the worst-off group, shape (batch,).
            - `sequence_lengths` (`torch.Tensor`):
                The lengths of the sequences in the query responses.
    """
    if isinstance(model, MaxMinRewardWrapper):

        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        # Compute rewards for ALL groups via a SINGLE backbone pass
        with torch.no_grad():
            rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )
    
        sequence_lengths = first_true_indices(query_responses[:, context_length:] == pad_token_id) - 1 + context_length
        
        # rewards.logits shape: (batch, seq, 2) = [minority, majority]
        # Extract per-sample rewards at the last non-padding token position
        # (same logic as the original get_reward in trl.experimental.utils)
        reward_logits = rewards.logits  # (batch, seq, 2)
        batch_idx = torch.arange(reward_logits.size(0), device=reward_logits.device)
        # reward at the last valid token for each group: (batch, 2)
        all_group_rewards = reward_logits[batch_idx, sequence_lengths]  # (batch, 2)

        # Optional z-score normalization (from calibration/ JSON)
        all_group_rewards = _apply_reward_normalization_maxmin(all_group_rewards)

        # Batch-level group selection (Algorithm 1): pick the group with lowest mean reward
        group_means_raw = all_group_rewards.mean(dim=0)
        group_means = all_group_rewards.mean(dim=0)  # (2,)
        worst_group_idx = group_means.argmin().item()
        selected_rewards = all_group_rewards[:, worst_group_idx]
        
        # Record group selection stats for logging
        _group_selection_info["worst_group_idx"] = worst_group_idx
        _group_selection_info["group_means"] = [group_means[0].item(), group_means[1].item()]
        _group_selection_info["group_means_raw"] = [group_means_raw[0].item(), group_means_raw[1].item()]
        _group_selection_info["group_names"] = ["minority", "majority"]
        
        return (
            reward_logits,
            selected_rewards,
            sequence_lengths,
        )
    
    else:
        # Value model or other model: use the ORIGINAL get_reward (not the overridden one)
        return _original_get_reward(model, query_responses, pad_token_id, context_length)


def prepare_get_gold_reward(policy_tokenizer, reward_tokenizer, alpha, max_length=512, norm_divisor=250):
    """
    Prepares a closure that computes gold reward based on positiveness and conciseness.

    Uses the same group preference formulas as create_chosen_rejected.py:
    - Minority: (1-normalized_length)*alpha + (1-alpha)*positiveness (prefers brevity)
    - Majority: (1-normalized_length)*(1-alpha) + alpha*positiveness (prefers positivity)

    Args:
        policy_tokenizer: Tokenizer for the policy model
        reward_tokenizer: Tokenizer for the reward model (sentiment model)
        alpha: Weight parameter controlling group preference strength (0.5-1.0)
        max_length: Maximum length for reward tokenization (default: 512)
        norm_divisor: Divisor for normalizing token length to [0,1] (default: 500)

    Returns:
        A function that computes gold rewards using MaxMin across groups
    """
    def compute_gold_reward(model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Computes the reward logits and the rewards for a given model and query responses.

        Args:
            model (`torch.nn.Module`):
                The model used to compute the reward logits.
            query_responses (`torch.Tensor`):
                The tensor containing the query responses.
            pad_token_id (`int`):
                The token ID representing the pad token.
            context_length (`int`):
                The length of the context in the query responses.

        Returns:
            tuple:
                - `reward_logits` (`torch.Tensor`):
                    The logits for the reward model.
                - `final_rewards` (`torch.Tensor`):
                    The final rewards for each query response.
                - `sequence_lengths` (`torch.Tensor`):
                    The lengths of the sequences in the query responses.
        """

        # Differentiate gold reward model from value model by num_labels
        # Value model has 1 label (regression), reward model has 2 labels (classification)
        if model.config.num_labels == 1:
            # This is the value model, use the ORIGINAL get_reward (not the overridden one)
            return _original_get_reward(model, query_responses, pad_token_id, context_length)
        else:
            # This is the gold reward model (sentiment classifier)
            pos_id = model.config.label2id["POSITIVE"]

            # Decode to text
            list_query_responses_text = policy_tokenizer.batch_decode(
                query_responses,
                skip_special_tokens=True
            )

            # Re-encode with reward tokenizer (sentiment model tokenizer)
            query_responses_tokens_reward = reward_tokenizer(
                list_query_responses_text,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=max_length,
            ).to(query_responses.device)

            reward_input_ids = query_responses_tokens_reward["input_ids"]
            reward_attention_mask = query_responses_tokens_reward["attention_mask"]

            # Compute positiveness scores
            with torch.no_grad():
                outputs = model(
                    input_ids=reward_input_ids,
                    attention_mask=reward_attention_mask,
                )
            probs = torch.softmax(outputs.logits, dim=-1)
            positiveness = probs[:, pos_id]

            # Compute normalized length (same as in create_chosen_rejected.py)
            # Token lengths
            sequence_lengths = first_true_indices(reward_input_ids == model.config.pad_token_id)
            # Normalized length [0, 1] using norm_divisor (matches scoring pipeline)
            normalized_length = torch.clamp(sequence_lengths.float() / norm_divisor, min=0.0, max=1.0)

            # Compute group scores using the SAME formulas as create_chosen_rejected.py
            # Minority group: prefers brevity (conciseness)
            # score = (1-normalized_length)*alpha + (1-alpha)*positiveness
            group_score_minority = (1 - normalized_length) * alpha + (1 - alpha) * positiveness
            
            # Majority group: prefers positivity
            # score = (1-normalized_length)*(1-alpha) + alpha*positiveness
            group_score_majority = (1 - normalized_length) * (1 - alpha) + alpha * positiveness

            # Batch-level group selection (Algorithm 1 of MaxMin-RLHF):
            # Pick the group with lower mean reward across the batch, then use
            # that group's per-sample rewards for the PPO update.
            min_mean = group_score_minority.mean().item()
            maj_mean = group_score_majority.mean().item()
            if min_mean <= maj_mean:
                selected_rewards = group_score_minority
                worst_idx = 0
            else:
                selected_rewards = group_score_majority
                worst_idx = 1
            
            # Record group selection stats for logging
            _group_selection_info["worst_group_idx"] = worst_idx
            _group_selection_info["group_means"] = [min_mean, maj_mean]
            _group_selection_info["group_names"] = ["minority", "majority"]
            
            return (
                selected_rewards,
                selected_rewards,
                sequence_lengths,
            )
    return compute_gold_reward


def prepare_get_gold_vanilla_reward(policy_tokenizer, reward_tokenizer, alpha, proportion, max_length=512, norm_divisor=250):
    """
    Prepares a closure that computes oracle rewards with per-sample group
    assignment for single-reward (vanilla) PPO — no MaxMin group selection.

    Each sample is randomly assigned to minority (prob=proportion) or majority
    (prob=1-proportion), then scored with that group's oracle formula.  All
    per-sample rewards are fed into a standard PPO update as a flat batch.

    This is an ablation baseline that isolates the effect of MaxMin vs vanilla
    PPO when both use the exact same oracle per-sample rewards.

    Args:
        policy_tokenizer: Tokenizer for the policy model
        reward_tokenizer: Tokenizer for the reward model (sentiment model)
        alpha: Weight parameter controlling group preference strength (0.5-1.0)
        proportion: Probability that a sample belongs to the minority group
        max_length: Maximum length for reward tokenization (default: 512)
        norm_divisor: Divisor for normalizing token length to [0,1] (default: 500)

    Returns:
        A function compatible with trl's ``get_reward`` signature.
    """
    def compute_gold_vanilla_reward(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Value model (num_labels == 1) → fall back to original get_reward
        if model.config.num_labels == 1:
            return _original_get_reward(model, query_responses, pad_token_id, context_length)

        # ---- Oracle reward computation (identical to gold) ----
        pos_id = model.config.label2id["POSITIVE"]

        list_query_responses_text = policy_tokenizer.batch_decode(
            query_responses, skip_special_tokens=True
        )

        query_responses_tokens_reward = reward_tokenizer(
            list_query_responses_text,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        ).to(query_responses.device)

        reward_input_ids = query_responses_tokens_reward["input_ids"]
        reward_attention_mask = query_responses_tokens_reward["attention_mask"]

        with torch.no_grad():
            outputs = model(
                input_ids=reward_input_ids,
                attention_mask=reward_attention_mask,
            )
        probs = torch.softmax(outputs.logits, dim=-1)
        positiveness = probs[:, pos_id]

        sequence_lengths = first_true_indices(reward_input_ids == model.config.pad_token_id)
        normalized_length = torch.clamp(sequence_lengths.float() / norm_divisor, min=0.0, max=1.0)

        group_score_minority = (1 - normalized_length) * alpha + (1 - alpha) * positiveness
        group_score_majority = (1 - normalized_length) * (1 - alpha) + alpha * positiveness

        # ---- Vanilla: per-sample group assignment, no MaxMin ----
        # Randomly assign each sample to a group (same distribution as
        # create_chosen_rejected.py).  Each sample gets its own group's
        # oracle score; all rewards are treated equally by PPO.
        batch_size = query_responses.size(0)
        # 1 = majority (prob 1-proportion), 0 = minority (prob proportion)
        group_ids = torch.bernoulli(
            torch.full((batch_size,), 1 - proportion, device=query_responses.device)
        ).long()

        selected_rewards = torch.where(
            group_ids == 0, group_score_minority, group_score_majority
        )

        return (
            selected_rewards,
            selected_rewards,
            sequence_lengths,
        )

    return compute_gold_vanilla_reward


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["query"],
        padding="max_length",
        max_length=8,
        truncation=True,
        padding_side="left",
        return_tensors="pt"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Unified PPO Training with configurable policy optimization method.")
    parser.add_argument("--method", type=str, required=True, choices=["sharedrep", "maxmin", "gold", "gold_vanilla", "vanilla"],
                        help="Policy optimization method: 'sharedrep', 'maxmin', 'gold', 'gold_vanilla', or 'vanilla'")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, default=None,
                        help="Proportion of minority group (required for sharedrep, maxmin, gold_vanilla, and vanilla methods).")
    parser.add_argument("--k", type=int, default=None,
                        help="Inner hidden dimension size (required for sharedrep method).")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value for the scores (controls group preference strength, default: 1.0). Used in all methods.")
    parser.add_argument(
        "--reward_normalization_json",
        type=str,
        default=None,
        help=(
            "Optional path to calibration JSON from normalize_reward_model.py (under calibration/). "
            "Applies z-score (r - mean) / std per reward model. Hub ids must match loaded RMs (ALPHA-len naming)."
        ),
    )
    parser.add_argument(
        "--no_reward_normalization",
        action="store_true",
        help="Disable reward normalization even when calibration JSON exists. Use raw RM logits for MaxMin/vanilla PPO.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens used for reward model/dataset naming (default 512). Must match train_reward_models/build_dataset.",
    )
    parser.add_argument("--vanilla_rm_hub_id", type=str, default=None,
                        help="Override vanilla reward model hub ID (e.g. {user}/vanilla-imdb-reward-ALPHA1.0-len512-gpt2).")
    parser.add_argument("--maxmin_minority_rm_hub_id", type=str, default=None,
                        help="Override MaxMin minority reward model hub ID.")
    parser.add_argument("--maxmin_majority_rm_hub_id", type=str, default=None,
                        help="Override MaxMin majority reward model hub ID.")
    parser.add_argument("--sharedrep_rm_hub_id", type=str, default=None,
                        help="Override SharedRep reward model hub ID.")
    parser.add_argument("--policy_model_name", type=str, default="gpt2",
                        help="Base model for policy/ref/value (default: gpt2). Use 'gpt2' to train from raw GPT-2, or 'lvwerra/gpt2-imdb' for IMDB-pretrained.")
    parser.add_argument("--policy_hub_id", type=str, default=None,
                        help="Override policy hub ID for saving. If omitted, derived from method/user_id/seed/proportion/alpha.")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL penalty coefficient (default 0.1). Set to 0 for ablation (no KL penalty; may cause policy collapse).")
    return parser.parse_args()


def load_reward_model(method, user_id, seed, proportion, k, alpha, device, max_new_tokens=512,
                      vanilla_rm_hub_id=None, maxmin_minority_rm_hub_id=None, maxmin_majority_rm_hub_id=None, sharedrep_rm_hub_id=None):
    """Load and configure the reward model based on the selected method.
    Returns (reward_model, reward_tokenizer, hub_id_or_ids).
    hub_id_or_ids: for vanilla a str (RM hub id); for maxmin a dict {"minority": id, "majority": id}; else None.
    Optional *_rm_hub_id args override the default constructed hub IDs.
    """
    if method == "sharedrep":
        if proportion is None or k is None:
            raise ValueError("--proportion and --k are required for sharedrep method")
        
        reward_model_name = sharedrep_rm_hub_id or f"{user_id}/sharedrep-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}-k{k}"
        reward_model = SharedRepGPT2RM.from_pretrained(reward_model_name)
        reward_model.enable_maxmin()
        reward_model.eval()
        # Move to device for consistency (though Accelerator handles this, explicit is clearer)
        reward_model.to(device)
        
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, padding_side="left")
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        
        return reward_model, reward_tokenizer, None
    
    elif method == "maxmin":
        if proportion is None:
            raise ValueError("--proportion is required for maxmin method")
        
        # Naming aligned with train_reward_models / normalize_reward_model (ALPHA-len, no seed in hub id)
        reward_model_name_majority = maxmin_majority_rm_hub_id or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-majority-gpt2"
        reward_model_majority = AutoModelForSequenceClassification.from_pretrained(reward_model_name_majority)
        reward_model_majority.eval()
        fc_maj = reward_model_majority.score
        del reward_model_majority
        
        reward_model_name_minority = maxmin_minority_rm_hub_id or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-minority-gpt2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name_minority)
        reward_model.eval()
        fc_min = reward_model.score
        
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_majority)
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

        fc_maj.to(device)
        fc_min.to(device)
        reward_model.to(device)
        maxmin_reward_model = MaxMinRewardWrapper(
            reward_model=reward_model,
            fc_maj=fc_maj,
            fc_min=fc_min
        )
        maxmin_reward_model.eval()
        
        hub_ids = {"minority": reward_model_name_minority, "majority": reward_model_name_majority}
        return maxmin_reward_model, reward_tokenizer, hub_ids
    
    elif method in ("gold", "gold_vanilla"):
        if method == "gold_vanilla" and proportion is None:
            raise ValueError("--proportion is required for gold_vanilla method")
        
        reward_model_name = "lvwerra/distilbert-imdb"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        reward_model.eval()
        reward_model.to(device)
        
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        
        return reward_model, reward_tokenizer, None
    
    elif method == "vanilla":
        if proportion is None:
            raise ValueError("--proportion is required for vanilla method")
        
        reward_model_name = vanilla_rm_hub_id or f"{user_id}/vanilla-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-gpt2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
        reward_model.eval()
        reward_model.to(device)
        
        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, padding_side="left")
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
        reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
        
        return reward_model, reward_tokenizer, reward_model_name
    
    else:
        raise ValueError(f"Unknown method: {method}")


def get_policy_hub_id(method, user_id, seed, proportion, k, alpha):
    """Generate the policy hub ID based on the selected method."""
    if method == "sharedrep":
        return f"{user_id}/sharedrep-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}-k{k}"
    elif method == "maxmin":
        return f"{user_id}/maxmin-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
    elif method == "gold":
        # Gold uses alpha to define group preferences in the MaxMin objective
        return f"{user_id}/gold-imdb-ppo-alpha{alpha}-seed{seed}"
    elif method == "gold_vanilla":
        # Gold vanilla: oracle per-sample group scores + single-reward PPO (no MaxMin)
        return f"{user_id}/gold-vanilla-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
    elif method == "vanilla":
        return f"{user_id}/vanilla-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
    else:
        raise ValueError(f"Unknown method: {method}")


def setup_reward_function(method, policy_tokenizer, reward_tokenizer, reward_model, alpha, proportion=None):
    """Set up the reward function based on the selected method.
    
    Args:
        method: Training method ('sharedrep', 'maxmin', 'gold', 'gold_vanilla', 'vanilla')
        policy_tokenizer: Tokenizer for the policy model
        reward_tokenizer: Tokenizer for the reward model
        reward_model: The reward model instance
        alpha: Alpha parameter for gold reward computation
        proportion: Minority group proportion (required for gold_vanilla)
    """
    # Override get_reward in BOTH locations where PPOTrainer resolves it:
    # 1. trl.experimental.utils module (the source module)
    # 2. PPOTrainer's own module namespace (where `from ..utils import get_reward` placed it)
    # Both must be set to ensure the override takes effect.
    
    if method == "sharedrep":
        custom_fn = get_sharedrep_reward
    elif method == "maxmin":
        custom_fn = get_maxmin_reward
    elif method == "gold":
        # Pass alpha to gold reward function for consistent group preference computation
        custom_fn = prepare_get_gold_reward(policy_tokenizer, reward_tokenizer, alpha, max_length=512, norm_divisor=250)
    elif method == "gold_vanilla":
        # Oracle per-sample group scoring, no MaxMin
        custom_fn = prepare_get_gold_vanilla_reward(policy_tokenizer, reward_tokenizer, alpha, proportion, max_length=512, norm_divisor=250)
    elif method == "vanilla":
        # Vanilla: use wrapper so we can apply z-score normalization when enabled
        custom_fn = get_vanilla_reward
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Apply the override to both modules:
    # 1. trl.experimental.utils (the source module)
    # 2. trl.experimental.ppo.ppo_trainer (where `from ..utils import get_reward` placed it)
    if trl_experimental_utils is not None:
        trl_experimental_utils.get_reward = custom_fn
    _experimental_ppo_module.get_reward = custom_fn


if __name__ == "__main__":

    accelerator = Accelerator()
    device = accelerator.device

    args = parse_args()
    method = args.method
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    k = args.k
    alpha = args.alpha
    # Dataset
    batch_size = 16
    dataset_name = f"stanfordnlp/imdb"
    dataset = load_dataset(dataset_name, split="train")

    # Load reward model based on method (returns hub_id_or_ids for normalization lookup)
    reward_model, reward_tokenizer, reward_model_hub_id_or_ids = load_reward_model(
        method, user_id, seed, proportion, k, alpha, device,
        max_new_tokens=args.max_new_tokens,
        vanilla_rm_hub_id=args.vanilla_rm_hub_id,
        maxmin_minority_rm_hub_id=args.maxmin_minority_rm_hub_id,
        maxmin_majority_rm_hub_id=args.maxmin_majority_rm_hub_id,
        sharedrep_rm_hub_id=args.sharedrep_rm_hub_id,
    )
    # Reward normalization: used for maxmin only (vanilla always uses raw RM logits).
    # Skip if --no_reward_normalization, or if method is vanilla.
    if (
        args.reward_normalization_json
        and reward_model_hub_id_or_ids is not None
        and not args.no_reward_normalization
        and method != "vanilla"
    ):
        load_reward_normalization_json(args.reward_normalization_json, reward_model_hub_id_or_ids)
        if accelerator.is_main_process:
            print(f"Loaded reward normalization (z-score): {args.reward_normalization_json}")
    elif method == "vanilla" and accelerator.is_main_process:
        print("Vanilla PPO: reward normalization not used (raw RM logits)")
    elif args.no_reward_normalization and accelerator.is_main_process:
        print("Reward normalization disabled (--no_reward_normalization)")

    # Policy Model (default gpt2 = raw, use lvwerra/gpt2-imdb for IMDB-pretrained)
    policy_model_name = args.policy_model_name
    if accelerator.is_main_process:
        print(f"Policy base model: {policy_model_name}")
    policy_hub_id = args.policy_hub_id or get_policy_hub_id(method, user_id, seed, proportion, k, alpha)
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name, padding_side="left")
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    # Policy model
    policy = AutoModelForCausalLM.from_pretrained(policy_model_name, trust_remote_code=True)

    # Ref policy
    ref_policy = AutoModelForCausalLM.from_pretrained(policy_model_name, trust_remote_code=True)

    # Value model
    value_model = AutoModelForSequenceClassification.from_pretrained(policy_model_name, trust_remote_code=True, num_labels=1)

    with accelerator.main_process_first():

        min_review_length = 200
        input_min_text_length = 2
        input_max_text_length = 8
        
        dataset = pre_processing_imdb(
            dataset,
            model_name=policy_model_name,
            min_review_length=min_review_length,
            input_min_text_length=input_min_text_length,
            input_max_text_length=input_max_text_length
        )
        dataset = dataset.map(lambda x: tokenize(x, policy_tokenizer), batched=True, batch_size=batch_size, num_proc=32, desc="Tokenizing")
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)
        train_dataset = dataset["train"]
        eval_dataset = dataset["test"]

    # Training Arguments
    training_args = PPOConfig(
        logging_steps=50,  # Reduced frequency to save time (was 10)
        num_train_epochs=3,  # Reduced from 5 to save time
        num_ppo_epochs=3,
        per_device_train_batch_size=32,  # Reduced from 64 to save memory
        gradient_accumulation_steps=2,  # Increased to maintain effective batch size
        bf16=True,
        learning_rate=3e-06,
        eval_on_start=False,  # Disabled to save time - evaluation happens at end anyway
        push_to_hub=True,
        save_strategy="no",
        hub_private_repo=True,
        hub_model_id=policy_hub_id,
        report_to=["wandb"],
        run_name=policy_hub_id,
        whiten_rewards=True,
        missing_eos_penalty=0.0,  # was 1.0; removed to prevent early-EOS collapse (rewards are in [0,1])
        total_episodes=50000,
        lr_scheduler_type="cosine",
        stop_token="eos",
        kl_coef=args.kl_coef,
    )

    # Set up reward function based on method
    setup_reward_function(method, policy_tokenizer, reward_tokenizer, reward_model, alpha, proportion)

    # Train
    callbacks = []
    if method in ("maxmin", "sharedrep", "gold"):
        callbacks.append(GroupSelectionCallback())
    
    trainer = PPOTrainer(
        args=training_args,
        model=policy,
        processing_class=policy_tokenizer,
        ref_model=ref_policy,
        reward_model=reward_model,
        value_model=value_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    trainer.train()

    # Save the model - explicitly save the policy model with its config and weights
    if accelerator.is_main_process:
        print("\n" + "="*80)
        print("SAVING MODEL TO HUGGINGFACE HUB")
        print("="*80)
        
        # Unwrap the model from accelerator
        unwrapped_policy = accelerator.unwrap_model(policy)
        
        # Ensure config has model_type (GPT2 models should have this, but ensure it's set)
        if not hasattr(unwrapped_policy.config, 'model_type') or unwrapped_policy.config.model_type is None:
            unwrapped_policy.config.model_type = "gpt2"
        
        print(f"Model ID: {policy_hub_id}")
        print(f"Config model_type: {unwrapped_policy.config.model_type}")
        
        # Push model and tokenizer to hub with weights
        try:
            print("Pushing model to Hub...")
            commit_info = unwrapped_policy.push_to_hub(
                policy_hub_id, 
                private=True,
                commit_message=f"Upload {policy_hub_id} model weights"
            )
            print(f"✓ Model pushed successfully!")
            print(f"  Commit URL: {commit_info.commit_url if hasattr(commit_info, 'commit_url') else 'N/A'}")
            
            print("Pushing tokenizer to Hub...")
            tokenizer_commit = policy_tokenizer.push_to_hub(policy_hub_id, private=True)
            print(f"✓ Tokenizer pushed successfully!")
            
        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR: Failed to push model/tokenizer to Hub!")
            print(f"{'!'*80}")
            print(f"Error: {e}")
            print(f"\nPlease manually upload the model:")
            print(f"  1. Save locally: unwrapped_policy.save_pretrained('./local_model')")
            print(f"  2. Push: model.push_to_hub('{policy_hub_id}', private=True)")
            raise
        
        print("="*80)
        print("MODEL SAVED SUCCESSFULLY")
        print("="*80 + "\n")
    
    # trainer.push_to_hub() pushes the dataset, not the model, so we handle model saving above
    if accelerator.is_main_process:
        try:
            trainer.push_to_hub(dataset_name=dataset_name)
        except Exception as e:
            print(f"Warning: Failed to push training dataset: {e}")
