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


# Patch is applied conditionally in main() based on --kl_div (see below).


# ---------------------------------------------------------------------------
# PCGrad patch – applied in main() when --method pcgrad is selected.
# Reads TRL source once and applies mean-KL normalisation (optional) plus
# 6 targeted PCGrad edits in a single getsource → exec round trip.
# ---------------------------------------------------------------------------

def _patch_ppo_for_pcgrad(apply_mean_kl: bool = True, apply_fn=None) -> None:
    """
    Patch PPOTrainer.train to perform two-objective multi-gradient PPO.

    Shared by PCGrad and SumGrad: both decompose the reward into per-group
    advantage streams, compute a per-group policy-gradient loss, and combine
    the two group gradients in a custom update step.  The ONLY difference is
    the combination rule, injected via ``apply_fn``:
      - PCGrad  → ``_pcgrad_compute_and_apply``  (conflict projection)
      - SumGrad → ``_sumgrad_compute_and_apply`` (plain vector sum)
    ``apply_fn`` defaults to the PCGrad update when not supplied.

    Makes 6 targeted edits to the TRL PPO training loop:
      1. Reset _pcgrad_state accumulators before each rollout.
      2. Concatenate per-group rewards after the rollout loop.
      3. Compute per-group advantages (KL-only, r1-only, r2-only).
      4a. Slice per-group mini-batch advantages.
      4b. Replace the single backward+step with _pcgrad_compute_and_apply().
      5. Update the mini-batch del block (remove `loss`, add new variables).
      6. Update the outer del block to include the new advantage tensors.

    Optionally also applies the mean-KL-per-token normalisation (same as
    _patch_ppo_kl_to_mean_per_token) within the same source pass so that both
    patches can share a single getsource → exec round trip.
    """
    try:
        source = inspect.getsource(_experimental_ppo_module.PPOTrainer.train)
    except (TypeError, OSError):
        return

    # Inject helpers into TRL's module namespace so the patched train body
    # can call them without an explicit import.
    gd = _experimental_ppo_module.__dict__
    gd["_pcgrad_state"]            = _pcgrad_state
    gd["_pcgrad_window"]           = _pcgrad_window
    gd["pcgrad_project_2"]         = pcgrad_project_2
    gd["_pcgrad_gae"]              = _pcgrad_gae
    gd["_pcgrad_compute_pg_loss"]  = _pcgrad_compute_pg_loss
    # The patched train body calls ``_pcgrad_compute_and_apply`` by name; bind
    # that name to the requested combination rule (PCGrad projection by default,
    # SumGrad vector sum when apply_fn is supplied).
    gd["_pcgrad_compute_and_apply"] = apply_fn if apply_fn is not None else _pcgrad_compute_and_apply

    # ── Optional mean-KL-per-token normalisation ──────────────────────────
    if apply_mean_kl:
        old_kl = "non_score_reward = -args.kl_coef * kl"
        new_kl = (
            "# PCGrad+mean_kl: use mean KL per token (length-invariant)\n"
            "                kl_mean_per_seq = kl.sum(1) / sequence_lengths_p1.clamp(min=1)\n"
            "                non_score_reward = -args.kl_coef * kl_mean_per_seq.unsqueeze(1).expand_as(kl)"
        )
        if old_kl in source:
            source = source.replace(old_kl, new_kl, 1)
            source = source.replace(
                "mean_kl = kl.sum(1).mean()",
                "mean_kl = kl_mean_per_seq.mean()",
                1,
            )

    # ── Change 1: reset side channel before the rollout block ─────────────
    old_c1 = (
        '            with torch.no_grad():\n'
        '                queries = data["input_ids"].to(device)'
    )
    new_c1 = (
        '            # PCGrad: reset per-group reward accumulators for this rollout\n'
        '            _pcgrad_state["scores_r1_list"] = []\n'
        '            _pcgrad_state["scores_r2_list"] = []\n'
        '            # Windowed PCGrad: expose the 1-indexed PPO update (== global_step)\n'
        '            _pcgrad_state["update"] = update\n'
        '            with torch.no_grad():\n'
        '                queries = data["input_ids"].to(device)'
    )
    if old_c1 not in source:
        return  # TRL version mismatch – skip rather than corrupt training
    source = source.replace(old_c1, new_c1, 1)

    # ── Change 2: concatenate per-group rewards after the rollout loop ─────
    old_c2 = (
        '                scores = torch.cat(scores, 0)\n'
        '                values = torch.cat(values, 0)'
    )
    new_c2 = (
        '                scores = torch.cat(scores, 0)\n'
        '                # PCGrad: concatenate per-group rewards accumulated during rollout\n'
        '                _pcgrad_state["scores_r1"] = torch.cat(_pcgrad_state["scores_r1_list"], 0)\n'
        '                _pcgrad_state["scores_r2"] = torch.cat(_pcgrad_state["scores_r2_list"], 0)\n'
        '                values = torch.cat(values, 0)'
    )
    source = source.replace(old_c2, new_c2, 1)

    # ── Change 3: compute per-group advantages after the combined ones ─────
    old_c3 = (
        '                advantages = torch.masked_fill(advantages, padding_mask, 0)\n'
        '                empty_cache()'
    )
    new_c3 = (
        '                advantages = torch.masked_fill(advantages, padding_mask, 0)\n'
        '                # PCGrad: separate advantages (KL-only, pure r1, pure r2)\n'
        '                scores_r1 = _pcgrad_state["scores_r1"]\n'
        '                scores_r2 = _pcgrad_state["scores_r2"]\n'
        '                advantages_kl = _pcgrad_gae(\n'
        '                    None, non_score_reward, values, padding_mask, padding_mask_p1,\n'
        '                    actual_start, actual_end, gen_length, args)\n'
        '                advantages_r1 = _pcgrad_gae(\n'
        '                    scores_r1, None, values, padding_mask, padding_mask_p1,\n'
        '                    actual_start, actual_end, gen_length, args)\n'
        '                advantages_r2 = _pcgrad_gae(\n'
        '                    scores_r2, None, values, padding_mask, padding_mask_p1,\n'
        '                    actual_start, actual_end, gen_length, args)\n'
        '                empty_cache()'
    )
    source = source.replace(old_c3, new_c3, 1)

    # ── Change 4a: slice per-group mini-batch advantages ───────────────────
    old_c4a = '                            mb_advantage = advantages[micro_batch_inds]'
    new_c4a = (
        '                            mb_advantage = advantages[micro_batch_inds]\n'
        '                            mb_advantage_kl = advantages_kl[micro_batch_inds]\n'
        '                            mb_advantage_r1 = advantages_r1[micro_batch_inds]\n'
        '                            mb_advantage_r2 = advantages_r2[micro_batch_inds]'
    )
    source = source.replace(old_c4a, new_c4a, 1)

    # ── Change 4b: replace single backward+step with PCGrad update ─────────
    old_c4b = (
        '                            loss = pg_loss + args.vf_coef * vf_loss\n'
        '                            accelerator.backward(loss)\n'
        '                            optimizer.step()\n'
        '                            optimizer.zero_grad()'
    )
    new_c4b = (
        '                            # PCGrad: per-component policy-gradient losses\n'
        '                            _mb_mask = ~padding_mask[micro_batch_inds]\n'
        '                            pg_loss_kl = _pcgrad_compute_pg_loss(\n'
        '                                mb_advantage_kl, ratio, args.cliprange, _mb_mask)\n'
        '                            pg_loss_r1 = _pcgrad_compute_pg_loss(\n'
        '                                mb_advantage_r1, ratio, args.cliprange, _mb_mask)\n'
        '                            pg_loss_r2 = _pcgrad_compute_pg_loss(\n'
        '                                mb_advantage_r2, ratio, args.cliprange, _mb_mask)\n'
        '                            # PCGrad update: 3 backward passes + projection\n'
        '                            _pcgrad_compute_and_apply(\n'
        '                                model, accelerator, optimizer,\n'
        '                                pg_loss_kl, pg_loss_r1, pg_loss_r2, vf_loss, args)'
    )
    source = source.replace(old_c4b, new_c4b, 1)

    # ── Change 5: update the mini-batch del block ──────────────────────────
    # Remove `loss` (no longer created); add PCGrad-specific variables.
    old_del_mb = (
        '                    del (\n'
        '                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,\n'
        '                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,\n'
        '                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,\n'
        '                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,\n'
        '                    )'
    )
    new_del_mb = (
        '                    del (\n'
        '                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,\n'
        '                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,\n'
        '                        pg_loss, pg_loss_kl, pg_loss_r1, pg_loss_r2,\n'
        '                        pg_clipfrac, prob_dist, entropy, approxkl, mb_return,\n'
        '                        mb_advantage, mb_advantage_kl, mb_advantage_r1, mb_advantage_r2,\n'
        '                        mb_values, mb_responses, mb_query_responses, mb_logprobs, _mb_mask,\n'
        '                    )'
    )
    source = source.replace(old_del_mb, new_del_mb, 1)

    # ── Change 6: update the outer del block to include new advantage tensors
    # (entries are indented 16 spaces in TRL's source, not 20).
    old_del_outer = '                advantages,\n                returns,'
    new_del_outer = (
        '                advantages,\n'
        '                advantages_kl, advantages_r1, advantages_r2,\n'
        '                scores_r1, scores_r2,\n'
        '                returns,'
    )
    source = source.replace(old_del_outer, new_del_outer, 1)

    # Compile and install
    source = textwrap.dedent(source)
    exec(compile(source, "<patched_ppo_train_pcgrad>", "exec"), gd)
    _experimental_ppo_module.PPOTrainer.train = gd["train"]


def _patch_ppo_for_vanilla_deconflict(apply_mean_kl: bool = True) -> None:
    """
    Patch PPOTrainer.train for windowed vanilla deconfliction (reviewer W4).

    Standard vanilla PPO keeps its single POOLED reward as the training signal, but
    a MINORITY guard advantage stream is added so that _deconflict_compute_and_apply
    can project the pooled update away from the minority gradient inside the window.

    Five targeted edits to the TRL PPO loop (a strict subset of the PCGrad edits —
    one guard group instead of two, and the pooled `advantages`/`pg_loss` remain the
    objective rather than being replaced by summed group losses):
      1. Reset the minority side channel and record the update index before rollout.
      2. Concatenate the minority per-group rewards after the rollout loop.
      3. Build the minority-only advantage stream (pure r1, KL-free).
      4a. Slice per-micro-batch minority advantages.
      4b. Replace the single backward+step with _deconflict_compute_and_apply().
      5. Update the mini-batch del block (drop `loss`, add the new variables).
      6. Update the outer del block to include the new advantage/score tensors.

    Optionally also applies the mean-KL-per-token normalisation in the same pass.
    """
    try:
        source = inspect.getsource(_experimental_ppo_module.PPOTrainer.train)
    except (TypeError, OSError):
        return

    gd = _experimental_ppo_module.__dict__
    gd["_pcgrad_state"]                 = _pcgrad_state
    gd["_pcgrad_window"]                = _pcgrad_window
    gd["_pcgrad_gae"]                   = _pcgrad_gae
    gd["_pcgrad_compute_pg_loss"]       = _pcgrad_compute_pg_loss
    gd["_deconflict_compute_and_apply"] = _deconflict_compute_and_apply

    # ── Optional mean-KL-per-token normalisation ──────────────────────────
    if apply_mean_kl:
        old_kl = "non_score_reward = -args.kl_coef * kl"
        new_kl = (
            "# deconflict+mean_kl: use mean KL per token (length-invariant)\n"
            "                kl_mean_per_seq = kl.sum(1) / sequence_lengths_p1.clamp(min=1)\n"
            "                non_score_reward = -args.kl_coef * kl_mean_per_seq.unsqueeze(1).expand_as(kl)"
        )
        if old_kl in source:
            source = source.replace(old_kl, new_kl, 1)
            source = source.replace(
                "mean_kl = kl.sum(1).mean()",
                "mean_kl = kl_mean_per_seq.mean()",
                1,
            )

    # ── Change 1: reset the minority side channel + record update index ───
    old_c1 = (
        '            with torch.no_grad():\n'
        '                queries = data["input_ids"].to(device)'
    )
    new_c1 = (
        '            # Deconflict: reset per-chunk minority reward accumulator for this rollout\n'
        '            _pcgrad_state["scores_r1_list"] = []\n'
        '            _pcgrad_state["update"] = update\n'
        '            with torch.no_grad():\n'
        '                queries = data["input_ids"].to(device)'
    )
    if old_c1 not in source:
        return  # TRL version mismatch – skip rather than corrupt training
    source = source.replace(old_c1, new_c1, 1)

    # ── Change 2: concatenate minority per-group rewards after rollout ─────
    old_c2 = (
        '                scores = torch.cat(scores, 0)\n'
        '                values = torch.cat(values, 0)'
    )
    new_c2 = (
        '                scores = torch.cat(scores, 0)\n'
        '                # Deconflict: concatenate minority rewards accumulated during rollout\n'
        '                _pcgrad_state["scores_r1"] = torch.cat(_pcgrad_state["scores_r1_list"], 0)\n'
        '                values = torch.cat(values, 0)'
    )
    source = source.replace(old_c2, new_c2, 1)

    # ── Change 3: minority-only advantage stream (pure r1, KL-free) ────────
    old_c3 = (
        '                advantages = torch.masked_fill(advantages, padding_mask, 0)\n'
        '                empty_cache()'
    )
    new_c3 = (
        '                advantages = torch.masked_fill(advantages, padding_mask, 0)\n'
        '                # Deconflict: minority-only advantages (pure r1, KL-free) for the guard gradient g1\n'
        '                scores_r1 = _pcgrad_state["scores_r1"]\n'
        '                advantages_r1 = _pcgrad_gae(\n'
        '                    scores_r1, None, values, padding_mask, padding_mask_p1,\n'
        '                    actual_start, actual_end, gen_length, args)\n'
        '                empty_cache()'
    )
    source = source.replace(old_c3, new_c3, 1)

    # ── Change 4a: slice per-micro-batch minority advantages ───────────────
    old_c4a = '                            mb_advantage = advantages[micro_batch_inds]'
    new_c4a = (
        '                            mb_advantage = advantages[micro_batch_inds]\n'
        '                            mb_advantage_r1 = advantages_r1[micro_batch_inds]'
    )
    source = source.replace(old_c4a, new_c4a, 1)

    # ── Change 4b: replace single backward+step with deconflict update ─────
    old_c4b = (
        '                            loss = pg_loss + args.vf_coef * vf_loss\n'
        '                            accelerator.backward(loss)\n'
        '                            optimizer.step()\n'
        '                            optimizer.zero_grad()'
    )
    new_c4b = (
        '                            # Deconflict: minority guard policy-gradient loss (g1)\n'
        '                            _mb_mask = ~padding_mask[micro_batch_inds]\n'
        '                            pg_loss_r1 = _pcgrad_compute_pg_loss(\n'
        '                                mb_advantage_r1, ratio, args.cliprange, _mb_mask)\n'
        '                            # Deconflict update: project pooled pg_loss gradient away from g1 in-window\n'
        '                            _deconflict_compute_and_apply(\n'
        '                                model, accelerator, optimizer,\n'
        '                                pg_loss, pg_loss_r1, vf_loss, args)'
    )
    source = source.replace(old_c4b, new_c4b, 1)

    # ── Change 5: update the mini-batch del block ──────────────────────────
    old_del_mb = (
        '                    del (\n'
        '                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,\n'
        '                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,\n'
        '                        pg_loss, loss, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,\n'
        '                        mb_advantage, mb_values, mb_responses, mb_query_responses, mb_logprobs,\n'
        '                    )'
    )
    new_del_mb = (
        '                    del (\n'
        '                        output, vpred_temp, logits, new_logprobs, vpred, vpredclipped,\n'
        '                        vf_losses1, vf_losses2, vf_loss, vf_clipfrac, logprobs_diff, ratio, pg_losses, pg_losses2, pg_loss_max,\n'
        '                        pg_loss, pg_loss_r1, pg_clipfrac, prob_dist, entropy, approxkl, mb_return,\n'
        '                        mb_advantage, mb_advantage_r1, mb_values, mb_responses, mb_query_responses, mb_logprobs, _mb_mask,\n'
        '                    )'
    )
    source = source.replace(old_del_mb, new_del_mb, 1)

    # ── Change 6: update the outer del block for the new tensors ───────────
    # (entries are indented 16 spaces in TRL's source, not 20).
    old_del_outer = '                advantages,\n                returns,'
    new_del_outer = (
        '                advantages,\n'
        '                advantages_r1, scores_r1,\n'
        '                returns,'
    )
    source = source.replace(old_del_outer, new_del_outer, 1)

    # Compile and install
    source = textwrap.dedent(source)
    exec(compile(source, "<patched_ppo_train_deconflict>", "exec"), gd)
    _experimental_ppo_module.PPOTrainer.train = gd["train"]

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

# ---------------------------------------------------------------------------
# PCGrad side channel.
# get_pcgrad_reward() appends per-group rewards to the lists on each
# rollout mini-batch call.  After the rollout loop the patch code
# concatenates them into scores_r1 / scores_r2.
# ---------------------------------------------------------------------------
_pcgrad_state: dict = {
    "scores_r1_list": None,   # list[Tensor] – accumulated per rollout chunk
    "scores_r2_list": None,
    "scores_r1": None,        # Tensor (batch,) – set after rollout loop
    "scores_r2": None,
    "update": 0,              # int – current PPO update index (1-indexed == global_step),
                              # set by the patched train loop; read by windowed PCGrad.
}

# Windowed-PCGrad experiment (reviewer W4: transient cause vs. persistent effect).
# When enabled, PCGrad conflict-projection is applied ONLY while the current PPO
# update index (1-indexed `update`, equal to the reported global_step) lies inside
# [start, end] (inclusive); outside the window the two group gradients are combined
# by a plain vector sum — identical to SumGrad / no gradient surgery. This lets us
# dissociate WHEN the gradient conflict is resolved from WHETHER the collapse is
# prevented (see _windowed_pcgrad_compute_and_apply).
_pcgrad_window: dict = {
    "enabled": False,
    "start": None,   # int, inclusive (1-indexed PPO update); None = unbounded below
    "end": None,     # int, inclusive; None = unbounded above
}

# SumGrad configuration. When "normalize" is True the two group gradients are
# mean-norm rescaled before summing (equalizes each objective's relative
# contribution while preserving overall step magnitude); otherwise a plain
# vector sum is used. Set in main() from --sumgrad_normalize.
_sumgrad_config: dict = {
    "normalize": False,
}

# MGDA configuration (Sener & Koltun 2018, "Multi-Task Learning as Multi-Objective
# Optimization", arXiv:1810.04650). Unlike SumGrad (g1 + g2) or PCGrad (conflict
# projection), MGDA combines the two group gradients by the *minimum-norm point in
# their convex hull*: it solves  min_{alpha in [0,1]} ||alpha*g1 + (1-alpha)*g2||^2
# and steps along d = alpha*g1 + (1-alpha)*g2. This d is a common-descent direction
# (<d, g_k> >= ||d||^2 >= 0 for both k), so the step never increases either group's
# objective when a mutually-improving direction exists; d = 0 iff the point is
# Pareto-stationary (0 in conv{g1, g2}). The mixing weight alpha adapts every update
# toward whichever group is currently being neglected.
#
# When "normalize" is True the two gradients are mean-norm rescaled BEFORE the
# min-norm solve (and before combining), so the convex-hull geometry is not
# dominated by whichever RM happens to have the larger reward scale — important
# here because the minority/majority RMs can differ in scale. See
# _mgda_compute_and_apply. Set in main() from --mgda_normalize.
_mgda_config: dict = {
    "normalize": False,
}

# SumGrad gradient-cosine diagnostic. When csv_path is set, each policy update
# records cos(g1, g2) between the minority (g1) and majority (g2) objective
# gradients — the direct measure of gradient conflict between the two groups.
_sumgrad_diag: dict = {
    "csv_path": None,
    "step": 0,
    "buffer": [],
}


def _sumgrad_record_cosine(g1: torch.Tensor, g2: torch.Tensor) -> None:
    """Record cos(g1, g2) + norms/dot for one update; buffered, flushed in chunks.

    Also records the current PPO update index (_pcgrad_state["update"]) so that,
    for windowed-PCGrad runs, each row can be labelled inside/outside the
    projection window in post-hoc analysis (step is a per-minibatch counter,
    whereas update is the 1-indexed PPO step matching the paper's figures).
    """
    if _sumgrad_diag["csv_path"] is None:
        return
    n1 = g1.norm()
    n2 = g2.norm()
    dot = torch.dot(g1, g2)
    cos = (dot / (n1 * n2).clamp(min=1e-12)).item()
    _sumgrad_diag["buffer"].append(
        (_sumgrad_diag["step"], _pcgrad_state.get("update", 0),
         cos, dot.item(), n1.item(), n2.item())
    )
    _sumgrad_diag["step"] += 1
    if len(_sumgrad_diag["buffer"]) >= 50:
        _sumgrad_flush_cosine()


def _sumgrad_flush_cosine() -> None:
    """Append buffered cosine rows to the CSV (writes header on first flush)."""
    path = _sumgrad_diag["csv_path"]
    if not path or not _sumgrad_diag["buffer"]:
        return
    import csv as _csv
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    write_header = (not os.path.exists(path)) or os.path.getsize(path) == 0
    with open(path, "a", newline="") as fh:
        w = _csv.writer(fh)
        if write_header:
            w.writerow(["step", "update", "cos_g1_g2", "dot", "norm_g1", "norm_g2"])
        w.writerows(_sumgrad_diag["buffer"])
    _sumgrad_diag["buffer"].clear()


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


# ---------------------------------------------------------------------------
# PCGrad helpers – Yu et al. (2020) "Gradient Surgery for Multi-Task Learning"
# ---------------------------------------------------------------------------

def pcgrad_project_2(
    g1: torch.Tensor,
    g2: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PCGrad projection for exactly 2 tasks.

    g1, g2 are 1-D flattened gradient vectors.  If dot(g1, g2) < 0 the
    conflicting component is subtracted from each gradient using the
    *original* (pre-projection) other gradient, matching Algorithm 1 of the
    paper.  When dot >= 0 neither gradient is modified.
    """
    g1_proj = g1.clone()
    g2_proj = g2.clone()
    dot = torch.dot(g1, g2)
    if dot < 0:
        g1_proj = g1 - (dot / g2.norm().pow(2).clamp(min=1e-12)) * g2
        g2_proj = g2 - (dot / g1.norm().pow(2).clamp(min=1e-12)) * g1
    return g1_proj, g2_proj


def _pcgrad_gae(
    scalar_scores: torch.Tensor | None,   # (batch,) scalar reward at EOS, or None
    token_rewards: torch.Tensor | None,   # (batch, seq) per-token rewards, or None
    values: torch.Tensor,                 # (batch, seq) value-model predictions
    padding_mask: torch.Tensor,           # (batch, seq) True = padding token
    padding_mask_p1: torch.Tensor,        # (batch, seq) shifted version
    actual_start: torch.Tensor,           # (batch,) batch indices
    actual_end: torch.Tensor,             # (batch,) EOS position in reward tensor
    gen_length: int,
    args,
) -> torch.Tensor:
    """
    Compute whitened GAE advantages from a custom reward decomposition.

    Used to compute separate advantage streams for PCGrad:
      - KL-only advantages: token_rewards = non_score_reward, scalar_scores = None
      - Pure group advantages: token_rewards = None, scalar_scores = r1 or r2
    """
    rewards = torch.zeros_like(values)
    if token_rewards is not None:
        rewards = rewards + token_rewards
    if scalar_scores is not None:
        rewards[actual_start, actual_end] = rewards[actual_start, actual_end] + scalar_scores

    if args.whiten_rewards:
        rewards = _safe_masked_whiten(rewards, mask=~padding_mask_p1, shift_mean=False)
        rewards = torch.masked_fill(rewards, padding_mask_p1, 0)

    lastgaelam = 0
    advantages_reversed = []
    for t in reversed(range(gen_length)):
        nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
        delta = rewards[:, t] + args.gamma * nextvalues - values[:, t]
        lastgaelam = delta + args.gamma * args.lam * lastgaelam
        advantages_reversed.append(lastgaelam)
    advantages = torch.stack(advantages_reversed[::-1], axis=1)
    advantages = _safe_masked_whiten(advantages, ~padding_mask)
    advantages = torch.masked_fill(advantages, padding_mask, 0)
    return advantages


def _pcgrad_compute_pg_loss(
    adv: torch.Tensor,
    ratio: torch.Tensor,
    cliprange: float,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Clipped PPO policy-gradient loss for a single advantage stream."""
    pg_l  = -adv * ratio
    pg_l2 = -adv * torch.clamp(ratio, 1.0 - cliprange, 1.0 + cliprange)
    # masked_mean lives in _experimental_ppo_module – access through the import
    from trl.experimental.ppo.ppo_trainer import masked_mean as _trl_masked_mean
    return _trl_masked_mean(torch.max(pg_l, pg_l2), mask)


def _pcgrad_compute_and_apply(
    model,
    accelerator,
    optimizer,
    pg_loss_kl: torch.Tensor,
    pg_loss_r1: torch.Tensor,
    pg_loss_r2: torch.Tensor,
    vf_loss: torch.Tensor,
    args,
) -> None:
    """
    PCGrad policy update for 2 group objectives.

    Performs 3 separate backward passes:
      pass 1 – minority reward policy loss  → g1 (policy params only)
      pass 2 – majority reward policy loss  → g2 (policy params only)
      pass 3 – KL penalty + value loss      → g_kl (policy) + g_vf (value)

    Final gradient on policy params = PCGrad(g1, g2) + g_kl.
    Value params are only updated by g_vf (pass 3, unaffected by PCGrad).

    Requires gradient_accumulation_steps == 1.
    """
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "PCGrad requires gradient_accumulation_steps=1. "
            "Lower per_device_train_batch_size instead of using gradient accumulation."
        )

    unwrapped = accelerator.unwrap_model(model)
    # Only trainable params carry gradients. Under LoRA the vast majority of the
    # policy is frozen; restricting to requires_grad keeps the flattened gradient
    # vectors tiny (LoRA adapters) instead of billions of zeros.
    policy_params = [p for p in unwrapped.policy.parameters() if p.requires_grad]

    def _flat_policy_grads() -> torch.Tensor:
        parts = []
        for p in policy_params:
            if p.grad is not None:
                parts.append(p.grad.detach().clone().flatten())
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        return torch.cat(parts)

    # ── pass 1: g1 (minority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r1, retain_graph=True)
    g1_flat = _flat_policy_grads()

    # ── pass 2: g2 (majority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r2, retain_graph=True)
    g2_flat = _flat_policy_grads()

    if accelerator.is_main_process:
        _sumgrad_record_cosine(g1_flat, g2_flat)

    # ── PCGrad projection ─────────────────────────────────────────────────
    g1_proj, g2_proj = pcgrad_project_2(g1_flat, g2_flat)
    g_pcgrad = g1_proj + g2_proj      # combined projected policy gradient

    # ── pass 3: KL penalty + value loss (shared regularisers) ────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_kl + args.vf_coef * vf_loss)

    # ── add PCGrad component on top of KL gradients ───────────────────────
    offset = 0
    for p in policy_params:
        n = p.numel()
        g_p = g_pcgrad[offset: offset + n].view_as(p).to(p.dtype)
        if p.grad is not None:
            p.grad.add_(g_p)
        else:
            p.grad = g_p.clone()
        offset += n

    optimizer.step()
    optimizer.zero_grad()


def _sumgrad_compute_and_apply(
    model,
    accelerator,
    optimizer,
    pg_loss_kl: torch.Tensor,
    pg_loss_r1: torch.Tensor,
    pg_loss_r2: torch.Tensor,
    vf_loss: torch.Tensor,
    args,
) -> None:
    """
    SumGrad policy update for 2 group objectives.

    Identical machinery to `_pcgrad_compute_and_apply` (3 separate backward
    passes over the policy) EXCEPT the two group gradients are combined WITHOUT
    the PCGrad conflict-projection:

      pass 1 – minority reward policy loss  → g1 (policy params only)
      pass 2 – majority reward policy loss  → g2 (policy params only)
      pass 3 – KL penalty + value loss      → g_kl (policy) + g_vf (value)

    The two group gradients are combined into `g_combined`:
      - default (``_sumgrad_config["normalize"]`` False): plain vector sum
        ``g1 + g2`` — the geometric (vector) sum of the two objectives'
        gradients, as requested by the reviewer.
      - normalized (``_sumgrad_config["normalize"]`` True): mean-norm
        rescaling ``s·(g1/‖g1‖) + s·(g2/‖g2‖)`` with ``s = (‖g1‖+‖g2‖)/2``,
        which equalizes each objective's relative contribution while keeping
        the combined magnitude comparable to the plain sum (so kl_coef / LR
        stay roughly valid).  Only the two group gradients are rescaled; the
        KL / value gradients (pass 3) are left at their raw scale.

    Final gradient on policy params = g_combined + g_kl.  No conflicting
    component is removed.  Value params are only updated by g_vf (pass 3).

    NOTE: because back-propagation is linear, (g1 + g2) is mathematically
    equivalent to a single backward pass over (pg_loss_r1 + pg_loss_r2).  We
    still compute the two gradients separately so that SumGrad shares the exact
    same code path as PCGrad and differs only in the combination step, making
    the two methods directly comparable.

    Requires gradient_accumulation_steps == 1.
    """
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "SumGrad requires gradient_accumulation_steps=1. "
            "Lower per_device_train_batch_size instead of using gradient accumulation."
        )

    unwrapped = accelerator.unwrap_model(model)
    # Only trainable params carry gradients (LoRA-friendly; see PCGrad note above).
    policy_params = [p for p in unwrapped.policy.parameters() if p.requires_grad]

    def _flat_policy_grads() -> torch.Tensor:
        parts = []
        for p in policy_params:
            if p.grad is not None:
                parts.append(p.grad.detach().clone().flatten())
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        return torch.cat(parts)

    # ── pass 1: g1 (minority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r1, retain_graph=True)
    g1_flat = _flat_policy_grads()

    # ── pass 2: g2 (majority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r2, retain_graph=True)
    g2_flat = _flat_policy_grads()

    # Record minority/majority gradient cosine (conflict measure) before combining.
    if accelerator.is_main_process:
        _sumgrad_record_cosine(g1_flat, g2_flat)

    # ── SumGrad combination (NO projection) ───────────────────────────────
    if _sumgrad_config["normalize"]:
        # Mean-norm rescaling: equalize each objective's relative contribution
        # while preserving overall step magnitude.
        n1 = g1_flat.norm().clamp(min=1e-12)
        n2 = g2_flat.norm().clamp(min=1e-12)
        s = 0.5 * (n1 + n2)
        g_combined = s * (g1_flat / n1) + s * (g2_flat / n2)
    else:
        # Plain vector sum — the default SumGrad update.
        g_combined = g1_flat + g2_flat

    # ── pass 3: KL penalty + value loss (shared regularisers) ────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_kl + args.vf_coef * vf_loss)

    # ── add combined group gradient on top of KL gradients ────────────────
    offset = 0
    for p in policy_params:
        n = p.numel()
        g_p = g_combined[offset: offset + n].view_as(p).to(p.dtype)
        if p.grad is not None:
            p.grad.add_(g_p)
        else:
            p.grad = g_p.clone()
        offset += n

    optimizer.step()
    optimizer.zero_grad()


def _mgda_solve_2(g1: torch.Tensor, g2: torch.Tensor) -> float:
    """
    Closed-form MGDA weight for exactly 2 objectives (Sener & Koltun 2018, Sec. 3.1;
    Désidéri 2012). Returns alpha in [0, 1] minimising ||alpha*g1 + (1-alpha)*g2||^2,
    i.e. the coefficient of the minimum-norm point in the convex hull of {g1, g2}.

    Setting the derivative to zero gives
        alpha* = ((g2 - g1)^T g2) / ||g1 - g2||^2
    clipped to [0, 1] (the paper's three-case rule: the interior solution, or a
    vertex g1 / g2 when the unclipped optimum falls outside [0, 1]). g1, g2 are
    1-D flattened gradient vectors. The combined direction is
    d = alpha*g1 + (1-alpha)*g2.
    """
    diff = g1 - g2
    denom = diff.pow(2).sum().clamp(min=1e-12)          # ||g1 - g2||^2
    alpha = (torch.dot(g2 - g1, g2) / denom).item()     # ((g2-g1)^T g2) / ||g1-g2||^2
    # Clip to [0, 1] — outside this range the min-norm point is a hull vertex.
    if alpha != alpha:      # NaN guard (g1 == g2 exactly)
        return 0.5
    return float(min(1.0, max(0.0, alpha)))


def _mgda_compute_and_apply(
    model,
    accelerator,
    optimizer,
    pg_loss_kl: torch.Tensor,
    pg_loss_r1: torch.Tensor,
    pg_loss_r2: torch.Tensor,
    vf_loss: torch.Tensor,
    args,
) -> None:
    """
    MGDA policy update for 2 group objectives (arXiv:1810.04650).

    Identical 3-backward-pass machinery to `_pcgrad_compute_and_apply` /
    `_sumgrad_compute_and_apply`, differing ONLY in how the two group gradients are
    combined: MGDA steps along the minimum-norm point of their convex hull.

      pass 1 – minority reward policy loss  → g1 (policy params only)
      pass 2 – majority reward policy loss  → g2 (policy params only)
      pass 3 – KL penalty + value loss      → g_kl (policy) + g_vf (value)

    Combination (this is the MGDA-specific part):
      - default (``_mgda_config["normalize"]`` False): solve
        alpha = argmin_{a in [0,1]} ||a*g1 + (1-a)*g2||^2 on the raw gradients and
        set g_combined = alpha*g1 + (1-alpha)*g2. Because this is a CONVEX
        combination, ||g_combined|| <= max(||g1||, ||g2||): MGDA's step is
        typically SMALLER than SumGrad's (g1 + g2). The descent *direction* is what
        MGDA guarantees, so raising --learning_rate (or lowering --kl_coef's
        relative weight) may be needed vs. sumgrad.
      - normalized (``_mgda_config["normalize"]`` True): mean-norm rescale first —
        s = (||g1|| + ||g2||)/2, ĝ_k = s * g_k/||g_k|| — then solve MGDA on
        (ĝ1, ĝ2) and set g_combined = alpha*ĝ1 + (1-alpha)*ĝ2. This makes the
        convex-hull weighting scale-invariant across the two RMs and keeps the
        combined magnitude near a single gradient's, so kl_coef / lr stay roughly
        valid. Recommended when the minority/majority RMs differ in reward scale.

    Final gradient on policy params = g_combined + g_kl. Value params are updated
    only by g_vf (pass 3, unaffected by MGDA). cos(g1, g2) is recorded every update
    (when --sumgrad_cosine_csv is set) using the same diagnostic as SumGrad, so the
    gradient-conflict trajectory is captured for MGDA runs too.

    Requires gradient_accumulation_steps == 1 (shared with PCGrad/SumGrad).
    """
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "MGDA requires gradient_accumulation_steps=1. "
            "Lower per_device_train_batch_size instead of using gradient accumulation."
        )

    unwrapped = accelerator.unwrap_model(model)
    # Only trainable params carry gradients (LoRA-friendly; see PCGrad note above).
    policy_params = [p for p in unwrapped.policy.parameters() if p.requires_grad]

    def _flat_policy_grads() -> torch.Tensor:
        parts = []
        for p in policy_params:
            if p.grad is not None:
                parts.append(p.grad.detach().clone().flatten())
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        return torch.cat(parts)

    # ── pass 1: g1 (minority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r1, retain_graph=True)
    g1_flat = _flat_policy_grads()

    # ── pass 2: g2 (majority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r2, retain_graph=True)
    g2_flat = _flat_policy_grads()

    # Record minority/majority gradient cosine (conflict measure) before combining.
    if accelerator.is_main_process:
        _sumgrad_record_cosine(g1_flat, g2_flat)

    # ── MGDA combination: min-norm point of conv{g1, g2} ─────────────────
    if _mgda_config["normalize"]:
        # Mean-norm rescale so the convex-hull weighting is scale-invariant.
        n1 = g1_flat.norm().clamp(min=1e-12)
        n2 = g2_flat.norm().clamp(min=1e-12)
        s = 0.5 * (n1 + n2)
        g1_use = s * (g1_flat / n1)
        g2_use = s * (g2_flat / n2)
    else:
        g1_use, g2_use = g1_flat, g2_flat

    alpha = _mgda_solve_2(g1_use, g2_use)               # in [0, 1]
    g_combined = alpha * g1_use + (1.0 - alpha) * g2_use

    # ── pass 3: KL penalty + value loss (shared regularisers) ────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_kl + args.vf_coef * vf_loss)

    # ── add MGDA-combined group gradient on top of KL gradients ──────────
    offset = 0
    for p in policy_params:
        n = p.numel()
        g_p = g_combined[offset: offset + n].view_as(p).to(p.dtype)
        if p.grad is not None:
            p.grad.add_(g_p)
        else:
            p.grad = g_p.clone()
        offset += n

    optimizer.step()
    optimizer.zero_grad()


def _windowed_pcgrad_compute_and_apply(
    model,
    accelerator,
    optimizer,
    pg_loss_kl: torch.Tensor,
    pg_loss_r1: torch.Tensor,
    pg_loss_r2: torch.Tensor,
    vf_loss: torch.Tensor,
    args,
) -> None:
    """
    Windowed-PCGrad policy update (reviewer W4: transient cause vs. persistent effect).

    Identical 3-backward-pass machinery to `_pcgrad_compute_and_apply` /
    `_sumgrad_compute_and_apply`, but the two group gradients are combined by a
    rule that depends on the CURRENT PPO update index (`_pcgrad_state["update"]`,
    1-indexed, equal to the reported global_step):

      - inside  [_pcgrad_window["start"], _pcgrad_window["end"]] (inclusive):
        PCGrad conflict-projection  (g1_proj + g2_proj) — gradient surgery ON.
      - outside the window:
        plain vector sum (g1 + g2)  — identical to SumGrad / surgery OFF.

    A `None` bound means "unbounded" on that side (start=None → from step 1;
    end=None → until the final step). This tests the effect of intervention
    timing; the outcome alone does not establish irreversibility or a mechanism.

    cos(g1, g2) is recorded every update (when --sumgrad_cosine_csv is set) so the
    conflict trajectory — and the effect of turning surgery on/off — is captured.

    Requires gradient_accumulation_steps == 1 (shared with PCGrad/SumGrad).
    """
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "Windowed PCGrad requires gradient_accumulation_steps=1. "
            "Lower per_device_train_batch_size instead of using gradient accumulation."
        )

    unwrapped = accelerator.unwrap_model(model)
    # Only trainable params carry gradients (LoRA-friendly; see PCGrad note above).
    policy_params = [p for p in unwrapped.policy.parameters() if p.requires_grad]

    def _flat_policy_grads() -> torch.Tensor:
        parts = []
        for p in policy_params:
            if p.grad is not None:
                parts.append(p.grad.detach().clone().flatten())
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        return torch.cat(parts)

    # ── pass 1: g1 (minority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r1, retain_graph=True)
    g1_flat = _flat_policy_grads()

    # ── pass 2: g2 (majority reward, no KL) ──────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r2, retain_graph=True)
    g2_flat = _flat_policy_grads()

    # Record minority/majority gradient cosine (conflict measure) before combining.
    if accelerator.is_main_process:
        _sumgrad_record_cosine(g1_flat, g2_flat)

    # ── window decision: surgery ON only inside [start, end] ─────────────
    update = _pcgrad_state.get("update", 0)
    start = _pcgrad_window["start"]
    end = _pcgrad_window["end"]
    in_window = (start is None or update >= start) and (end is None or update <= end)

    if in_window:
        # Gradient surgery ON — PCGrad conflict projection.
        g1_proj, g2_proj = pcgrad_project_2(g1_flat, g2_flat)
        g_combined = g1_proj + g2_proj
    else:
        # Gradient surgery OFF — plain vector sum (== SumGrad).
        g_combined = g1_flat + g2_flat

    # ── pass 3: KL penalty + value loss (shared regularisers) ────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_kl + args.vf_coef * vf_loss)

    # ── add combined group gradient on top of KL gradients ────────────────
    offset = 0
    for p in policy_params:
        n = p.numel()
        g_p = g_combined[offset: offset + n].view_as(p).to(p.dtype)
        if p.grad is not None:
            p.grad.add_(g_p)
        else:
            p.grad = g_p.clone()
        offset += n

    optimizer.step()
    optimizer.zero_grad()


def _deconflict_compute_and_apply(
    model,
    accelerator,
    optimizer,
    pg_loss: torch.Tensor,       # pooled vanilla PPO policy loss (reward + KL) → g_single
    pg_loss_r1: torch.Tensor,    # minority guard policy loss (pure r1, KL-free) → g1
    vf_loss: torch.Tensor,
    args,
) -> None:
    """
    Vanilla deconflict-only update (reviewer W4, VANILLA setting — Fig. 3 quantity).

    Fig. 3 plots cos(g_single, g1), the cosine between the pooled vanilla PPO update
    direction (g_single) and the minority-objective gradient (g1); it is negative
    early (the update fights the minority objective) then returns positive. This
    update neutralises that conflict ONLY while the current PPO update index
    (_pcgrad_state["update"], 1-indexed == global_step) lies inside
    [_pcgrad_window["start"], _pcgrad_window["end"]]:

      in window & dot(g_single, g1) < 0:
          g_update = g_single - (dot / ||g1||^2) * g1   # drop the conflicting component
      otherwise:
          g_update = g_single                            # unchanged (standard vanilla)

    The minority RM is used ONLY as a guard: g1 is never added as an objective (no
    new optimisation pressure toward minority), so this isolates "remove the
    conflict" — the cleanest causal test of W4. With the window empty every update
    is unchanged, i.e. bit-for-bit standard vanilla PPO.

    Three backward passes over the shared graph (like PCGrad/SumGrad):
      pass 1 – pooled PPO policy loss → g_single (includes KL via pooled advantages)
      pass 2 – minority guard policy loss → g1
      pass 3 – value loss → value-model grads (policy grads ~0)
    Final policy grad = g_update; value params updated by vf only.

    cos(g_single, g1) is recorded every update (when --sumgrad_cosine_csv is set) —
    a training-batch, KL-inclusive counterpart of Fig. 3 that also shows the
    intervention's effect. Requires gradient_accumulation_steps == 1.
    """
    if args.gradient_accumulation_steps != 1:
        raise ValueError(
            "Vanilla deconflict requires gradient_accumulation_steps=1. "
            "Lower per_device_train_batch_size instead of using gradient accumulation."
        )

    unwrapped = accelerator.unwrap_model(model)
    # Only trainable params carry gradients (LoRA-friendly; see PCGrad note above).
    policy_params = [p for p in unwrapped.policy.parameters() if p.requires_grad]

    def _flat_policy_grads() -> torch.Tensor:
        parts = []
        for p in policy_params:
            if p.grad is not None:
                parts.append(p.grad.detach().clone().flatten())
            else:
                parts.append(torch.zeros(p.numel(), device=p.device, dtype=torch.float32))
        return torch.cat(parts)

    # ── pass 1: g_single (pooled PPO update, incl. KL via pooled advantages) ──
    optimizer.zero_grad()
    accelerator.backward(pg_loss, retain_graph=True)
    g_single = _flat_policy_grads()

    # ── pass 2: g1 (minority guard, no KL) ───────────────────────────────────
    optimizer.zero_grad()
    accelerator.backward(pg_loss_r1, retain_graph=True)
    g1_flat = _flat_policy_grads()

    # Record cos(g_single, g1) — the Fig. 3 quantity on the training batch.
    if accelerator.is_main_process:
        _sumgrad_record_cosine(g_single, g1_flat)

    # ── deconflict decision: only inside the window, only when conflicting ────
    update = _pcgrad_state.get("update", 0)
    start = _pcgrad_window["start"]
    end = _pcgrad_window["end"]
    in_window = (start is None or update >= start) and (end is None or update <= end)

    g_update = g_single
    if in_window:
        dot = torch.dot(g_single, g1_flat)
        if dot < 0:
            # Project the vanilla update to remove its component opposing g1.
            g_update = g_single - (dot / g1_flat.norm().pow(2).clamp(min=1e-12)) * g1_flat

    # ── pass 3: value loss (value-model params only) ─────────────────────────
    optimizer.zero_grad()
    accelerator.backward(args.vf_coef * vf_loss)

    # ── write the (possibly projected) policy update on top of value grads ────
    offset = 0
    for p in policy_params:
        n = p.numel()
        g_p = g_update[offset: offset + n].view_as(p).to(p.dtype)
        if p.grad is not None:
            p.grad.add_(g_p)
        else:
            p.grad = g_p.clone()
        offset += n

    optimizer.step()
    optimizer.zero_grad()


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



class GradientCosineDiagnosticCallback(TrainerCallback):
    """
    Calls the offline gradient-cosine diagnostic on the live in-memory vanilla
    policy every `diag_every` PPO update steps.

    The diagnostic generates responses from the current policy, scores them with
    r_single / r1 / r2, computes per-reward gradients, and records pairwise cosine
    similarities.  KL (beta) applies only to g_single; g1/g2 are pure group-reward
    gradients (beta=0, no π_ref).  It does NOT touch the PPO optimizer, does NOT call trainer.step(),
    and zeros policy gradients before and after so that PPO training is unaffected.

    Results are appended to a CSV file and, when wandb is active, logged as
    scalars under the "diag/" prefix.
    """

    def __init__(
        self,
        *,
        policy,
        accelerator,
        ref_policy,
        reward_single,
        reward_single_tok,
        reward_r1,
        reward_r1_tok,
        reward_r2,
        reward_r2_tok,
        prompt_ids,
        prompt_mask,
        alpha: float,
        beta: float,
        diag_every: int,
        batch_size: int,
        max_new_tokens: int,
        reward_max_length: int,
        tokenizer,
        csv_path: str,
        device,
    ):
        super().__init__()
        self.policy = policy
        self.accelerator = accelerator
        self.ref_policy = ref_policy
        self.reward_single = reward_single
        self.reward_single_tok = reward_single_tok
        self.reward_r1 = reward_r1
        self.reward_r1_tok = reward_r1_tok
        self.reward_r2 = reward_r2
        self.reward_r2_tok = reward_r2_tok
        self.prompt_ids = prompt_ids
        self.prompt_mask = prompt_mask
        self.alpha = alpha
        self.beta = beta
        self.diag_every = diag_every
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.reward_max_length = reward_max_length
        self.tokenizer = tokenizer
        self.csv_path = csv_path
        self.device = device

        # Lazily import the diagnostic helper to avoid circular imports at module load
        from src.imdb.diagnostics.gradient_cosine_diagnostic import (
            run_online_diagnostic,
            ONLINE_DIAG_FIELDS,
        )
        self._run_online_diagnostic = run_online_diagnostic
        self._fields = ONLINE_DIAG_FIELDS

        # Prepare CSV: create parent dirs, write header if file is new/empty
        import os, csv as _csv
        os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
        self._csv_path = csv_path
        self._fields_ref = ONLINE_DIAG_FIELDS
        needs_header = (
            not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
        )
        if needs_header:
            with open(csv_path, "w", newline="") as fh:
                _csv.DictWriter(fh, fieldnames=ONLINE_DIAG_FIELDS).writeheader()

    def _write_csv(self, row: dict) -> None:
        import csv as _csv
        with open(self._csv_path, "a", newline="") as fh:
            _csv.DictWriter(fh, fieldnames=self._fields_ref).writerow(row)

    @staticmethod
    def _use_wandb() -> bool:
        try:
            import wandb
            return wandb.run is not None
        except ImportError:
            return False

    def on_step_end(self, args, state, control, **kwargs):
        if self.diag_every <= 0 or state.global_step % self.diag_every != 0:
            return control

        unwrapped = self.accelerator.unwrap_model(self.policy)
        row = self._run_online_diagnostic(
            policy=unwrapped,
            ref_policy=self.ref_policy,
            reward_single=self.reward_single,
            reward_single_tok=self.reward_single_tok,
            reward_r1=self.reward_r1,
            reward_r1_tok=self.reward_r1_tok,
            reward_r2=self.reward_r2,
            reward_r2_tok=self.reward_r2_tok,
            prompt_ids=self.prompt_ids,
            prompt_mask=self.prompt_mask,
            alpha=self.alpha,
            step=state.global_step,
            beta=self.beta,
            max_new_tokens=self.max_new_tokens,
            batch_size=self.batch_size,
            reward_max_length=self.reward_max_length,
            device=self.device,
            tokenizer=self.tokenizer,
        )

        self._write_csv(row)

        if self._use_wandb():
            import wandb
            wandb.log(
                {f"diag/{k}": v for k, v in row.items() if k not in ("alpha",)},
                step=state.global_step,
            )

        return control


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
from src.helper.model_utils import (
    configure_tokenizer,
    build_lora_config,
    default_lora_target_modules,
    add_lora_cli_args,
    resolve_use_lora,
    maybe_bnb_config,
)


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


def prepare_get_sharedrep_reward(policy_tokenizer, reward_tokenizer, max_length: int = 512):
    """
    Returns a get_reward-compatible function that re-tokenizes query_responses when
    policy and reward tokenizers differ.
    """
    need_alignment = len(policy_tokenizer) != len(reward_tokenizer)

    def get_sharedrep_reward_aligned(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(model, SharedRepGPT2RM) and need_alignment:
            qr, rpad, ctx = _align_query_responses_to_reward_tokenizer(
                policy_tokenizer,
                reward_tokenizer,
                query_responses,
                pad_token_id,
                context_length,
                query_responses.device,
                max_length=max_length,
            )
            return get_sharedrep_reward(model, qr, rpad, ctx)
        return get_sharedrep_reward(model, query_responses, pad_token_id, context_length)

    return get_sharedrep_reward_aligned


def _align_query_responses_to_reward_tokenizer(
    policy_tokenizer,
    reward_tokenizer,
    query_responses: torch.Tensor,
    pad_token_id: int,
    context_length: int,
    device,
    max_length: int = 512,
) -> tuple[torch.Tensor, int, int]:
    """
    Re-tokenize query_responses (policy token space) into reward tokenizer space.
    Used when policy and reward model use different tokenizers (e.g. Qwen policy + GPT-2 RM).
    Returns (aligned_query_responses, reward_pad_token_id, 0) for _original_get_reward.
    """
    texts = policy_tokenizer.batch_decode(query_responses, skip_special_tokens=True)
    encoded = reward_tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
        pad_to_multiple_of=1,
    ).to(device)
    reward_pad_id = reward_tokenizer.pad_token_id
    if reward_pad_id is None:
        reward_pad_id = reward_tokenizer.eos_token_id
    return encoded["input_ids"], reward_pad_id, 0


def get_vanilla_reward(model: torch.nn.Module, query_responses: torch.Tensor, pad_token_id: int, context_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vanilla (single RM) reward: use TRL's get_reward then apply z-score normalization if enabled.
    """
    reward_logits, final_rewards, sequence_lengths = _original_get_reward(model, query_responses, pad_token_id, context_length)
    if _reward_normalization["enabled"] and _reward_normalization["vanilla"]["mean"] is not None:
        final_rewards = _apply_reward_normalization(final_rewards)
    return reward_logits, final_rewards, sequence_lengths


def prepare_get_vanilla_reward(policy_tokenizer, reward_tokenizer, max_length: int = 512):
    """
    Returns a get_reward-compatible function that re-tokenizes query_responses when
    policy and reward tokenizers differ (e.g. Qwen policy + GPT-2 RM).
    Avoids CUDA index-out-of-bounds when policy token IDs exceed reward model vocab size.
    """
    policy_vocab_size = len(policy_tokenizer)
    reward_vocab_size = len(reward_tokenizer)
    need_alignment = policy_vocab_size != reward_vocab_size

    def get_vanilla_reward_aligned(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Value model (num_labels==1) uses policy tokenizer; no alignment needed
        if getattr(model.config, "num_labels", 0) == 1:
            return get_vanilla_reward(model, query_responses, pad_token_id, context_length)
        # Reward model: align if tokenizers differ
        if need_alignment:
            qr, rpad, ctx = _align_query_responses_to_reward_tokenizer(
                policy_tokenizer,
                reward_tokenizer,
                query_responses,
                pad_token_id,
                context_length,
                query_responses.device,
                max_length=max_length,
            )
            reward_logits, final_rewards, sequence_lengths = _original_get_reward(model, qr, rpad, ctx)
        else:
            reward_logits, final_rewards, sequence_lengths = _original_get_reward(
                model, query_responses, pad_token_id, context_length
            )
        if _reward_normalization["enabled"] and _reward_normalization["vanilla"]["mean"] is not None:
            final_rewards = _apply_reward_normalization(final_rewards)
        return reward_logits, final_rewards, sequence_lengths

    return get_vanilla_reward_aligned


def prepare_get_vanilla_deconflict_reward(
    policy_tokenizer,
    pooled_reward_model,
    pooled_reward_tokenizer,
    minority_reward_model,
    minority_reward_tokenizer,
    max_length: int = 512,
):
    """
    Vanilla PPO reward with a MINORITY-objective guard (reviewer W4, vanilla setting).

    The scalar reward that drives the PPO update, the value-model target, and the
    logged reward is the POOLED vanilla RM — identical to standard vanilla PPO
    (via prepare_get_vanilla_reward). In addition, on each rollout chunk the
    MINORITY group RM is scored out-of-band (no grad) and its per-sample rewards
    are appended to _pcgrad_state["scores_r1_list"], so the patched train loop can
    build a minority advantage stream and _deconflict_compute_and_apply can project
    the vanilla update away from the minority gradient g1 during the window.

    The guard scoring never touches the scalar reward, so when the intervention
    window is empty this is bit-for-bit standard vanilla PPO. The pooled reward
    model is identified by object identity (robust to num_labels collisions with
    the value model), so the minority RM is scored exactly once per rollout chunk.
    """
    pooled_fn = prepare_get_vanilla_reward(
        policy_tokenizer, pooled_reward_tokenizer, max_length=max_length
    )
    need_align_min = len(policy_tokenizer) != len(minority_reward_tokenizer)

    def get_vanilla_deconflict_reward(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        result = pooled_fn(model, query_responses, pad_token_id, context_length)
        # Only the pooled reward-model call scores the minority guard (the value
        # model call, and any other, is left untouched).
        if model is pooled_reward_model and _pcgrad_state["scores_r1_list"] is not None:
            with torch.no_grad():
                if need_align_min:
                    qr, rpad, ctx = _align_query_responses_to_reward_tokenizer(
                        policy_tokenizer,
                        minority_reward_tokenizer,
                        query_responses,
                        pad_token_id,
                        context_length,
                        query_responses.device,
                        max_length=max_length,
                    )
                    _, min_scores, _ = _original_get_reward(minority_reward_model, qr, rpad, ctx)
                else:
                    _, min_scores, _ = _original_get_reward(
                        minority_reward_model, query_responses, pad_token_id, context_length
                    )
            _pcgrad_state["scores_r1_list"].append(min_scores.detach())
        return result

    return get_vanilla_deconflict_reward


# ---------------------------------------------------------------------------
# PCGrad reward function
# ---------------------------------------------------------------------------

def get_pcgrad_reward(
    model: torch.nn.Module,
    query_responses: torch.Tensor,
    pad_token_id: int,
    context_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    PCGrad reward function using the MaxMinRewardWrapper to compute BOTH group
    rewards.  Stores them in _pcgrad_state for later use by the PCGrad patch.
    Returns the per-sample AVERAGE reward as the scalar score so that TRL's
    standard value-model target and logging paths keep working normally.

    Falls back to _original_get_reward for the value model (num_labels == 1).
    """
    if isinstance(model, MaxMinRewardWrapper):
        attention_mask = query_responses != pad_token_id
        position_ids = attention_mask.cumsum(1) - attention_mask.long()
        input_ids = torch.masked_fill(query_responses, ~attention_mask, 0)

        with torch.no_grad():
            rewards = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        sequence_lengths = (
            first_true_indices(query_responses[:, context_length:] == pad_token_id)
            - 1 + context_length
        )
        reward_logits = rewards.logits   # (batch, seq, 2)
        batch_idx = torch.arange(reward_logits.size(0), device=reward_logits.device)
        all_group_rewards = reward_logits[batch_idx, sequence_lengths]  # (batch, 2)

        all_group_rewards = _apply_reward_normalization_maxmin(all_group_rewards)

        scores_r1 = all_group_rewards[:, 0]  # minority
        scores_r2 = all_group_rewards[:, 1]  # majority

        # Accumulate per-chunk rewards in the side channel
        if _pcgrad_state["scores_r1_list"] is not None:
            _pcgrad_state["scores_r1_list"].append(scores_r1.detach())
            _pcgrad_state["scores_r2_list"].append(scores_r2.detach())

        # Return average reward for TRL's value-model training and reward logging
        avg_scores = (scores_r1 + scores_r2) * 0.5
        return reward_logits, avg_scores, sequence_lengths

    else:
        return _original_get_reward(model, query_responses, pad_token_id, context_length)


def prepare_get_pcgrad_reward(
    policy_tokenizer,
    reward_tokenizer,
    max_length: int = 512,
):
    """
    Returns a get_reward-compatible closure that re-tokenises query_responses
    when the policy and reward tokenisers differ (same as prepare_get_maxmin_reward).
    """
    policy_vocab_size = len(policy_tokenizer)
    reward_vocab_size = len(reward_tokenizer)
    need_alignment = policy_vocab_size != reward_vocab_size

    def get_pcgrad_reward_aligned(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(model, MaxMinRewardWrapper) and need_alignment:
            qr, rpad, ctx = _align_query_responses_to_reward_tokenizer(
                policy_tokenizer,
                reward_tokenizer,
                query_responses,
                pad_token_id,
                context_length,
                query_responses.device,
                max_length=max_length,
            )
            return get_pcgrad_reward(model, qr, rpad, ctx)
        return get_pcgrad_reward(model, query_responses, pad_token_id, context_length)

    return get_pcgrad_reward_aligned


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


def prepare_get_maxmin_reward(policy_tokenizer, reward_tokenizer, max_length: int = 512):
    """
    Returns a get_reward-compatible function that re-tokenizes query_responses when
    policy and reward tokenizers differ (e.g. Qwen policy + GPT-2 RM).
    """
    policy_vocab_size = len(policy_tokenizer)
    reward_vocab_size = len(reward_tokenizer)
    need_alignment = policy_vocab_size != reward_vocab_size

    def get_maxmin_reward_aligned(
        model: torch.nn.Module,
        query_responses: torch.Tensor,
        pad_token_id: int,
        context_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if isinstance(model, MaxMinRewardWrapper) and need_alignment:
            qr, rpad, ctx = _align_query_responses_to_reward_tokenizer(
                policy_tokenizer,
                reward_tokenizer,
                query_responses,
                pad_token_id,
                context_length,
                query_responses.device,
                max_length=max_length,
            )
            return get_maxmin_reward(model, qr, rpad, ctx)
        return get_maxmin_reward(model, query_responses, pad_token_id, context_length)

    return get_maxmin_reward_aligned


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
    parser.add_argument("--method", type=str, required=True,
                        choices=["sharedrep", "maxmin", "gold", "gold_vanilla", "vanilla", "pcgrad", "sumgrad", "mgda"],
                        help="Policy optimization method: 'sharedrep', 'maxmin', 'gold', 'gold_vanilla', 'vanilla', "
                             "'pcgrad', 'sumgrad', or 'mgda' (MGDA min-norm multi-objective, arXiv:1810.04650)")
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
                        help="Base model for policy/ref (default: gpt2). Use 'lvwerra/gpt2-imdb' for the "
                             "IMDB-pretrained GPT-2, 'EleutherAI/pythia-160m' for the GPT-2-scale Pythia backbone, "
                             "'meta-llama/Llama-3.2-1B' for a mid-size Llama backbone, or 'Qwen/Qwen3-4B-Base' "
                             "for the larger Qwen3 backbone.")
    parser.add_argument("--value_model_name", type=str, default=None,
                        help="Base model for the value head. Defaults to --policy_model_name. "
                             "Tip: with a large LoRA policy (e.g. Qwen3-4B) set a smaller SAME-tokenizer model "
                             "(e.g. Qwen/Qwen3-0.6B-Base) so the full-fine-tuned value model is cheap.")
    parser.add_argument("--policy_hub_id", type=str, default=None,
                        help="Override policy hub ID for saving. If omitted, derived from method/user_id/seed/proportion/alpha.")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing on policy/value to cut activation memory (recommended for 4B).")
    parser.add_argument("--per_device_train_batch_size", type=int, default=None,
                        help="Base per-device PPO batch size for vanilla/maxmin (pcgrad/sumgrad/mgda use 2x with grad_accum=1 "
                             "to match effective batch). Default None -> 32 (legacy). Lower (e.g. 8) to fit a 4B policy on 32GB.")
    parser.add_argument("--load_in_4bit_frozen", action="store_true",
                        help="QLoRA: load the frozen reward model (and full-FT ref model) in NF4 4-bit to save VRAM. "
                             "Requires bitsandbytes>=0.45 (Blackwell/RTX 5090).")
    parser.add_argument(
        "--local_policy_output_dir",
        type=str,
        default=None,
        help=(
            "Local directory for saving the trained policy before Hub upload. "
            "Defaults to local_models/<policy_hub_id with / replaced by __>."
        ),
    )
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL penalty coefficient (default 0.1). Set to 0 for ablation (no KL penalty; may cause policy collapse).")
    parser.add_argument("--total_episodes", type=int, default=50000,
                        help="Total PPO episodes (controls num_total_batches ≈ total_episodes/batch_size). "
                             "Lower for quick diagnostic/pre-check runs (e.g. 10000 ≈ 156 updates).")
    parser.add_argument("--learning_rate", type=float, default=3e-6,
                        help="PPO policy/value learning rate. Default 3e-6 (legacy). The standard TRL "
                             "sentiment-PPO recipe uses ~1.4e-5; raise this if the reward (objective/scores) "
                             "stays flat (policy not learning).")
    parser.add_argument("--no_push", action="store_true",
                        help="Skip all Hub pushes and local model saving (diagnostic/pre-check runs). "
                             "Diagnostic CSVs are still written during training.")
    parser.add_argument(
        "--sumgrad_normalize",
        action="store_true",
        help=(
            "SumGrad only: mean-norm rescale the two group gradients before summing "
            "(equalizes each objective's relative contribution). Default: plain vector sum. "
            "Note: changes step magnitude little, but re-tuning kl_coef/lr may still help."
        ),
    )
    parser.add_argument(
        "--sumgrad_cosine_csv",
        type=str,
        default=None,
        help=(
            "SumGrad/MGDA only: path to a CSV that records cos(g1, g2) between the minority "
            "and majority objective gradients at every policy update (gradient-conflict "
            "diagnostic). Disabled when unset."
        ),
    )
    parser.add_argument(
        "--mgda_normalize",
        action="store_true",
        help=(
            "MGDA only: mean-norm rescale the two group gradients BEFORE the min-norm "
            "convex-hull solve (makes the mixing weight scale-invariant across the "
            "minority/majority RMs and keeps the step magnitude near a single gradient's). "
            "Default: solve on raw gradients (faithful to arXiv:1810.04650; note the "
            "convex-combination step is smaller than sumgrad's, so lr/kl_coef may need "
            "re-tuning)."
        ),
    )
    parser.add_argument(
        "--vanilla_deconflict",
        action="store_true",
        help=(
            "Vanilla method only (reviewer W4, Fig. 3 setting): load the minority group "
            "reward model as a GUARD and, inside the intervention window, project the "
            "pooled vanilla PPO update to remove its component that conflicts with the "
            "minority gradient g1 (deconflict-only; g1 is never added as an objective). "
            "Outside the window the update is standard vanilla. "
            "Combine with --pcgrad_window_start/--pcgrad_window_end "
            "for early/late windows; omit both for deconflict on every step. The guard RM "
            "defaults to the maxmin minority RM (override via --maxmin_minority_rm_hub_id)."
        ),
    )
    parser.add_argument(
        "--pcgrad_window_start",
        type=int,
        default=None,
        help=(
            "Windowed intervention (pcgrad, or vanilla with --vanilla_deconflict): "
            "first PPO update (1-indexed, "
            "inclusive; == global_step) at which PCGrad conflict-projection is applied. "
            "Outside [start, end], pcgrad plain-sums the two group gradients "
            "(SumGrad); vanilla deconfliction retains the pooled gradient. "
            "Omit for unbounded-below (from step 1). Setting either "
            "--pcgrad_window_start or --pcgrad_window_end enables windowed mode."
        ),
    )
    parser.add_argument(
        "--pcgrad_window_end",
        type=int,
        default=None,
        help=(
            "Windowed intervention (pcgrad, or vanilla with --vanilla_deconflict): "
            "last PPO update (inclusive) eligible for projection. "
            "Omit for unbounded-above (until the final step)."
        ),
    )
    parser.add_argument("--kl_div", type=str, default="mean_kl", choices=["total", "mean_kl"],
                        help=(
                            "How to apply KL penalty: 'total' = TRL default (sum of KL over tokens, length-dependent); "
                            "'mean_kl' = mean KL per token (length-invariant). Default: mean_kl."
                        ))

    add_lora_cli_args(parser)

    # ── Online gradient-cosine diagnostic (vanilla method only) ──────────
    parser.add_argument(
        "--diag_every", type=int, default=0,
        help=(
            "Run the gradient-cosine diagnostic every N PPO update steps. "
            "0 (default) disables the diagnostic entirely. Only active for --method vanilla."
        ),
    )
    parser.add_argument(
        "--diag_beta", type=float, default=0.0,
        help=(
            "KL coefficient for g_single only (matches PPO when >0). "
            "g1/g2 always use beta=0 (no π_ref). Default 0 = no KL on g_single either."
        ),
    )
    parser.add_argument(
        "--diag_max_prompts", type=int, default=128,
        help="Number of fixed prompts used for the diagnostic (sampled once from IMDB test set).",
    )
    parser.add_argument(
        "--diag_max_new_tokens", type=int, default=200,
        help="Max response tokens when generating for the diagnostic.",
    )
    parser.add_argument(
        "--diag_batch_size", type=int, default=16,
        help="Mini-batch size for diagnostic generation, scoring, and gradient accumulation.",
    )
    parser.add_argument(
        "--diag_reward_max_length", type=int, default=512,
        help="Max tokeniser length when scoring responses with reward models in the diagnostic.",
    )
    parser.add_argument(
        "--diag_output_path", type=str, default="results/online_gradient_cosine.csv",
        help="Path for the diagnostic CSV output (rows appended each time the diagnostic runs).",
    )
    parser.add_argument(
        "--diag_r1_hub_id", type=str, default=None,
        help=(
            "Hub id for group reward model r1 (minority). "
            "Defaults to {user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-minority-gpt2."
        ),
    )
    parser.add_argument(
        "--diag_r2_hub_id", type=str, default=None,
        help=(
            "Hub id for group reward model r2 (majority). "
            "Defaults to {user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-majority-gpt2."
        ),
    )

    return parser.parse_args()


def load_reward_model(method, user_id, seed, proportion, k, alpha, device, max_new_tokens=512,
                      vanilla_rm_hub_id=None, maxmin_minority_rm_hub_id=None, maxmin_majority_rm_hub_id=None, sharedrep_rm_hub_id=None,
                      rm_load_kwargs=None):
    """Load and configure the reward model based on the selected method.
    Returns (reward_model, reward_tokenizer, hub_id_or_ids).
    hub_id_or_ids: for vanilla a str (RM hub id); for maxmin a dict {"minority": id, "majority": id}; else None.
    Optional *_rm_hub_id args override the default constructed hub IDs.
    rm_load_kwargs: extra from_pretrained kwargs (e.g. torch_dtype, quantization_config)
    applied to the (frozen) reward backbone so a 4B RM can be loaded in bf16 or NF4 4-bit.
    """
    rm_load_kwargs = dict(rm_load_kwargs or {})
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
    
    elif method in ("maxmin", "pcgrad", "sumgrad", "mgda"):
        if proportion is None:
            raise ValueError(f"--proportion is required for {method} method")
        
        # Naming aligned with train_reward_models / normalize_reward_model (ALPHA-len, no seed in hub id)
        reward_model_name_majority = maxmin_majority_rm_hub_id or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-majority-gpt2"
        reward_model_majority = AutoModelForSequenceClassification.from_pretrained(reward_model_name_majority, **rm_load_kwargs)
        reward_model_majority.eval()
        fc_maj = reward_model_majority.score
        del reward_model_majority

        reward_model_name_minority = maxmin_minority_rm_hub_id or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-minority-gpt2"
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name_minority, **rm_load_kwargs)
        reward_model.eval()
        fc_min = reward_model.score

        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name_majority)
        configure_tokenizer(reward_tokenizer)
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
        reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name, **rm_load_kwargs)
        reward_model.eval()
        reward_model.to(device)

        reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name, padding_side="left")
        configure_tokenizer(reward_tokenizer, padding_side="left")
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
    elif method == "pcgrad":
        return f"{user_id}/pcgrad-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
    elif method == "sumgrad":
        return f"{user_id}/sumgrad-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
    elif method == "mgda":
        return f"{user_id}/mgda-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
    else:
        raise ValueError(f"Unknown method: {method}")


def setup_reward_function(method, policy_tokenizer, reward_tokenizer, reward_model, alpha, proportion=None,
                          deconflict_minority_model=None, deconflict_minority_tok=None):
    """Set up the reward function based on the selected method.

    Args:
        method: Training method ('sharedrep', 'maxmin', 'gold', 'gold_vanilla', 'vanilla')
        policy_tokenizer: Tokenizer for the policy model
        reward_tokenizer: Tokenizer for the reward model
        reward_model: The reward model instance
        alpha: Alpha parameter for gold reward computation
        proportion: Minority group proportion (required for gold_vanilla)
        deconflict_minority_model / deconflict_minority_tok: when set (vanilla method,
            --vanilla_deconflict), the minority group RM used as the guard for the W4
            deconfliction intervention. The scalar reward stays the pooled vanilla RM.
    """
    # Override get_reward in BOTH locations where PPOTrainer resolves it:
    # 1. trl.experimental.utils module (the source module)
    # 2. PPOTrainer's own module namespace (where `from ..utils import get_reward` placed it)
    # Both must be set to ensure the override takes effect.

    max_length = 512
    if method == "sharedrep":
        custom_fn = prepare_get_sharedrep_reward(policy_tokenizer, reward_tokenizer, max_length=max_length)
    elif method == "maxmin":
        custom_fn = prepare_get_maxmin_reward(policy_tokenizer, reward_tokenizer, max_length=max_length)
    elif method == "gold":
        # Pass alpha to gold reward function for consistent group preference computation
        custom_fn = prepare_get_gold_reward(policy_tokenizer, reward_tokenizer, alpha, max_length=max_length, norm_divisor=250)
    elif method == "gold_vanilla":
        # Oracle per-sample group scoring, no MaxMin
        custom_fn = prepare_get_gold_vanilla_reward(policy_tokenizer, reward_tokenizer, alpha, proportion, max_length=max_length, norm_divisor=250)
    elif method == "vanilla":
        if deconflict_minority_model is not None:
            # W4 deconfliction: pooled vanilla reward drives the update; minority RM is
            # scored out-of-band as the guard (stashed in _pcgrad_state for the patch).
            custom_fn = prepare_get_vanilla_deconflict_reward(
                policy_tokenizer, reward_model, reward_tokenizer,
                deconflict_minority_model, deconflict_minority_tok, max_length=max_length,
            )
        else:
            # Vanilla: use wrapper so we can apply z-score normalization when enabled
            custom_fn = prepare_get_vanilla_reward(policy_tokenizer, reward_tokenizer, max_length=max_length)
    elif method in ("pcgrad", "sumgrad", "mgda"):
        # SumGrad/MGDA reuse PCGrad's reward extraction: all need per-group rewards
        # (minority/majority) accumulated in the shared side channel _pcgrad_state.
        custom_fn = prepare_get_pcgrad_reward(policy_tokenizer, reward_tokenizer, max_length=max_length)
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

    # Apply TRL training-loop patches.
    # PCGrad and SumGrad share the same multi-gradient patch (includes optional
    # mean-KL normalisation); they differ only in how the two group gradients
    # are combined (projection vs plain sum) via the injected apply_fn.
    # All other methods use the standalone mean-KL patch when --kl_div mean_kl.
    # Vanilla deconfliction (reviewer W4, Fig. 3 setting) — only valid for vanilla.
    _vanilla_deconflict = bool(args.vanilla_deconflict)
    if _vanilla_deconflict and args.method != "vanilla":
        raise ValueError(
            "--vanilla_deconflict is only valid with --method vanilla "
            f"(got --method {args.method})."
        )
    # Windowed intervention request (either bound set). Valid for --method pcgrad,
    # or --method vanilla WITH --vanilla_deconflict.
    _windowed_requested = (
        args.pcgrad_window_start is not None or args.pcgrad_window_end is not None
    )
    if _windowed_requested and not (args.method == "pcgrad" or _vanilla_deconflict):
        raise ValueError(
            "--pcgrad_window_start/--pcgrad_window_end require --method pcgrad or "
            f"--method vanilla with --vanilla_deconflict (got --method {args.method}"
            f"{' without --vanilla_deconflict' if args.method == 'vanilla' else ''})."
        )
    if (
        args.pcgrad_window_start is not None
        and args.pcgrad_window_end is not None
        and args.pcgrad_window_start > args.pcgrad_window_end
    ):
        raise ValueError(
            f"--pcgrad_window_start ({args.pcgrad_window_start}) must be "
            f"<= --pcgrad_window_end ({args.pcgrad_window_end})."
        )

    if args.method in ("pcgrad", "sumgrad", "mgda"):
        apply_mean_kl = (args.kl_div == "mean_kl")
        apply_fn = {
            "pcgrad": _pcgrad_compute_and_apply,
            "sumgrad": _sumgrad_compute_and_apply,
            "mgda": _mgda_compute_and_apply,
        }[args.method]
        if args.method == "sumgrad":
            _sumgrad_config["normalize"] = args.sumgrad_normalize
            if args.sumgrad_cosine_csv and accelerator.is_main_process:
                _sumgrad_diag["csv_path"] = args.sumgrad_cosine_csv
        if args.method == "pcgrad" and args.sumgrad_cosine_csv and accelerator.is_main_process:
            _sumgrad_diag["csv_path"] = args.sumgrad_cosine_csv
        if args.method == "mgda":
            _mgda_config["normalize"] = args.mgda_normalize
            if args.sumgrad_cosine_csv and accelerator.is_main_process:
                _sumgrad_diag["csv_path"] = args.sumgrad_cosine_csv
        if _windowed_requested:
            # Windowed-PCGrad: projection ON only inside [start, end]; plain sum outside.
            _pcgrad_window["enabled"] = True
            _pcgrad_window["start"] = args.pcgrad_window_start
            _pcgrad_window["end"] = args.pcgrad_window_end
            apply_fn = _windowed_pcgrad_compute_and_apply
            # Reuse the SumGrad cosine diagnostic to log cos(g1, g2) each update
            # (the conflict trajectory + on/off effect). Recommended for windowed runs.
            if args.sumgrad_cosine_csv and accelerator.is_main_process:
                _sumgrad_diag["csv_path"] = args.sumgrad_cosine_csv
        _patch_ppo_for_pcgrad(apply_mean_kl=apply_mean_kl, apply_fn=apply_fn)
        if accelerator.is_main_process:
            kl_tag = "mean_kl (length-invariant)" if apply_mean_kl else "total (sum over tokens)"
            print(f"{args.method} PPO patch applied. KL penalty: {kl_tag}")
            if _windowed_requested:
                lo = args.pcgrad_window_start if args.pcgrad_window_start is not None else 1
                hi = args.pcgrad_window_end if args.pcgrad_window_end is not None else "end"
                print(
                    f"Windowed PCGrad: conflict-projection ON for PPO updates "
                    f"[{lo}, {hi}] (inclusive); plain vector sum (no surgery) outside."
                )
            if args.method == "sumgrad":
                combine_tag = "mean-norm rescaled" if args.sumgrad_normalize else "plain vector sum"
                print(f"SumGrad gradient combination: {combine_tag}")
            if args.method == "mgda":
                combine_tag = (
                    "mean-norm rescaled before min-norm solve"
                    if args.mgda_normalize else "raw gradients (faithful to arXiv:1810.04650)"
                )
                print(f"MGDA gradient combination: min-norm point of conv{{g1, g2}}; {combine_tag}")
            if _sumgrad_diag["csv_path"]:
                print(f"gradient-cosine diagnostic → {_sumgrad_diag['csv_path']}")
    elif _vanilla_deconflict:
        apply_mean_kl = (args.kl_div == "mean_kl")
        # start/end may be None → deconflict on every step (the "full" condition).
        _pcgrad_window["enabled"] = True
        _pcgrad_window["start"] = args.pcgrad_window_start
        _pcgrad_window["end"] = args.pcgrad_window_end
        # Reuse the cosine diagnostic to log cos(g_single, g1) each update (Fig. 3).
        if args.sumgrad_cosine_csv and accelerator.is_main_process:
            _sumgrad_diag["csv_path"] = args.sumgrad_cosine_csv
        _patch_ppo_for_vanilla_deconflict(apply_mean_kl=apply_mean_kl)
        if accelerator.is_main_process:
            kl_tag = "mean_kl (length-invariant)" if apply_mean_kl else "total (sum over tokens)"
            lo = args.pcgrad_window_start if args.pcgrad_window_start is not None else 1
            hi = args.pcgrad_window_end if args.pcgrad_window_end is not None else "end"
            print(f"vanilla deconflict patch applied. KL penalty: {kl_tag}")
            print(
                f"Vanilla deconflict: project the pooled update off the minority "
                f"gradient g1 for PPO updates [{lo}, {hi}] (inclusive); standard vanilla outside."
            )
            if _sumgrad_diag["csv_path"]:
                print(f"cos(g_single, g1) diagnostic → {_sumgrad_diag['csv_path']}")
    elif args.kl_div == "mean_kl":
        _patch_ppo_kl_to_mean_per_token()
        if accelerator.is_main_process:
            print("KL penalty: mean_kl (length-invariant, mean KL per token)")
    else:
        if accelerator.is_main_process:
            print("KL penalty: total (TRL default, sum of KL over tokens)")
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

    # Reward-model load kwargs: bf16 by default; optional NF4 4-bit for the frozen RM.
    rm_load_kwargs = {"torch_dtype": torch.bfloat16, "trust_remote_code": True}
    _bnb = maybe_bnb_config(args.load_in_4bit_frozen)
    if _bnb is not None:
        rm_load_kwargs["quantization_config"] = _bnb

    # Load reward model based on method (returns hub_id_or_ids for normalization lookup)
    reward_model, reward_tokenizer, reward_model_hub_id_or_ids = load_reward_model(
        method, user_id, seed, proportion, k, alpha, device,
        max_new_tokens=args.max_new_tokens,
        vanilla_rm_hub_id=args.vanilla_rm_hub_id,
        maxmin_minority_rm_hub_id=args.maxmin_minority_rm_hub_id,
        maxmin_majority_rm_hub_id=args.maxmin_majority_rm_hub_id,
        sharedrep_rm_hub_id=args.sharedrep_rm_hub_id,
        rm_load_kwargs=rm_load_kwargs,
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

    # Policy / value backbones (default Qwen3-4B; --value_model_name may point at a
    # smaller SAME-tokenizer model to keep the full-FT value head cheap).
    policy_model_name = args.policy_model_name
    value_model_name = args.value_model_name or policy_model_name
    use_lora = resolve_use_lora(args.use_lora, policy_model_name)
    if accelerator.is_main_process:
        print(f"Policy base model: {policy_model_name}  (LoRA: {use_lora})")
        print(f"Value base model:  {value_model_name}")
    policy_hub_id = args.policy_hub_id or get_policy_hub_id(method, user_id, seed, proportion, k, alpha)
    # Distinguish windowed-intervention runs when the hub id is auto-derived (so the
    # early-only / late-only / full runs don't overwrite each other).
    if _pcgrad_window["enabled"] and not args.policy_hub_id:
        _ws = _pcgrad_window["start"] if _pcgrad_window["start"] is not None else "1"
        _we = _pcgrad_window["end"] if _pcgrad_window["end"] is not None else "end"
        _tag = "deconflict-win" if _vanilla_deconflict else "win"
        policy_hub_id = f"{policy_hub_id}-{_tag}{_ws}_{_we}"
    policy_tokenizer = AutoTokenizer.from_pretrained(policy_model_name, padding_side="left")
    configure_tokenizer(policy_tokenizer, padding_side="left")

    # Policy model (bf16). Under LoRA the base weights stay frozen; only adapters train.
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_name, trust_remote_code=True, torch_dtype=torch.bfloat16
    )
    policy.config.pad_token_id = policy_tokenizer.pad_token_id

    if use_lora:
        # LoRA on the policy. With a PEFT policy TRL can use the adapter-disabled
        # base model as the reference (null-ref), so we do NOT allocate a separate
        # frozen ref copy — saving one full model's worth of VRAM.
        from peft import get_peft_model
        policy_lora = build_lora_config(
            task_type="CAUSAL_LM",
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=default_lora_target_modules(policy),
        )
        policy = get_peft_model(policy, policy_lora)
        if args.gradient_checkpointing:
            policy.enable_input_require_grads()
        if accelerator.is_main_process:
            policy.print_trainable_parameters()
        ref_policy = None
    else:
        # Full fine-tuning: explicit frozen reference copy (optionally 4-bit).
        ref_kwargs = {"trust_remote_code": True, "torch_dtype": torch.bfloat16}
        _bnb_ref = maybe_bnb_config(args.load_in_4bit_frozen)
        if _bnb_ref is not None:
            ref_kwargs["quantization_config"] = _bnb_ref
        ref_policy = AutoModelForCausalLM.from_pretrained(policy_model_name, **ref_kwargs)

    # Value model (bf16, full fine-tuning — keep on a small backbone for large policies).
    value_model = AutoModelForSequenceClassification.from_pretrained(
        value_model_name, trust_remote_code=True, num_labels=1, torch_dtype=torch.bfloat16
    )
    value_model.config.pad_token_id = policy_tokenizer.pad_token_id
    # Gradient checkpointing itself is enabled by PPOConfig(gradient_checkpointing=...)
    # inside TRL; the enable_input_require_grads() call above is what makes it work
    # with a frozen (LoRA) embedding table.

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

    # PCGrad/SumGrad and vanilla-deconflict require gradient_accumulation_steps=1
    # (multiple sequential backward passes + a custom step). Double the per-device
    # batch size to keep the same effective batch size (32*2 = 64).
    _needs_single_accum = method in ("pcgrad", "sumgrad", "mgda") or _vanilla_deconflict
    _base_bs = args.per_device_train_batch_size if args.per_device_train_batch_size else 32
    grad_accum_steps = 1 if _needs_single_accum else 2
    pcgrad_batch_size = _base_bs * 2 if _needs_single_accum else _base_bs

    # HF repo privacy: HF_PRIVATE=0 pushes public (needed when the private
    # storage quota is full). Defaults to private to preserve prior behavior.
    _hf_private = os.environ.get("HF_PRIVATE", "1") != "0"
    if accelerator.is_main_process:
        print(f"HF push visibility: {'private' if _hf_private else 'public'}")

    # Training Arguments
    training_args = PPOConfig(
        logging_steps=50,  # Reduced frequency to save time (was 10)
        num_train_epochs=3,  # Reduced from 5 to save time
        num_ppo_epochs=3,
        per_device_train_batch_size=pcgrad_batch_size,
        gradient_accumulation_steps=grad_accum_steps,  # PCGrad: 1 (batch doubled); else 2
        bf16=True,
        learning_rate=args.learning_rate,
        eval_on_start=False,  # Disabled to save time - evaluation happens at end anyway
        push_to_hub=(not args.no_push),
        save_strategy="no",
        hub_private_repo=_hf_private,
        hub_model_id=policy_hub_id,
        report_to=["wandb"],
        run_name=policy_hub_id,
        whiten_rewards=True,
        missing_eos_penalty=0.0,  # was 1.0; removed to prevent early-EOS collapse (rewards are in [0,1])
        total_episodes=args.total_episodes,
        lr_scheduler_type="cosine",
        stop_token="eos",
        kl_coef=args.kl_coef,
        gradient_checkpointing=args.gradient_checkpointing,
    )

    # ── W4 deconfliction: load the minority group RM as a guard (vanilla only) ──
    _deconflict_min_model = None
    _deconflict_min_tok = None
    if _vanilla_deconflict:
        _deconflict_min_id = (
            args.maxmin_minority_rm_hub_id
            or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{args.max_new_tokens}-minority-gpt2"
        )
        if accelerator.is_main_process:
            print(f"Deconflict guard (minority) RM: {_deconflict_min_id}")
        _deconflict_min_model = AutoModelForSequenceClassification.from_pretrained(
            _deconflict_min_id, **rm_load_kwargs
        )
        _deconflict_min_model.eval()
        _deconflict_min_model.to(device)
        for _p in _deconflict_min_model.parameters():
            _p.requires_grad_(False)
        _deconflict_min_tok = AutoTokenizer.from_pretrained(_deconflict_min_id, padding_side="left")
        configure_tokenizer(_deconflict_min_tok, padding_side="left")
        _deconflict_min_model.config.pad_token_id = _deconflict_min_tok.pad_token_id

    # Set up reward function based on method
    setup_reward_function(
        method, policy_tokenizer, reward_tokenizer, reward_model, alpha, proportion,
        deconflict_minority_model=_deconflict_min_model,
        deconflict_minority_tok=_deconflict_min_tok,
    )

    # ── Online gradient-cosine diagnostic setup (vanilla only) ───────────────
    diag_callback = None
    if method == "vanilla" and args.diag_every > 0:
        from src.imdb.diagnostics.gradient_cosine_diagnostic import (
            run_online_diagnostic as _run_online_diag,  # noqa: F401 (imported into callback)
            load_prompts as _diag_load_prompts,
            tokenize_prompts as _diag_tokenize_prompts,
            load_reward_model as _diag_load_rm,
            load_ref_model as _diag_load_ref,
            ONLINE_DIAG_FIELDS as _ONLINE_DIAG_FIELDS,  # noqa: F401
        )

        if accelerator.is_main_process:
            print(
                f"\nOnline gradient-cosine diagnostic enabled: every {args.diag_every} steps, "
                f"beta={args.diag_beta}, prompts={args.diag_max_prompts}, "
                f"output={args.diag_output_path}"
            )

        # Fixed prompt batch — sampled once from IMDB test set, reused every K steps
        _diag_prompts = _diag_load_prompts(None, args.diag_max_prompts, seed)
        _diag_prompt_ids, _diag_prompt_mask = _diag_tokenize_prompts(
            _diag_prompts, policy_tokenizer, device
        )

        # r_single is the vanilla reward_model already loaded above; reuse it
        _diag_r1_path = (
            args.diag_r1_hub_id
            or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{args.max_new_tokens}-minority-gpt2"
        )
        _diag_r2_path = (
            args.diag_r2_hub_id
            or f"{user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{args.max_new_tokens}-majority-gpt2"
        )
        _diag_reward_r1, _diag_reward_r1_tok = _diag_load_rm(_diag_r1_path, device)
        _diag_reward_r2, _diag_reward_r2_tok = _diag_load_rm(_diag_r2_path, device)

        # ref_policy for KL term — only load if beta != 0 to save memory
        _diag_ref = (
            _diag_load_ref(args.policy_model_name, device)
            if args.diag_beta != 0.0
            else None
        )

        diag_callback = GradientCosineDiagnosticCallback(
            policy=policy,
            accelerator=accelerator,
            ref_policy=_diag_ref,
            reward_single=reward_model,
            reward_single_tok=reward_tokenizer,
            reward_r1=_diag_reward_r1,
            reward_r1_tok=_diag_reward_r1_tok,
            reward_r2=_diag_reward_r2,
            reward_r2_tok=_diag_reward_r2_tok,
            prompt_ids=_diag_prompt_ids,
            prompt_mask=_diag_prompt_mask,
            alpha=alpha,
            beta=args.diag_beta,
            diag_every=args.diag_every,
            batch_size=args.diag_batch_size,
            max_new_tokens=args.diag_max_new_tokens,
            reward_max_length=args.diag_reward_max_length,
            tokenizer=policy_tokenizer,
            csv_path=args.diag_output_path,
            device=device,
        )

    # Train
    callbacks = []
    if method in ("maxmin", "sharedrep", "gold"):
        callbacks.append(GroupSelectionCallback())
    if diag_callback is not None:
        callbacks.append(diag_callback)

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

    # Flush any remaining SumGrad cosine-diagnostic rows.
    if accelerator.is_main_process and _sumgrad_diag["csv_path"]:
        _sumgrad_flush_cosine()
        print(f"SumGrad cosine diagnostic saved ({_sumgrad_diag['step']} updates) → {_sumgrad_diag['csv_path']}")

    # Save the model - explicitly save the policy model with its config and weights
    if accelerator.is_main_process and args.no_push:
        print("\n--no_push set: skipping local save and Hub push (diagnostic run).")

    if accelerator.is_main_process and not args.no_push:
        print("\n" + "="*80)
        print("SAVING MODEL TO HUGGINGFACE HUB")
        print("="*80)
        
        # Unwrap the model from accelerator
        unwrapped_policy = accelerator.unwrap_model(policy)

        # Under LoRA, merge adapters into the base weights so the pushed model is a
        # standard causal-LM checkpoint that eval/generation can load directly with
        # AutoModelForCausalLM (no PEFT dependency downstream).
        if use_lora and hasattr(unwrapped_policy, "merge_and_unload"):
            print("Merging LoRA adapters into base policy weights...")
            unwrapped_policy = unwrapped_policy.merge_and_unload()

        # model_type must reflect the ACTUAL backbone (e.g. qwen3), not a hardcoded
        # value — otherwise the saved config is corrupt for non-GPT-2 backbones.
        if not getattr(unwrapped_policy.config, "model_type", None):
            raise ValueError(
                "Policy config is missing model_type; refusing to save a corrupt "
                "checkpoint. Check the base model loaded correctly."
            )

        print(f"Model ID: {policy_hub_id}")
        print(f"Config model_type: {unwrapped_policy.config.model_type}")

        local_policy_output_dir = args.local_policy_output_dir or os.path.join(
            "local_models", policy_hub_id.replace("/", "__")
        )
        print(f"Saving local model copy to {local_policy_output_dir}...")
        unwrapped_policy.save_pretrained(local_policy_output_dir, safe_serialization=True)
        policy_tokenizer.save_pretrained(local_policy_output_dir)
        print("✓ Local model copy saved.")
        
        # Push model and tokenizer to hub with weights
        try:
            print("Pushing model to Hub...")
            commit_info = unwrapped_policy.push_to_hub(
                policy_hub_id,
                private=_hf_private,
                commit_message=f"Upload {policy_hub_id} model weights"
            )
            print(f"✓ Model pushed successfully!")
            print(f"  Commit URL: {commit_info.commit_url if hasattr(commit_info, 'commit_url') else 'N/A'}")

            print("Pushing tokenizer to Hub...")
            tokenizer_commit = policy_tokenizer.push_to_hub(policy_hub_id, private=_hf_private)
            print(f"✓ Tokenizer pushed successfully!")

        except Exception as e:
            print(f"\n{'!'*80}")
            print(f"ERROR: Failed to push model/tokenizer to Hub!")
            print(f"{'!'*80}")
            print(f"Error: {e}")
            print(f"\nPlease manually upload the model:")
            print(f"  1. Local copy is saved at: {local_policy_output_dir}")
            print(f"  2. Free Hub storage or make the repo public, then push that directory to '{policy_hub_id}'")
            raise
        
        print("="*80)
        print("MODEL SAVED SUCCESSFULLY")
        print("="*80 + "\n")
    
    # NOTE: trainer.push_to_hub() ALSO re-uploads trainer.model to hub_model_id.
    # Under LoRA that is the (unmerged) PEFT policy, which overwrites the merged
    # full model we just pushed above with a bare adapter — corrupting the repo.
    # So only call it for full fine-tuning; for LoRA the merged model + tokenizer
    # were already pushed above and we skip the dataset-card push.
    if accelerator.is_main_process and not use_lora and not args.no_push:
        try:
            trainer.push_to_hub(dataset_name=dataset_name)
        except Exception as e:
            print(f"Warning: Failed to push training dataset: {e}")

        # IMPORTANT: run this LAST. trainer.push_to_hub() re-uploads TRL's
        # trainer.model, which carries a (Qwen-referencing) PEFT adapter config —
        # even for full-FT — and would otherwise re-add a stale adapter_config.json
        # + empty adapter_model.safetensors that hijacks AutoModelForCausalLM at
        # eval time (loads the adapter's base model instead of this checkpoint).
        from src.helper.model_utils import cleanup_stale_hub_files
        cleanup_stale_hub_files(policy_hub_id, ["adapter_config.json", "adapter_model.safetensors"])
        print("✓ Stale adapter files (if any) removed from policy repo.")
