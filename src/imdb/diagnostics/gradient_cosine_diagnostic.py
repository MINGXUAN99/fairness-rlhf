"""
Post-hoc gradient cosine diagnostic for TRL PPO RLHF experiments.

For each alpha value, this script runs two diagnostics:

1. At theta = final vanilla policy:
   - Generates responses from the vanilla policy on a fixed prompt batch.
   - Computes g_vanilla (gradient under r_single), g1 (under r1), g2 (under r2).
   - Saves cos(g_vanilla, g1), cos(g_vanilla, g2), cos(g1, g2).

2. At theta = final MaxMin policy:
   - Generates responses from the MaxMin policy on the same prompt batch.
   - Computes g1 (under r1), g2 (under r2).
   - Sets g_maxmin = g1 if mean(r1) <= mean(r2), else g2  (hard-min).
   - Saves cos(g_maxmin, g1), cos(g_maxmin, g2), cos(g1, g2).

Diagnostic loss for reward r:
    A = r_norm - beta * (log_pi - log_pi_ref)
    L = -mean(A.detach() * log_pi)

Rewards are batch-normalised (z-score) across all prompts before constructing
the loss.  Gradients are computed only over trainable policy parameters (all
parameters for a full-weight model; LoRA-only if a PeftModel is detected).
Log-probabilities are summed over response tokens only (excluding prompt tokens
and any padding after the first EOS).

Output CSV columns:
    alpha, policy_type, cos_obj_g1, cos_obj_g2, cos_g1_g2,
    norm_obj, norm_g1, norm_g2,
    mean_reward_g1, mean_reward_g2, mean_reward_obj, worst_group

Usage example:
    python src/imdb/diagnostics/gradient_cosine_diagnostic.py \\
        --alphas 0.5 0.7 1.0 \\
        --user_id myuser \\
        --seed 42 \\
        --proportion 0.2 \\
        --beta 0.05 \\
        --batch_size 16 \\
        --max_new_tokens 200 \\
        --max_prompts 128 \\
        --output_path gradient_cosine_diagnostic.csv
"""

from __future__ import annotations

import sys
import os

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import argparse
import csv
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
)

try:
    from peft import PeftModel  # type: ignore
    _PEFT_AVAILABLE = True
except ImportError:
    _PEFT_AVAILABLE = False
    PeftModel = None  # type: ignore

from src.helper.imdb_utils import pre_processing_imdb

# Prompts are left-padded to this fixed length, matching the training convention.
PROMPT_PAD_LEN: int = 8

CSV_FIELDS: List[str] = [
    "alpha", "policy_type",
    "cos_obj_g1", "cos_obj_g2", "cos_g1_g2",
    "norm_obj", "norm_g1", "norm_g2",
    "mean_reward_g1", "mean_reward_g2", "mean_reward_obj",
    "worst_group",
]


# ──────────────────────────────────────────────────────────────────────────
# Argument parsing
# ──────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Post-hoc gradient cosine diagnostic for PPO RLHF experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Alpha sweep
    p.add_argument("--alphas", nargs="+", type=float, required=True,
                   help="Alpha values to evaluate, e.g. --alphas 0.5 0.7 1.0")

    # HF naming convention helpers (used to build default templates)
    p.add_argument("--user_id", type=str, default=None,
                   help="HuggingFace user/org id for default path template expansion.")
    p.add_argument("--seed", type=int, default=42,
                   help="Seed used in policy hub-id naming.")
    p.add_argument("--proportion", type=float, default=0.2,
                   help="Minority proportion used in policy hub-id naming.")

    # Path templates — {alpha} (and others) are substituted per iteration.
    # Defaults follow the naming conventions in unified_ppo_training.py.
    p.add_argument("--vanilla_policy_template", type=str, default=None,
                   help=(
                       "Template for the vanilla policy path. "
                       "Placeholders: {alpha}, {user_id}, {seed}, {proportion}. "
                       "Default: {user_id}/vanilla-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
                   ))
    p.add_argument("--maxmin_policy_template", type=str, default=None,
                   help=(
                       "Template for the MaxMin policy path. Same placeholders. "
                       "Default: {user_id}/maxmin-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}"
                   ))
    p.add_argument("--ref_policy_path", type=str, default="lvwerra/gpt2-imdb",
                   help="Path or hub-id for the reference policy (shared across all alphas).")
    p.add_argument("--reward_single_template", type=str, default=None,
                   help=(
                       "Template for the single pooled reward model (r_single). "
                       "Placeholders: {alpha}, {user_id}, {seed}, {proportion}, {max_new_tokens}. "
                       "Default: {user_id}/vanilla-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-gpt2"
                   ))
    p.add_argument("--reward_r1_template", type=str, default=None,
                   help=(
                       "Template for group reward model r1 (minority). "
                       "Default: {user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-minority-gpt2"
                   ))
    p.add_argument("--reward_r2_template", type=str, default=None,
                   help=(
                       "Template for group reward model r2 (majority). "
                       "Default: {user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-majority-gpt2"
                   ))

    # Prompts
    p.add_argument("--prompt_file", type=str, default=None,
                   help=(
                       "Text file with one prompt per line. "
                       "If omitted, samples from the IMDB test set via pre_processing_imdb."
                   ))
    p.add_argument("--max_prompts", type=int, default=128,
                   help="Maximum number of prompts to use.")

    # Generation / loss
    p.add_argument("--max_new_tokens", type=int, default=200,
                   help="Maximum response length (tokens) for generation.")
    p.add_argument("--batch_size", type=int, default=16,
                   help=(
                       "Mini-batch size for generation, reward scoring, and gradient "
                       "accumulation."
                   ))
    p.add_argument("--beta", type=float, default=0.05,
                   help="KL penalty coefficient (beta) in the diagnostic loss.")

    # Reward model scoring
    p.add_argument("--reward_max_length", type=int, default=512,
                   help="Maximum tokeniser length when scoring with reward models.")

    # Output
    p.add_argument("--output_path", type=str, default="gradient_cosine_diagnostic.csv",
                   help="Path for the output CSV file.")
    p.add_argument("--device", type=str, default=None,
                   help=(
                       "Compute device override ('cuda', 'cpu', 'cuda:1', …). "
                       "Defaults to 'cuda' if available, else 'cpu'."
                   ))
    return p.parse_args()


# ──────────────────────────────────────────────────────────────────────────
# Path resolution
# ──────────────────────────────────────────────────────────────────────────

def _resolve(
    template: str,
    alpha: float,
    user_id: Optional[str],
    seed: int,
    proportion: float,
    max_new_tokens: int,
) -> str:
    """Substitute named placeholders in a path template."""
    return template.format(
        alpha=alpha,
        user_id=user_id or "",
        seed=seed,
        proportion=proportion,
        max_new_tokens=max_new_tokens,
    )


def build_templates(args: argparse.Namespace) -> dict:
    """
    Return a dict of template strings, filling in defaults where not supplied.
    Raises ValueError if any default template is needed but --user_id is absent.
    """
    uid = args.user_id
    needs_default = any(
        t is None for t in [
            args.vanilla_policy_template,
            args.maxmin_policy_template,
            args.reward_single_template,
            args.reward_r1_template,
            args.reward_r2_template,
        ]
    )
    if needs_default and uid is None:
        raise ValueError(
            "--user_id is required when any path template is not explicitly provided. "
            "Supply --user_id or override all five templates with explicit paths."
        )
    return {
        "vanilla_policy": args.vanilla_policy_template or (
            f"{uid}/vanilla-imdb-ppo-prop{{proportion}}-alpha{{alpha}}-seed{{seed}}"
        ),
        "maxmin_policy": args.maxmin_policy_template or (
            f"{uid}/maxmin-imdb-ppo-prop{{proportion}}-alpha{{alpha}}-seed{{seed}}"
        ),
        "reward_single": args.reward_single_template or (
            f"{uid}/vanilla-imdb-reward-ALPHA{{alpha}}-len{{max_new_tokens}}-gpt2"
        ),
        "reward_r1": args.reward_r1_template or (
            f"{uid}/maxmin-imdb-reward-ALPHA{{alpha}}-len{{max_new_tokens}}-minority-gpt2"
        ),
        "reward_r2": args.reward_r2_template or (
            f"{uid}/maxmin-imdb-reward-ALPHA{{alpha}}-len{{max_new_tokens}}-majority-gpt2"
        ),
    }


# ──────────────────────────────────────────────────────────────────────────
# Model loaders
# ──────────────────────────────────────────────────────────────────────────

def load_policy_model(path: str, device: torch.device) -> AutoModelForCausalLM:
    """
    Load a causal-LM policy.  All parameters are set to require_grad=True.
    Model starts in eval mode; compute_grad() will switch to train mode.
    """
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    model.to(device)
    for param in model.parameters():
        param.requires_grad_(True)
    model.eval()
    return model


def load_ref_model(path: str, device: torch.device) -> AutoModelForCausalLM:
    """Load a frozen reference causal LM (no gradients)."""
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def load_reward_model(
    path: str, device: torch.device
) -> Tuple[AutoModelForSequenceClassification, AutoTokenizer]:
    """Load a frozen reward model and its tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(path, trust_remote_code=True)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    # Use right-padding (default GPT-2 padding side) so that
    # GPT2ForSequenceClassification correctly identifies the last content token.
    tokenizer = AutoTokenizer.from_pretrained(path)
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


# ──────────────────────────────────────────────────────────────────────────
# Prompt loading
# ──────────────────────────────────────────────────────────────────────────

def load_prompts(
    prompt_file: Optional[str], max_prompts: int, seed: int
) -> List[str]:
    """
    Load a fixed prompt batch.  If prompt_file is given, reads one prompt per
    line.  Otherwise, samples from the IMDB test set using pre_processing_imdb
    (the same preprocessing used during PPO training).
    """
    if prompt_file is not None:
        with open(prompt_file) as fh:
            prompts = [ln.strip() for ln in fh if ln.strip()]
        return prompts[:max_prompts]

    ds = load_dataset("stanfordnlp/imdb", split="test")
    ds = pre_processing_imdb(
        ds,
        model_name="lvwerra/gpt2-imdb",
        min_review_length=200,
        input_min_text_length=2,
        input_max_text_length=8,
    )
    return list(ds["query"])[:max_prompts]


# ──────────────────────────────────────────────────────────────────────────
# Tokenisation
# ──────────────────────────────────────────────────────────────────────────

def tokenize_prompts(
    prompts: List[str],
    tokenizer: AutoTokenizer,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Left-pad prompts to PROMPT_PAD_LEN tokens, matching the training convention.
    Returns (input_ids, attention_mask) both on `device`.
    """
    tokenizer.padding_side = "left"
    enc = tokenizer(
        prompts,
        padding="max_length",
        max_length=PROMPT_PAD_LEN,
        truncation=True,
        return_tensors="pt",
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


# ──────────────────────────────────────────────────────────────────────────
# Generation
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def generate_responses(
    policy: AutoModelForCausalLM,
    prompt_ids: torch.Tensor,    # (N, PROMPT_PAD_LEN)
    prompt_mask: torch.Tensor,   # (N, PROMPT_PAD_LEN)
    max_new_tokens: int,
    batch_size: int,
    pad_token_id: int,
    eos_token_id: int,
) -> List[torch.Tensor]:
    """
    Generate responses in mini-batches.

    Returns a list of tensors (one per mini-batch), each of shape
    (B_i, PROMPT_PAD_LEN + actual_gen_len).  Keeping them in separate
    tensors avoids cross-batch padding when response lengths differ.
    """
    policy.eval()
    batches: List[torch.Tensor] = []
    N = prompt_ids.shape[0]
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        ids = policy.generate(
            input_ids=prompt_ids[start:end],
            attention_mask=prompt_mask[start:end],
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=0,
            temperature=0.7,
            top_p=0.95,
            eos_token_id=eos_token_id,
            pad_token_id=pad_token_id,
        )
        batches.append(ids)
    return batches


def decode_all(
    full_id_batches: List[torch.Tensor],
    tokenizer: AutoTokenizer,
) -> List[str]:
    """Decode all mini-batch tensors into a flat list of strings."""
    texts: List[str] = []
    for ids in full_id_batches:
        texts.extend(tokenizer.batch_decode(ids, skip_special_tokens=True))
    return texts


# ──────────────────────────────────────────────────────────────────────────
# Reward scoring
# ──────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_texts(
    reward_model: AutoModelForSequenceClassification,
    reward_tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> torch.Tensor:
    """
    Score a list of decoded texts.  Returns a (N,) CPU float tensor.
    GPT2ForSequenceClassification uses the last non-padding token's hidden
    state, which requires right-padding (the default for these models).
    """
    all_scores: List[torch.Tensor] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = reward_tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        out = reward_model(**enc)
        # logits: (B, num_labels) — squeeze the label dim for single-head models
        scores = out.logits
        if scores.dim() > 1:
            scores = scores[:, 0]
        all_scores.append(scores.float().cpu())
    return torch.cat(all_scores, dim=0)  # (N,)


# ──────────────────────────────────────────────────────────────────────────
# Response log-probability
# ──────────────────────────────────────────────────────────────────────────

def compute_response_log_prob(
    model: AutoModelForCausalLM,
    full_ids: torch.Tensor,  # (B, prompt_len + resp_len)
    prompt_length: int,
    pad_token_id: int,
) -> torch.Tensor:
    """
    Compute the sum of per-token log-probs over response tokens only.

    Masking: all positions up to and including the first EOS/pad token are
    counted (so the end-of-response marker is included); everything after the
    first EOS/pad is excluded.  If no EOS/pad appears in the response, all
    generated tokens are counted.

    Returns: (B,) scalar log-prob per sequence.
    """
    out = model(input_ids=full_ids)
    # Upcast to float32 for numerical stability
    log_probs = F.log_softmax(out.logits.float(), dim=-1)  # (B, seq, V)

    # token[t] is predicted by logits[t-1]
    # response token positions: prompt_length … seq-1
    # their logit positions:    prompt_length-1 … seq-2
    response_ids = full_ids[:, prompt_length:]                       # (B, R)
    response_lp = log_probs[:, prompt_length - 1 : -1, :]           # (B, R, V)

    # Gather log-prob of the actual token at each response position
    token_lp = response_lp.gather(
        2, response_ids.unsqueeze(-1)
    ).squeeze(-1)                                                    # (B, R)

    # Build validity mask: keep positions up to and including the first EOS/pad.
    # cumsum==0  → before first pad; ==1 → at first pad; >1 → after first pad
    is_pad = response_ids == pad_token_id                            # (B, R) bool
    cumsum = is_pad.long().cumsum(dim=1)
    valid = cumsum <= 1                                              # (B, R) bool

    return (token_lp * valid.float()).sum(dim=1)                     # (B,)


# ──────────────────────────────────────────────────────────────────────────
# Trainable-parameter helpers
# ──────────────────────────────────────────────────────────────────────────

def _get_trainable_params(
    policy: AutoModelForCausalLM,
) -> List[torch.nn.Parameter]:
    """
    Return the list of parameters over which gradients are computed.
    When a PeftModel (LoRA) is detected, only LoRA parameters are returned;
    otherwise, all requires_grad parameters are returned.
    """
    if _PEFT_AVAILABLE and isinstance(policy, PeftModel):
        return [
            p for n, p in policy.named_parameters()
            if "lora_" in n and p.requires_grad
        ]
    return [p for p in policy.parameters() if p.requires_grad]


def _flatten_grads(params: List[torch.nn.Parameter]) -> torch.Tensor:
    """Flatten and concatenate gradients into a 1-D CPU float tensor."""
    parts: List[torch.Tensor] = []
    for p in params:
        g = p.grad.detach().float().view(-1) if p.grad is not None else torch.zeros(
            p.numel(), device=p.device
        )
        parts.append(g)
    return torch.cat(parts).cpu()


# ──────────────────────────────────────────────────────────────────────────
# Gradient computation (with mini-batch accumulation)
# ──────────────────────────────────────────────────────────────────────────

def compute_grad(
    policy: AutoModelForCausalLM,
    ref_policy: AutoModelForCausalLM,
    raw_rewards: torch.Tensor,               # (N,) CPU
    full_id_batches: List[torch.Tensor],     # list of (B_i, seq_i) on device
    pad_token_id: int,
    beta: float,
) -> torch.Tensor:
    """
    Compute the policy gradient under the diagnostic loss:

        r_norm = batch_zscore(raw_rewards)
        A      = r_norm − beta * (log_pi − log_pi_ref)
        L      = −mean(A.detach() * log_pi)

    Gradients are accumulated over mini-batches.  Each mini-batch loss is
    weighted by its fraction of the total N so the accumulation equals the
    full-batch mean loss gradient.

    Returns: 1-D CPU float gradient vector over trainable policy parameters.
    """
    device = next(policy.parameters()).device
    N = int(raw_rewards.shape[0])

    # Batch-level reward normalisation (z-score across all N samples)
    r_mean = raw_rewards.float().mean()
    r_std = raw_rewards.float().std().clamp(min=1e-8)
    r_norm = (raw_rewards.float() - r_mean) / r_std  # (N,) CPU

    policy.train()
    policy.zero_grad()

    sample_offset = 0
    for batch_ids in full_id_batches:
        B = batch_ids.shape[0]
        mb_r_norm = r_norm[sample_offset : sample_offset + B].to(device)

        with torch.no_grad():
            log_pi_ref = compute_response_log_prob(
                ref_policy, batch_ids, PROMPT_PAD_LEN, pad_token_id
            )  # (B,)

        # Policy log-prob with gradient
        log_pi = compute_response_log_prob(
            policy, batch_ids, PROMPT_PAD_LEN, pad_token_id
        )  # (B,)

        A = mb_r_norm - beta * (log_pi - log_pi_ref.to(device))

        # Weight by B/N so that summing over mini-batches gives the full-batch mean
        L = -(A.detach() * log_pi).sum() * (1.0 / N)
        L.backward()

        sample_offset += B

    params = _get_trainable_params(policy)
    grad_vec = _flatten_grads(params)
    policy.zero_grad()
    return grad_vec  # (P,) CPU float


# ──────────────────────────────────────────────────────────────────────────
# Cosine similarity
# ──────────────────────────────────────────────────────────────────────────

def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return F.cosine_similarity(
        a.float().unsqueeze(0), b.float().unsqueeze(0)
    ).item()


# ──────────────────────────────────────────────────────────────────────────
# Per-policy diagnostic
# ──────────────────────────────────────────────────────────────────────────

def run_one_policy(
    *,
    policy_path: str,
    policy_type: str,                          # "vanilla" | "maxmin"
    ref_policy: AutoModelForCausalLM,
    reward_single: Optional[AutoModelForSequenceClassification],
    reward_single_tok: Optional[AutoTokenizer],
    reward_r1: AutoModelForSequenceClassification,
    reward_r1_tok: AutoTokenizer,
    reward_r2: AutoModelForSequenceClassification,
    reward_r2_tok: AutoTokenizer,
    prompt_ids: torch.Tensor,
    prompt_mask: torch.Tensor,
    alpha: float,
    beta: float,
    max_new_tokens: int,
    batch_size: int,
    reward_max_length: int,
    device: torch.device,
    tokenizer: AutoTokenizer,
) -> dict:
    """
    Run the full diagnostic for a single (policy_type, alpha) combination.

    Returns a dict matching CSV_FIELDS.
    """
    print(f"  [{policy_type}] loading policy: {policy_path}")
    policy = load_policy_model(policy_path, device)
    pad_tok = tokenizer.pad_token_id
    eos_tok = tokenizer.eos_token_id

    # ── Step 1: generate responses (single pass, no grad) ─────────────────
    print(f"  [{policy_type}] generating responses ...")
    full_id_batches = generate_responses(
        policy, prompt_ids, prompt_mask,
        max_new_tokens, batch_size, pad_tok, eos_tok,
    )
    texts = decode_all(full_id_batches, tokenizer)
    N = len(texts)
    print(f"  [{policy_type}] generated {N} responses.")

    # ── Step 2: score responses with all applicable reward models ──────────
    print(f"  [{policy_type}] scoring responses ...")
    r1_scores = score_texts(reward_r1, reward_r1_tok, texts, batch_size, reward_max_length, device)
    r2_scores = score_texts(reward_r2, reward_r2_tok, texts, batch_size, reward_max_length, device)

    if policy_type == "vanilla":
        assert reward_single is not None and reward_single_tok is not None
        r_single_scores = score_texts(
            reward_single, reward_single_tok, texts, batch_size, reward_max_length, device
        )

    mean_r1 = r1_scores.mean().item()
    mean_r2 = r2_scores.mean().item()

    # ── Step 3: compute gradients ──────────────────────────────────────────
    # g1 and g2 are computed fresh for THIS policy (responses generated above).
    # Gradients from different reward models are computed independently (zero_grad
    # is called inside compute_grad at both the start and end of each call).
    print(f"  [{policy_type}] computing gradients ...")

    g1 = compute_grad(policy, ref_policy, r1_scores, full_id_batches, pad_tok, beta)
    g2 = compute_grad(policy, ref_policy, r2_scores, full_id_batches, pad_tok, beta)

    if policy_type == "vanilla":
        g_obj = compute_grad(
            policy, ref_policy, r_single_scores, full_id_batches, pad_tok, beta
        )
        mean_reward_obj = r_single_scores.mean().item()
    else:
        # MaxMin hard-min: g_maxmin is whichever group has the lower mean reward
        worst_group = "r1" if mean_r1 <= mean_r2 else "r2"
        g_obj = g1 if worst_group == "r1" else g2
        mean_reward_obj = min(mean_r1, mean_r2)

    worst_group = "r1" if mean_r1 <= mean_r2 else "r2"

    # ── Step 4: cosine similarities & norms ───────────────────────────────
    cos_obj_g1 = cosine_sim(g_obj, g1)
    cos_obj_g2 = cosine_sim(g_obj, g2)
    cos_g1_g2  = cosine_sim(g1, g2)

    del policy
    torch.cuda.empty_cache()

    return {
        "alpha":          alpha,
        "policy_type":    policy_type,
        "cos_obj_g1":     cos_obj_g1,
        "cos_obj_g2":     cos_obj_g2,
        "cos_g1_g2":      cos_g1_g2,
        "norm_obj":       g_obj.norm().item(),
        "norm_g1":        g1.norm().item(),
        "norm_g2":        g2.norm().item(),
        "mean_reward_g1": mean_r1,
        "mean_reward_g2": mean_r2,
        "mean_reward_obj":mean_reward_obj,
        "worst_group":    worst_group,
    }


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    device = torch.device(
        args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"Device: {device}")
    print(f"Beta (KL coefficient): {args.beta}")
    print(f"Batch size: {args.batch_size}  |  Max new tokens: {args.max_new_tokens}")
    print(f"Max prompts: {args.max_prompts}")

    templates = build_templates(args)

    # ── Load prompts once — reused across all alphas ───────────────────────
    print("\nLoading prompts ...")
    prompts = load_prompts(args.prompt_file, args.max_prompts, args.seed)
    print(f"  {len(prompts)} prompts ready.")

    # Shared policy tokenizer (GPT-2 IMDB fine-tuned)
    policy_tokenizer = AutoTokenizer.from_pretrained(
        "lvwerra/gpt2-imdb", padding_side="left"
    )
    policy_tokenizer.pad_token = policy_tokenizer.eos_token

    prompt_ids, prompt_mask = tokenize_prompts(prompts, policy_tokenizer, device)

    # ── Open output CSV ────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(args.output_path))
    os.makedirs(out_dir, exist_ok=True)
    out_file = open(args.output_path, "w", newline="")
    writer = csv.DictWriter(out_file, fieldnames=CSV_FIELDS)
    writer.writeheader()
    out_file.flush()
    print(f"\nResults will be written to: {args.output_path}")

    # ── Per-alpha loop ─────────────────────────────────────────────────────
    for alpha in args.alphas:
        print(f"\n{'='*64}")
        print(f"  Alpha = {alpha}")
        print(f"{'='*64}")

        def R(tmpl: str) -> str:
            return _resolve(tmpl, alpha, args.user_id, args.seed,
                            args.proportion, args.max_new_tokens)

        vanilla_path  = R(templates["vanilla_policy"])
        maxmin_path   = R(templates["maxmin_policy"])
        r_single_path = R(templates["reward_single"])
        r1_path       = R(templates["reward_r1"])
        r2_path       = R(templates["reward_r2"])

        print(f"  Vanilla policy : {vanilla_path}")
        print(f"  MaxMin policy  : {maxmin_path}")
        print(f"  r_single       : {r_single_path}")
        print(f"  r1 (minority)  : {r1_path}")
        print(f"  r2 (majority)  : {r2_path}")
        print(f"  ref policy     : {args.ref_policy_path}")

        # Load models shared between the two diagnostics for this alpha
        print("\n  Loading reference policy ...")
        ref_policy = load_ref_model(args.ref_policy_path, device)

        print("  Loading reward models ...")
        reward_single, reward_single_tok = load_reward_model(r_single_path, device)
        reward_r1, reward_r1_tok = load_reward_model(r1_path, device)
        reward_r2, reward_r2_tok = load_reward_model(r2_path, device)

        # ── Diagnostic 1: vanilla policy ──────────────────────────────────
        print(f"\n  {'─'*28} Diagnostic 1: vanilla policy {'─'*4}")
        row_vanilla = run_one_policy(
            policy_path=vanilla_path,
            policy_type="vanilla",
            ref_policy=ref_policy,
            reward_single=reward_single,
            reward_single_tok=reward_single_tok,
            reward_r1=reward_r1,
            reward_r1_tok=reward_r1_tok,
            reward_r2=reward_r2,
            reward_r2_tok=reward_r2_tok,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            alpha=alpha,
            beta=args.beta,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            reward_max_length=args.reward_max_length,
            device=device,
            tokenizer=policy_tokenizer,
        )
        writer.writerow(row_vanilla)
        out_file.flush()
        print(
            f"  [vanilla] cos(g_obj,g1)={row_vanilla['cos_obj_g1']:.4f}  "
            f"cos(g_obj,g2)={row_vanilla['cos_obj_g2']:.4f}  "
            f"cos(g1,g2)={row_vanilla['cos_g1_g2']:.4f}  "
            f"worst={row_vanilla['worst_group']}"
        )

        # ── Diagnostic 2: MaxMin policy ────────────────────────────────────
        print(f"\n  {'─'*28} Diagnostic 2: MaxMin policy {'─'*5}")
        row_maxmin = run_one_policy(
            policy_path=maxmin_path,
            policy_type="maxmin",
            ref_policy=ref_policy,
            reward_single=None,
            reward_single_tok=None,
            reward_r1=reward_r1,
            reward_r1_tok=reward_r1_tok,
            reward_r2=reward_r2,
            reward_r2_tok=reward_r2_tok,
            prompt_ids=prompt_ids,
            prompt_mask=prompt_mask,
            alpha=alpha,
            beta=args.beta,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            reward_max_length=args.reward_max_length,
            device=device,
            tokenizer=policy_tokenizer,
        )
        writer.writerow(row_maxmin)
        out_file.flush()
        print(
            f"  [maxmin]  cos(g_obj,g1)={row_maxmin['cos_obj_g1']:.4f}  "
            f"cos(g_obj,g2)={row_maxmin['cos_obj_g2']:.4f}  "
            f"cos(g1,g2)={row_maxmin['cos_g1_g2']:.4f}  "
            f"worst={row_maxmin['worst_group']}"
        )

        # ── Release this alpha's models ────────────────────────────────────
        del ref_policy, reward_single, reward_r1, reward_r2
        torch.cuda.empty_cache()

    out_file.close()
    print(f"\nDone. Results written to: {args.output_path}")


if __name__ == "__main__":
    main()
