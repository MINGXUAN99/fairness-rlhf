"""
Backbone-agnostic helpers for switching between GPT-2 and modern decoder LMs
(Qwen3, LLaMA, Pythia/GPT-NeoX, etc.) as the reward / policy / value backbone.

The original pipeline was hard-wired to GPT-2 (``model.transformer``,
``pad_token = eos_token``, ``model_type = "gpt2"``).  These helpers centralise
the small number of backbone-specific decisions so the training scripts stay
model-agnostic:

* :func:`configure_tokenizer` — sets a pad token only when the model does not
  already define one (Qwen3 ships a real ``<|endoftext|>`` pad token; GPT-2 has
  none and must borrow the EOS token).
* :func:`get_backbone` — returns the transformer trunk regardless of whether the
  attribute is ``.transformer`` (GPT-2) or ``.model`` (Qwen3/LLaMA).
* :func:`build_lora_config` / :func:`maybe_bnb_config` — LoRA + optional 4-bit
  (QLoRA) configuration so a 4B backbone fits on a single 32 GB GPU.
"""
from __future__ import annotations

from typing import Optional


# Attention + MLP projection names shared by Qwen2/Qwen3/LLaMA/Mistral.  GPT-2
# uses fused ``c_attn``/``c_proj`` Conv1D layers instead; we special-case it.
QWEN_LIKE_LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"]
GPT2_LORA_TARGETS = ["c_attn", "c_proj", "c_fc"]
# GPT-NeoX (Pythia) uses a fused ``query_key_value`` attention projection and
# ``dense``/``dense_h_to_4h``/``dense_4h_to_h`` MLP layers -- the *_proj names
# above do not exist, so LoRA needs this dedicated set.
GPTNEOX_LORA_TARGETS = ["query_key_value", "dense",
                        "dense_h_to_4h", "dense_4h_to_h"]


def default_lora_target_modules(model) -> list[str]:
    """Pick sensible LoRA target modules from the model's architecture."""
    model_type = getattr(getattr(model, "config", None), "model_type", "") or ""
    if "gpt2" in model_type:
        return GPT2_LORA_TARGETS
    if "gpt_neox" in model_type:  # Pythia / GPT-NeoX family
        return GPTNEOX_LORA_TARGETS
    # Qwen3/Qwen2/LLaMA/Mistral and most modern decoders use *_proj names.
    return QWEN_LIKE_LORA_TARGETS


def configure_tokenizer(tokenizer, padding_side: Optional[str] = None):
    """
    Ensure the tokenizer has a pad token without clobbering a model that already
    defines one.

    GPT-2 has no pad token → fall back to EOS (historical behaviour).
    Qwen3 already defines a distinct pad token → leave it untouched, so padding
    is not confused with end-of-sequence.
    """
    if padding_side is not None:
        tokenizer.padding_side = padding_side
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer defines neither pad_token nor eos_token; cannot pad."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_backbone(model):
    """
    Return the transformer trunk of a (possibly PEFT-wrapped) model.

    Works for GPT-2 (``.transformer``) and Qwen3/LLaMA (``.model``).  Unwraps a
    single PEFT layer via ``get_base_model`` when present.
    """
    if hasattr(model, "get_base_model"):
        model = model.get_base_model()
    prefix = getattr(model, "base_model_prefix", None)
    if prefix and hasattr(model, prefix):
        return getattr(model, prefix)
    if hasattr(model, "model"):
        return model.model
    if hasattr(model, "transformer"):
        return model.transformer
    raise AttributeError(
        f"{type(model).__name__} has no recognised backbone (.model/.transformer)"
    )


def freeze_backbone(model) -> None:
    """Freeze every parameter of the transformer trunk (reward-head-only training)."""
    for param in get_backbone(model).parameters():
        param.requires_grad = False


def build_lora_config(
    task_type: str,
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    target_modules: Optional[list[str]] = None,
    modules_to_save: Optional[list[str]] = None,
):
    """
    Build a PEFT ``LoraConfig``.

    Args:
        task_type: ``"CAUSAL_LM"`` for the policy, ``"SEQ_CLS"`` for reward/value
            models (so the classification ``score`` head is trained via
            ``modules_to_save``).
        target_modules: LoRA target module names; inferred per-backbone when None
            (caller should pass ``default_lora_target_modules(model)``).
        modules_to_save: extra modules trained in full precision (e.g. the reward
            ``score`` head, which is randomly initialised and must adapt).
    """
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type=task_type,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
    )


def cleanup_stale_hub_files(repo_id, filenames, repo_type: str = "model") -> None:
    """
    Delete leftover files from a Hub repo after a full/merged-model push.

    Both reward-training push paths (full-FT ``trainer.push_to_hub`` and LoRA
    ``merged.push_to_hub``) upload a *plain* AutoModel — no PEFT adapter should
    remain. But ``push_to_hub`` is additive: if the repo previously held a PEFT
    adapter (e.g. a Qwen LoRA run reusing the same repo name), a stale
    ``adapter_config.json`` + empty ``adapter_model.safetensors`` survive and
    hijack ``AutoModelForSequenceClassification.from_pretrained`` (it follows the
    adapter to the wrong base model). This removes such leftovers. Missing files
    are ignored.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    for fn in filenames:
        try:
            api.delete_file(
                path_in_repo=fn, repo_id=repo_id, repo_type=repo_type,
                commit_message=f"Remove stale {fn} after full-model push",
            )
        except Exception:
            pass  # not present → nothing to clean


def maybe_bnb_config(load_in_4bit: bool):
    """
    Return a ``BitsAndBytesConfig`` for NF4 4-bit loading (QLoRA), or None.

    Used to shrink the frozen reference / reward models so all four PPO model
    copies fit alongside a LoRA-trained policy.  Requires bitsandbytes with
    Blackwell (sm_120) support (>= 0.45) on RTX 5090.
    """
    if not load_in_4bit:
        return None
    import torch
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


def add_lora_cli_args(parser) -> None:
    """Register the shared LoRA / quantisation CLI flags on an argparse parser."""
    parser.add_argument(
        "--use_lora",
        dest="use_lora",
        action="store_true",
        default=None,
        help="Train with LoRA adapters instead of full fine-tuning. Recommended "
             "for large (>=1B) backbones. Default: on for non-GPT-2 backbones.",
    )
    parser.add_argument(
        "--no_lora",
        dest="use_lora",
        action="store_false",
        help="Force full-parameter fine-tuning (disable LoRA).",
    )
    parser.add_argument("--lora_r", type=int, default=16, help="LoRA rank.")
    parser.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument(
        "--load_in_4bit",
        action="store_true",
        help="QLoRA: load the backbone in NF4 4-bit (needs bitsandbytes>=0.45 on Blackwell).",
    )


def resolve_use_lora(use_lora_flag, model_name: str) -> bool:
    """
    Decide whether to use LoRA when the CLI flag is left unset (None):
    default to LoRA for any non-GPT-2 backbone, off for GPT-2 (preserves the
    original small-model full-FT behaviour).
    """
    if use_lora_flag is not None:
        return bool(use_lora_flag)
    return "gpt2" not in model_name.lower()
