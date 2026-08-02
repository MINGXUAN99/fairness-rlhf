# Modern backbone options (Pythia, Llama-3.2, Qwen3)

The backbone is now **selectable**. **GPT-2 remains the default** (all legacy
behaviour is preserved). Any HF causal-LM id works via a flag or env var — these
backbones are wired as first-class options:

| Backbone | Model id | Hub-id suffix | Arch | Notes |
|----------|----------|---------------|------|-------|
| GPT-2 (default) | `gpt2` / `lvwerra/gpt2-imdb` | `gpt2` | gpt2 | Legacy full-FT path, unchanged |
| **Pythia-160m** | `EleutherAI/pythia-160m` | `pythia160m` | gpt_neox | **~GPT-2 scale (162M), non-gated — best for reproducing GPT-2** |
| Pythia-410m / 1.4b | `EleutherAI/pythia-{410m,1.4b}` | `pythia410m` / `pythia14b` | gpt_neox | Same suite → clean scaling ablation |
| Llama-3.2-1B | `meta-llama/Llama-3.2-1B` | `llama1b` | llama | Mid-size (~1.2B), gated |
| Qwen3-4B | `Qwen/Qwen3-4B-Base` | `qwen4b` | qwen3 | Largest; LoRA + small critic recommended |

## Why Pythia-160m (closest to GPT-2)

Pythia (EleutherAI, GPT-NeoX architecture, trained on the Pile) is a controlled
scaling suite. **`pythia-160m` (~162M) is the closest modern LM to GPT-2 base
(124M)**, is non-gated, and full fine-tunes exactly like GPT-2 — the strongest
option for reproducing the GPT-2 results with a modern backbone while answering a
reviewer's "use a modern LM (Pythia/variants)" request. The 410m/1.4b siblings
share the architecture and tokenizer, giving a clean single-family scaling study.

GPT-NeoX uses a fused `query_key_value` attention projection and `dense_*` MLP
layers, so its LoRA targets differ from the Llama `*_proj` set;
`default_lora_target_modules` now returns the correct `GPTNEOX_LORA_TARGETS` for
`model_type == "gpt_neox"`, so **both full-FT and LoRA work**. Pythia's tokenizer
differs from GPT-2's, so reusing GPT-2 reward models would hit the cross-tokenizer
path — train the RMs on Pythia so both sides share a tokenizer.

Ready-to-run (reward models + PPO policies, alpha=1.0, prop=0.2, vanilla/maxmin/pcgrad):

```bash
bash scripts/train_pythia160m_rm_and_ppo.sh
# Phase 1 trains vanilla + maxmin RMs on pythia-160m (full-FT); Phase 2 trains the
# three PPO policies. Jobs are DISTRIBUTED across GPUs by job-level parallelism
# (one run per GPU, concurrent), NOT DDP -- each run keeps the single-GPU batch
# size so the RL dynamics match the GPT-2 baseline. Knobs: GPUS="0 1 2 3",
# SKIP_RM=1 to reuse existing RMs, USE_LORA=1 to LoRA-tune.
```

Why job-level parallelism instead of DDP here: DDP (`accelerate ... num_processes=4`)
multiplies PPO's effective batch by the world size (`per_device_train_batch_size`
32 → 128), changing the RL dynamics — undesirable when reproducing the GPT-2
results. At 160M each run fits on one GPU, so pinning one run per GPU
(`CUDA_VISIBLE_DEVICES`) fills the hardware while keeping every run identical to
the single-GPU recipe. Reserve DDP/FSDP for backbones too big for one GPU.

## Why Llama-3.2-1B

- Reward: `--model_hub_id <id>` (or `REWARD_MODEL_BASE=...`)
- Policy: `--policy_model_name <id>` (or `POLICY_MODEL_NAMES=...`)

## Why Llama-3.2-1B (reproducing the GPT-2 results)

Qwen3-4B is ~30× larger than GPT-2, and that size gap — plus LoRA-only adaptation
— is the likely reason it does not reproduce the GPT-2 findings. **Llama-3.2-1B**
is only ~8× GPT-2 (124M → 1.2B) and small enough to **full fine-tune** on a single
32 GB GPU, so it mirrors the original full-FT recipe far more closely while still
answering the reviewer's "use a modern LM" request.

Recommended recipe (full-FT, mirrors GPT-2):

```bash
# Llama models are gated: accept the license on huggingface.co and authenticate.
hf auth login          # or: export HF_TOKEN=...

# 1. Reward models on Llama-3.2-1B, full fine-tuning (USE_LORA=0).
REWARD_MODEL_BASE=meta-llama/Llama-3.2-1B USE_LORA=0 \
  bash scripts/train_reward_models.sh
#    → pushes ...-reward-...-llama1b (+ -minority-llama1b / -majority-llama1b)

# 2. PPO: full-FT Llama-3.2-1B policy. The critic reuses the same 1B backbone
#    (no separate tiny critic needed at 1B), on 4-GPU DDP. RM_SUFFIX=llama1b so it
#    finds the reward models from step 1.
POLICY_MODEL_NAMES=meta-llama/Llama-3.2-1B USE_LORA=0 \
RM_SUFFIX=llama1b GRAD_CKPT=1 \
LAUNCHER="accelerate launch --config_file accelerate_config_ddp.yaml" \
  bash scripts/train_ppo_policy.sh
```

Set `USE_LORA=1` instead to LoRA-tune the 1B backbone if you are VRAM-constrained
(LoRA is the auto-default for any non-GPT-2 backbone; here we override it off to
match GPT-2's full fine-tuning). Everything else — cross-tokenizer alignment being
bypassed when policy and reward share a tokenizer, the merged-adapter push, the
`model_type`-from-backbone save — behaves exactly as documented below for Qwen.

---

## Qwen3-4B

Scope of this option: **vanilla, maxmin, pcgrad** methods. SharedRep still uses
the GPT-2-only `SharedRepGPT2RM` class and was intentionally left unported.

LoRA turns on automatically for non-GPT-2 backbones and stays off for GPT-2
(so the GPT-2 path is full fine-tuning exactly as before). Override with
`--use_lora` / `--no_lora` (or `USE_LORA=1|0`).

## Why both must be Qwen (not GPT-2 reward + Qwen policy)

A GPT-2 reward model + Qwen policy needs cross-tokenizer alignment (decode →
re-encode) on every rollout, which is slow and lossy. With Qwen on both sides the
tokenizers are identical (`len(policy_tokenizer) == len(reward_tokenizer)`), so
the alignment path is bypassed entirely — this is the failure mode you hit before.

The value (critic) may use a **smaller** Qwen model (`Qwen/Qwen3-0.6B-Base`): it
shares Qwen3-4B's tokenizer, so it consumes the policy's token ids directly while
keeping the full-fine-tuned critic cheap.

## Memory design for 4× RTX 5090 (32 GB each)

PPO holds up to four model copies. A full-FT 4B critic alone needs ~64 GB of
optimizer state, so the default recipe is:

| Model   | Backbone        | Training      | ~VRAM / GPU (DDP) |
|---------|-----------------|---------------|-------------------|
| Policy  | Qwen3-4B-Base   | LoRA (bf16)   | ~8 GB base + adapters |
| Ref     | (policy, adapters disabled) | — (null-ref) | 0 (reuses policy) |
| Value   | Qwen3-0.6B-Base | full FT       | ~11 GB |
| Reward  | Qwen3-4B (trained RM) | frozen bf16 | ~8 GB |

≈ 27 GB + activations per GPU; gradient checkpointing (`GRAD_CKPT=1`, on by
default) keeps activations in budget. Four GPUs run four DDP replicas (~4×
throughput). Set `--load_in_4bit_frozen` to NF4-quantize the frozen reward/ref
models if you need more headroom (needs `bitsandbytes>=0.45` for Blackwell).

For **full fine-tuning** of the 4B policy (`USE_LORA=0`), use FSDP:
`ACCEL_CONFIG=accelerate_config_fsdp.yaml USE_LORA=0 bash scripts/train_ppo_policy.sh`.

## Launch

### Default (GPT-2, unchanged)

```bash
bash scripts/train_reward_models.sh      # gpt2 backbone, -gpt2 hub-id suffix
bash scripts/train_ppo_policy.sh         # gpt2 policy, single GPU, full FT
```

### Qwen3-4B (opt-in)

```bash
# Environment (Blackwell needs CUDA 12.8 wheels)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

# 1. Reward models on Qwen3-4B (head-only linear probe by default; USE_LORA=1 for LoRA)
REWARD_MODEL_BASE=Qwen/Qwen3-4B-Base bash scripts/train_reward_models.sh
#    → pushes ...-reward-...-qwen4b  (and -minority-qwen4b / -majority-qwen4b)

# 2. PPO: LoRA Qwen3-4B policy + Qwen3-0.6B critic, 4-GPU DDP.
#    RM_SUFFIX=qwen4b so it finds the Qwen reward models from step 1.
POLICY_MODEL_NAMES=Qwen/Qwen3-4B-Base \
VALUE_MODEL_NAME=Qwen/Qwen3-0.6B-Base \
RM_SUFFIX=qwen4b GRAD_CKPT=1 \
LAUNCHER="accelerate launch --config_file accelerate_config_ddp.yaml" \
  bash scripts/train_ppo_policy.sh

# 3. Evaluate (lower --batch_size for the 4B policy to avoid OOM)
accelerate launch --config_file accelerate_config_ddp.yaml \
  src/helper/generate_responses_unified.py --mode evaluation --batch_size 8 ...
```

## Key knobs (env vars / CLI)

- `USE_LORA=0|1` — force full-FT / LoRA (default: LoRA for non-GPT-2 backbones).
- `--lora_r / --lora_alpha / --lora_dropout` — LoRA hyperparameters.
- `--value_model_name` / `VALUE_MODEL_NAME` — critic backbone (default 0.6B).
- `--gradient_checkpointing` / `GRAD_CKPT=1` — activation-memory savings.
- `--load_in_4bit_frozen` — QLoRA 4-bit for the frozen reward/ref models.
- `RM_SUFFIX` — hub-id suffix; defaults to `qwen4b` (was `gpt2`).

## Loading behaviour notes

- Tokenizer pad tokens are no longer force-set to EOS: `configure_tokenizer`
  only falls back to EOS when the model defines no pad token (GPT-2). Qwen's real
  pad token is preserved.
- LoRA adapters are **merged into the base weights before push** (reward and
  policy), so downstream loaders use plain `AutoModel*` with no PEFT dependency.
- The PPO save no longer hardcodes `model_type = "gpt2"`; it uses the actual
  backbone's `model_type` and refuses to save a checkpoint missing it.

## Not runtime-verified here

These edits were written against a machine without the training env installed, so
they are syntax-checked only. Verify on your GPU box:
1. TRL accepts a PEFT policy + `ref_model=None` (null-ref) in your TRL version.
2. The PCGrad/SumGrad monkey-patch still matches your TRL source (it degrades
   gracefully / skips on mismatch, but confirm the patch applied).
3. `bitsandbytes` 4-bit works on sm_120 if you enable `--load_in_4bit_frozen`.
