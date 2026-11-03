#!/bin/bash
# Train PPO policies over methods and alpha values.
#
# Methods (--method): vanilla, maxmin, sharedrep, gold, gold_vanilla, pcgrad,
#   sumgrad, mgda. The multi-objective combiners (pcgrad/sumgrad/mgda) reuse the
#   MaxMin minority/majority reward models. mgda = MGDA min-norm common-descent
#   (Sener & Koltun 2018, arXiv:1810.04650).
#
# Logging: two log files (same pattern as build_dataset.sh / train_reward_models.sh):
#   - Echo log: only this script's echo lines (clean + readable)
#   - Full log: everything (including Python output), for debugging
#
# Naming: aligned with train_reward_models (ALPHA-len-seed); uses reward_normalization JSON when present.
#
# =============================================================================

set -e  # Exit on error

# =============================================================================
# Ensure we're in the project root directory
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# =============================================================================
# Logging: create TWO log files (same pattern as train_reward_models.sh)
# =============================================================================
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_PREFIX="$(basename "$0" .sh)"
TS="${FAIRNESS_RLHF_LOG_TS:-$(date +%Y%m%d_%H%M%S)}"
ECHO_LOG_FILE="${FAIRNESS_RLHF_ECHO_LOG_FILE:-$LOG_DIR/${LOG_PREFIX}_${TS}_echo.log}"
FULL_LOG_FILE="${FAIRNESS_RLHF_FULL_LOG_FILE:-$LOG_DIR/${LOG_PREFIX}_${TS}_full.log}"

# Wrapper: re-exec under `script` to capture full output; internal mode runs the pipeline.
if [ "${1:-}" != "__internal" ] && [ -z "${FAIRNESS_RLHF_INTERNAL_RUN:-}" ]; then
    : > "$ECHO_LOG_FILE"
    builtin echo "Echo log: $ECHO_LOG_FILE"
    builtin echo "Full log: $FULL_LOG_FILE"
    builtin echo ""

    export FAIRNESS_RLHF_LOG_TS="$TS"
    export FAIRNESS_RLHF_ECHO_LOG_FILE="$ECHO_LOG_FILE"
    export FAIRNESS_RLHF_FULL_LOG_FILE="$FULL_LOG_FILE"
    export FAIRNESS_RLHF_INTERNAL_RUN=1

    if command -v script >/dev/null 2>&1; then
        script -q -e -f "$FULL_LOG_FILE" -c "bash \"$0\" __internal"
        exit $?
    else
        builtin echo "WARNING: 'script' not found; falling back to tee for full log."
        builtin echo "         (This may cause tqdm progress bars to spam the full log.)"
        bash "$0" __internal 2>&1 | tee -a "$FULL_LOG_FILE"
        exit ${PIPESTATUS[0]}
    fi
fi

# Internal mode: append only echo lines to the echo log (and print to terminal).
echo() {
    builtin echo "$@" | tee -a "$ECHO_LOG_FILE"
}

echo "Echo log: $ECHO_LOG_FILE"
echo "Full log: $FULL_LOG_FILE"
echo ""

# =============================================================================
# Configuration (naming aligned with train_reward_models / build_dataset)
# =============================================================================
USER_ID="${USER_ID:-mxcui}"
SEED="${SEED:-42}"
PROPORTION="${PROPORTION:-0.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ALPHA="${ALPHA:-1.0}"
# Policy base models: e.g. "gpt2" (default), "meta-llama/Llama-3.2-1B", or
# "Qwen/Qwen3-4B-Base" (space-separated to train multiple in one run).
POLICY_MODEL_NAMES="${POLICY_MODEL_NAMES:-gpt2}"
POLICY_MODEL_NAMES_ARRAY=($POLICY_MODEL_NAMES)
# Value (critic) backbone. Empty -> use the policy backbone (default). For a large
# LoRA policy (e.g. Qwen3-4B) set a small SAME-tokenizer critic, e.g.
# VALUE_MODEL_NAME=Qwen/Qwen3-0.6B-Base, to keep the full-FT value model cheap.
VALUE_MODEL_NAME="${VALUE_MODEL_NAME:-}"
VALUE_ARG=()
[ -n "$VALUE_MODEL_NAME" ] && VALUE_ARG=(--value_model_name "$VALUE_MODEL_NAME")
# LoRA on the policy: unset -> Python default (on for non-GPT-2, off for gpt2). USE_LORA=0/1 to force.
USE_LORA="${USE_LORA:-}"
LORA_ARG=()
[ "$USE_LORA" = "1" ] && LORA_ARG=(--use_lora)
[ "$USE_LORA" = "0" ] && LORA_ARG=(--no_lora)
# Gradient checkpointing (recommended for large backbones). GRAD_CKPT=1 to enable.
GRAD_CKPT="${GRAD_CKPT:-0}"
GRAD_CKPT_ARG=()
[ "$GRAD_CKPT" = "1" ] && GRAD_CKPT_ARG=(--gradient_checkpointing)
# NF4-quantize the frozen reward/ref models (QLoRA-style) for extra VRAM headroom.
# LOAD_IN_4BIT_FROZEN=1 to enable (needs bitsandbytes>=0.45 on Blackwell).
LOAD_4BIT_ARG=()
[ "${LOAD_IN_4BIT_FROZEN:-0}" = "1" ] && LOAD_4BIT_ARG=(--load_in_4bit_frozen)
# Learning rate (default 3e-6 legacy; ~1.4e-5 is the standard TRL sentiment-PPO
# value — raise if objective/scores stays flat, i.e. the policy isn't learning).
LEARNING_RATE="${LEARNING_RATE:-3e-6}"
# Total PPO episodes (more = more updates ≈ total_episodes/batch). Default 50000.
TOTAL_EPISODES="${TOTAL_EPISODES:-50000}"
# Optional tag appended to the policy hub id so tuning runs don't clobber existing
# policies, e.g. RUN_TAG=lr1.4e-5 -> ...-EleutherAI-pythia-160m-lr1.4e-5
RUN_TAG="${RUN_TAG:-}"
# KL coefficient (default 0.1; use KL_COEF=0 for ablation - no KL penalty)
KL_COEF="${KL_COEF:-0.1}"
# KL penalty type: "total" = TRL default (sum over tokens), "mean_kl" = mean per token (length-invariant)
KL_DIV="${KL_DIV:-mean_kl}"
[ "$KL_DIV" = "mean_kl" ] && KL_SUFFIX="mean_kl" || KL_SUFFIX="kl"
# Set NO_REWARD_NORM=1 to disable reward normalization (use raw RM logits)
NO_REWARD_NORM="${NO_REWARD_NORM:-1}"
# MGDA only: mean-norm rescale the two group gradients before the min-norm solve
# (recommended when the minority/majority RMs differ in reward scale). MGDA_NORMALIZE=1 to enable.
MGDA_NORMALIZE="${MGDA_NORMALIZE:-0}"
# pcgrad/sumgrad/mgda: record cos(g1, g2) between the minority/majority objective
# gradients every update to a CSV under diagnostics/ (gradient-conflict trace).
# RECORD_COSINE=1 to enable.
RECORD_COSINE="${RECORD_COSINE:-0}"
# PCGrad only: inclusive OUTER PPO update bounds. Empty means unbounded.
# Outside the window use SumGrad; omit both for full PCGrad.
PCGRAD_WINDOW_START="${PCGRAD_WINDOW_START:-}"
PCGRAD_WINDOW_END="${PCGRAD_WINDOW_END:-}"
WINDOW_SUFFIX=""
if [ -n "$PCGRAD_WINDOW_START" ] || [ -n "$PCGRAD_WINDOW_END" ]; then
    WINDOW_SUFFIX="-win${PCGRAD_WINDOW_START:-1}-${PCGRAD_WINDOW_END:-end}"
fi

# HuggingFace token for private repos (export HF_TOKEN or run: hf auth login)
# Python/transformers use HF_TOKEN automatically for from_pretrained/push_to_hub

# Optional RM hub ID overrides (when set, passed to PPO training)
# RM hub-id suffix should match train_reward_models.sh RM_SUFFIX (default gpt2;
# e.g. pythia160m for EleutherAI/pythia-160m, llama1b for meta-llama/Llama-3.2-1B,
# qwen4b for a Qwen3-4B reward backbone).
RM_SUFFIX="${RM_SUFFIX:-gpt2}"
VANILLA_RM_HUB_ID="${USER_ID}/vanilla-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-${RM_SUFFIX}"
MAXMIN_MINORITY_RM_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-minority-${RM_SUFFIX}"
MAXMIN_MAJORITY_RM_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-majority-${RM_SUFFIX}"

PYTHON="${PYTHON:-$(which python3)}"
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
fi

# Methods and alphas (space-separated for multiple)
METHODS="${METHODS:-vanilla maxmin}"
METHODS_ARRAY=($METHODS)
if [ -n "$WINDOW_SUFFIX" ]; then
    for METHOD in "${METHODS_ARRAY[@]}"; do
        if [ "$METHOD" != "pcgrad" ]; then
            echo "ERROR: PCGRAD_WINDOW_START/END require METHODS=pcgrad."
            exit 1
        fi
    done
fi
ALPHAS="${ALPHAS:-1.0}"
ALPHAS_ARRAY=($ALPHAS)

echo "=========================================="
echo "Training PPO Policies - Vanilla & MaxMin"
echo "=========================================="
echo "User ID:         $USER_ID"
echo "Seed:            $SEED"
echo "Proportion:      $PROPORTION"
echo "MAX_NEW_TOKENS:  $MAX_NEW_TOKENS"
echo "Methods:        ${METHODS_ARRAY[*]}"
echo "Alpha values:   ${ALPHAS_ARRAY[*]}"
echo "Policy models:  ${POLICY_MODEL_NAMES_ARRAY[*]}"
echo "KL coefficient: $KL_COEF"
echo "KL div:         $KL_DIV (suffix: ${KL_SUFFIX})"
echo "Python:         $PYTHON"
echo "Normalization:  calibration/\${USER_ID}_gpt2-imdb-preference-ALPHA\${ALPHA}-len\${MAX_NEW_TOKENS}-random.json (if present)"
[ -n "$VANILLA_RM_HUB_ID" ] && echo "Vanilla RM:     $VANILLA_RM_HUB_ID"
[ -n "$MAXMIN_MINORITY_RM_HUB_ID" ] && echo "MaxMin RM min:  $MAXMIN_MINORITY_RM_HUB_ID"
[ -n "$MAXMIN_MAJORITY_RM_HUB_ID" ] && echo "MaxMin RM maj:  $MAXMIN_MAJORITY_RM_HUB_ID"
echo "=========================================="
echo ""

TOTAL_RUNS=$((${#METHODS_ARRAY[@]} * ${#POLICY_MODEL_NAMES_ARRAY[@]}))
CURRENT_RUN=0

for METHOD in "${METHODS_ARRAY[@]}"; do
    for POLICY_MODEL_NAME in "${POLICY_MODEL_NAMES_ARRAY[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Reward normalization JSON (same filename as normalize_reward_model.py)
        NORM_JSON="calibration/${USER_ID}_gpt2-imdb-preference-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random.json"
        NORM_ARG=()
        if [ "$NO_REWARD_NORM" = "1" ]; then
            NORM_ARG=(--no_reward_normalization)
        elif [ -f "$NORM_JSON" ]; then
            NORM_ARG=(--reward_normalization_json "$NORM_JSON")
        fi

        RM_ARGS=()
        [ -n "$VANILLA_RM_HUB_ID" ] && RM_ARGS+=(--vanilla_rm_hub_id "$VANILLA_RM_HUB_ID")
        [ -n "$MAXMIN_MINORITY_RM_HUB_ID" ] && RM_ARGS+=(--maxmin_minority_rm_hub_id "$MAXMIN_MINORITY_RM_HUB_ID")
        [ -n "$MAXMIN_MAJORITY_RM_HUB_ID" ] && RM_ARGS+=(--maxmin_majority_rm_hub_id "$MAXMIN_MAJORITY_RM_HUB_ID")

        echo ""
        echo "=========================================="
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: $METHOD + $POLICY_MODEL_NAME"
        echo "=========================================="
        echo "Method:          $METHOD"
        echo "Policy model:    $POLICY_MODEL_NAME"
        echo "Alpha:           $ALPHA"
        echo "MAX_NEW_TOKENS:  $MAX_NEW_TOKENS"
        echo "KL coefficient:  $KL_COEF"
        if [ "$NO_REWARD_NORM" = "1" ]; then
            echo "Normalization:   (disabled via NO_REWARD_NORM)"
        elif [ ${#NORM_ARG[@]} -gt 0 ]; then
            echo "Normalization:   $NORM_JSON"
        else
            echo "Normalization:   (none)"
        fi
        if [ ${#RM_ARGS[@]} -gt 0 ]; then
            echo "RM overrides:    ${RM_ARGS[*]}"
        fi
        echo "=========================================="
        echo ""

        # Sanitize model name for hub ID (replace / with -)
        POLICY_MODEL_SUFFIX="${POLICY_MODEL_NAME//\//-}"
        POLICY_HUB_ID="${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-${KL_SUFFIX}${KL_COEF}-${POLICY_MODEL_SUFFIX}${WINDOW_SUFFIX}${RUN_TAG:+-${RUN_TAG}}"

        # Method-specific extra args (MGDA normalization + gradient-conflict CSV).
        METHOD_EXTRA_ARGS=()
        if [ "$METHOD" = "pcgrad" ] && [ -n "$WINDOW_SUFFIX" ]; then
            [ -n "$PCGRAD_WINDOW_START" ] && METHOD_EXTRA_ARGS+=(--pcgrad_window_start "$PCGRAD_WINDOW_START")
            [ -n "$PCGRAD_WINDOW_END" ] && METHOD_EXTRA_ARGS+=(--pcgrad_window_end "$PCGRAD_WINDOW_END")
            echo "PCGrad window:  [${PCGRAD_WINDOW_START:-1}, ${PCGRAD_WINDOW_END:-end}] inclusive; SumGrad outside"
        fi
        if [ "$METHOD" = "mgda" ] && [ "$MGDA_NORMALIZE" = "1" ]; then
            METHOD_EXTRA_ARGS+=(--mgda_normalize)
            echo "MGDA:            mean-norm rescale before min-norm solve (--mgda_normalize)"
        fi
        if { [ "$METHOD" = "pcgrad" ] || [ "$METHOD" = "mgda" ] || [ "$METHOD" = "sumgrad" ]; } && [ "$RECORD_COSINE" = "1" ]; then
            mkdir -p "$PROJECT_ROOT/diagnostics"
            COSINE_CSV="diagnostics/cosine_${METHOD}-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-${KL_SUFFIX}${KL_COEF}-${POLICY_MODEL_SUFFIX}${WINDOW_SUFFIX}${RUN_TAG:+-${RUN_TAG}}.csv"
            METHOD_EXTRA_ARGS+=(--sumgrad_cosine_csv "$COSINE_CSV")
            echo "Conflict CSV:    $COSINE_CSV"
        fi

        "$PYTHON" src/imdb/ppo_training/unified_ppo_training.py \
            --method "$METHOD" \
            --policy_model_name "$POLICY_MODEL_NAME" \
            "${VALUE_ARG[@]}" \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --kl_coef "$KL_COEF" \
            --kl_div "$KL_DIV" \
            --learning_rate "$LEARNING_RATE" \
            --total_episodes "$TOTAL_EPISODES" \
            --policy_hub_id "$POLICY_HUB_ID" \
            "${LORA_ARG[@]}" \
            "${GRAD_CKPT_ARG[@]}" \
            "${LOAD_4BIT_ARG[@]}" \
            "${NORM_ARG[@]}" \
            "${RM_ARGS[@]}" \
            "${METHOD_EXTRA_ARGS[@]}"
        echo ""
        echo "✅ Completed: $METHOD + $POLICY_MODEL_NAME"
        echo "   Policy: $POLICY_HUB_ID"
        echo ""
    done
done

echo ""
echo "=========================================="
echo "✅ All PPO Training Completed!"
echo "=========================================="
echo ""
echo "Trained policies:"
for METHOD in "${METHODS_ARRAY[@]}"; do
    for POLICY_MODEL_NAME in "${POLICY_MODEL_NAMES_ARRAY[@]}"; do
        POLICY_MODEL_SUFFIX="${POLICY_MODEL_NAME//\//-}"
        echo "  - ${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-${KL_SUFFIX}${KL_COEF}-${POLICY_MODEL_SUFFIX}${WINDOW_SUFFIX}${RUN_TAG:+-${RUN_TAG}}"
    done
done
echo ""
