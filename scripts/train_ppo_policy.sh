#!/bin/bash
# Train PPO policies for vanilla and maxmin methods over alpha values.
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
        script -q -f "$FULL_LOG_FILE" -c "bash \"$0\" __internal"
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
# Policy base models: e.g. "gpt2" or "Qwen/Qwen3-0.6B" (space-separated for multiple)
POLICY_MODEL_NAMES="${POLICY_MODEL_NAMES:-gpt2 Qwen/Qwen3-0.6B}"
POLICY_MODEL_NAMES_ARRAY=($POLICY_MODEL_NAMES)
# KL coefficient (default 0.1; use KL_COEF=0 for ablation - no KL penalty)
KL_COEF="${KL_COEF:-0.1}"
# Set NO_REWARD_NORM=1 to disable reward normalization (use raw RM logits)
NO_REWARD_NORM="${NO_REWARD_NORM:-1}"

# HuggingFace token for private repos (export HF_TOKEN or run: hf auth login)
# Python/transformers use HF_TOKEN automatically for from_pretrained/push_to_hub

# Optional RM hub ID overrides (when set, passed to PPO training)
VANILLA_RM_HUB_ID="${USER_ID}/vanilla-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-gpt2"
MAXMIN_MINORITY_RM_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-minority-gpt2"
MAXMIN_MAJORITY_RM_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-majority-gpt2"

PYTHON="${PYTHON:-$(which python3)}"
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
fi

# Methods and alphas (space-separated for multiple)
METHODS="${METHODS:-vanilla maxmin}"
METHODS_ARRAY=($METHODS)
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
        POLICY_HUB_ID="${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-mean_kl${KL_COEF}-${POLICY_MODEL_SUFFIX}"

        CUDA_VISIBLE_DEVICES=1 "$PYTHON" src/imdb/ppo_training/unified_ppo_training.py \
            --method "$METHOD" \
            --policy_model_name "$POLICY_MODEL_NAME" \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            --kl_coef "$KL_COEF" \
            --policy_hub_id "$POLICY_HUB_ID" \
            "${NORM_ARG[@]}" \
            "${RM_ARGS[@]}"
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
        echo "  - ${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-mean_kl${KL_COEF}-${POLICY_MODEL_SUFFIX}"
    done
done
echo ""
