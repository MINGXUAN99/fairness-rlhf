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

PYTHON="${PYTHON:-$(which python3)}"
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
fi

# Methods and alphas
METHODS=("vanilla" "maxmin")
ALPHAS="${ALPHAS:-1.0}"
# Support space-separated or array
ALPHAS_ARRAY=($ALPHAS)

echo "=========================================="
echo "Training PPO Policies - Vanilla & MaxMin"
echo "=========================================="
echo "User ID:         $USER_ID"
echo "Seed:            $SEED"
echo "Proportion:      $PROPORTION"
echo "MAX_NEW_TOKENS:  $MAX_NEW_TOKENS"
echo "Methods:        ${METHODS[*]}"
echo "Alpha values:   ${ALPHAS_ARRAY[*]}"
echo "Python:         $PYTHON"
echo "Normalization:  calibration/\${USER_ID}_gpt2-imdb-preference-ALPHA\${ALPHA}-len\${MAX_NEW_TOKENS}-random.json (if present)"
echo "=========================================="
echo ""

TOTAL_RUNS=$((${#METHODS[@]} * ${#ALPHAS_ARRAY[@]}))
CURRENT_RUN=0

for METHOD in "${METHODS[@]}"; do
    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))

        # Reward normalization JSON (same filename as normalize_reward_model.py)
        NORM_JSON="calibration/${USER_ID}_gpt2-imdb-preference-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random.json"
        NORM_ARG=()
        if [ -f "$NORM_JSON" ]; then
            NORM_ARG=(--reward_normalization_json "$NORM_JSON")
        fi

        echo ""
        echo "=========================================="
        echo "Run $CURRENT_RUN/$TOTAL_RUNS: $METHOD alpha=$ALPHA"
        echo "=========================================="
        echo "Method:          $METHOD"
        echo "Alpha:           $ALPHA"
        echo "MAX_NEW_TOKENS:  $MAX_NEW_TOKENS"
        if [ ${#NORM_ARG[@]} -gt 0 ]; then
            echo "Normalization:   $NORM_JSON"
        else
            echo "Normalization:   (none)"
        fi
        echo "=========================================="
        echo ""

        "$PYTHON" src/imdb/ppo_training/unified_ppo_training.py \
            --method "$METHOD" \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA" \
            --max_new_tokens "$MAX_NEW_TOKENS" \
            "${NORM_ARG[@]}"

        POLICY_HUB_ID="${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}"
        echo ""
        echo "✅ Completed: $METHOD alpha=$ALPHA"
        echo "   Policy: $POLICY_HUB_ID"
        echo ""
    done
done

echo ""
echo "=========================================="
echo "✅ All PPO Training Completed!"
echo "=========================================="
echo ""
echo "Trained policies (naming from unified_ppo_training.get_policy_hub_id):"
for METHOD in "${METHODS[@]}"; do
    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        echo "  - ${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}"
    done
done
echo ""
