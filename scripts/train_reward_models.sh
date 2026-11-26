#!/bin/bash
# Train all reward models: Vanilla (baseline) and MaxMin (both groups)
# For multiple alpha values
#
# Logging: two log files (like build_dataset.sh):
#   - Echo log: only this script's echo lines (clean + readable)
#   - Full log: everything (including Python output), for debugging
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
# Logging: create TWO log files (same pattern as build_dataset.sh)
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

echo "🔍 DEBUG: Changed to project root: $(pwd)"
echo "🔍 DEBUG: Script directory: $SCRIPT_DIR"

# =============================================================================
# Configuration (naming aligned with build_dataset.sh)
# =============================================================================
USER_ID="${USER_ID:-mxcui}"  # Set via environment or replace with your HF username
SEED="${SEED:-42}"
PROPORTION="${PROPORTION:-0.2}"
NUM_RESPONSES="${NUM_RESPONSES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
# Use venv Python if available, otherwise use system python3
PYTHON="${PYTHON:-$(which python3)}"
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
fi

# Alpha values configuration
ALPHAS="${ALPHAS:-1.0}"  # Alpha values to train

# Reward model backbone. Default: gpt2 (legacy). Modern backbones are opt-in, e.g.
#   REWARD_MODEL_BASE=EleutherAI/pythia-160m    bash scripts/train_reward_models.sh
#   REWARD_MODEL_BASE=meta-llama/Llama-3.2-1B   bash scripts/train_reward_models.sh
#   REWARD_MODEL_BASE=Qwen/Qwen3-4B-Base        bash scripts/train_reward_models.sh
# EleutherAI/pythia-160m (~162M) is the closest modern LM to GPT-2 base (124M),
# non-gated, and full fine-tunable exactly like GPT-2 -- the best pick for
# reproducing the GPT-2 results with a modern backbone. Llama-3.2-1B is a larger
# (~1.2B) mid-size option. (Llama/Qwen may be gated: accept the license and set
# HF_TOKEN / `hf auth login` first; Pythia is not gated.)
REWARD_MODEL_BASE="${REWARD_MODEL_BASE:-gpt2}"
# Suffix for RM hub IDs (e.g. gpt2, qwen4b, qwen06b, llama1b). Default derived from backbone.
RM_SUFFIX="${RM_SUFFIX:-}"
if [ -z "$RM_SUFFIX" ]; then
    case "$REWARD_MODEL_BASE" in
        gpt2|*gpt2*) RM_SUFFIX="gpt2" ;;
        *Qwen3-4B*)  RM_SUFFIX="qwen4b" ;;
        *Qwen3-0.6B*) RM_SUFFIX="qwen06b" ;;
        *Llama-3.2-1B*|*Llama-3.2-1b*|*llama-3.2-1b*) RM_SUFFIX="llama1b" ;;
        *Llama-3.2-3B*|*Llama-3.2-3b*|*llama-3.2-3b*) RM_SUFFIX="llama3b" ;;
        *Llama*|*llama*|*LLaMA*) RM_SUFFIX="llama" ;;
        *pythia-160m*|*Pythia-160m*) RM_SUFFIX="pythia160m" ;;
        *pythia-410m*|*Pythia-410m*) RM_SUFFIX="pythia410m" ;;
        *pythia-1b*|*Pythia-1b*) RM_SUFFIX="pythia1b" ;;
        *pythia-1.4b*|*Pythia-1.4b*) RM_SUFFIX="pythia14b" ;;
        *pythia*|*Pythia*) RM_SUFFIX="pythia" ;;
        *) RM_SUFFIX="qwen4b" ;;
    esac
fi
# LoRA: unset lets the Python default decide (on for non-GPT-2). Set USE_LORA=0 to
# force head-only linear-probe training, USE_LORA=1 to force LoRA.
USE_LORA="${USE_LORA:-}"
LORA_ARG=()
[ "$USE_LORA" = "1" ] && LORA_ARG=(--use_lora)
[ "$USE_LORA" = "0" ] && LORA_ARG=(--no_lora)

# Subset of RM heads to train (enables per-GPU parallel dispatch: run this script
# once per head, each pinned to a different GPU). Default: all three. Valid tokens:
# vanilla, maxmin-minority, maxmin-majority.
RM_METHODS="${RM_METHODS:-vanilla maxmin-minority maxmin-majority}"
_rm_selected() { case " $RM_METHODS " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

# Skip flags (set to non-empty to skip, e.g. SKIP_TRAIN=1 or SKIP_NORM=1)
SKIP_TRAIN="${SKIP_TRAIN:-}"
SKIP_NORM="${SKIP_NORM:-}"

echo "🚀 Starting Reward Model Training Pipeline"
echo "================================================"
echo "User ID:         $USER_ID"
echo "Seed:            $SEED"
echo "Proportion:      $PROPORTION"
echo "NUM_RESPONSES:   $NUM_RESPONSES"
echo "MAX_NEW_TOKENS:  $MAX_NEW_TOKENS"
echo "Alphas:          $ALPHAS"
echo "Reward model:    $REWARD_MODEL_BASE"
echo "Python:          $PYTHON"
echo "SKIP_TRAIN:      ${SKIP_TRAIN:-0}"
echo "SKIP_NORM:       ${SKIP_NORM:-0}"
echo "Mode:            Single GPU (no accelerate)"
echo "Preference IDs:  \${USER_ID}/gpt2-imdb-preference-ALPHA\${ALPHA}-len\${MAX_NEW_TOKENS}-random"
echo "================================================"
echo ""

# Convert alphas to array for counting
ALPHAS_ARRAY=($ALPHAS)
TOTAL_ALPHAS=${#ALPHAS_ARRAY[@]}
# Training methods: vanilla, maxmin-minority, maxmin-majority
TOTAL_METHODS=3
TOTAL_MODELS=$((TOTAL_ALPHAS * TOTAL_METHODS))

echo "📊 Training $TOTAL_MODELS models:"
echo "  - Methods: Vanilla, MaxMin-Minority, MaxMin-Majority"
echo "  - Alphas: $ALPHAS"
echo "  - Order: Iterate through methods first, then alphas for each method"
echo ""

# =============================================================================
# Part 1: Train models (params always defined; skip only Python calls if SKIP_TRAIN)
# =============================================================================
MODEL_COUNTER=0

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 Part 1: Training Models (Vanilla → MaxMin)"
echo "════════════════════════════════════════════════════════════════"
echo ""

# =========================================================================
# Method 1: Train Vanilla (Baseline) Reward Model for all alphas
# =========================================================================
METHOD_COUNTER=1
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 Method $METHOD_COUNTER/$TOTAL_METHODS: Vanilla (Baseline) Reward Model"
echo "════════════════════════════════════════════════════════════════"
echo ""

ALPHA_COUNTER=0
for ALPHA in $ALPHAS; do
    ALPHA_COUNTER=$((ALPHA_COUNTER + 1))
    MODEL_COUNTER=$((MODEL_COUNTER + 1))
    PREFERENCE_EXP_TAG="ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random"
    PREFERENCE_DATASET_HUB_ID="${USER_ID}/gpt2-imdb-preference-${PREFERENCE_EXP_TAG}"
    VANILLA_REWARD_HUB_ID="${USER_ID}/vanilla-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-${RM_SUFFIX}"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "📊 [$MODEL_COUNTER/$TOTAL_MODELS] Vanilla - Alpha=$ALPHA ($ALPHA_COUNTER/$TOTAL_ALPHAS)"
    echo "────────────────────────────────────────────────────────────────"
    echo "Architecture: Single $REWARD_MODEL_BASE → Linear → Reward"
    echo "Preference dataset: $PREFERENCE_DATASET_HUB_ID"
    echo "Hub model id:       $VANILLA_REWARD_HUB_ID"
    
    if [ -z "$SKIP_TRAIN" ] && _rm_selected vanilla; then
        "$PYTHON" src/imdb/reward_training/vanilla_reward_training.py \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA" \
            --preference_dataset_hub_id "$PREFERENCE_DATASET_HUB_ID" \
            --hub_model_id "$VANILLA_REWARD_HUB_ID" \
            --model_hub_id "$REWARD_MODEL_BASE" \
            "${LORA_ARG[@]}"
        echo "✅ Vanilla model (alpha=$ALPHA) training complete"
    else
        echo "⏭️  Vanilla model (alpha=$ALPHA) training skipped (SKIP_TRAIN)"
    fi
    echo ""
done

# =========================================================================
# Method 2: Train MaxMin Reward Model - Minority Group for all alphas
# =========================================================================
METHOD_COUNTER=2
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 Method $METHOD_COUNTER/$TOTAL_METHODS: MaxMin-Minority Reward Model"
echo "════════════════════════════════════════════════════════════════"
echo ""

ALPHA_COUNTER=0
for ALPHA in $ALPHAS; do
    ALPHA_COUNTER=$((ALPHA_COUNTER + 1))
    MODEL_COUNTER=$((MODEL_COUNTER + 1))
    PREFERENCE_EXP_TAG="ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random"
    PREFERENCE_DATASET_HUB_ID="${USER_ID}/gpt2-imdb-preference-${PREFERENCE_EXP_TAG}"
    MAXMIN_REWARD_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-minority-${RM_SUFFIX}"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "📊 [$MODEL_COUNTER/$TOTAL_MODELS] MaxMin-Minority - Alpha=$ALPHA ($ALPHA_COUNTER/$TOTAL_ALPHAS)"
    echo "────────────────────────────────────────────────────────────────"
    echo "Architecture: Separate $REWARD_MODEL_BASE → Linear → Reward (for minority)"
    echo "Preference dataset: $PREFERENCE_DATASET_HUB_ID"
    echo "Hub model id:       $MAXMIN_REWARD_HUB_ID"
    
    if [ -z "$SKIP_TRAIN" ] && _rm_selected maxmin-minority; then
        "$PYTHON" src/imdb/reward_training/maxmin_reward_training.py \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA" \
            --group 0 \
            --preference_dataset_hub_id "$PREFERENCE_DATASET_HUB_ID" \
            --hub_model_id "$MAXMIN_REWARD_HUB_ID" \
            --model_hub_id "$REWARD_MODEL_BASE" \
            "${LORA_ARG[@]}"
        echo "✅ MaxMin minority model (alpha=$ALPHA) training complete"
    else
        echo "⏭️  MaxMin minority model (alpha=$ALPHA) training skipped (SKIP_TRAIN)"
    fi
    echo ""
done

# =========================================================================
# Method 3: Train MaxMin Reward Model - Majority Group for all alphas
# =========================================================================
METHOD_COUNTER=3
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 Method $METHOD_COUNTER/$TOTAL_METHODS: MaxMin-Majority Reward Model"
echo "════════════════════════════════════════════════════════════════"
echo ""

ALPHA_COUNTER=0
for ALPHA in $ALPHAS; do
    ALPHA_COUNTER=$((ALPHA_COUNTER + 1))
    MODEL_COUNTER=$((MODEL_COUNTER + 1))
    PREFERENCE_EXP_TAG="ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random"
    PREFERENCE_DATASET_HUB_ID="${USER_ID}/gpt2-imdb-preference-${PREFERENCE_EXP_TAG}"
    MAXMIN_REWARD_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-majority-${RM_SUFFIX}"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "📊 [$MODEL_COUNTER/$TOTAL_MODELS] MaxMin-Majority - Alpha=$ALPHA ($ALPHA_COUNTER/$TOTAL_ALPHAS)"
    echo "────────────────────────────────────────────────────────────────"
    echo "Architecture: Separate $REWARD_MODEL_BASE → Linear → Reward (for majority)"
    echo "Preference dataset: $PREFERENCE_DATASET_HUB_ID"
    echo "Hub model id:       $MAXMIN_REWARD_HUB_ID"
    
    if [ -z "$SKIP_TRAIN" ] && _rm_selected maxmin-majority; then
        "$PYTHON" src/imdb/reward_training/maxmin_reward_training.py \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA" \
            --group 1 \
            --preference_dataset_hub_id "$PREFERENCE_DATASET_HUB_ID" \
            --hub_model_id "$MAXMIN_REWARD_HUB_ID" \
            --model_hub_id "$REWARD_MODEL_BASE" \
            "${LORA_ARG[@]}"
        echo "✅ MaxMin majority model (alpha=$ALPHA) training complete"
    else
        echo "⏭️  MaxMin majority model (alpha=$ALPHA) training skipped (SKIP_TRAIN)"
    fi
    echo ""
done

if [ -z "$SKIP_TRAIN" ]; then
    echo "✅ Part 1 complete: all reward models trained"
else
    echo "✅ Part 1 complete: params defined; training skipped (SKIP_TRAIN)"
fi

# =============================================================================
# Part 2: RM normalization (z-score params per dataset, unless SKIP_NORM is set)
# =============================================================================
if [ -z "$SKIP_NORM" ]; then
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "🎯 Part 2: RM normalization (calibration JSON per preference dataset)"
echo "════════════════════════════════════════════════════════════════"
echo ""

NORM_COUNTER=0
for ALPHA in $ALPHAS; do
    PREFERENCE_EXP_TAG="ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random"
    PREFERENCE_DATASET_HUB_ID="${USER_ID}/gpt2-imdb-preference-${PREFERENCE_EXP_TAG}"
    
    echo ""
    echo "────────────────────────────────────────────────────────────────"
    echo "📐 Normalization for preference dataset: $PREFERENCE_DATASET_HUB_ID"
    echo "────────────────────────────────────────────────────────────────"
    
    VANILLA_REWARD_HUB_ID="${USER_ID}/vanilla-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-${RM_SUFFIX}"
    echo "  Vanilla RM z-score..."
    "$PYTHON" src/imdb/reward_training/normalize_reward_model.py \
        --rm_type vanilla \
        --user_id "$USER_ID" \
        --seed "$SEED" \
        --proportion "$PROPORTION" \
        --alpha "$ALPHA" \
        --max_new_tokens "$MAX_NEW_TOKENS" \
        --preference_dataset_hub_id "$PREFERENCE_DATASET_HUB_ID" \
        --vanilla_hub_model_id "$VANILLA_REWARD_HUB_ID"
    
    MAXMIN_MINORITY_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-minority-${RM_SUFFIX}"
    MAXMIN_MAJORITY_HUB_ID="${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-majority-${RM_SUFFIX}"
    echo "  MaxMin RM z-score (minority + majority)${MAXMIN_UNIFIED:+ [unified scale]}..."
    MAXMIN_NORM_ARGS=(--rm_type maxmin --user_id "$USER_ID" --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --max_new_tokens "$MAX_NEW_TOKENS" --preference_dataset_hub_id "$PREFERENCE_DATASET_HUB_ID" --maxmin_minority_hub_model_id "$MAXMIN_MINORITY_HUB_ID" --maxmin_majority_hub_model_id "$MAXMIN_MAJORITY_HUB_ID")
    [ -n "${MAXMIN_UNIFIED:-}" ] && MAXMIN_NORM_ARGS+=(--maxmin_unified)
    "$PYTHON" src/imdb/reward_training/normalize_reward_model.py "${MAXMIN_NORM_ARGS[@]}"
    
    NORM_COUNTER=$((NORM_COUNTER + 1))
    echo "✅ Normalization for alpha=$ALPHA complete"
    echo ""
done

echo "✅ Part 2 complete: calibration JSONs written to calibration/"
else
echo "⏭️  Part 2 skipped (SKIP_NORM is set)"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo "🎉 Reward Model Pipeline Complete!"
echo "════════════════════════════════════════════════════════════════"
if [ -z "$SKIP_TRAIN" ]; then
    echo "Trained $TOTAL_MODELS models"
fi
if [ -z "$SKIP_NORM" ]; then
    echo "Wrote calibration JSONs to calibration/ (one per preference dataset)"
fi
echo ""
echo "Models on HuggingFace Hub (naming aligned with build_dataset):"
echo ""
for ALPHA in $ALPHAS; do
    echo "Alpha = $ALPHA (preference: ${USER_ID}/gpt2-imdb-preference-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random):"
    echo "  • Vanilla:   ${USER_ID}/vanilla-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-${RM_SUFFIX}"
    echo "  • MaxMin-0:  ${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-minority-${RM_SUFFIX}"
    echo "  • MaxMin-1:  ${USER_ID}/maxmin-imdb-reward-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-majority-${RM_SUFFIX}"
    echo ""
done
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "📝 Next steps:"
echo "   1. Check WandB for training metrics and comparisons"
echo "   2. Proceed to PPO training with these reward models (use calibration/ JSONs for z-score if needed)"
echo "   3. Evaluate policies on fairness metrics"
echo "   4. Compare performance across different alpha values"

