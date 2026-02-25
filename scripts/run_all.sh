#!/bin/bash
# ============================================================================
# Fairness-RLHF: Full Experiment Pipeline
# ============================================================================
#
# Runs all 4 stages end-to-end:
#   Stage 1: Dataset preparation (generate responses, score, create pairs)
#   Stage 2: Reward model training (vanilla, maxmin, sharedrep)
#   Stage 3: PPO policy training (gold, vanilla, maxmin, sharedrep)
#   Stage 4: Evaluation (generate responses from policies, score them)
#
# Usage:
#   bash scripts/run_all.sh                      # Run with defaults
#   ALPHAS="0.5 0.8" bash scripts/run_all.sh     # Custom alphas
#   METHODS="vanilla maxmin" bash scripts/run_all.sh  # Subset of methods
#   SKIP_DATASET=1 bash scripts/run_all.sh        # Skip dataset stage
#   SKIP_REWARD=1 bash scripts/run_all.sh         # Skip reward training
#   SKIP_PPO=1 bash scripts/run_all.sh            # Skip PPO training
#   SKIP_EVAL=1 bash scripts/run_all.sh           # Skip evaluation
#
# ============================================================================

set -e  # Exit on error

# ============================================================================
# Project root
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# Logging: create TWO log files:
# - echo log: only this script's `echo` lines (clean + readable)
# - full log: everything (including Python output), useful for debugging
#
# Key detail: piping output through `tee` breaks TTY detection and makes tqdm
# print one-update-per-line. To keep tqdm normal in the terminal *and* still
# capture all output, we run the pipeline under `script` (pseudo-TTY) for the
# full log, while independently tee'ing `echo` lines into the echo log.
# ============================================================================
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
TS="${FAIRNESS_RLHF_LOG_TS:-$(date +%Y%m%d_%H%M%S)}"
ECHO_LOG_FILE="${FAIRNESS_RLHF_ECHO_LOG_FILE:-$LOG_DIR/run_all_${TS}_echo.log}"
FULL_LOG_FILE="${FAIRNESS_RLHF_FULL_LOG_FILE:-$LOG_DIR/run_all_${TS}_full.log}"

# Wrapper mode: re-exec ourselves under `script` to capture full output without
# turning tqdm into log spam. Internal mode runs the real pipeline.
if [ "${1:-}" != "__internal" ] && [ -z "${FAIRNESS_RLHF_INTERNAL_RUN:-}" ]; then
    : > "$ECHO_LOG_FILE"
    echo "Echo log: $ECHO_LOG_FILE"
    echo "Full log: $FULL_LOG_FILE"
    echo ""

    export FAIRNESS_RLHF_LOG_TS="$TS"
    export FAIRNESS_RLHF_ECHO_LOG_FILE="$ECHO_LOG_FILE"
    export FAIRNESS_RLHF_FULL_LOG_FILE="$FULL_LOG_FILE"
    export FAIRNESS_RLHF_INTERNAL_RUN=1

    if command -v script >/dev/null 2>&1; then
        # `script` records everything (stdout/stderr) while keeping a pseudo-TTY.
        script -q -f "$FULL_LOG_FILE" -c "bash \"$0\" __internal"
        exit $?
    else
        echo "WARNING: 'script' not found; falling back to tee for full log."
        echo "         (This may cause tqdm progress bars to spam the full log.)"
        bash "$0" __internal 2>&1 | tee -a "$FULL_LOG_FILE"
        exit ${PIPESTATUS[0]}
    fi
fi

# Internal mode: append ONLY `echo` lines to the echo log (but still print to terminal).
echo() {
    builtin echo "$@" | tee -a "$ECHO_LOG_FILE"
}

echo "Echo log: $ECHO_LOG_FILE"
echo "Full log: $FULL_LOG_FILE"
echo ""

# ============================================================================
# Configuration (all overridable via environment variables)
# ============================================================================
USER_ID="${USER_ID:-mxcui}"
SEED="${SEED:-42}"
PROPORTION="${PROPORTION:-0.2}"
K="${K:-64}"                       # SharedRep inner hidden dimension
NUM_RESPONSES="${NUM_RESPONSES:-2}" # Responses per prompt for dataset creation

# Methods to train and evaluate
# Options: gold, vanilla, maxmin, sharedrep
METHODS="${METHODS:-gold vanilla maxmin sharedrep}"

# Alpha values to sweep
ALPHAS="${ALPHAS:-0.5}"

# Stage skip flags (set to 1 to skip)
SKIP_DATASET="${SKIP_DATASET:-0}"
SKIP_REWARD="${SKIP_REWARD:-0}"
SKIP_PPO="${SKIP_PPO:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# method:alpha combinations to skip in PPO and Eval stages.
# Useful when some combinations are already trained / evaluated correctly.
# Example: SKIP_COMBOS="gold:0.5 vanilla:0.5"
SKIP_COMBOS="${SKIP_COMBOS:-}"

# Python interpreter
PYTHON="${PYTHON:-python3}"
if [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
    PYTHON="$PROJECT_ROOT/venv/bin/python3"
fi

# ============================================================================
# Timing helper
# ============================================================================
PIPELINE_START=$(date +%s)

stage_timer_start() {
    STAGE_START=$(date +%s)
}

stage_timer_end() {
    local elapsed=$(( $(date +%s) - STAGE_START ))
    local mins=$((elapsed / 60))
    local secs=$((elapsed % 60))
    echo "  Time: ${mins}m ${secs}s"
}

pipeline_timer_end() {
    local elapsed=$(( $(date +%s) - PIPELINE_START ))
    local hours=$((elapsed / 3600))
    local mins=$(( (elapsed % 3600) / 60 ))
    local secs=$((elapsed % 60))
    echo "Total time: ${hours}h ${mins}m ${secs}s"
}

# ============================================================================
# Derived variables
# ============================================================================
METHODS_ARRAY=($METHODS)
ALPHAS_ARRAY=($ALPHAS)

# Count total work items
TOTAL_REWARD=0
TOTAL_PPO=0
TOTAL_EVAL=0
for METHOD in "${METHODS_ARRAY[@]}"; do
    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        case "$METHOD" in
            vanilla)  TOTAL_REWARD=$((TOTAL_REWARD + 1)); TOTAL_PPO=$((TOTAL_PPO + 1)); TOTAL_EVAL=$((TOTAL_EVAL + 1)) ;;
            maxmin)   TOTAL_REWARD=$((TOTAL_REWARD + 2)); TOTAL_PPO=$((TOTAL_PPO + 1)); TOTAL_EVAL=$((TOTAL_EVAL + 1)) ;;  # 2 = minority + majority
            sharedrep) TOTAL_REWARD=$((TOTAL_REWARD + 1)); TOTAL_PPO=$((TOTAL_PPO + 1)); TOTAL_EVAL=$((TOTAL_EVAL + 1)) ;;
            gold)     TOTAL_PPO=$((TOTAL_PPO + 1)); TOTAL_EVAL=$((TOTAL_EVAL + 1)) ;;  # No reward training needed
        esac
    done
done

# ============================================================================
# Print configuration
# ============================================================================
echo ""
echo "=================================================================="
echo " Fairness-RLHF: Full Experiment Pipeline"
echo "=================================================================="
echo ""
echo " Configuration:"
echo "   User ID:       $USER_ID"
echo "   Seed:          $SEED"
echo "   Proportion:    $PROPORTION"
echo "   K (SharedRep): $K"
echo "   Num responses: $NUM_RESPONSES"
echo "   Methods:       ${METHODS_ARRAY[*]}"
echo "   Alphas:        ${ALPHAS_ARRAY[*]}"
echo "   Python:        $PYTHON"
echo ""
echo " Stages:"
echo "   [$([ "$SKIP_DATASET" = 1 ] && builtin echo "SKIP" || builtin echo " ON ")] Stage 1: Dataset preparation"
echo "   [$([ "$SKIP_REWARD" = 1 ] && builtin echo "SKIP" || builtin echo " ON ")] Stage 2: Reward training ($TOTAL_REWARD models)"
echo "   [$([ "$SKIP_PPO" = 1 ] && builtin echo "SKIP" || builtin echo " ON ")] Stage 3: PPO training ($TOTAL_PPO policies)"
echo "   [$([ "$SKIP_EVAL" = 1 ] && builtin echo "SKIP" || builtin echo " ON ")] Stage 4: Evaluation ($TOTAL_EVAL runs)"
echo ""
echo "=================================================================="
echo ""


# ############################################################################
# STAGE 1: Dataset Preparation
# ############################################################################
if [ "$SKIP_DATASET" = 1 ]; then
    echo "[STAGE 1/4] Dataset preparation -- SKIPPED"
    echo ""
else
    echo "=================================================================="
    echo "[STAGE 1/4] Dataset Preparation"
    echo "=================================================================="
    echo ""

    # Step 1a: Generate responses from base model
    echo "  [1a] Generating $NUM_RESPONSES responses per prompt..."
    stage_timer_start
    "$PYTHON" src/imdb/dataset/generate_responses.py \
        --user_id "$USER_ID" \
        --num_responses "$NUM_RESPONSES"
    stage_timer_end
    echo ""

    # Step 1b: Score responses (positiveness + length)
    echo "  [1b] Scoring responses..."
    stage_timer_start
    "$PYTHON" src/imdb/dataset/score_responses.py \
        --user_id "$USER_ID" \
        --num_responses "$NUM_RESPONSES"
    stage_timer_end
    echo ""

    # Step 1c: Create chosen/rejected pairs for each alpha
    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        echo "  [1c] Creating preference pairs (alpha=$ALPHA)..."
        stage_timer_start
        "$PYTHON" src/imdb/dataset/create_chosen_rejected.py \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA"
        stage_timer_end
    done

    echo ""
    echo "  Dataset artifacts:"
    echo "    Responses:  $USER_ID/gpt2-imdb-responses"
    echo "    Scored:     $USER_ID/gpt2-imdb-responses-scored"
    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        echo "    Pairs:      $USER_ID/gpt2-imdb-seed${SEED}-prop${PROPORTION}-alpha${ALPHA}"
    done
    echo ""
fi


# ############################################################################
# Helper function: Train reward models for a specific alpha
# ############################################################################
train_reward_models_for_alpha() {
    local TARGET_ALPHA=$1
    local COUNTER_START=$2
    local COUNTER_END=$3
    
    local REWARD_COUNTER=$((COUNTER_START - 1))  # Initialize to START-1 since we increment before printing
    
    for METHOD in "${METHODS_ARRAY[@]}"; do
        case "$METHOD" in

            vanilla)
                REWARD_COUNTER=$((REWARD_COUNTER + 1))
                echo "  [$REWARD_COUNTER/$COUNTER_END] Vanilla reward (alpha=$TARGET_ALPHA)..."
                stage_timer_start
                "$PYTHON" src/imdb/reward_training/vanilla_reward_training.py \
                    --user_id "$USER_ID" \
                    --seed "$SEED" \
                    --proportion "$PROPORTION" \
                    --alpha "$TARGET_ALPHA"
                stage_timer_end
                echo ""
                ;;

            maxmin)
                REWARD_COUNTER=$((REWARD_COUNTER + 1))
                echo "  [$REWARD_COUNTER/$COUNTER_END] MaxMin reward - minority (alpha=$TARGET_ALPHA)..."
                stage_timer_start
                "$PYTHON" src/imdb/reward_training/maxmin_reward_training.py \
                    --user_id "$USER_ID" \
                    --seed "$SEED" \
                    --proportion "$PROPORTION" \
                    --alpha "$TARGET_ALPHA" \
                    --group 0
                stage_timer_end
                echo ""

                REWARD_COUNTER=$((REWARD_COUNTER + 1))
                echo "  [$REWARD_COUNTER/$COUNTER_END] MaxMin reward - majority (alpha=$TARGET_ALPHA)..."
                stage_timer_start
                "$PYTHON" src/imdb/reward_training/maxmin_reward_training.py \
                    --user_id "$USER_ID" \
                    --seed "$SEED" \
                    --proportion "$PROPORTION" \
                    --alpha "$TARGET_ALPHA" \
                    --group 1
                stage_timer_end
                echo ""
                ;;

            sharedrep)
                REWARD_COUNTER=$((REWARD_COUNTER + 1))
                echo "  [$REWARD_COUNTER/$COUNTER_END] SharedRep reward (alpha=$TARGET_ALPHA, k=$K)..."
                stage_timer_start
                "$PYTHON" src/imdb/reward_training/sharedrep_reward_training.py \
                    --user_id "$USER_ID" \
                    --seed "$SEED" \
                    --proportion "$PROPORTION" \
                    --alpha "$TARGET_ALPHA" \
                    --k "$K"
                stage_timer_end
                echo ""
                ;;

            gold)
                # Gold uses a pre-trained sentiment classifier, no reward training needed
                ;;
        esac
    done
}

# ############################################################################
# STAGE 2: Reward Model Training
# ############################################################################
if [ "$SKIP_REWARD" = 1 ]; then
    echo "[STAGE 2/4] Reward model training -- SKIPPED"
    echo ""
else
    echo "=================================================================="
    echo "[STAGE 2/4] Reward Model Training"
    echo "=================================================================="
    echo ""

    REWARD_COUNTER=0

    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        train_reward_models_for_alpha "$ALPHA" "$((REWARD_COUNTER + 1))" "$TOTAL_REWARD"
        # Update counter based on how many models were trained for this alpha
        for METHOD in "${METHODS_ARRAY[@]}"; do
            case "$METHOD" in
                vanilla)  REWARD_COUNTER=$((REWARD_COUNTER + 1)) ;;
                maxmin)   REWARD_COUNTER=$((REWARD_COUNTER + 2)) ;;
                sharedrep) REWARD_COUNTER=$((REWARD_COUNTER + 1)) ;;
            esac
        done
    done

    echo "  Reward models on HF Hub:"
    for ALPHA in "${ALPHAS_ARRAY[@]}"; do
        for METHOD in "${METHODS_ARRAY[@]}"; do
            case "$METHOD" in
                vanilla)   echo "    $USER_ID/vanilla-imdb-reward-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}" ;;
                maxmin)    echo "    $USER_ID/maxmin-imdb-reward-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-minority"
                           echo "    $USER_ID/maxmin-imdb-reward-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-majority" ;;
                sharedrep) echo "    $USER_ID/sharedrep-imdb-reward-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-k${K}" ;;
            esac
        done
    done
    echo ""
fi


# ############################################################################
# STAGE 3: PPO Policy Training
# ############################################################################
if [ "$SKIP_PPO" = 1 ]; then
    echo "[STAGE 3/4] PPO policy training -- SKIPPED"
    echo ""
else
    echo "=================================================================="
    echo "[STAGE 3/4] PPO Policy Training"
    echo "=================================================================="
    echo ""

    PPO_COUNTER=0

    for METHOD in "${METHODS_ARRAY[@]}"; do
        for ALPHA in "${ALPHAS_ARRAY[@]}"; do
            PPO_COUNTER=$((PPO_COUNTER + 1))

            # Check if this method:alpha combo should be skipped
            if [[ " $SKIP_COMBOS " == *" $METHOD:$ALPHA "* ]]; then
                echo "  [$PPO_COUNTER/$TOTAL_PPO] PPO $METHOD (alpha=$ALPHA)... SKIPPED (in SKIP_COMBOS)"
                echo ""
                continue
            fi

            # Build args
            PPO_ARGS=(
                --method "$METHOD"
                --user_id "$USER_ID"
                --seed "$SEED"
                --proportion "$PROPORTION"
                --alpha "$ALPHA"
            )
            if [ "$METHOD" = "sharedrep" ]; then
                PPO_ARGS+=(--k "$K")
            fi

            echo "  [$PPO_COUNTER/$TOTAL_PPO] PPO $METHOD (alpha=$ALPHA)..."
            stage_timer_start
            "$PYTHON" src/imdb/ppo_training/unified_ppo_training.py "${PPO_ARGS[@]}"
            stage_timer_end
            echo ""
        done
    done

    echo "  Policies on HF Hub:"
    for METHOD in "${METHODS_ARRAY[@]}"; do
        for ALPHA in "${ALPHAS_ARRAY[@]}"; do
            case "$METHOD" in
                gold)      echo "    $USER_ID/gold-imdb-ppo-alpha${ALPHA}-seed${SEED}" ;;
                vanilla)   echo "    $USER_ID/vanilla-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}" ;;
                maxmin)    echo "    $USER_ID/maxmin-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}" ;;
                sharedrep) echo "    $USER_ID/sharedrep-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-k${K}" ;;
            esac
        done
    done
    echo ""
fi


# ############################################################################
# STAGE 4: Evaluation
# ############################################################################
if [ "$SKIP_EVAL" = 1 ]; then
    echo "[STAGE 4/4] Evaluation -- SKIPPED"
    echo ""
else
    echo "=================================================================="
    echo "[STAGE 4/4] Evaluation"
    echo "=================================================================="
    echo ""

    EVAL_COUNTER=0

    for METHOD in "${METHODS_ARRAY[@]}"; do
        for ALPHA in "${ALPHAS_ARRAY[@]}"; do
            EVAL_COUNTER=$((EVAL_COUNTER + 1))

            # Check if this method:alpha combo should be skipped
            if [[ " $SKIP_COMBOS " == *" $METHOD:$ALPHA "* ]]; then
                echo "  [$EVAL_COUNTER/$TOTAL_EVAL] Evaluate $METHOD (alpha=$ALPHA)... SKIPPED (in SKIP_COMBOS)"
                echo ""
                continue
            fi

            # Build common eval args
            EVAL_ARGS=(
                --user_id "$USER_ID"
                --seed "$SEED"
                --method "$METHOD"
                --alpha "$ALPHA"
            )
            if [ "$METHOD" != "gold" ]; then
                EVAL_ARGS+=(--proportion "$PROPORTION")
            fi
            if [ "$METHOD" = "sharedrep" ]; then
                EVAL_ARGS+=(--k "$K")
            fi

            echo "  [$EVAL_COUNTER/$TOTAL_EVAL] Evaluate $METHOD (alpha=$ALPHA)"

            # Step 4a: Generate responses from trained policy
            echo "    [4a] Generating responses..."
            stage_timer_start
            if "$PYTHON" src/imdb/evaluation/generate_responses.py "${EVAL_ARGS[@]}"; then
                stage_timer_end
            else
                echo "    FAILED -- skipping scoring"
                stage_timer_end
                echo ""
                continue
            fi

            # Step 4b: Score responses
            echo "    [4b] Scoring responses..."
            stage_timer_start
            if "$PYTHON" src/imdb/evaluation/score_responses.py "${EVAL_ARGS[@]}"; then
                stage_timer_end
            else
                echo "    FAILED"
                stage_timer_end
            fi

            # Step 4c: Print summary
            echo "    [4c] Reading scores..."
            READ_ARGS=(
                --user_id "$USER_ID"
                --seed "$SEED"
                --method "$METHOD"
                --alpha "$ALPHA"
            )
            if [ "$METHOD" != "gold" ]; then
                READ_ARGS+=(--proportion "$PROPORTION")
            fi
            if [ "$METHOD" = "sharedrep" ]; then
                READ_ARGS+=(--k "$K")
            fi
            "$PYTHON" src/imdb/evaluation/read_evaluation_scores.py "${READ_ARGS[@]}" || true

            echo ""
        done
    done

    echo "  Evaluation datasets on HF Hub:"
    for METHOD in "${METHODS_ARRAY[@]}"; do
        for ALPHA in "${ALPHAS_ARRAY[@]}"; do
            case "$METHOD" in
                gold)      echo "    $USER_ID/gold-imdb-ppo-eval-alpha${ALPHA}-seed${SEED}" ;;
                vanilla)   echo "    $USER_ID/vanilla-imdb-ppo-eval-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}" ;;
                maxmin)    echo "    $USER_ID/maxmin-imdb-ppo-eval-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}" ;;
                sharedrep) echo "    $USER_ID/sharedrep-imdb-ppo-eval-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-k${K}" ;;
            esac
        done
    done
    echo ""
fi


# ############################################################################
# Summary
# ############################################################################
echo ""
echo "=================================================================="
echo " Pipeline Complete"
echo "=================================================================="
echo ""
pipeline_timer_end
echo ""
echo " Methods: ${METHODS_ARRAY[*]}"
echo " Alphas:  ${ALPHAS_ARRAY[*]}"
echo " Seed:    $SEED"
echo ""
echo " Next steps:"
echo "   1. Check WandB for training curves"
echo "   2. Run notebooks/analyze_ppo_evaluations.ipynb for analysis"
echo "   3. Compare group fairness across methods and alphas"
echo ""
echo "=================================================================="
