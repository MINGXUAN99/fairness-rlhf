#!/bin/bash

# Evaluate PPO models (vanilla and maxmin with gpt2 backbone).
# This script runs two evaluation steps:
#   1. generate_responses.py - Generate responses using trained policy models
#   2. score_responses.py - Score the generated responses using gold reward model
# Usage: ./scripts/evaluate_ppo_policy.sh

set -e  # Exit on error

# =============================================================================
# Ensure we're in the project root directory
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# =============================================================================
# Configuration
# =============================================================================
USER_ID="${USER_ID:-mxcui}"  # Set via environment or replace with your HF username
SEED="${SEED:-42}"
PROPORTION=0.2
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"  # Fixed max_new_tokens for evaluation generation

PYTHON="${PYTHON:-$(which python3 2>/dev/null || which python)}"
if [ -f "venv/bin/python3" ]; then
    PYTHON="venv/bin/python3"
fi

# Policies to evaluate (gpt2 backbone).
# Override with POLICIES env var, e.g. POLICIES="mxcui/vanilla-imdb-ppo-prop0.2-alpha1.0-seed42" to evaluate vanilla only.
if [ -n "${POLICIES:-}" ]; then
    POLICIES_ARRAY=($POLICIES)
else
    POLICIES_ARRAY=(
        "mxcui/vanilla-imdb-ppo-prop0.2-alpha1.0-seed42"
        "mxcui/maxmin-imdb-ppo-prop0.2-alpha1.0-seed42"
    )
fi

# Script paths
GENERATE_SCRIPT="src/imdb/evaluation/generate_responses.py"
SCORE_SCRIPT="src/imdb/evaluation/score_responses.py"

echo "=========================================="
echo "Evaluating PPO Models (gpt2)"
echo "=========================================="
echo "User ID:         $USER_ID"
echo "Policies:        ${POLICIES_ARRAY[*]}"
echo "Max new tokens:  $EVAL_MAX_NEW_TOKENS"
echo "=========================================="
echo ""

# Track total number of evaluation runs
TOTAL_RUNS=${#POLICIES_ARRAY[@]}
CURRENT_RUN=0

for POLICY_HUB_ID in "${POLICIES_ARRAY[@]}"; do
    CURRENT_RUN=$((CURRENT_RUN + 1))
    METHOD=$(echo "$POLICY_HUB_ID" | sed 's|.*/||' | cut -d'-' -f1)

    echo ""
    echo "=========================================="
    echo "Evaluation Run $CURRENT_RUN/$TOTAL_RUNS"
    echo "=========================================="
    echo "Policy:         $POLICY_HUB_ID"
    echo "=========================================="
    echo ""

    # Step 1: Generate responses
    echo "Step 1/2: Generating responses..."
    echo "----------------------------------------"
    if "$PYTHON" "$GENERATE_SCRIPT" \
        --method "$METHOD" \
        --user_id "$USER_ID" \
        --seed "$SEED" \
        --proportion "$PROPORTION" \
        --alpha 1.0 \
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --policy_hub_id "$POLICY_HUB_ID"; then
        EVAL_DATASET="${POLICY_HUB_ID/-ppo-/-ppo-eval-}"
        echo "✅ Response generation completed for $POLICY_HUB_ID"
        echo "   Dataset: $EVAL_DATASET"
    else
        echo "❌ Response generation failed for $POLICY_HUB_ID"
        echo "   Skipping scoring step..."
        continue
    fi
    echo ""

    # Step 2: Score responses
    echo "Step 2/2: Scoring responses..."
    echo "----------------------------------------"
    if "$PYTHON" "$SCORE_SCRIPT" \
        --method "$METHOD" \
        --user_id "$USER_ID" \
        --seed "$SEED" \
        --proportion "$PROPORTION" \
        --alpha 1.0 \
        --max_new_tokens "$EVAL_MAX_NEW_TOKENS" \
        --policy_hub_id "$POLICY_HUB_ID"; then
        echo "✅ Response scoring completed for $POLICY_HUB_ID"
    else
        echo "❌ Response scoring failed for $POLICY_HUB_ID"
    fi

    echo ""
    echo "=========================================="
    echo "✅ Evaluation completed for $POLICY_HUB_ID"
    echo "=========================================="
    echo ""
done

echo ""
echo "=========================================="
echo "✅ All Evaluations Completed!"
echo "=========================================="
echo ""
echo "Evaluated models:"
for POLICY_HUB_ID in "${POLICIES_ARRAY[@]}"; do
    EVAL_DATASET="${POLICY_HUB_ID/-ppo-/-ppo-eval-}"
    echo "  - $POLICY_HUB_ID"
    echo "    → Responses: $EVAL_DATASET"
done
echo ""

