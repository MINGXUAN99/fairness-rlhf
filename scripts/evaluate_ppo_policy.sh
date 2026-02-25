#!/bin/bash

# Evaluate PPO models for vanilla and maxmin methods with alpha values from 0.7 to 1.0
# This script runs two evaluation steps:
#   1. generate_responses.py - Generate responses using trained policy models
#   2. score_responses.py - Score the generated responses using gold reward model
# Usage: ./scripts/evaluate_ppo_policies.sh

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

# Methods to evaluate
METHODS=("vanilla" "maxmin")

# Alpha values to evaluate (0.6, 0.7, 0.8, 0.9, 1.0)
ALPHAS=(1.0)

# Script paths
GENERATE_SCRIPT="src/imdb/evaluation/generate_responses.py"
SCORE_SCRIPT="src/imdb/evaluation/score_responses.py"

echo "=========================================="
echo "Evaluating PPO Models - Vanilla & MaxMin"
echo "=========================================="
echo "User ID:         $USER_ID"
echo "Methods:        ${METHODS[*]}"
echo "Seed:           $SEED"
echo "Proportion:     $PROPORTION"
echo "Alpha values:   ${ALPHAS[*]}"
echo "=========================================="
echo ""

# Track total number of evaluation runs
TOTAL_RUNS=$((${#METHODS[@]} * ${#ALPHAS[@]}))
CURRENT_RUN=0

# Loop over methods
for METHOD in "${METHODS[@]}"; do
    # Loop over alpha values
    for ALPHA in "${ALPHAS[@]}"; do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        
        echo ""
        echo "=========================================="
        echo "Evaluation Run $CURRENT_RUN/$TOTAL_RUNS"
        echo "=========================================="
        echo "Method:          $METHOD"
        echo "Alpha:           $ALPHA"
        echo "Seed:            $SEED"
        echo "Proportion:      $PROPORTION"
        echo "=========================================="
        echo ""
        
        # Step 1: Generate responses
        echo "Step 1/2: Generating responses..."
        echo "----------------------------------------"
        if python "$GENERATE_SCRIPT" \
            --method "$METHOD" \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA"; then
            echo "✅ Response generation completed for $METHOD with alpha=$ALPHA"
            echo "   Dataset: $USER_ID/$METHOD-imdb-ppo-eval-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}"
        else
            echo "❌ Response generation failed for $METHOD with alpha=$ALPHA"
            echo "   Skipping scoring step..."
            continue
        fi
        echo ""
        
        # Step 2: Score responses
        echo "Step 2/2: Scoring responses..."
        echo "----------------------------------------"
        if python "$SCORE_SCRIPT" \
            --method "$METHOD" \
            --user_id "$USER_ID" \
            --seed "$SEED" \
            --proportion "$PROPORTION" \
            --alpha "$ALPHA"; then
            echo "✅ Response scoring completed for $METHOD with alpha=$ALPHA"
        else
            echo "❌ Response scoring failed for $METHOD with alpha=$ALPHA"
        fi
        
        echo ""
        echo "=========================================="
        echo "✅ Evaluation completed for $METHOD with alpha=$ALPHA"
        echo "=========================================="
        echo ""
    done
done

echo ""
echo "=========================================="
echo "✅ All Evaluations Completed!"
echo "=========================================="
echo ""
echo "Evaluated models:"
for METHOD in "${METHODS[@]}"; do
    for ALPHA in "${ALPHAS[@]}"; do
        echo "  - $USER_ID/$METHOD-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}"
        echo "    → Responses: $USER_ID/$METHOD-imdb-ppo-eval-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}"
    done
done
echo ""

