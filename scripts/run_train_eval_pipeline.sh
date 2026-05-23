#!/bin/bash
#
# Full pipeline: train reward models → train PPO policies → generate & score responses.
#
# Usage:
#   bash scripts/run_train_eval_pipeline.sh
#
# Examples:
#   # Train + eval maxmin @ alpha 1.5, seed 43 (RMs already on Hub)
#   USER_ID=mxcui SEED=43 ALPHAS="1.5" METHODS=maxmin KL_DIV=mean_kl \
#   SKIP_REWARD_TRAINING=1 bash scripts/run_train_eval_pipeline.sh
#
#   # Eval only (policy already trained)
#   POLICIES="mxcui/maxmin-imdb-ppo-prop0.2-alpha1.5-seed43-mean_kl0.1-gpt2" \
#   SKIP_REWARD_TRAINING=1 SKIP_PPO_TRAINING=1 bash scripts/run_train_eval_pipeline.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# =============================================================================
# Configuration (override any default via environment)
# =============================================================================
USER_ID="${USER_ID:-mxcui}"
SEED="${SEED:-42}"
PROPORTION="${PROPORTION:-0.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ALPHAS="${ALPHAS:-1.0}"
METHODS="${METHODS:-vanilla maxmin}"
POLICY_MODEL_NAMES="${POLICY_MODEL_NAMES:-gpt2}"
KL_COEF="${KL_COEF:-0.1}"
# KL div: "total" (TRL default, hub suffix "kl") or "mean_kl" (length-invariant)
KL_DIV="${KL_DIV:-mean_kl}"
[ "$KL_DIV" = "mean_kl" ] && KL_SUFFIX="mean_kl" || KL_SUFFIX="kl"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
NO_REWARD_NORM="${NO_REWARD_NORM:-1}"
SKIP_REWARD_TRAINING="${SKIP_REWARD_TRAINING:-0}"
SKIP_PPO_TRAINING="${SKIP_PPO_TRAINING:-0}"
# Optional: explicit hub ids for eval (space-separated). If unset, built from ALPHAS x METHODS.
POLICIES="${POLICIES:-}"

METHODS_ARRAY=($METHODS)
POLICY_MODEL_NAMES_ARRAY=($POLICY_MODEL_NAMES)

echo "============================================================"
echo " Run Train + Eval Pipeline"
echo "============================================================"
echo "USER_ID:            $USER_ID"
echo "SEED:               $SEED"
echo "PROPORTION:         $PROPORTION"
echo "ALPHAS:             $ALPHAS"
echo "METHODS:            ${METHODS_ARRAY[*]}"
echo "POLICY_MODEL_NAMES: ${POLICY_MODEL_NAMES_ARRAY[*]}"
echo "KL_COEF:            $KL_COEF"
echo "KL_DIV:             $KL_DIV (suffix: ${KL_SUFFIX})"
echo "SKIP_RM_TRAIN:      $SKIP_REWARD_TRAINING"
echo "SKIP_PPO_TRAIN:     $SKIP_PPO_TRAINING"
echo "MAX_NEW_TOKENS:     $MAX_NEW_TOKENS"
echo "EVAL_MAX_TOKENS:    $EVAL_MAX_NEW_TOKENS"
[ -n "$POLICIES" ] && echo "POLICIES (eval):    $POLICIES"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Train reward models (skipped when SKIP_REWARD_TRAINING=1)
# =============================================================================
if [ "$SKIP_REWARD_TRAINING" = "1" ]; then
  echo "[1/3] Training reward models -- SKIPPED"
  echo ""
else
  echo "[1/3] Training reward models for alpha(s) $ALPHAS..."
  echo "--------------------------------------------------------------------"
  USER_ID="$USER_ID" SEED="$SEED" PROPORTION="$PROPORTION" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  ALPHAS="$ALPHAS" REWARD_MODEL_BASE="${REWARD_MODEL_BASE:-gpt2}" SKIP_NORM="${SKIP_NORM:-1}" \
  bash scripts/train_reward_models.sh __internal
  echo ""
fi

# =============================================================================
# Step 2: Train PPO policies for each alpha
# =============================================================================
if [ "$SKIP_PPO_TRAINING" = "1" ]; then
  echo "[2/3] Training PPO policies -- SKIPPED"
  echo ""
else
  echo "[2/3] Training PPO policies (${METHODS_ARRAY[*]}) for alpha(s) $ALPHAS..."
  echo "--------------------------------------------------------------------"
  for ALPHA in $ALPHAS; do
    echo ">>> Alpha=$ALPHA"
    USER_ID="$USER_ID" SEED="$SEED" PROPORTION="$PROPORTION" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    ALPHA="$ALPHA" KL_COEF="$KL_COEF" KL_DIV="$KL_DIV" \
    METHODS="$METHODS" POLICY_MODEL_NAMES="$POLICY_MODEL_NAMES" NO_REWARD_NORM="$NO_REWARD_NORM" \
    bash scripts/train_ppo_policy.sh __internal
    echo ""
  done
fi

# =============================================================================
# Step 3: Generate and score responses for all trained policies
# =============================================================================
echo "[3/3] Generate and score responses for all policies..."
echo "--------------------------------------------------------------------"
if [ -n "$POLICIES" ]; then
  POLICIES_LIST="$POLICIES"
else
  POLICIES_LIST=""
  for ALPHA in $ALPHAS; do
    for METHOD in "${METHODS_ARRAY[@]}"; do
      for POLICY_MODEL_NAME in "${POLICY_MODEL_NAMES_ARRAY[@]}"; do
        POLICY_MODEL_SUFFIX="${POLICY_MODEL_NAME//\//-}"
        P="${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-${KL_SUFFIX}${KL_COEF}-${POLICY_MODEL_SUFFIX}"
        POLICIES_LIST="$POLICIES_LIST $P"
      done
    done
  done
  POLICIES_LIST="${POLICIES_LIST# }"  # trim leading space
fi

USER_ID="$USER_ID" SEED="$SEED" PROPORTION="$PROPORTION" \
EVAL_MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS" POLICIES="$POLICIES_LIST" \
bash scripts/evaluate_ppo_policy.sh

echo ""
echo "============================================================"
echo " Pipeline complete."
echo "============================================================"
if [ "$SKIP_REWARD_TRAINING" = "1" ]; then
  echo "Reward models:  (skipped)"
else
  echo "Reward models:  trained for alpha(s) $ALPHAS"
fi
if [ "$SKIP_PPO_TRAINING" = "1" ]; then
  echo "PPO policies:   (skipped)"
else
  echo "PPO policies:   ${METHODS_ARRAY[*]} (alpha $ALPHAS)"
fi
echo "Eval policies:  $POLICIES_LIST"
echo "Eval datasets:  *-ppo-eval-* (on Hub)"
echo "============================================================"
