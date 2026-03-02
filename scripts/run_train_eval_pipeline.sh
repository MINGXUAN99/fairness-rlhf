#!/bin/bash
#
# Full pipeline: train reward models → train PPO policies → generate & score responses.
# Configuration: alpha=0.05 | total_kl | skip RM training | vanilla + maxmin
#
# Usage:
#   bash scripts/run_train_eval_pipeline.sh
#   USER_ID=... bash scripts/run_train_eval_pipeline.sh
#
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# =============================================================================
# Configuration (alpha 0.5,0.7, total_kl, kl_coef=0.1, skip RM training, vanilla+maxmin)
# =============================================================================
USER_ID="${USER_ID:-mxcui}"
SEED="${SEED:-42}"
PROPORTION="${PROPORTION:-0.2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
ALPHAS="0.6"
KL_COEF="${KL_COEF:-0.1}"
# KL div: "total" (TRL default, hub suffix "kl") or "mean_kl" (length-invariant)
KL_DIV="${KL_DIV:-total}"
[ "$KL_DIV" = "mean_kl" ] && KL_SUFFIX="mean_kl" || KL_SUFFIX="kl"
EVAL_MAX_NEW_TOKENS="${EVAL_MAX_NEW_TOKENS:-256}"
SKIP_REWARD_TRAINING="${SKIP_REWARD_TRAINING:-0}"

echo "============================================================"
echo " Run Train + Eval Pipeline (alpha=0.6,0.81 | total_kl | kl=0.1 | skip RM)"
echo "============================================================"
echo "USER_ID:          $USER_ID"
echo "SEED:             $SEED"
echo "PROPORTION:       $PROPORTION"
echo "ALPHAS:           $ALPHAS"
echo "KL_COEF:          $KL_COEF"
echo "KL_DIV:           $KL_DIV"
echo "SKIP_RM_TRAIN:    $SKIP_REWARD_TRAINING"
echo "MAX_NEW_TOKENS:   $MAX_NEW_TOKENS"
echo "EVAL_MAX_TOKENS:  $EVAL_MAX_NEW_TOKENS"
echo "============================================================"
echo ""

# =============================================================================
# Step 1: Train reward models (skipped when SKIP_REWARD_TRAINING=1)
# =============================================================================
if [ "$SKIP_REWARD_TRAINING" = "1" ]; then
  echo "[1/3] Training reward models -- SKIPPED"
  echo ""
else
  echo "[1/3] Training reward models (vanilla + maxmin) for alpha $ALPHAS..."
  echo "--------------------------------------------------------------------"
  USER_ID="$USER_ID" SEED="$SEED" PROPORTION="$PROPORTION" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
  ALPHAS="$ALPHAS" REWARD_MODEL_BASE="gpt2" SKIP_NORM=1 \
  bash scripts/train_reward_models.sh __internal
  echo ""
fi

# =============================================================================
# Step 2: Train PPO policies for each alpha
# =============================================================================
echo "[2/3] Training PPO policies (vanilla + maxmin) for each alpha..."
echo "--------------------------------------------------------------------"
for ALPHA in $ALPHAS; do
    echo ">>> Alpha=$ALPHA"
    USER_ID="$USER_ID" SEED="$SEED" PROPORTION="$PROPORTION" MAX_NEW_TOKENS="$MAX_NEW_TOKENS" \
    ALPHA="$ALPHA" KL_COEF="$KL_COEF" KL_DIV="$KL_DIV" METHODS="vanilla maxmin" POLICY_MODEL_NAMES="gpt2" NO_REWARD_NORM=1 \
    bash scripts/train_ppo_policy.sh __internal
    echo ""
done

# =============================================================================
# Step 3: Generate and score responses for all trained policies
# =============================================================================
echo "[3/3] Generate and score responses for all policies..."
echo "--------------------------------------------------------------------"
POLICIES_LIST=""
for ALPHA in $ALPHAS; do
    for METHOD in vanilla maxmin; do
        P="${USER_ID}/${METHOD}-imdb-ppo-prop${PROPORTION}-alpha${ALPHA}-seed${SEED}-${KL_SUFFIX}${KL_COEF}-gpt2"
        POLICIES_LIST="$POLICIES_LIST $P"
    done
done
POLICIES_LIST="${POLICIES_LIST# }"  # trim leading space

USER_ID="$USER_ID" SEED="$SEED" PROPORTION="$PROPORTION" \
EVAL_MAX_NEW_TOKENS="$EVAL_MAX_NEW_TOKENS" POLICIES="$POLICIES_LIST" \
bash scripts/evaluate_ppo_policy.sh

echo ""
echo "============================================================"
echo " Pipeline complete."
echo "============================================================"
echo "Reward models:  (skipped - use existing)"
echo "PPO policies:   vanilla, maxmin (alpha $ALPHAS)"
echo "Eval datasets:  *-ppo-eval-* (on Hub)"
echo "============================================================"
