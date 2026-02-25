#!/bin/bash
#
# Runs generate → score → create_chosen_rejected.
# Produces: raw (scored) dataset + preference dataset (chosen/rejected pairs).
#
# Logging: two log files (like run_all.sh):
#   - Echo log: only this script's echo lines (clean + readable)
#   - Full log: everything (including Python output), for debugging
#
# Usage:
#   bash scripts/build_dataset.sh
#   USER_ID=... HF_TOKEN=... bash scripts/build_dataset.sh
#
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT" || exit 1

# ============================================================================
# Logging: create TWO log files (same pattern as run_all.sh)
# ============================================================================
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

SKIP_GENERATE="${SKIP_GENERATE:-0}"
SKIP_SCORE="${SKIP_SCORE:-0}"
SKIP_PREFERENCE="${SKIP_PREFERENCE:-0}"

# ============================================================================
# Configuration
# ============================================================================
USER_ID="${USER_ID:-mxcui}"
SEED="${SEED:-42}"
ALPHA="${ALPHA:-1.0}"
PROPORTION="${PROPORTION:-0.2}"
NUM_RESPONSES="${NUM_RESPONSES:-2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"

# Unique raw dataset hub id (includes exp tag to avoid overwriting prior runs)
RAW_EXP_TAG="${RAW_EXP_TAG:-n${NUM_RESPONSES}-len${MAX_NEW_TOKENS}-random}"
RAW_DATASET_HUB_ID="${RAW_DATASET_HUB_ID:-${USER_ID}/gpt2-imdb-raw-${RAW_EXP_TAG}}"
PREFERENCE_EXP_TAG="${PREFERENCE_EXP_TAG:-ALPHA${ALPHA}-len${MAX_NEW_TOKENS}-random}"
PREFERENCE_DATASET_HUB_ID="${PREFERENCE_DATASET_HUB_ID:-${USER_ID}/gpt2-imdb-preference-${PREFERENCE_EXP_TAG}}"

PYTHON="${PYTHON:-python3}"
if [ -f "$PROJECT_ROOT/venv/bin/python3" ]; then
  PYTHON="$PROJECT_ROOT/venv/bin/python3"
fi

echo "============================================================"
echo " Dataset build: seed=$SEED, max_new_tokens=$MAX_NEW_TOKENS"
echo "============================================================"
echo "USER_ID:            $USER_ID"
echo "SEED:               $SEED"
echo "ALPHA:              $ALPHA"
echo "PROPORTION:         $PROPORTION"
echo "NUM_RESPONSES:      $NUM_RESPONSES"
echo "MAX_NEW_TOKENS:     $MAX_NEW_TOKENS"
echo "RAW_DATASET_HUB_ID:       $RAW_DATASET_HUB_ID"
echo "PREFERENCE_DATASET_HUB_ID: $PREFERENCE_DATASET_HUB_ID"
echo "PYTHON:             $PYTHON"
echo "============================================================"
echo ""

if [ "$SKIP_GENERATE" = 1 ]; then
  echo "[1/3] Generate raw responses (max_new_tokens=$MAX_NEW_TOKENS) -- SKIPPED"
  echo ""
else
echo "[1/3] Generate raw responses (max_new_tokens=$MAX_NEW_TOKENS)"
"$PYTHON" src/imdb/dataset/generate_responses.py \
  --user_id "$USER_ID" \
  --num_responses "$NUM_RESPONSES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --random_max_new_tokens \
  --dataset_hub_id "$RAW_DATASET_HUB_ID"
echo ""
fi

if [ "$SKIP_SCORE" = 1 ]; then
  echo "[2/3] Score raw responses (positiveness + length) -- SKIPPED"
  echo ""
else
echo "[2/3] Score raw responses (positiveness + length)"
"$PYTHON" src/imdb/dataset/score_responses.py \
  --user_id "$USER_ID" \
  --num_responses "$NUM_RESPONSES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --dataset_hub_id "$RAW_DATASET_HUB_ID"
echo ""
fi

if [ "$SKIP_PREFERENCE" = 1 ]; then
  echo "[3/3] Create preference dataset (chosen/rejected) -- SKIPPED"
  echo ""
else
echo "[3/3] Create preference dataset (chosen/rejected)"
"$PYTHON" src/imdb/dataset/create_chosen_rejected.py \
  --user_id "$USER_ID" \
  --seed "$SEED" \
  --proportion "$PROPORTION" \
  --alpha "$ALPHA" \
  --num_responses "$NUM_RESPONSES" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --raw_dataset_hub_id "$RAW_DATASET_HUB_ID" \
  --output_hub_id "$PREFERENCE_DATASET_HUB_ID" \
  --drop_ties
echo ""
fi

echo "============================================================"
echo "Done."
echo "  Raw (scored) dataset: $RAW_DATASET_HUB_ID"
echo "  Preference dataset:   $PREFERENCE_DATASET_HUB_ID"
echo "============================================================"
