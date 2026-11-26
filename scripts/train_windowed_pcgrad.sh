#!/usr/bin/env bash
# Prepare once, run Early/Late sequentially, then score locally. No Hub pushes.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-venv/bin/python}"
ARM="${1:-both}"
HORIZON=800
HALF=400
MODE=full
if [[ "$ARM" == smoke ]]; then
    ARM=both
    HORIZON=4
    HALF=2
    MODE=smoke
fi
LATE_START=$((HALF + 1))
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
[[ "$ROLLOUT_BATCH_SIZE" =~ ^[0-9]+$ ]] && (( ROLLOUT_BATCH_SIZE >= 8 )) || { echo "ROLLOUT_BATCH_SIZE must be an integer >= 8" >&2; exit 2; }
EPISODES=$((HORIZON * ROLLOUT_BATCH_SIZE))
case "$ARM" in both|early|late) ;; *) echo 'Usage: train_windowed_pcgrad.sh [both|early|late|smoke]' >&2; exit 2;; esac
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
ROOT_DIR="${EXPERIMENT_ROOT:-runs/windowed-${MODE}-len53-batch${ROLLOUT_BATCH_SIZE}-alpha0.8-seed43-${RUN_TAG}}"
INPUTS="${EXPERIMENT_INPUTS:-${ROOT_DIR}/inputs}"
USER_ID="${USER_ID:-mxcui}"
if [[ ! -f "$INPUTS/inputs.json" ]]; then
    PREP_ARGS=()
    [[ -z "${NONE_ARTIFACT:-}" ]] || PREP_ARGS+=(--none_artifact "$NONE_ARTIFACT")
    [[ -z "${FULL_ARTIFACT:-}" ]] || PREP_ARGS+=(--full_artifact "$FULL_ARTIFACT")
    "$PYTHON" src/helper/windowed_experiment.py prepare --output "$INPUTS" \
        --minority_rm "${MINORITY_RM:-${USER_ID}/maxmin-imdb-reward-ALPHA0.8-len512-minority-gpt2}" \
        --majority_rm "${MAJORITY_RM:-${USER_ID}/maxmin-imdb-reward-ALPHA0.8-len512-majority-gpt2}" \
        "${PREP_ARGS[@]}"
fi
ARMS=(early late)
[[ "$ARM" == both ]] || ARMS=("$ARM")
mkdir -p "$ROOT_DIR/logs"
for arm in "${ARMS[@]}"; do
    start=1; end=$HALF
    [[ "$arm" == early ]] || { start=$LATE_START; end=$HORIZON; }
    run_dir="$ROOT_DIR/${arm}-win${start}-${end}"
    [[ ! -e "$run_dir" ]] || { echo "Refusing to overwrite $run_dir" >&2; exit 1; }
    "$PYTHON" src/imdb/ppo_training/unified_ppo_training.py \
        --method pcgrad --policy_model_name gpt2 --model_parameter_dtype float32 --no_lora --user_id "$USER_ID" \
        --seed 43 --alpha 0.8 --proportion 0.2 --max_new_tokens 512 --training_response_length 53 \
        --learning_rate 3e-6 --kl_coef 0.1 --kl_div mean_kl \
        --experiment_batch_size "$ROLLOUT_BATCH_SIZE" --total_episodes "$EPISODES" --expected_updates "$HORIZON" --no_reward_normalization --no_push \
        --pcgrad_window_start "$start" --pcgrad_window_end "$end" \
        --experiment_inputs "$INPUTS" --experiment_dir "$run_dir" \
        2>&1 | tee "$ROOT_DIR/logs/${arm}-train.log"
    mkdir -p "$run_dir/logs"
    cp "$ROOT_DIR/logs/${arm}-train.log" "$run_dir/logs/train.log"
done
# Score after both requested training arms so a scoring failure cannot suppress Late.
for arm in "${ARMS[@]}"; do
    start=1; end=$HALF
    [[ "$arm" == early ]] || { start=$LATE_START; end=$HORIZON; }
    "$PYTHON" src/helper/windowed_experiment.py analyze \
        --run_dir "$ROOT_DIR/${arm}-win${start}-${end}" --device "${SCORING_DEVICE:-cpu}"
done

if [[ "$ARM" == both ]]; then
    "$PYTHON" src/helper/windowed_experiment.py compare \
        --early "$ROOT_DIR/early-win1-${HALF}" --late "$ROOT_DIR/late-win${LATE_START}-${HORIZON}" \
        --output "$ROOT_DIR/comparison"
fi
