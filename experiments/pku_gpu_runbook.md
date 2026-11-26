# Running the PKU SafeRLHF experiment on a GPU server

Run the commands below from the repository root in a Bash session on the GPU
server. This runbook covers setup, a small execution pilot, the full experiment,
and evaluation. No full reward training, PPO, or 7B evaluator inference has been
run yet. CPU tests with tiny local models pass.

## 1 Transfer the current source

Branch: `experiment/pku-saferlhf`. Use the latest branch revision, which includes
evaluation and this runbook. The earlier policy commit `f1d3c21` alone does not
contain the complete pipeline. On the GPU server:

```bash
git clone --branch experiment/pku-saferlhf --single-branch \
  https://github.com/MINGXUAN99/fairness-rlhf.git fairness-rlhf-pku
cd fairness-rlhf-pku
```

Alternatively, transfer the local source archive `output/pku-gpu-handoff.tar.gz`
produced alongside this runbook. The archive is not stored in Git. It includes
`src/`, `experiments/`, the root README, and root requirements; it does not include
credentials, Python environments, model caches, training outputs, or Git history.
Extract it into a new directory on the GPU server:

```bash
mkdir -p fairness-rlhf-pku
tar -xzf pku-gpu-handoff.tar.gz -C fairness-rlhf-pku
cd fairness-rlhf-pku
```

Do not copy the macOS virtual environments. Recreate Linux environments below.
The prepared dataset is already on Hugging Face, so it need not be transferred:

- Repository: `mxcui/pku-saferlhf-preferences` (private).
- Revision: `4730d5ec7e69f2d42ab3ffc14dab4ea9eae4119f`.
- Train: 62,644 pairs; test: 8,211 pairs.
- Group 0: harmlessness; group 1: helpfulness; approximately 20/80.

## 2 Set up two Python environments

Use Python 3.11 on the server. The reward and policy environments intentionally
pin different Transformers versions. Evaluation runs in the policy environment.
Run one process on one GPU; this PPO entry point does not support distributed
training. Use a persistent session such as tmux or your server's job scheduler.

The commands use the PyTorch 2.8 CUDA 12.8 wheel, listed in the
[official PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/#v280).
Choose the appropriate published wheel for the server's driver/GPU before running
these commands; do not install a CPU wheel on the training host.

```bash
set -euo pipefail
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false

python3.11 -m venv tmp/reward-venv
python3.11 -m venv tmp/policy-venv
export PKU_RM_PY="$PWD/tmp/reward-venv/bin/python"
export PKU_PPO_PY="$PWD/tmp/policy-venv/bin/python"

for pku_python in "$PKU_RM_PY" "$PKU_PPO_PY"; do
  "$pku_python" -m pip install --upgrade pip
  "$pku_python" -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
done
"$PKU_RM_PY" -m pip install -r src/pku_saferlhf/requirements-training.txt
"$PKU_RM_PY" -m pip install sentencepiece
"$PKU_PPO_PY" -m pip install -r src/pku_saferlhf/requirements-policy.txt
"$PKU_PPO_PY" -m pip install -r src/pku_saferlhf/requirements-evaluation.txt

nvidia-smi
"$PKU_PPO_PY" - <<'PY'
import torch
assert torch.cuda.is_available(), 'CUDA is unavailable'
print('Torch:', torch.__version__, 'CUDA:', torch.version.cuda)
print('GPU:', torch.cuda.get_device_name(0))
print('VRAM GiB:', torch.cuda.get_device_properties(0).total_memory / 2**30)
print('BF16 supported:', torch.cuda.is_bf16_supported())
PY
```

Authenticate locally on the server; never put the token in scripts or Git:

```bash
tmp/policy-venv/bin/hf auth login
"$PKU_PPO_PY" - <<'PY'
from huggingface_hub import HfApi
print('Authenticated user:', HfApi().whoami()['name'])
PY
```

The account needs read access to the private dataset and any gated base model.
Run the offline regression tests in their respective environments:

```bash
OMP_NUM_THREADS=1 "$PKU_RM_PY" -m unittest \
  src.pku_saferlhf.test_prepare_dataset src.pku_saferlhf.test_train_reward
OMP_NUM_THREADS=1 "$PKU_PPO_PY" -m unittest \
  src.pku_saferlhf.test_train_policy src.pku_saferlhf.test_evaluation
```

## 3 Choose and freeze the configuration

The base model remains to be chosen. Select an instruction-capable causal LM
supported by both pinned Transformers versions and small enough for the server.
Use the same base snapshot for all policies and reward models in the first study.
The values below are starting settings for the execution pilot, not tuned results.

```bash
export PKU_BASE_MODEL='REPLACE_WITH_BASE_MODEL_HUB_ID'
export PKU_BASE_REVISION='main'
export PKU_BASE_DIR="$PWD/local_models/pku-base"
export PKU_RUN="$PWD/output/pku-main"
export PKU_DATASET='mxcui/pku-saferlhf-preferences'
export PKU_DATA_REV='4730d5ec7e69f2d42ab3ffc14dab4ea9eae4119f'
export PKU_PRECISION='bf16'
export PKU_RM_MODE='head'
export PKU_POLICY_MODE='lora'
export PKU_PPO_BATCH=8
export PKU_RM_BATCH=4
export PKU_SCORE_BATCH=2
mkdir -p "$PKU_RUN/logs"

pku_run() {
  local pku_log_name="$1"
  shift
  "$@" 2>&1 | tee "$PKU_RUN/logs/${pku_log_name}.log"
}
```

`head` keeps the reward backbone frozen; `lora` adapts the policy and saves a
merged full checkpoint. Change these modes only before the main experiment and
use the same choice for every comparator. If BF16 is unsupported, `fp32` is the
common supported fallback but uses more memory. PPO does not expose FP16 here.

Download one fixed base-model snapshot and record its immutable revision:

```bash
"$PKU_PPO_PY" - <<'PY'
import json, os
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
name = os.environ['PKU_BASE_MODEL']
if name.startswith('REPLACE_'):
    raise ValueError('Set PKU_BASE_MODEL first')
revision = HfApi().model_info(name, revision=os.environ['PKU_BASE_REVISION']).sha
snapshot_download(name, revision=revision, local_dir=os.environ['PKU_BASE_DIR'])
Path(os.environ['PKU_RUN'], 'base_snapshot.json').write_text(
    json.dumps({'model': name, 'revision': revision}, indent=2) + '\n')
PY
"$PKU_RM_PY" -m pip freeze > "$PKU_RUN/reward-environment.txt"
"$PKU_PPO_PY" -m pip freeze > "$PKU_RUN/policy-environment.txt"
nvidia-smi > "$PKU_RUN/gpu.txt"
```

Keep enough disk space for three complete reward checkpoints, twelve complete
policy checkpoints, the base snapshot, two 7B evaluator downloads, and HF caches.
Even head-only and LoRA runs export full model weights. GPU memory fit is not yet
verified; do not infer it from successful CPU smoke tests.

## 4 Run a small end-to-end pilot

Exercise the real architecture, all four update methods, and the real external
evaluators before spending the full training budget. To keep the official test
set untouched during execution checks, make a disposable pilot dataset entirely
from training rows. Its `test` key is only a schema-compatible smoke-test input;
its scores are not experiment results or a new validation set.

```bash
"$PKU_PPO_PY" - <<'PY'
import os
from datasets import DatasetDict, load_dataset
rows = load_dataset(os.environ['PKU_DATASET'], revision=os.environ['PKU_DATA_REV'],
                    split='train').shuffle(seed=42)
DatasetDict(train=rows.select(range(256)), test=rows.select(range(256, 272))).save_to_disk(
    os.path.join(os.environ['PKU_RUN'], 'pilot-data'))
PY

for group in helpfulness harmlessness pooled; do
  pku_run "pilot-rm-${group}" "$PKU_RM_PY" -m src.pku_saferlhf.train_reward \
    --dataset "$PKU_RUN/pilot-data" --model "$PKU_BASE_DIR" --group "$group" \
    --output_dir "$PKU_RUN/pilot-rewards/$group" --seed 42 \
    --training_mode "$PKU_RM_MODE" --precision "$PKU_PRECISION" \
    --max_steps 2 --max_length 1024 --batch_size "$PKU_RM_BATCH" --eval_batch_size 2
done

for method in vanilla maxmin sumgrad pcgrad; do
  pku_run "pilot-ppo-${method}" "$PKU_PPO_PY" -m src.pku_saferlhf.train_policy \
    --method "$method" --policy_model "$PKU_BASE_DIR" --dataset "$PKU_RUN/pilot-data" \
    --pooled_rm "$PKU_RUN/pilot-rewards/pooled" \
    --harmlessness_rm "$PKU_RUN/pilot-rewards/harmlessness" \
    --helpfulness_rm "$PKU_RUN/pilot-rewards/helpfulness" \
    --output_dir "$PKU_RUN/pilot-policies/$method" --seed 42 \
    --training_mode "$PKU_POLICY_MODE" --precision "$PKU_PRECISION" \
    --batch_size "$PKU_PPO_BATCH" --reward_batch_size "$PKU_SCORE_BATCH" \
    --total_episodes "$((2 * PKU_PPO_BATCH))" --ppo_epochs 1 \
    --max_prompt_length 512 --response_length 256
done
```

Generate a pilot evaluation configuration and test both 7B scorers:

```bash
"$PKU_PPO_PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['PKU_RUN'])
runs = [{'run_id': 'initial', 'method': 'initial', 'training_seed': None,
         'model': os.environ['PKU_BASE_DIR'], 'prompt_format': 'auto'}]
runs += [{'run_id': m, 'method': m, 'training_seed': 42,
          'model': str(root / 'pilot-policies' / m)}
         for m in ('vanilla', 'maxmin', 'sumgrad', 'pcgrad')]
(root / 'pilot-policies.json').write_text(json.dumps(runs, indent=2))
PY
pku_run pilot-generate "$PKU_PPO_PY" -m src.pku_saferlhf.generate_responses \
  --policies "$PKU_RUN/pilot-policies.json" --dataset "$PKU_RUN/pilot-data" \
  --output_dir "$PKU_RUN/pilot-generation" --max_prompts 8 \
  --max_prompt_length 512 --max_new_tokens 256 --batch_size 2 \
  --device cuda --precision "$PKU_PRECISION"
pku_run pilot-score "$PKU_PPO_PY" -m src.pku_saferlhf.evaluate_responses \
  --generation_dir "$PKU_RUN/pilot-generation" --output_dir "$PKU_RUN/pilot-scored" \
  --batch_size "$PKU_SCORE_BATCH" --device cuda --precision "$PKU_PRECISION"
pku_run pilot-summary "$PKU_PPO_PY" -m src.pku_saferlhf.summarize_results \
  --scored_dir "$PKU_RUN/pilot-scored" --output_dir "$PKU_RUN/pilot-summary" \
  --reference_run initial
```

Proceed when all four policies save/reload, reward outputs and PPO losses are
finite, SumGrad/PCGrad produce gradient traces, and both real Beaver checkpoints
score successfully. Check a few generated texts for correct instruction format.
Inspect reward tokenization exclusions and retained PPO prompts. Pilot scores
from minimally trained models are not evidence about the methods.

If memory is insufficient, first lower RM training/evaluation batch size and
reward/evaluator scoring batch size. PPO batch size must remain at least 8.
Try policy LoRA, `--gradient_checkpointing`, or a smaller common base model if
PPO still does not fit. Lowering reward batch size alone does not lower policy
rollout memory. Test any changed configuration across all methods, then freeze it.
Do not silently change context lengths or training budgets for one method only.

## 5 Train the three final reward models

Use the actual uploaded train/test splits. No calibration or validation split is
added. Discard pilot weights and start again from the fixed base snapshot.

```bash
for group in helpfulness harmlessness pooled; do
  pku_run "rm-${group}" "$PKU_RM_PY" -m src.pku_saferlhf.train_reward \
    --dataset "$PKU_DATASET" --dataset_revision "$PKU_DATA_REV" \
    --model "$PKU_BASE_DIR" --group "$group" \
    --output_dir "$PKU_RUN/rewards/$group" --seed 42 \
    --training_mode "$PKU_RM_MODE" --precision "$PKU_PRECISION" \
    --epochs 4 --learning_rate 1e-4 --max_length 1024 \
    --batch_size "$PKU_RM_BATCH" --eval_batch_size 2 --gradient_accumulation_steps 4
done
```

Review `run_config.json` for group counts and tokenization exclusions, and archive
the final `test_results.json`. These are final test reports; do not select
checkpoints or tune hyperparameters using these scores. If pilot memory changes
altered RM accumulation, use that frozen setting consistently here.

## 6 Train twelve final policies

Train Vanilla, MaxMin, SumGrad, and PCGrad with policy seeds 42, 43, and 44. Reuse
the same three final reward models across all policy runs. This measures policy
seed variability conditional on those reward models, not reward-training variance.

```bash
for seed in 42 43 44; do
  for method in vanilla maxmin sumgrad pcgrad; do
    pku_run "ppo-${method}-${seed}" "$PKU_PPO_PY" -m src.pku_saferlhf.train_policy \
      --method "$method" --policy_model "$PKU_BASE_DIR" \
      --dataset "$PKU_DATASET" --dataset_revision "$PKU_DATA_REV" \
      --pooled_rm "$PKU_RUN/rewards/pooled" \
      --harmlessness_rm "$PKU_RUN/rewards/harmlessness" \
      --helpfulness_rm "$PKU_RUN/rewards/helpfulness" \
      --output_dir "$PKU_RUN/policies/${method}-${seed}" --seed "$seed" \
      --training_mode "$PKU_POLICY_MODE" --precision "$PKU_PRECISION" \
      --batch_size "$PKU_PPO_BATCH" --reward_batch_size "$PKU_SCORE_BATCH" \
      --total_episodes 50000 --ppo_epochs 3 --learning_rate 3e-6 \
      --kl_coef 0.1 --kl_div mean_kl --max_prompt_length 512 --response_length 256
  done
done
```

Keep each run's configuration, final weights/tokenizer, `trainer_state.json`,
`reward_trace.jsonl`, and gradient CSV where applicable. The scripts save final
models only and do not implement resume from an interrupted PPO run. An interrupted
run must restart with a new empty output directory; preserve its logs separately.
Do not evaluate a half-written model directory as a completed run.

## 7 Evaluate the initial policy and all twelve trained policies

Build one configuration listing all runs so prompt eligibility is intersected
across every checkpoint. Do not evaluate each method on a separately selected set.

```bash
"$PKU_PPO_PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['PKU_RUN'])
runs = [{'run_id': 'initial', 'method': 'initial', 'training_seed': None,
         'model': os.environ['PKU_BASE_DIR'], 'prompt_format': 'auto'}]
runs += [{'run_id': f'{method}-{seed}', 'method': method, 'training_seed': seed,
          'model': str(root / 'policies' / f'{method}-{seed}')}
         for seed in (42, 43, 44) for method in ('vanilla', 'maxmin', 'sumgrad', 'pcgrad')]
(root / 'evaluation-policies.json').write_text(json.dumps(runs, indent=2))
# Freeze the evaluator revisions that passed the real-model pilot.
pilot = json.loads((root / 'pilot-scored' / 'manifest.json').read_text())
(root / 'evaluator-revisions.json').write_text(json.dumps(pilot['evaluators'], indent=2))
PY

pku_run generate "$PKU_PPO_PY" -m src.pku_saferlhf.generate_responses \
  --policies "$PKU_RUN/evaluation-policies.json" \
  --dataset "$PKU_DATASET" --dataset_revision "$PKU_DATA_REV" \
  --output_dir "$PKU_RUN/generation" --seed 42 \
  --max_prompt_length 512 --max_new_tokens 256 --batch_size 2 \
  --device cuda --precision "$PKU_PRECISION"

export PKU_HELP_REV="$("$PKU_PPO_PY" -c 'import json,os; print(json.load(open(os.path.join(os.environ["PKU_RUN"], "evaluator-revisions.json")))["helpfulness_reward"]["revision"])')"
export PKU_COST_REV="$("$PKU_PPO_PY" -c 'import json,os; print(json.load(open(os.path.join(os.environ["PKU_RUN"], "evaluator-revisions.json")))["harmfulness_cost"]["revision"])')"

pku_run score "$PKU_PPO_PY" -m src.pku_saferlhf.evaluate_responses \
  --generation_dir "$PKU_RUN/generation" --output_dir "$PKU_RUN/scored" \
  --helpfulness_revision "$PKU_HELP_REV" --cost_revision "$PKU_COST_REV" \
  --batch_size "$PKU_SCORE_BATCH" --max_length 2048 \
  --device cuda --precision "$PKU_PRECISION"
pku_run summary "$PKU_PPO_PY" -m src.pku_saferlhf.summarize_results \
  --scored_dir "$PKU_RUN/scored" --output_dir "$PKU_RUN/summary" \
  --reference_run initial --bootstrap_samples 2000 --seed 42
```

Generation is greedy by default. The initial policy is generated once and is the
common paired reference. Beaver reward and cost models load sequentially. Their
raw outputs have different scales: helpfulness reward is higher-better, harmfulness
cost is lower-better. Do not average the two scores or turn cost into an unsafe
percentage without a separately justified classifier/threshold.

Scoring fails rather than silently truncating long prompt/response pairs. If this
happens, inspect the affected lengths and choose an explicit evaluation protocol
for all methods; `--allow_truncation` exists but flags affected scores and changes
what is evaluated. Do not drop problematic examples only from one method.

## 8 Bring back the results

Keep the full `$PKU_RUN` directory on durable storage. For analysis on the CPU
machine, transfer at least:

- `summary/`, `scored/`, and `generation/`, including their manifests and JSONL.
- `logs/`, `base_snapshot.json`, environment snapshots, and `gpu.txt`.
- Reward-model `run_config.json` and `test_results.json` files.
- Policy `run_config.json`, `trainer_state.json`, `reward_trace.jsonl`, and
  `gradient_cosine.csv` files.

`summary/per_run.csv` contains individual seed results and prompt-bootstrap
intervals. `summary/per_method.csv` summarizes variation across the three policy
seeds. `summary/paired_differences.csv` reports differences from the initial
policy. Prompt uncertainty and training-seed variability answer different questions.
Full model weights are needed only for rerunning inference or further experiments.

Beaver evaluation is a learned proxy from the same dataset family, not an oracle.
Audit a sample with blinded human or separate LLM judgments before making strong
claims about helpfulness/safety improvements. No result is predetermined by this
runbook.
