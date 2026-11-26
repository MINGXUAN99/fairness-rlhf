# PKU SafeRLHF data preparation

Prepare one Hugging Face dataset with `train` and `test` splits. Each row has
`prompt`, `chosen`, `rejected`, and `group_id`. Group 0 is harmlessness (20% by
default), and group 1 is helpfulness (80%). Response fields do not include the
prompt; reward training must format the prompt together with each response.

No GPU, reward calibration, or validation split is needed. Install only the data
dependencies for this step:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install datasets huggingface-hub
.venv/bin/python -m src.pku_saferlhf.prepare_dataset
```

The default local output is `output/pku-preferences-seed42-prop0.2`. It includes
the dataset, a dataset card, and `preparation_report.json` with counts and the
resolved source commit. Existing nonempty output directories are not overwritten.

To prepare and upload one private dataset using your existing HF login/token:

```bash
.venv/bin/python -m src.pku_saferlhf.prepare_dataset \
  --seed 42 --proportion 0.2 \
  --output_dir output/pku-preferences-upload \
  --output_hub_id YOUR_USERNAME/pku-preferences-seed42-prop0.2
```

Use `--public` only if the repository should be public. `--source`, `--config`,
and `--revision` select the source; defaults are PKU-Alignment/PKU-SafeRLHF,
default, and main. The resolved commit is recorded for reproducibility.

## Group assignment

Preserve official test data and remove training rows whose whitespace-normalized
prompts appear in test. Within each split, uniformly select `floor(N * proportion)`
pairs from those that are not both safe and assign them to harmlessness. Assign
all remaining pairs to helpfulness. Each source pair appears once; no pair is
duplicated, and both-safe prompts remain available. Fail if there are too few
eligible pairs rather than silently changing the ratio.

For group 0 use `safer_response_id`; for group 1 use `better_response_id`. The
selected response becomes `chosen` and the other `rejected`. Check that mixed-safe
pairs prefer the safe response. This conditional assignment makes the group data
distributions different; the group proportion is approximate only due to integer
rounding. Both-safe pairs contribute no harmlessness ranking loss.

Group reward training filters on `group_id`; Vanilla trains on all training rows.
PPO uses unique training prompts and evaluation uses unique test prompts.

Run CPU-only preparation checks:

```bash
python3 -m unittest src.pku_saferlhf.test_prepare_dataset
```

## Reward training

Train three reward models from the uploaded dataset. The model ID is required;
choose it before launching the full runs. The default dataset revision pins the
verified upload. No validation split is created, no reward calibration is applied,
and test evaluation runs only after training finishes. Only the final model is
saved. The script never uploads model weights automatically.

Install the tested training dependencies in the training environment (install the
appropriate CUDA build of PyTorch first on the GPU machine):

```bash
python -m pip install -r src/pku_saferlhf/requirements-training.txt
```

Run from the repository root, replacing `YOUR_BASE_MODEL` with the selected model:

```bash
for group in helpfulness harmlessness pooled; do
  python -m src.pku_saferlhf.train_reward \
    --model YOUR_BASE_MODEL \
    --group "$group" \
    --training_mode head \
    --output_dir "output/pku-reward-${group}-seed42" \
    --seed 42 --precision bf16
done
```

The `helpfulness` run filters group 1, `harmlessness` filters group 0, and `pooled`
uses all rows without reweighting or resampling. Existing nonempty output
directories are rejected. Use `--dataset PATH` to load the locally prepared
DatasetDict; otherwise the default is `mxcui/pku-saferlhf-preferences`.

Training modes:

- `head` (default): freeze the backbone and train the scalar scoring head.
- `lora`: adapt attention/MLP layers and the scoring head; export a merged complete
  reward model. PPO must then load both complete group models, rather than swap
  their scoring heads onto one backbone.
- `full`: train all parameters; also requires independent full group models in PPO.

The loss is `-logsigmoid(reward_chosen - reward_rejected)`. It uses a small
Transformers Trainer subclass so the preference loss and final test evaluation
are explicit; this entry point does not depend on TRL. It reuses the existing
tokenizer, backbone-freezing, and LoRA helpers.

`--format auto` uses the tokenizer's chat template if present, otherwise
`User: {prompt}\nAssistant: {response}` plus EOS. `--format chat` requires a chat
template; `--format plain` forces the plain format. All scoring includes the
prompt. Reuse this formatting for later PPO reward inference.

Sequences are right-truncated at `--max_length` (default 1024). Pairs that become
identical or empty after tokenization are excluded, with counts and the remaining
group proportions recorded. Inspect these counts before full runs; a large removal
rate calls for a longer context budget. Padding is on the left.

Each output directory contains complete model weights, tokenizer files,
`run_config.json`, `train_results.json`, and `test_results.json`. Test metrics
include strict preference accuracy (ties are not wins), tie rate, mean reward
margin, and loss. Pooled runs additionally report separate helpfulness and
harmlessness metrics. No test-based checkpoint selection is performed.

Batch size, accumulation, learning rate, epochs, maximum length, model revision,
seed, precision, and optional W&B logging are CLI options; see `--help`. Defaults
are starting settings, not tuned hyperparameters. `--cpu --precision fp32
--max_steps 2` supports a short smoke run; full experiments require suitable compute.

Offline tests create a tiny random model and tokenizer locally, exercise all three
groups and all training modes, and check that exported models reload:

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  python -m unittest src.pku_saferlhf.test_train_reward
```

## Policy training

After producing the three reward checkpoints, use `train_policy.py` for Vanilla,
MaxMin, SumGrad, and PCGrad. The script reuses the existing PPO patch and gradient
combination functions in `src/imdb/ppo_training/unified_ppo_training.py`. It has a
separate task entry point, full instruction prompts, and reward inference that
uses each reward model's complete backbone and scoring head.

Use an isolated policy environment: PPO is tested with TRL 0.24.0 and Transformers
4.56.2, while the reward-training environment pins Transformers 4.51.3. Install the
appropriate CUDA build of torch on the GPU host before the remaining requirements.

```bash
python3 -m venv tmp/policy-venv
tmp/policy-venv/bin/python -m pip install -r src/pku_saferlhf/requirements-policy.txt
```

Run all four methods from the same policy initialization and reward checkpoints:

```bash
for method in vanilla maxmin sumgrad pcgrad; do
  tmp/policy-venv/bin/python -m src.pku_saferlhf.train_policy \
    --method "$method" \
    --policy_model YOUR_BASE_MODEL \
    --pooled_rm output/pku-reward-pooled-seed42 \
    --harmlessness_rm output/pku-reward-harmlessness-seed42 \
    --helpfulness_rm output/pku-reward-helpfulness-seed42 \
    --output_dir "output/pku-policy-${method}-seed42" \
    --seed 42 --training_mode full --precision bf16 \
    --batch_size 16 --reward_batch_size 4 \
    --max_prompt_length 512 --response_length 256 \
    --total_episodes 50000 --ppo_epochs 3 --kl_coef 0.1
done
```

Reward checkpoint arguments accept local directories or Hub model IDs. Vanilla
loads only the pooled model; the other methods load the same two complete group
models. When present, the reward checkpoint's `run_config.json` supplies its
training format and context limit. Otherwise `--reward_format` and
`--reward_max_length` supply explicit fallbacks. Upload that metadata alongside
reward weights when moving checkpoints to the Hub. Hub revisions resolved at
startup are recorded in the policy run configuration.

### Method behavior

| Method | Reward and policy update |
|---|---|
| Vanilla | Pooled reward with standard PPO |
| MaxMin | Select the lowest mean raw group reward over the full rollout batch, then use that group's per-response rewards |
| SumGrad | Sum both group policy gradients, then add the KL gradient |
| PCGrad | Project conflicting components from both group gradients, sum them, then add the KL gradient |

No reward calibration or group-gradient normalization is applied. Existing PPO
reward/advantage whitening remains enabled. SumGrad and PCGrad use the mean of
the two scores for the shared value target, and separate group advantage streams
for policy updates, matching the existing implementation. Group order is
harmlessness first and helpfulness second.

All methods use one rollout batch and one minibatch, with gradient accumulation
fixed at 1. `--batch_size` must be at least 8 for whitening. Reward-model forward
passes can be split using `--reward_batch_size`, but MaxMin selection happens
after all scores are collected, so it remains a full-batch decision. Lowering
reward batch size does not reduce policy rollout memory. Only single-process
training is supported in this entry point. TRL can round the episode budget up
to the next complete batch; the trainer state records actual episodes.

`--training_mode lora` adapts the policy and uses the adapter-disabled base as its
reference. `full` uses a separate frozen reference. Both modes train a value
model from the same base architecture and export a complete final policy. The
two frozen reward models remain independent in both modes. These choices can
require substantial GPU memory; choose model size and batch size accordingly.

Only unique prompts from the dataset's **train** split enter PPO. Long prompts
are excluded and counted rather than truncated into incomplete instructions.
The original prompt is retained for reward formatting; completions are decoded
and formatted separately with each reward tokenizer. A distinct policy pad token
is added if necessary to avoid treating EOS as padding. The script never uses
test prompts or creates a validation split. TRL's required eval-dataloader
placeholder uses training prompts with sampling and evaluation disabled.

Outputs include final policy/tokenizer files, `run_config.json`,
`trainer_state.json`, and `reward_trace.jsonl` (raw group means and MaxMin's selected
group). SumGrad/PCGrad additionally write `gradient_cosine.csv`. These are training
diagnostics; independent policy evaluation remains a separate step. Model upload
is not automatic. Outputs are protected against accidental overwrites.

The default KL convention is `--kl_div mean_kl`, reusing the IMDb implementation;
`total` keeps TRL's sum convention. The PPO source patch is checked before
SumGrad/PCGrad training and raises an error on incompatible source instead of
silently running standard PPO. Use a fresh process for each method.

Offline CPU smoke tests train all four methods on tiny local models, test LoRA
export, verify full-batch MaxMin selection, and compare independent reward scores:

```bash
OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false \
  tmp/policy-venv/bin/python -m unittest src.pku_saferlhf.test_train_policy
```

These tests verify the implementation path, not model quality or GPU memory fit.
The real reward models must be trained before launching the full policy runs.
