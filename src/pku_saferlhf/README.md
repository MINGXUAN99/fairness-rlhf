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
