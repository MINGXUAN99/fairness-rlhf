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
