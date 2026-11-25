# Run the conciseness-gradient experiment

Implementation of the [measurement checklist](conciseness_gradient_collapse.md#measurement-checklist-what-we-will-collect). No project training or GPU evaluation has been run on this machine.

## 1. Prepare one configuration

Copy `experiments/attribute_gradient_config.example.json` to a run-specific JSON file in `experiments/`.

- Replace `minority_model.path` with the verified alpha-matched reward checkpoint. Pin revisions for the tokenizer and reward models. Model access is checked only when loading, not by schema validation.
- Replace `prompts_file` with JSONL records like `{"id":"imdb-test-review-123","text":"This movie had"}`. IDs must identify source reviews. The four supplied prompts are illustrations, not an evaluation dataset.
- Confirm `reward`: length tokenizer/source, denominator, clipping, text scope, and scoring cap. The example is a configuration template, not a recovered historical run manifest.
- Set `seed`, `alpha`, and a fresh `output_dir` for each run. Paths in JSON are relative to the JSON file. Use 128 prompts × 4 responses for confirmation, or a smaller pilot panel.
- Optionally set `evaluation_prompts_file` to a second JSONL panel with disjoint source-review IDs. It measures outcomes under the same clean-panel sampler, independently of the gradient panel.
- Checkpoint saves are optional and large. Adjust `save_checkpoint_updates` and `save_final_checkpoint` to available disk space.

Validate without loading models, importing Torch, or using a GPU:

```bash
python -m src.imdb.diagnostics.attribute_gradient_runner \
  --config experiments/attribute_gradient_config.example.json --validate-only
```

## 2. Collect during vanilla PPO on the GPU machine

Use the repository's working training environment, including its `trl.experimental.ppo` installation. Do not replace that environment merely to run diagnostics. The diagnostic hooks were source-checked against TRL 0.27, 0.28, and 0.29; unknown loop structures fail before training. GPU/runtime compatibility still needs a short pilot on that machine.

After preparing `experiments/attribute_gradient_pilot.json`, run:

```bash
python src/imdb/ppo_training/unified_ppo_training.py \
  --method vanilla --user_id YOUR_HF_USER --seed 42 --proportion 0.2 --alpha 0.75 \
  --policy_model_name gpt2 \
  --vanilla_rm_hub_id YOUR_VERIFIED_POOLED_REWARD_CHECKPOINT \
  --max_new_tokens 512 --rollout_response_length 512 \
  --kl_div mean_kl --kl_coef 0.1 --total_episodes 50000 \
  --local_only --local_policy_output_dir local_models/attribute-alpha075-seed42 \
  --attribute_diag_config experiments/attribute_gradient_pilot.json
```

The run is single-process, full-fine-tuning vanilla PPO. Do not combine it with the old cosine callback, LoRA, or gradient-intervention flags. Set alpha and seed identically in JSON and CLI. `--rollout_response_length` explicitly sets the actual PPO generation cap; `--max_new_tokens` historically controls dataset/model naming. With diagnostics disabled, existing defaults are preserved.

`--local_only` saves the final model locally and disables Hub uploads. Diagnostic checkpoint snapshots contain policy, value, optimizer, scheduler, and RNG state for an explicit continuation implementation. They are **not an automatic exact-resume interface**: the existing PPO trainer resets its loop and does not restore dataloader position.

Every rollout records reward/length statistics. Scheduled rollouts additionally record raw-scale gradients and the **first non-skipped synchronized optimizer displacement**. This is one actual optimizer step, not the displacement over the entire multi-epoch PPO rollout update. Statistics for all training steps, including KL/clipping and whitening scales, are saved separately.

The fixed clean panel uses matched, unfiltered sampling and log probabilities. The training stream uses actual rollout tokens, the actual generation temperature, and the pooled scores passed to PPO. The oracle gradients use evaluation-mode forwards; they are explicitly separate from the realized accumulated PPO policy-loss gradient.

## 3. Evaluate existing checkpoints without training

Install only the offline dependencies if needed:

```bash
python -m pip install -r experiments/requirements-attribute-diagnostics.txt
```

In a separate configuration with a fresh `output_dir`, add:

```json
"checkpoints": [
  {"path": "../local_models/initial-policy", "revision": "local", "update": 0},
  {"path": "../local_models/late-policy", "revision": "local", "update": 700}
]
```

Then run:

```bash
python -m src.imdb.diagnostics.attribute_gradient_runner \
  --config experiments/attribute_gradient_offline.json
```

Offline evaluation measures reward/length distributions, gradient norms, alignment, prompt blocks, and counterfactual alpha weights. It does not invent actual-update measurements from unrelated checkpoint pairs. Checkpoints must be complete causal-LM models, not adapter-only directories. An optional `pooled_model` spec with `path`, `revision`, and `padding_side` enables pooled-reward gradient comparisons offline; online PPO already supplies its actual pooled reward scores.

## 4. Inspect the measurements

```bash
python -m src.imdb.diagnostics.analyze_attribute_gradients \
  results/conciseness_gradient_collapse/gpt2-alpha075-seed42-pilot
```

Or use [the analysis notebook](../notebooks/analyze_attribute_gradients.ipynb). Add `--csv-only` for an export without plotting dependencies.

| Output | Main fields |
|---|---|
| `rollout_summary.csv` | `policy_length_mean/std`, `cap_hit_rate`, `eos_rate`, `conciseness_mean/std`, `sentiment_mean/std`, within-prompt variances on the repeated-response panel |
| `gradient_summary.csv` | `weighted_norm_conciseness`, `weighted_norm_sentiment`, `rho`, pairwise gradient cosines, mixture reconstruction error |
| `update_summary.csv` | `norm_update`, `cos_update__*`, `dot_update__*`, `contribution_conciseness`, `contribution_sentiment`, accumulated `ppo_ascent` gradient |
| `counterfactual_alpha.csv` | Same-policy results at evaluation alpha 0.5, 0.75, 0.9, and 1.0 |
| `training_controls.csv` | Training KL, clipping, entropy, learning rate, and other TRL metrics |
| `evaluation.csv` | Optional separate-panel outcome measurements |
| `figures/` | Four-panel mechanism plots, signed update contributions, and counterfactual-weight plots |

JSONL files are the append-only canonical records; CSV files are generated by the analysis command. `rollouts.jsonl` preserves token actions and attributes for the panel and selected training rollouts (`save_all_rollouts: true` retains every training rollout). `manifest.json` records configuration, model revisions where available, package versions, parameter layout, and code hashes. `completed.json` is written only after successful completion.

Do not average `clean_panel` and `training_rollout` rows: they have different sampling/baseline semantics. Training batches have per-row IDs and a token-prefix hash, not recovered source-review IDs; within-review variance is only estimated on the explicitly identified repeated-response panel. Near-zero cosines and ratios are null when numerically undefined. A numeric cosine above the numerical floor still requires sampling-stability checks.

## CPU-only verification

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python -m unittest src.imdb.diagnostics.test_attribute_gradient_diagnostic -v
```

Checks cover an analytically known categorical policy, mixture reconstruction, reward scaling, zero-variance rewards, first-EOS masking, RNG/training-gradient preservation, accumulation-aware update capture, and a tiny locally initialized checkpoint pipeline. No pretrained model download or GPU is needed. These tests establish implementation behavior, not evidence for the research hypothesis.
