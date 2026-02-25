# Training Metrics Reference

All training runs log metrics to [Weights & Biases](https://wandb.ai) (`report_to=["wandb"]`). This document covers every metric recorded during both **reward model training** (Step 2) and **PPO policy training** (Step 3).

---

## 1. Reward Model Training

Reward models are trained using TRL's `RewardTrainer` (vanilla, maxmin) or the custom `SharedRepRMTrainer` (sharedrep). Both extend HuggingFace's `Trainer`, so they share a base set of metrics plus reward-specific additions.

### 1.1 Vanilla & MaxMin Reward Training

These use `trl.RewardTrainer` directly. The loss is the Bradley-Terry pairwise ranking loss:

```
loss = -log sigmoid(reward_chosen - reward_rejected)
```

#### Training Metrics (logged every `logging_steps=10`)

| Metric | Description |
|--------|-------------|
| `train/loss` | Bradley-Terry pairwise ranking loss on the training batch. |
| `train/learning_rate` | Current learning rate from the scheduler. |
| `train/epoch` | Current training epoch (fractional). |
| `train/grad_norm` | Gradient norm (if gradient clipping is enabled). |
| `train/num_tokens` | Cumulative number of tokens processed. |
| `train/min_reward` | Minimum predicted reward across all samples (chosen + rejected) in the batch. Tracks reward scale. |
| `train/mean_reward` | Mean predicted reward across all samples in the batch. |
| `train/max_reward` | Maximum predicted reward across all samples in the batch. |
| `train/accuracy` | Fraction of pairs where `reward_chosen > reward_rejected`. Primary quality signal. |
| `train/margin` | Mean difference `reward_chosen - reward_rejected`. Higher margin = more confident separation. |

#### Evaluation Metrics (logged every `eval_steps=10`)

| Metric | Description |
|--------|-------------|
| `eval/loss` | Bradley-Terry loss on the held-out eval set. |
| `eval/num_tokens` | Total tokens in the eval set. |
| `eval/min_reward` | Minimum predicted reward across all eval samples. |
| `eval/mean_reward` | Mean predicted reward across all eval samples. |
| `eval/max_reward` | Maximum predicted reward across all eval samples. |
| `eval/accuracy` | Pairwise accuracy on eval set. Primary metric for reward model quality. |
| `eval/margin` | Mean reward margin on eval set. |
| `eval/runtime` | Evaluation wall-clock time (seconds). |
| `eval/samples_per_second` | Evaluation throughput. |
| `eval/steps_per_second` | Evaluation throughput (in steps). |

#### What to watch

- **`train/accuracy` and `eval/accuracy`**: Should be well above 0.5 (random). Typical good values are 0.65-0.80 for IMDb.
- **`eval/accuracy` vs `train/accuracy`**: Large gap indicates overfitting.
- **`train/margin`**: Should be positive and increasing. Very large margins may indicate overconfident predictions.
- **`train/mean_reward`**: Monitor for reward drift/collapse (all rewards converging to same value).

### 1.2 SharedRep Reward Training

Uses `SharedRepRMTrainer`, a custom extension of `RewardTrainer`. The model has a shared GPT-2 backbone with group-specific heads, and uses group IDs to route samples to the correct head during training.

The loss is the same Bradley-Terry ranking loss, but computed per-group-head:

```
loss = -log sigmoid(reward_chosen_head[group] - reward_rejected_head[group])
```

> **Note**: `SharedRepRMTrainer` overrides `compute_loss` and does NOT populate TRL's built-in reward statistics (`min_reward`, `mean_reward`, `max_reward`, `accuracy`, `margin`) during training steps. These are only available during evaluation via `compute_metrics`.

#### Training Metrics (logged every `logging_steps=10`)

| Metric | Description |
|--------|-------------|
| `train/loss` | Bradley-Terry pairwise ranking loss (routed through group-specific heads). |
| `train/learning_rate` | Current learning rate from the scheduler. |
| `train/epoch` | Current training epoch (fractional). |
| `train/grad_norm` | Gradient norm. |

#### Evaluation Metrics

| Metric | Description |
|--------|-------------|
| `eval/loss` | Bradley-Terry loss on eval set. |
| `eval/accuracy` | Pairwise accuracy from `compute_accuracy` callback. |
| `eval/runtime` | Evaluation wall-clock time (seconds). |
| `eval/samples_per_second` | Evaluation throughput. |
| `eval/steps_per_second` | Evaluation throughput (in steps). |

### 1.3 Reward Training Hyperparameters (for reference)

| Parameter | Vanilla | MaxMin | SharedRep |
|-----------|---------|--------|-----------|
| Batch size | 16 | 16 | 16 |
| Learning rate | 5e-5 | 5e-5 | 5e-4 |
| Epochs | 3 | 3 | 3 |
| Backbone | GPT-2 (frozen) | GPT-2 (frozen) | GPT-2 (frozen) |
| Trainable params | Linear head | Linear head | Group-specific heads |
| Data | All samples | Group-filtered | All samples (group-routed) |

---

## 2. PPO Policy Training

PPO training is handled by TRL's `PPOTrainer` via `unified_ppo_training.py`. The training loop generates responses, scores them with the reward model, computes advantages using GAE, and optimizes the policy with clipped PPO.

All four methods (gold, maxmin, sharedrep, vanilla) produce the same base metrics. The fairness-aware methods (gold, maxmin, sharedrep) additionally produce group selection metrics.

### 2.1 Objective Metrics

These capture the core RLHF signals: what reward the policy achieves and how far it drifts from the reference.

| Metric | Description |
|--------|-------------|
| `objective/scores` | Mean reward score from the reward model across the batch. This is the primary task signal -- should increase over training as the policy learns to produce higher-reward responses. |
| `objective/kl` | KL divergence between the current policy and the frozen reference policy, summed per sequence and averaged across the batch. Measures policy drift. Should stay bounded (typically < 10-15). |
| `objective/entropy` | Mean token-level entropy of the policy (sum of `-log p(token)` per sequence). Higher values indicate more diverse/exploratory generation. Expected to decrease somewhat during training as the policy becomes more focused. |
| `objective/non_score_reward` | Mean KL penalty reward: `-kl_coef * KL`, summed per sequence. This is the implicit reward the policy receives for staying close to the reference. Always negative. |
| `objective/rlhf_reward` | Total RLHF reward: `non_score_reward + scores`. This is the actual combined signal the policy optimizes. Balances task performance against distribution drift. |

### 2.2 Policy Optimization Metrics

These monitor the PPO optimization process itself across the `num_ppo_epochs=3` inner epochs and minibatches.

| Metric | Description |
|--------|-------------|
| `loss/policy_avg` | Mean clipped PPO surrogate policy loss, averaged over all PPO epochs, minibatches, and gradient accumulation steps. The core optimization objective. |
| `loss/value_avg` | Mean clipped value function (critic) loss. The value model learns to predict expected returns. |
| `policy/approxkl_avg` | Approximate KL divergence of the policy update: `0.5 * mean((log pi_new - log pi_old)^2)`. Monitors the magnitude of each PPO update. Very high values (> 0.1) may indicate instability. |
| `policy/clipfrac_avg` | Fraction of samples where the PPO clipped surrogate was active (ratio fell outside `[1-clip, 1+clip]`). Values around 0.1-0.3 are typical. Very high values mean the clip is constantly binding. |
| `policy/entropy_avg` | Mean entropy of the policy logits during the PPO optimization steps. Distinct from `objective/entropy` -- this is measured during the backward pass, not during generation. |

### 2.3 Value Function Metrics

These monitor the value model (critic) that estimates expected returns for GAE advantage computation.

| Metric | Description |
|--------|-------------|
| `val/clipfrac_avg` | Fraction of samples where the value function clip was active. Similar interpretation to `policy/clipfrac_avg`. |
| `val/ratio` | Mean importance sampling ratio (`pi_new(a|s) / pi_old(a|s)`), averaged over all PPO optimization steps. Should stay close to 1.0. |
| `val/ratio_var` | Variance of the importance sampling ratio. High variance (> 1.0) signals unstable updates where some samples have very different probabilities under old vs. new policy. |
| `val/num_eos_tokens` | Number of generated responses in the batch that contain an EOS token. Tracks whether the model produces proper sentence endings. A sudden drop may indicate mode collapse. |

### 2.4 Group Selection Metrics (maxmin, sharedrep, gold only)

These are logged by the `GroupSelectionCallback` and track the batch-level worst-group selection that implements the MaxMin objective (Algorithm 1).

| Metric | Description |
|--------|-------------|
| `group_selection/worst_group_idx` | Index of the worst-off group selected for the current PPO update. For **maxmin** and **gold**: `0` = minority, `1` = majority. For **sharedrep**: `0` to `n_heads-1`. |
| `group_selection/worst_group_name` | Human-readable name of the selected worst group. Values: `minority`, `majority` (maxmin/gold) or `head_0`, `head_1`, ... (sharedrep). |
| `group_selection/mean_reward_minority` | Mean reward of the minority group across the batch (**maxmin/gold only**). |
| `group_selection/mean_reward_majority` | Mean reward of the majority group across the batch (**maxmin/gold only**). |
| `group_selection/mean_reward_head_<i>` | Mean reward of head `i` across the batch (**sharedrep only**, one entry per head). |

#### What to watch

- **`worst_group_idx` alternating** between groups is healthy -- the policy is balancing both sub-populations.
- **`worst_group_idx` stuck on one group** means the policy persistently underperforms for that sub-population.
- **`mean_reward_minority` and `mean_reward_majority` converging** indicates the MaxMin objective is successfully closing the fairness gap between groups.
- **One group's mean reward dropping** while the other increases suggests the policy is over-optimizing for one group at the expense of the other.

### 2.5 Training State Metrics

| Metric | Description |
|--------|-------------|
| `eps` | Episodes (samples) processed per second. Measures training throughput. |
| `lr` | Current learning rate from the cosine scheduler. |
| `episode` | Total number of episodes (samples) processed so far across all updates. |

### 2.6 PPO Hyperparameters (for reference)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `total_episodes` | 50,000 | Total training samples |
| `per_device_train_batch_size` | 32 | Batch size per device |
| `gradient_accumulation_steps` | 2 | Effective batch size = 64 |
| `num_ppo_epochs` | 3 | Inner PPO optimization epochs per batch |
| `learning_rate` | 3e-6 | Initial learning rate |
| `lr_scheduler_type` | cosine | Learning rate schedule |
| `kl_coef` | 0.05 | KL penalty coefficient |
| `whiten_rewards` | true | Whiten rewards before advantage computation |
| `missing_eos_penalty` | 1.0 | Penalty subtracted from score if response lacks EOS |
| `bf16` | true | Use bfloat16 mixed precision |
| `logging_steps` | 10 | Log metrics every N steps |

---

## 3. Evaluation Metrics

After PPO training, the evaluation pipeline generates responses and scores them. The final scored dataset (pushed to HF Hub) contains per-sample columns:

| Column | Description |
|--------|-------------|
| `prompt` | The input IMDb review fragment. |
| `completion` | The generated continuation from the trained policy. |
| `positiveness` | Sentiment score from `lvwerra/distilbert-imdb`, in \[0, 1\]. Higher = more positive sentiment. |
| `length` | Raw token length of the generated response. |
| `normalized_length` | `clamp(length / max_length, 0, 1)`. Normalized response length. |

The `read_evaluation_scores.py` script computes aggregate statistics from these columns:

| Statistic | Description |
|-----------|-------------|
| Mean, Std, Min, Max, Median | Standard descriptive statistics per metric. |
| 25th / 75th percentile | Quartile values for each metric. |
| Metric correlations | Pairwise Pearson correlation between positiveness and normalized_length. |

### Derived Group Scores (computed offline for analysis)

Using the same formulas as training, per-sample group scores can be derived:

```
conciseness = 1 - normalized_length
minority_score = conciseness * alpha + positiveness * (1 - alpha)
majority_score = conciseness * (1 - alpha) + positiveness * alpha
```

The key fairness metric is the **gap** between `mean(minority_score)` and `mean(majority_score)` -- the MaxMin objective aims to minimize this gap by improving the worst-off group.
