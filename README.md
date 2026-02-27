# Fairness-RLHF

Fairness-aware Reinforcement Learning from Human Feedback on the IMDb sentiment dataset. This project implements and compares four RLHF methods with different reward modeling strategies, focusing on group fairness across sub-populations with diverse preferences.

Based on [sharedrep-rlhf](https://github.com/marcellobullo/sharedrep-rlhf), with the MaxMin-RLHF component replicating [MaxMinRLHF](https://github.com/CharlesQ9/MaxMinRLHF).

**Key references:**

- MaxMin-RLHF: [Maxmin-RLHF: Alignment with Diverse Human Preferences](https://arxiv.org/abs/2402.08925)
- SharedRep-RLHF: [Shared Representations in RLHF](https://arxiv.org/abs/2509.03672)

## Methods

| Method | Reward Model | PPO Objective | Description |
|--------|-------------|---------------|-------------|
| **Vanilla** | Single GPT-2 reward head trained on all data | Standard PPO with learned reward | Baseline RLHF without fairness considerations |
| **MaxMin** | Two separate GPT-2 reward heads (minority + majority) | Batch-level worst-group selection | Optimizes for the worst-off group per Algorithm 1 of MaxMin-RLHF |
| **SharedRep** | Single GPT-2 backbone with multiple group-specific heads | Batch-level worst-head selection | Shared representation with per-group reward heads and MaxMin objective |
| **Gold** | Pre-trained sentiment classifier (`lvwerra/distilbert-imdb`) | Batch-level worst-group selection using analytical scores | Uses known group preference formulas (no learned reward model) |

### Group Preferences

Two simulated sub-populations are defined by their preference over two traits:

- **Positiveness**: Sentiment score from a DistilBERT classifier, in \[0, 1\]
- **Conciseness**: `1 - normalized_length`, where `normalized_length = clamp(token_length / max_length, 0, 1)`

The `alpha` parameter controls preference strength:

- **Minority group**: `score = conciseness * alpha + positiveness * (1 - alpha)`
- **Majority group**: `score = conciseness * (1 - alpha) + positiveness * alpha`

When `alpha = 1.0`, minority cares only about conciseness and majority only about positiveness. When `alpha = 0.5`, both groups have identical preferences.

### MaxMin Objective

All fairness-aware methods (maxmin, sharedrep, gold) implement Algorithm 1 from the MaxMin-RLHF paper:

```
max_policy  min_group  E[reward_group(response)]
```

At each PPO step, this is approximated by **batch-level group selection**: compute the mean reward per group across the batch, select the group with the lowest mean, and use that group's per-sample rewards for the PPO update.

## Project Structure

```
fairness-rlhf/
├── scripts/                          # Shell scripts for running the full pipeline
│   ├── train_all_reward_models.sh    # Train vanilla + maxmin reward models
│   ├── train_all_ppo_policy.sh       # Train PPO policies (vanilla + maxmin)
│   └── evaluate_ppo_policies.sh      # Generate & score evaluation responses
│
├── src/
│   ├── helper/                       # Shared utilities
│   │   ├── imdb_utils.py             # IMDb preprocessing, length scoring
│   │   ├── generate_responses_unified.py  # Unified response generation logic
│   │   ├── score_responses_unified.py     # Unified response scoring logic
│   │   └── sharedrep_reward_trainer.py    # Custom RewardTrainer for SharedRep
│   │
│   ├── models/
│   │   └── sr_gpt2/                  # SharedRep GPT-2 reward model
│   │       ├── sr_gpt2_configuration.py   # Model config (n_heads, k, maxmin flag)
│   │       └── sr_gpt2_modeling.py        # SharedRepGPT2RM: shared backbone + group heads
│   │
│   └── imdb/                         # IMDb experiment code
│       ├── dataset/                  # Dataset preparation (Step 1)
│       │   ├── generate_responses.py       # Generate k responses per prompt
│       │   ├── score_responses.py          # Score responses (positiveness + length)
│       │   └── create_chosen_rejected.py   # Create preference pairs per group
│       │
│       ├── reward_training/          # Reward model training (Step 2)
│       │   ├── vanilla_reward_training.py      # Single reward model on all data
│       │   ├── maxmin_reward_training.py        # Per-group reward models (--group 0/1)
│       │   └── sharedrep_reward_training.py     # SharedRep multi-head reward model
│       │
│       ├── ppo_training/             # PPO policy training (Step 3)
│       │   └── unified_ppo_training.py     # Unified entry point for all methods
│       │
│       └── evaluation/               # Policy evaluation (Step 4)
│           ├── generate_responses.py       # Generate responses from trained policy
│           ├── score_responses.py          # Score generated responses
│           └── read_evaluation_scores.py   # Analyze and display evaluation metrics
│
├── notebooks/
│   └── analyze_ppo_evaluations.ipynb # Analysis notebook
│
├── accelerate_config.yaml            # Single-GPU accelerate config
└── requirements.txt                  # Python dependencies
```

## Setup

1. Create and activate a Python environment:

```bash
conda create -n rlhf python=3.13
conda activate rlhf
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Log in to Hugging Face and Weights & Biases:

```bash
huggingface-cli login
wandb login
```

## Training Pipeline

The full pipeline has four stages. Each stage uploads artifacts to the Hugging Face Hub.

### Step 1: Dataset Preparation

Generate multiple responses per IMDb prompt and score them:

```bash
# Generate k=2 responses per prompt from the base GPT-2 model
python src/imdb/dataset/generate_responses.py \
    --user_id <HF_USER> --seed 42 --num_responses 2

# Score all responses (positiveness + normalized length)
python src/imdb/dataset/score_responses.py \
    --user_id <HF_USER> --seed 42 --num_responses 2

# Create chosen/rejected preference pairs per group
python src/imdb/dataset/create_chosen_rejected.py \
    --user_id <HF_USER> --seed 42 --proportion 0.2 --alpha 0.5
```

### Step 2: Reward Model Training

```bash
# Train all reward models (vanilla + maxmin minority + maxmin majority)
bash scripts/train_all_reward_models.sh

# Or train individually:
# Vanilla
python src/imdb/reward_training/vanilla_reward_training.py \
    --user_id <HF_USER> --seed 42 --proportion 0.2 --alpha 0.5

# MaxMin (train one model per group)
python src/imdb/reward_training/maxmin_reward_training.py \
    --user_id <HF_USER> --seed 42 --proportion 0.2 --alpha 0.5 --group 0  # minority
python src/imdb/reward_training/maxmin_reward_training.py \
    --user_id <HF_USER> --seed 42 --proportion 0.2 --alpha 0.5 --group 1  # majority

# SharedRep
python src/imdb/reward_training/sharedrep_reward_training.py \
    --user_id <HF_USER> --seed 42 --proportion 0.2 --alpha 0.5 --k 64
```

### Step 3: PPO Policy Training

All methods use the unified training script:

```bash
# Train all PPO policies
bash scripts/train_all_ppo_policy.sh

# Or train individually:
python src/imdb/ppo_training/unified_ppo_training.py \
    --method <gold|maxmin|sharedrep|vanilla> \
    --user_id <HF_USER> --seed 42 --proportion 0.2 --alpha 0.5
    # Add --k 64 for sharedrep
```

### Step 4: Evaluation

```bash
# Evaluate all trained policies
bash scripts/evaluate_ppo_policies.sh

# Or evaluate individually:
# Generate responses from trained policy
python src/imdb/evaluation/generate_responses.py \
    --user_id <HF_USER> --seed 42 --method maxmin --proportion 0.2 --alpha 0.5

# Score the generated responses
python src/imdb/evaluation/score_responses.py \
    --user_id <HF_USER> --seed 42 --method maxmin --proportion 0.2 --alpha 0.5

# Display evaluation metrics
python src/imdb/evaluation/read_evaluation_scores.py \
    --user_id <HF_USER> --seed 42 --method maxmin --proportion 0.2 --alpha 0.5
```

## Key Parameters

| Parameter | Flag | Description | Default |
|-----------|------|-------------|---------|
| Method | `--method` | Training method: `gold`, `maxmin`, `sharedrep`, `vanilla` | Required |
| User ID | `--user_id` | HuggingFace username for model/dataset hub IDs | Required |
| Seed | `--seed` | Random seed for reproducibility | Required |
| Proportion | `--proportion` | Fraction of minority group (e.g., 0.2 = 20% minority) | Required for maxmin/sharedrep/vanilla |
| Alpha | `--alpha` | Group preference strength (0.5 = identical, 1.0 = maximally different) | 1.0 |
| K | `--k` | SharedRep inner hidden dimension | Required for sharedrep |

## HuggingFace Hub Naming Convention

All models and datasets are pushed to the HF Hub with systematic naming:

### Reward Models

| Method | Hub ID |
|--------|--------|
| Vanilla | `<user>/vanilla-imdb-reward-ALPHA<A>-len<L>-gpt2` |
| MaxMin (minority) | `<user>/maxmin-imdb-reward-ALPHA<A>-len<L>-minority-gpt2` |
| MaxMin (majority) | `<user>/maxmin-imdb-reward-ALPHA<A>-len<L>-majority-gpt2` |
| SharedRep | `<user>/sharedrep-imdb-reward-prop<P>-alpha<A>-seed<S>-k<K>` |

### PPO Policies

| Method | Hub ID |
|--------|--------|
| Gold | `<user>/gold-imdb-ppo-alpha<A>-seed<S>` |
| Vanilla | `<user>/vanilla-imdb-ppo-prop<P>-alpha<A>-seed<S>` |
| MaxMin | `<user>/maxmin-imdb-ppo-prop<P>-alpha<A>-seed<S>` |
| SharedRep | `<user>/sharedrep-imdb-ppo-prop<P>-alpha<A>-seed<S>-k<K>` |

### Evaluation Datasets

| Method | Hub ID |
|--------|--------|
| Gold | `<user>/gold-imdb-ppo-eval-alpha<A>-seed<S>` |
| Vanilla | `<user>/vanilla-imdb-ppo-eval-prop<P>-alpha<A>-seed<S>` |
| MaxMin | `<user>/maxmin-imdb-ppo-eval-prop<P>-alpha<A>-seed<S>` |
| SharedRep | `<user>/sharedrep-imdb-ppo-eval-prop<P>-alpha<A>-seed<S>-k<K>` |

## WandB Metrics (PPO Training)

During PPO training, the following metrics are logged to Weights & Biases:

### Objective Metrics

| Metric | Meaning |
|--------|---------|
| `objective/scores` | Mean reward from the reward model. Primary signal -- should increase over training. |
| `objective/kl` | KL divergence from the reference policy. Measures policy drift. |
| `objective/entropy` | Policy entropy (token-level). Higher = more diverse generation. |
| `objective/non_score_reward` | KL penalty reward (`-kl_coef * KL`). Implicit reward for staying close to reference. |
| `objective/rlhf_reward` | Total RLHF reward = `non_score_reward + scores`. |

### Policy & Loss Metrics

| Metric | Meaning |
|--------|---------|
| `loss/policy_avg` | Clipped PPO policy loss, averaged over epochs/minibatches. |
| `loss/value_avg` | Clipped value function loss. |
| `policy/approxkl_avg` | Approximate KL of the policy update. Monitors update magnitude. |
| `policy/clipfrac_avg` | Fraction of samples where PPO clip threshold was active. |
| `policy/entropy_avg` | Mean entropy of policy logits during optimization. |

### Value Function Metrics

| Metric | Meaning |
|--------|---------|
| `val/clipfrac_avg` | Fraction of samples where value function clip was active. |
| `val/ratio` | Mean importance sampling ratio (should stay near 1.0). |
| `val/ratio_var` | Variance of importance sampling ratio. High = unstable updates. |
| `val/num_eos_tokens` | Count of generated responses containing an EOS token. |

### Group Selection Metrics (maxmin, sharedrep, gold only)

| Metric | Meaning |
|--------|---------|
| `group_selection/worst_group_idx` | Index of the worst-off group selected for the PPO update. |
| `group_selection/worst_group_name` | Name of the selected group (`minority`/`majority` or `head_0`/`head_1`/...). |
| `group_selection/mean_reward_<name>` | Mean reward per group across the batch (one entry per group). |

**What to watch:**

- `objective/scores` trending up = policy is learning
- `objective/kl` staying bounded = policy hasn't collapsed
- `group_selection/worst_group_idx` alternating between groups = MaxMin objective is balancing both groups
- `mean_reward_minority` and `mean_reward_majority` converging = fairness gap is closing

### Training State

| Metric | Meaning |
|--------|---------|
| `eps` | Episodes processed per second (throughput). |
| `lr` | Current learning rate. |
| `episode` | Total episodes processed. |

## Architecture Details

### Base Models

- **Policy model**: `lvwerra/gpt2-imdb` (GPT-2 fine-tuned on IMDb)
- **Value model**: `lvwerra/gpt2-imdb` with `num_labels=1` (regression head)
- **Gold reward model**: `lvwerra/distilbert-imdb` (DistilBERT sentiment classifier)
- **Learned reward models**: Raw GPT-2 backbone (`gpt2`) with frozen weights, trainable linear head(s)

### SharedRep Model (`SharedRepGPT2RM`)

A custom GPT-2-based reward model with:
- Shared GPT-2 backbone (frozen during reward training)
- `n_heads` group-specific reward heads, each: `hidden_dim -> k -> 1`
- In `maxmin` mode, returns all head scores `(batch, n_heads)` for batch-level group selection

### MaxMin Reward Wrapper (`MaxMinRewardWrapper`)

Wraps a GPT-2 backbone with two separate linear heads (minority and majority). Returns stacked rewards `(batch, 2)` for batch-level group selection during PPO.

### Reward Function Override

The TRL `PPOTrainer` uses a `get_reward` function to compute rewards. For custom methods, this function is monkey-patched at `trl.experimental.ppo.ppo_trainer.get_reward` to implement method-specific reward logic while preserving the original function for value model computation.
