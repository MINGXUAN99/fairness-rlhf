# Controlled Sentiment Analysis

**Goal:** The model is required to output responses aligned to a specified sentiment, like positive, negative or neutral

**Dataset:** IMDb movie review dataset (Maas et al., 2011)

## 1. 📚 Dataset
The IMDb dataset is pre-processed as follows: for each review in the training-split, the first 2–8 tokens are extracted as the prompt, and for each of them 2 responses  $\boldsymbol{y}_1, \boldsymbol{y}_2 \sim\pi(\cdot|\boldsymbol{x})$ are sampled from `lvwerra/gpt2-imdb`.

**Note:** The generation and scoring scripts now use unified implementations in `src/helper/` that support both dataset creation and evaluation modes.

```bash
# Step 1: Generate responses (uses unified script in dataset mode)
python sharedrep-rlhf/src/imdb/dataset/generate_responses.py --user_id "HF_USER_ID"
```

```bash
# Step 2: Score responses using conciseness and positiveness (uses unified script in dataset mode)
python sharedrep-rlhf/src/imdb/dataset/score_responses.py --user_id "HF_USER_ID"
```

```bash
# Step 3: Create dataset with group_id, chosen, rejected columns
# Pushes to hub at {user_id}/gpt2-imdb-seed{SEED}-prop{PROP}-alpha{ALPHA}
python sharedrep-rlhf/src/imdb/dataset/create_chosen_rejected.py --user_id "HF_USER_ID" --seed SEED --proportion PROP --alpha ALPHA
```


## 3. 🎯 Reward Training
Train reward models according to the following methodologies:

1. **Vanilla (Baseline)**: Single reward model trained on all data
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/reward_training/vanilla_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --user_id "$HF_USER_ID"
```

2. **MaxMin**: Train 2 separate models (one per group). The min() operation happens during PPO.
```bash
# Train minority model (group 0)
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/reward_training/maxmin_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --group 0 --user_id "$HF_USER_ID"

# Train majority model (group 1)
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/reward_training/maxmin_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --group 1 --user_id "$HF_USER_ID"
```

3. **SharedRep**: Single model with shared backbone + group-specific heads
```bash
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/reward_training/sharedrep_reward_training.py --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --k "$K" --user_id "$HF_USER_ID"
```


## 4. 🤖 Reinforcement Learning
Train a RLHF policy using PPO using the different reward models that follow:

**Note:** All training now uses the unified PPO training script which supports all methods.

### Gold (Oracle with True Group Preferences)
```bash
# Gold uses alpha to define group preferences for the MaxMin objective
# No proportion/k needed since it uses ground-truth reward model (distilbert-imdb)
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/ppo_training/unified_ppo_training.py --method "gold" --seed "$SEED" --user_id "$HF_USER_ID" --alpha "$ALPHA"
```

### MaxMin (Separate Reward Models)
```bash
# Requires proportion to load the two group-specific reward models
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/ppo_training/unified_ppo_training.py --method "maxmin" --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --user_id "$HF_USER_ID"
```

### Vanilla (Single Reward Model Baseline)
```bash
# Trains with single reward model (no group considerations)
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/ppo_training/unified_ppo_training.py --method "vanilla" --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --user_id "$HF_USER_ID"
```

### SharedRep (Shared Backbone + Group Heads)
```bash
# Requires both proportion and k (shared representation dimension)
accelerate launch --config_file path/to/config.yaml sharedrep-rlhf/src/imdb/ppo_training/unified_ppo_training.py --method "sharedrep" --seed "$SEED" --proportion "$PROPORTION" --alpha "$ALPHA" --k "$K" --user_id "$HF_USER_ID"
```


## 5. 🎓 Evaluation
Method evaluation consists of:
1. Sampling responses $y$ from the trained policy $\pi_{\rm method}$ given input prompts $x$
2. Scoring $(x,y)$ pairs using the gold reward models

**Note:** The generation and scoring scripts now use unified implementations that handle all methods. Arguments `proportion` and `k` are only required when needed (no placeholder values).

```bash
GENERATION_SCRIPT=sharedrep-rlhf/src/imdb/evaluation/generate_responses.py
SCORING_SCRIPT=sharedrep-rlhf/src/imdb/evaluation/score_responses.py
```

### Evaluate $\pi_{\rm gold}$
```bash
# Gold now uses alpha parameter (defines group preferences)
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "gold" --user_id "$HF_USER_ID" --alpha "$ALPHA"
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "gold" --user_id "$HF_USER_ID" --alpha "$ALPHA"
```

### Evaluate $\pi_{\rm maxmin}$
```bash
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "maxmin" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --alpha "$ALPHA"
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "maxmin" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --alpha "$ALPHA"
```

### Evaluate $\pi_{\rm vanilla}$
```bash
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "vanilla" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --alpha "$ALPHA"
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "vanilla" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --alpha "$ALPHA"
```

### Evaluate $\pi_{\rm sharedrep}$
```bash
accelerate launch --config_file path/to/config.yaml "$GENERATION_SCRIPT" --seed "$SEED" --method "sharedrep" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --alpha "$ALPHA" --k "$K"
accelerate launch --config_file path/to/config.yaml "$SCORING_SCRIPT" --seed "$SEED" --method "sharedrep" --user_id "$HF_USER_ID" --proportion "$PROPORTION" --alpha "$ALPHA" --k "$K"
```



