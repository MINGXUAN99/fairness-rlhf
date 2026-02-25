# IMDb Dataset Pipeline

This directory contains the pipeline for creating a preference dataset for fairness-aware RLHF training.

## Pipeline Overview

```
1. generate_responses.py  →  2. score_responses.py  →  3. create_chosen_rejected.py
   (Generate texts)           (Score quality)            (Create preference pairs)
```

## Files & Functions

### 1. `generate_responses.py`
**Purpose**: Generates multiple response completions for IMDb prompts using GPT-2.

**Usage**:
```bash
python generate_responses.py --user_id YOUR_HF_USERNAME
```

**Output**: Creates dataset `{user_id}/gpt2-imdb-raw` with:
- `prompt` - Original prompt text
- `response_0`, `response_1` - Two generated completions per prompt

---

### 2. `score_responses.py`
**Purpose**: Scores generated responses on positiveness and length metrics.


**Usage**:
```bash
python score_responses.py --user_id YOUR_HF_USERNAME
```

**Output**: Updates `{user_id}/gpt2-imdb-raw` with additional columns:
- `positiveness_0`, `positiveness_1` - Sentiment scores (0-1)
- `length_0`, `length_1` - Raw length scores
- `normalized_length_0`, `normalized_length_1` - Normalized length scores

---

### 3. `create_chosen_rejected.py`
**Purpose**: Creates preference pairs (chosen/rejected) with group-aware scoring for fairness analysis.

**Main Functions**:
- `assign_group_id(dataset, seed, proportion)` - Assigns examples to minority/majority groups
- `compute_scores(example, alpha)` - Computes group-dependent preference scores
  - **Minority group**: Prefers conciseness (shorter responses)
    - Formula: `score = (1-normalized_length)*α + (1-α)*positiveness`
  - **Majority group**: Prefers positivity (positive sentiment)
    - Formula: `score = (1-normalized_length)*(1-α) + α*positiveness`
- `create_chosen_rejected(example)` - Creates chosen/rejected pairs based on scores
- `create_chosen_rejected_dataset(...)` - Main pipeline function

**Usage**:
```bash
python create_chosen_rejected.py \
    --user_id YOUR_HF_USERNAME \
    --seed 42 \
    --proportion 0.2 \
    --alpha 0.8
```

**Output**: Creates dataset `{user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}` with:
- `chosen` - Preferred response (higher score)
- `rejected` - Non-preferred response (lower score)
- `group_id` - Group assignment (0=minority, 1=majority)

---

## Complete Pipeline Example

```bash
# Step 1: Generate responses (takes ~30 min on single GPU)
python src/imdb/dataset/generate_responses.py --user_id "YOUR_ID"

# Step 2: Score responses (takes ~10 min)
python src/imdb/dataset/score_responses.py --user_id "YOUR_ID"

# Step 3: Create preference pairs (takes <1 min)
python src/imdb/dataset/create_chosen_rejected.py \
    --user_id "YOUR_ID" \
    --seed 42 \
    --proportion 0.2 \
    --alpha 0.8
```

## Key Parameters

- **seed**: Random seed for reproducible group assignment
- **proportion**: Fraction of examples in minority group (e.g., 0.2 = 20%)
- **alpha**: Weight parameter (0.5-1.0) controlling strength of group preferences:
  - α=1.0: Strong group preferences
    - Minority: 100% conciseness (shortest responses preferred)
    - Majority: 100% positivity (most positive responses preferred)
  - α=0.5: Equal weighting for both groups
    - Both: 50% conciseness + 50% positivity
  - Recommended range: 0.5 to 1.0

## Notes

- All scripts use `accelerate` for distributed GPU processing
- Datasets are automatically pushed to HuggingFace Hub (private by default)
- The pipeline creates group-aware preference data for studying fairness in RLHF

