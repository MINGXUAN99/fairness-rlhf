# Cleanup Old Models Script

This script helps you remove old models from both **wandb** and **Hugging Face Hub** after resetting your code.

## Quick Start

### 1. List all models (no deletion)

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --list_only
```

### 1b. List models + datasets (no deletion)

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --list_only --hf_datasets
```

### 2. List and interactively delete models

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui
```

### 3. Filter by pattern (e.g., only reward models)

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --pattern "reward"
```

### 4. Only cleanup Hugging Face models

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --hf_only
```

### 4b. Cleanup Hugging Face datasets too

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --hf_only --hf_datasets
```

### 5. Only cleanup wandb runs

```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --wandb_only
```

## Model Naming Patterns

Based on your codebase, models follow these patterns:

### Reward Models:
- `{user_id}/vanilla-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}`
- `{user_id}/maxmin-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}-majority`
- `{user_id}/maxmin-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}-minority`
- `{user_id}/sharedrep-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}-k{k}`

### PPO Policy Models:
- `{user_id}/vanilla-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}`
- `{user_id}/maxmin-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}`
- `{user_id}/sharedrep-imdb-ppo-prop{proportion}-alpha{alpha}-seed{seed}-k{k}`
- `{user_id}/gold-imdb-ppo-alpha{alpha}-seed{seed}`

## Examples

### Delete all reward models:
```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --pattern "reward" --hf_only
```

### Delete all PPO models:
```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --pattern "ppo" --hf_only
```

### Delete all models for a specific seed:
```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --pattern "seed42" --hf_only
```

## Requirements

Make sure you have:
1. **Hugging Face Hub** authentication:
   ```bash
   huggingface-cli login
   ```

2. **wandb** authentication:
   ```bash
   wandb login
   ```

## Troubleshooting

### wandb "Could not find project uncategorized" error

If you see this error, the script will now automatically:
- Detect your default wandb entity
- List all projects for that entity
- Collect runs from all projects

If it still fails, you can manually specify your entity and project:
```bash
python3 scripts/cleanup_old_models.py --user_id mxcui --wandb_entity YOUR_ENTITY --wandb_project YOUR_PROJECT
```

## Safety Features

- **File size information**: Shows the size of each model/run and total size to be freed
- The script shows you all models/runs before deletion
- Interactive mode requires confirmation before deleting
- You can use `--list_only` to just view without deleting
- Use `--pattern` to filter before viewing/deleting

## Notes

- Deletion is **permanent** - make sure you really want to delete these models
- wandb runs can be restored from the wandb UI if needed (within retention period)
- Hugging Face model deletion is permanent and cannot be undone

