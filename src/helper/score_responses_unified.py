"""
Unified response scoring script for both dataset creation and evaluation.

Modes:
- 'dataset': Score multiple responses for preference dataset creation
- 'evaluation': Score single completion from trained policy for evaluation
"""
import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))
sys.path.insert(0, ROOT)

import torch
import argparse
import numpy as np
from tqdm import tqdm
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.helper.imdb_utils import compute_length_score


def sync_dataset_card_readme(dataset, repo_id: str) -> None:
    """Update Hub dataset README features to match scored parquet columns."""
    from huggingface_hub import HfApi

    feature_lines = []
    for name in dataset.column_names:
        feat = dataset.features[name]
        dtype = getattr(feat, "dtype", "string")
        feature_lines.append(f"  - name: {name}\n    dtype: {dtype}")

    readme = (
        "---\n"
        "dataset_info:\n"
        "  features:\n"
        + "\n".join(feature_lines)
        + "\n"
        "  splits:\n"
        "  - name: train\n"
        f"    num_examples: {len(dataset)}\n"
        "configs:\n"
        "- config_name: default\n"
        "  data_files:\n"
        "  - split: train\n"
        "    path: data/train-*\n"
        "---\n"
    )
    HfApi().upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Sync dataset card features with scored columns",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Score responses (unified for dataset and evaluation)")
    
    # Common arguments
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID")
    parser.add_argument("--mode", type=str, required=True, choices=["dataset", "evaluation"],
                        help="Mode: 'dataset' for creating training data, 'evaluation' for policy evaluation")
    
    # Dataset mode arguments
    parser.add_argument("--num_responses", type=int, default=2,
                        help="Number of responses per prompt (dataset mode only, default: 2)")
    parser.add_argument(
        "--dataset_hub_id",
        type=str,
        default=None,
        help=(
            "Dataset hub ID for mode=dataset. If omitted, defaults to {user_id}/gpt2-imdb-raw. "
            "Use this to score a seed-tagged raw dataset without overwriting others."
        ),
    )
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens used in generation; used to define conciseness = max_new_tokens - length (default: 512)")
    
    # Evaluation mode arguments
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (evaluation mode only)")
    parser.add_argument("--method", type=str, default=None,
                        choices=["gold", "gold_vanilla", "maxmin", "sharedrep", "vanilla", "pcgrad", "sumgrad", "mgda"],
                        help="Method name (evaluation mode only)")
    parser.add_argument("--proportion", type=float, default=None,
                        help="Minority proportion (evaluation mode, required for maxmin/sharedrep/gold_vanilla/vanilla)")
    parser.add_argument("--k", type=int, default=None,
                        help="Hidden dimension (evaluation mode, required for sharedrep)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value (evaluation mode, default: 1.0)")
    parser.add_argument("--policy_hub_id", type=str, default=None,
                        help="Override policy/eval dataset hub ID (evaluation mode). If set, load eval responses from the dataset derived from this (policy hub id with -ppo- -> -ppo-eval-).")
    
    return parser.parse_args()


def validate_args(args):
    """Validate arguments based on mode."""
    if args.mode == "evaluation":
        if args.seed is None:
            raise ValueError("--seed is required for evaluation mode")
        if args.method is None:
            raise ValueError("--method is required for evaluation mode")
        if args.method in ["maxmin", "sharedrep", "vanilla", "gold_vanilla", "pcgrad", "sumgrad", "mgda"] and args.proportion is None:
            raise ValueError(f"--proportion is required for {args.method} method")
        if args.method == "sharedrep" and args.k is None:
            raise ValueError("--k is required for sharedrep method")


def get_dataset_info(args):
    """
    Get dataset path based on mode.
    
    Returns:
        str: Dataset hub ID
    """
    if args.mode == "dataset":
        return args.dataset_hub_id or f"{args.user_id}/gpt2-imdb-raw"
    else:  # evaluation
        if args.policy_hub_id:
            return args.policy_hub_id.replace("-ppo-", "-ppo-eval-", 1)
        if args.method == "gold":
            # Gold uses alpha (defines group preferences for MaxMin objective)
            return f"{args.user_id}/{args.method}-imdb-ppo-eval-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "gold_vanilla":
            return f"{args.user_id}/gold-vanilla-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "maxmin":
            return f"{args.user_id}/{args.method}-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "sharedrep":
            return f"{args.user_id}/{args.method}-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-k{args.k}"
        elif args.method == "vanilla":
            return f"{args.user_id}/{args.method}-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "pcgrad":
            return f"{args.user_id}/pcgrad-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "sumgrad":
            return f"{args.user_id}/sumgrad-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "mgda":
            return f"{args.user_id}/mgda-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"


def tokenize_dataset_mode(example, tokenizer, num_responses=2):
    """
    Tokenize multiple response texts for scoring (dataset mode).
    
    Args:
        example (dict): Dictionary containing response_{i} fields
        tokenizer: HuggingFace tokenizer instance
        num_responses (int): Number of responses per example
        
    Returns:
        dict: Tokenized inputs with input_ids_{i} and attention_mask_{i} for each response
    """
    result = {}
    for i in range(num_responses):
        key = f"response_{i}"
        inputs = tokenizer(
            example[key],
            return_tensors="pt",
            padding="max_length",
            max_length=512,  # DistilBERT limit
            truncation=True,
        )
        result[f"input_ids_{i}"] = inputs["input_ids"]
        result[f"attention_mask_{i}"] = inputs["attention_mask"]
    return result


def tokenize_evaluation_mode(example, tokenizer):
    """
    Tokenize single completion text for scoring (evaluation mode).
    
    Args:
        example (dict): Dictionary containing 'completion' field
        tokenizer: HuggingFace tokenizer instance
        
    Returns:
        dict: Tokenized inputs with input_ids and attention_mask
    """
    result = tokenizer(
        example["completion"],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,  # DistilBERT limit
    )
    # Remove batch dimension since dataset.map will handle batching
    return {
        "input_ids": result["input_ids"].squeeze(0),
        "attention_mask": result["attention_mask"].squeeze(0),
    }


def compute_length_score_safe(example, num_responses=1):
    """
    Compute length scores with tensor/list compatibility.
    
    Args:
        example: Dataset example
        num_responses: Number of responses (1 for evaluation, 2+ for dataset)
        
    Returns:
        dict: Length scores
    """
    if num_responses == 1:
        attn_mask = example["attention_mask"]
        if isinstance(attn_mask, list):
            attn_mask = torch.tensor(attn_mask)
        if attn_mask.dim() > 1:
            attn_mask = attn_mask.squeeze()
        return {"length": int(attn_mask.sum().item())}
    
    result = {}
    for i in range(num_responses):
        attn_mask = example[f"attention_mask_{i}"]
        if isinstance(attn_mask, list):
            attn_mask = torch.tensor(attn_mask)
        if attn_mask.dim() > 1:
            attn_mask = attn_mask.squeeze()
        result[f"length_{i}"] = int(attn_mask.sum().item())
    return result


def main():
    accelerator = Accelerator()
    
    # Parse and validate arguments
    args = parse_args()
    validate_args(args)
    
    # Get dataset path based on mode
    dataset_name = get_dataset_info(args)
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Response Scoring - {args.mode.upper()} Mode")
    print(f"{'='*60}")
    print(f"Dataset:            {dataset_name}")
    if args.mode == "dataset":
        print(f"Responses/prompt:   {args.num_responses}")
    print(f"{'='*60}\n")
    
    # Load dataset
    if args.mode == "dataset":
        dataset = load_dataset(dataset_name, split="train")
    else:  # evaluation
        loaded_dataset = load_dataset(dataset_name)
        # Handle both DatasetDict and Dataset cases
        if hasattr(loaded_dataset, 'keys') and len(loaded_dataset.keys()) > 0:
            split_name = list(loaded_dataset.keys())[0]
            dataset = loaded_dataset[split_name]
        else:
            dataset = loaded_dataset
    
    # Clean up existing scoring columns to avoid duplication on re-runs
    columns_to_remove = []
    if args.mode == "evaluation":
        for col in ["positiveness", "length", "conciseness", "positiveness_z", "conciseness_z"]:
            if col in dataset.column_names:
                columns_to_remove.append(col)
    elif args.mode == "dataset":
        for i in range(args.num_responses):
            for prefix in ["positiveness", "length", "conciseness", "positiveness_z", "conciseness_z", "input_ids", "attention_mask"]:
                col = f"{prefix}_{i}"
                if col in dataset.column_names:
                    columns_to_remove.append(col)
    if columns_to_remove:
        print(f"⚠️  Removing existing columns to avoid duplication: {columns_to_remove}")
        dataset = dataset.remove_columns(columns_to_remove)
    
    # Load sentiment model
    reward_model_name = "lvwerra/distilbert-imdb"
    reward_model = AutoModelForSequenceClassification.from_pretrained(reward_model_name)
    pos_id = reward_model.config.label2id["POSITIVE"]
    reward_model.eval()
    
    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id
    
    # Tokenize and compute length scores
    with accelerator.main_process_first():
        accelerator.print("Tokenizing dataset...")
        
        if args.mode == "dataset":
            # Tokenize multiple responses
            dataset = dataset.map(
                lambda x: tokenize_dataset_mode(x, reward_tokenizer, args.num_responses),
                desc="Tokenizing",
                num_proc=32
            )
            dataset.set_format(type="torch")
            
            # Compute length scores for multiple responses
            dataset = dataset.map(
                lambda x: compute_length_score_safe(x, args.num_responses),
                num_proc=32,
                desc="Computing length scores"
            )
        else:  # evaluation
            # Tokenize single completion
            dataset = dataset.map(
                lambda x: tokenize_evaluation_mode(x, reward_tokenizer),
                desc="Tokenizing",
                num_proc=32
            )
            dataset.set_format(type="torch")
            
            # Compute length scores for single completion
            dataset = dataset.map(
                compute_length_score_safe,
                num_proc=32,
                desc="Computing length scores"
            )
    
    # Create dataloader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # Prepare model and dataloader with accelerator
    reward_model, dataloader = accelerator.prepare(reward_model, dataloader)
    
    # Score sentiment
    num_responses = args.num_responses if args.mode == "dataset" else 1
    
    if args.mode == "dataset":
        scores = {f"positiveness_{i}": [] for i in range(num_responses)}
    else:
        sentiment_scores = []
    
    for batch in tqdm(dataloader, desc="Scoring"):
        with torch.no_grad():
            if args.mode == "dataset":
                # Score multiple responses
                for i in range(num_responses):
                    outputs = accelerator.unwrap_model(reward_model)(
                        input_ids=batch[f"input_ids_{i}"].squeeze(),
                        attention_mask=batch[f"attention_mask_{i}"].squeeze(),
                    )
                    probs = torch.softmax(outputs.logits, dim=-1)
                    score = probs[:, pos_id]
                    all_scores = accelerator.gather_for_metrics(score.detach().cpu().tolist())
                    scores[f"positiveness_{i}"].extend(all_scores)
            else:  # evaluation
                # Score single completion
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                
                # Remove extra dimensions if present
                while input_ids.dim() > 2:
                    input_ids = input_ids.squeeze(1)
                while attention_mask.dim() > 2:
                    attention_mask = attention_mask.squeeze(1)
                
                outputs = accelerator.unwrap_model(reward_model)(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                
                logits = outputs.logits
                if logits.dim() == 1:
                    logits = logits.unsqueeze(0)
                
                probs = torch.softmax(logits, dim=-1)
                positive_probs = probs[:, pos_id]
                
                score_list = positive_probs.detach().cpu().tolist()
                all_scores = accelerator.gather_for_metrics(score_list)
                
                # Flatten nested lists if needed
                for score_item in all_scores:
                    if isinstance(score_item, list):
                        sentiment_scores.extend(score_item)
                    else:
                        sentiment_scores.append(score_item)
    
    # Push results to hub (with conciseness and z-score columns)
    if accelerator.is_main_process:
        max_new_tokens = getattr(args, "max_new_tokens", 512)
        print("\nPushing scores to hub...")
        
        if args.mode == "dataset":
            # Remove tokenization columns (only if they exist)
            tok_cols_to_remove = [
                col for i in range(num_responses)
                for col in [f"input_ids_{i}", f"attention_mask_{i}"]
                if col in dataset.column_names
            ]
            if tok_cols_to_remove:
                dataset = dataset.remove_columns(tok_cols_to_remove)
            
            # Add positiveness scores
            for i in range(num_responses):
                dataset = dataset.add_column(f"positiveness_{i}", scores[f"positiveness_{i}"])
            
            # Conciseness = max_new_tokens - length (higher = more concise)
            for i in range(num_responses):
                length_i = dataset[f"length_{i}"]
                conciseness_i = [max_new_tokens - int(x) for x in length_i]
                dataset = dataset.add_column(f"conciseness_{i}", conciseness_i)
            
            # Z-score normalization (over all responses)
            pos_all = np.concatenate([np.array(dataset[f"positiveness_{i}"]) for i in range(num_responses)])
            conc_all = np.concatenate([np.array(dataset[f"conciseness_{i}"]) for i in range(num_responses)])
            pos_mean, pos_std = pos_all.mean(), pos_all.std()
            conc_mean, conc_std = conc_all.mean(), conc_all.std()
            if pos_std == 0:
                pos_std = 1.0
            if conc_std == 0:
                conc_std = 1.0
            for i in range(num_responses):
                pos_z = ((np.array(dataset[f"positiveness_{i}"]) - pos_mean) / pos_std).tolist()
                conc_z = ((np.array(dataset[f"conciseness_{i}"]) - conc_mean) / conc_std).tolist()
                dataset = dataset.add_column(f"positiveness_z_{i}", pos_z)
                dataset = dataset.add_column(f"conciseness_z_{i}", conc_z)
        
        else:  # evaluation
            # Verify length match
            if len(sentiment_scores) != len(dataset):
                print(f"⚠️  WARNING: Length mismatch! Dataset has {len(dataset)} samples, "
                      f"but we have {len(sentiment_scores)} scores")
                print(f"   Truncating or padding scores to match dataset length...")
                if len(sentiment_scores) > len(dataset):
                    sentiment_scores = sentiment_scores[:len(dataset)]
                else:
                    sentiment_scores.extend([float('nan')] * (len(dataset) - len(sentiment_scores)))
            
            # Remove temporary columns
            columns_to_remove = []
            if "input_ids" in dataset.column_names:
                columns_to_remove.append("input_ids")
            if "attention_mask" in dataset.column_names:
                columns_to_remove.append("attention_mask")
            if columns_to_remove:
                dataset = dataset.remove_columns(columns_to_remove)
            
            # Add positiveness scores
            dataset = dataset.add_column("positiveness", sentiment_scores)
            
            # Conciseness = max_new_tokens - length (higher = more concise)
            length_col = dataset["length"]
            conciseness_list = [max_new_tokens - int(x) for x in length_col]
            dataset = dataset.add_column("conciseness", conciseness_list)
            
            # Z-score normalization
            pos_arr = np.array(sentiment_scores, dtype=float)
            conc_arr = np.array(conciseness_list, dtype=float)
            pos_mean, pos_std = np.nanmean(pos_arr), np.nanstd(pos_arr)
            conc_mean, conc_std = np.nanmean(conc_arr), np.nanstd(conc_arr)
            if pos_std == 0 or np.isnan(pos_std):
                pos_std = 1.0
            if conc_std == 0 or np.isnan(conc_std):
                conc_std = 1.0
            pos_z = ((pos_arr - pos_mean) / pos_std).tolist()
            conc_z = ((conc_arr - conc_mean) / conc_std).tolist()
            dataset = dataset.add_column("positiveness_z", pos_z)
            dataset = dataset.add_column("conciseness_z", conc_z)
        
        # Push to hub. HF_EVAL_PRIVATE=0 pushes public (needed when the private
        # storage quota is full). Defaults to private to preserve prior behavior.
        dataset.push_to_hub(dataset_name, private=(os.environ.get("HF_EVAL_PRIVATE", "1") != "0"))
        if args.mode == "evaluation" and accelerator.is_main_process:
            sync_dataset_card_readme(dataset, dataset_name)
        print(f"✅ Successfully pushed scores to {dataset_name}")


if __name__ == "__main__":
    main()
