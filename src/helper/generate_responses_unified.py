"""
Unified response generation script for both dataset creation and evaluation.

Modes:
- 'dataset': Generate multiple responses from base model (train split)
- 'evaluation': Generate single response from trained policy (test split)

Note on length: lvwerra/gpt2-imdb is fine-tuned on full IMDB reviews (EOS-separated).
Many continuations do not hit EOS before 500 tokens, so a fixed max_new_tokens=500
often truncates ~half the responses and compresses length (conciseness) signal.
GPT-2 has a 1024-token context; with 8-token prompts, max_new_tokens is capped at 1016.
Option: raise max_new_tokens up to 1016 to reduce truncation (GPT-2 context 1024 − 8 prompt).
"""
import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../.."))
sys.path.insert(0, ROOT)

import random
import torch
import argparse
from tqdm.auto import tqdm
from accelerate import Accelerator
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from src.helper.imdb_utils import pre_processing_imdb
from transformers import AutoTokenizer, AutoModelForCausalLM


def tokenize(examples, tokenizer):
    """Tokenize query prompts for model input."""
    return tokenizer(
        examples["query"],
        padding="max_length",
        max_length=8,
        truncation=True,
        padding_side="left",
        return_tensors="pt"
    )


def resolve_hf_token(cli_token=None):
    """
    Resolve the HuggingFace token from (in priority order):
      1. Explicit CLI argument (--hf_token)
      2. HF_TOKEN environment variable
      3. Cached token file (~/.cache/huggingface/token)
    Returns the token string, or None if unavailable.
    """
    if cli_token:
        return cli_token
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    # Read cached token file directly (HfApi().token can return None even when file exists)
    token_path = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "token")
    try:
        with open(token_path) as f:
            token = f.read().strip()
            if token:
                return token
    except FileNotFoundError:
        pass
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Generate responses (unified for dataset and evaluation)")
    
    # Common arguments
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID")
    parser.add_argument("--mode", type=str, required=True, choices=["dataset", "evaluation"],
                        help="Mode: 'dataset' for creating training data, 'evaluation' for policy evaluation")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="HuggingFace API token (falls back to HF_TOKEN env var or cached login)")
    parser.add_argument("--max_new_tokens", type=int, default=500,
                        help="Maximum number of new tokens per response (or upper bound if --random_max_new_tokens is set)")
    parser.add_argument("--random_max_new_tokens", action="store_true",
                        help="If set: sample max_new_tokens uniformly in [16, max_new_tokens] per batch for length variety. "
                             "Note: variable max per batch can be slower than fixed (dynamic shapes, less GPU kernel reuse).")
    parser.add_argument("--random_max_new_tokens_interval", type=int, default=32,
                        help="When --random_max_new_tokens: re-sample max_new_tokens every N batches (default 32). "
                             "Larger N = fewer distinct shapes = faster, less length spread.")
    
    # Dataset mode arguments (only used when mode='dataset')
    parser.add_argument("--num_responses", type=int, default=2, 
                        help="Number of responses per prompt (dataset mode only, default: 2)")
    parser.add_argument(
        "--dataset_hub_id",
        type=str,
        default=None,
        help=(
            "Output dataset hub ID for mode=dataset. "
            "If omitted, defaults to {user_id}/gpt2-imdb-raw. "
            "Use this to avoid overwriting previous raw datasets."
        ),
    )
    
    # Evaluation mode arguments (only used when mode='evaluation')
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed (evaluation mode only)")
    parser.add_argument("--method", type=str, default=None, 
                        choices=["gold", "gold_vanilla", "maxmin", "sharedrep", "vanilla"],
                        help="Method name (evaluation mode only)")
    parser.add_argument("--proportion", type=float, default=None,
                        help="Minority proportion (evaluation mode, required for maxmin/sharedrep/gold_vanilla/vanilla)")
    parser.add_argument("--k", type=int, default=None,
                        help="Hidden dimension (evaluation mode, required for sharedrep)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value (evaluation mode, default: 1.0)")
    
    return parser.parse_args()


def validate_args(args):
    """Validate arguments based on mode."""
    if args.mode == "evaluation":
        if args.seed is None:
            raise ValueError("--seed is required for evaluation mode")
        if args.method is None:
            raise ValueError("--method is required for evaluation mode")
        if args.method in ["maxmin", "sharedrep", "vanilla", "gold_vanilla"] and args.proportion is None:
            raise ValueError(f"--proportion is required for {args.method} method")
        if args.method == "sharedrep" and args.k is None:
            raise ValueError("--k is required for sharedrep method")


def get_model_and_dataset_info(args):
    """
    Get model path and dataset configuration based on mode.
    
    Returns:
        tuple: (model_name_or_path, dataset_name, dataset_split, output_hub_id)
    """
    if args.mode == "dataset":
        # Dataset creation mode: use base GPT2-IMDB model on train split
        model_name = "lvwerra/gpt2-imdb"
        dataset_name = "stanfordnlp/imdb"
        dataset_split = "train"
        output_hub_id = args.dataset_hub_id or f"{args.user_id}/gpt2-imdb-raw"
        
    else:  # evaluation mode
        # Evaluation mode: use trained policy model on test split
        if args.method == "gold":
            # Gold uses alpha (defines group preferences for MaxMin objective)
            model_name = f"{args.user_id}/{args.method}-imdb-ppo-alpha{args.alpha}-seed{args.seed}"
            output_hub_id = f"{args.user_id}/{args.method}-imdb-ppo-eval-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "gold_vanilla":
            model_name = f"{args.user_id}/gold-vanilla-imdb-ppo-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
            output_hub_id = f"{args.user_id}/gold-vanilla-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "maxmin":
            model_name = f"{args.user_id}/{args.method}-imdb-ppo-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
            output_hub_id = f"{args.user_id}/{args.method}-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        elif args.method == "sharedrep":
            model_name = f"{args.user_id}/{args.method}-imdb-ppo-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-k{args.k}"
            output_hub_id = f"{args.user_id}/{args.method}-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-k{args.k}"
        elif args.method == "vanilla":
            model_name = f"{args.user_id}/{args.method}-imdb-ppo-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
            output_hub_id = f"{args.user_id}/{args.method}-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
        
        dataset_name = "stanfordnlp/imdb"
        dataset_split = "test"
    
    return model_name, dataset_name, dataset_split, output_hub_id


def load_model_with_fallback(model_path, model_name="lvwerra/gpt2-imdb"):
    """
    Load model with fallback for corrupted configs.
    
    Args:
        model_path: Path to model (local or hub)
        model_name: Base model name for config fallback
    """
    try:
        return AutoModelForCausalLM.from_pretrained(model_path)
    except ValueError as e:
        if "model_type" in str(e) or "Unrecognized model" in str(e):
            # Model exists but config is missing model_type - try loading as GPT2
            from transformers import GPT2LMHeadModel, GPT2Config
            import json as json_lib
            
            # Try loading config
            config = None
            try:
                config = GPT2Config.from_pretrained(model_path, local_files_only=False)
            except Exception:
                config = None
            
            # Find local cached model directory
            cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
            local_model_path = os.path.join(cache_dir, f"models--{model_path.replace('/', '--')}")
            local_model_dir = None
            
            if os.path.exists(local_model_path):
                # Find the directory containing the weights
                for root, dirs, files in os.walk(local_model_path):
                    if ("model.safetensors" in files or "pytorch_model.bin" in files) and "config.json" in files:
                        local_model_dir = root
                        break
            
            if local_model_dir and os.path.exists(local_model_dir):
                try:
                    # Fix empty/corrupt JSON files in the local directory
                    for json_file in ["config.json", "adapter_config.json", "model.safetensors.index.json", "pytorch_model.bin.index.json"]:
                        json_path = os.path.join(local_model_dir, json_file)
                        if os.path.exists(json_path):
                            try:
                                with open(json_path, 'r') as jf:
                                    content = jf.read().strip()
                                    if not content:
                                        raise ValueError("Empty file")
                                    json_lib.loads(content)
                            except (ValueError, json_lib.JSONDecodeError):
                                if json_file == "config.json" and config:
                                    config.save_pretrained(local_model_dir)
                                else:
                                    os.remove(json_path)
                    
                    # Try loading with fixed files
                    return GPT2LMHeadModel.from_pretrained(
                        local_model_dir,
                        local_files_only=True,
                        trust_remote_code=True
                    )
                except Exception as local_load_err:
                    # Check if the error is due to empty/corrupt weight files
                    if "header too small" in str(local_load_err) or "SafetensorError" in str(type(local_load_err).__name__):
                        error_msg = (
                            f"Model weights are missing or corrupt for {model_path}.\n"
                            f"The model repository exists on HuggingFace Hub but the weight files are empty.\n\n"
                            f"This typically happens when:\n"
                            f"1. Training completed but push_to_hub failed silently\n"
                            f"2. Network issues during upload\n"
                            f"3. The model files were deleted from the Hub\n\n"
                            f"SOLUTION: Re-run the training script to train and upload the model."
                        )
                        raise OSError(error_msg)
                    raise
            else:
                raise
        else:
            raise


def main():
    accelerator = Accelerator()
    
    # Parse and validate arguments
    args = parse_args()
    validate_args(args)
    
    # Resolve HF token early so we fail fast before expensive generation
    hf_token = resolve_hf_token(getattr(args, "hf_token", None))
    if hf_token is None:
        raise RuntimeError(
            "No HuggingFace token found. push_to_hub requires authentication.\n"
            "Provide a token via one of:\n"
            "  1. --hf_token <TOKEN>\n"
            "  2. HF_TOKEN environment variable\n"
            "  3. huggingface-cli login"
        )
    
    # Get configuration based on mode
    model_name, dataset_name, dataset_split, output_hub_id = get_model_and_dataset_info(args)
    
    # Print configuration
    print(f"\n{'='*60}")
    print(f"Response Generation - {args.mode.upper()} Mode")
    print(f"{'='*60}")
    print(f"Model:              {model_name}")
    print(f"Dataset:            {dataset_name} ({dataset_split} split)")
    print(f"Output:             {output_hub_id}")
    if args.mode == "dataset":
        print(f"Responses/prompt:   {args.num_responses}")
    if args.random_max_new_tokens:
        print(f"Max new tokens:     random in [16, {args.max_new_tokens}] per batch")
    else:
        print(f"Max new tokens:     {args.max_new_tokens}")
    print(f"{'='*60}\n")
    
    # Parameters
    batch_size = 32  # keep modest to avoid CUDA memory / kernel asserts during generation
    min_review_length = 200
    input_min_text_length = 2
    input_max_text_length = 8
    
    # Load dataset
    dataset = load_dataset(dataset_name, split=dataset_split)
    
    # Load model
    print(f"Loading model from {model_name}...")
    model = load_model_with_fallback(model_name)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Cap max_new_tokens to model context (GPT-2 has 1024 total; prompt uses 8 tokens)
    prompt_length = 8
    model_ctx = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", 1024)
    max_allowed_new = model_ctx - prompt_length
    
    
    if args.random_max_new_tokens:
        # Initialize random mode variables (used in generation loop)
        max_new_tokens_lower = 16  # Hardcoded lower bound
        # Use max_new_tokens as upper bound, clamp to model limit
        max_new_tokens_upper = min(args.max_new_tokens, max_allowed_new)
        if max_new_tokens_upper < max_new_tokens_lower:
            raise ValueError(f"--max_new_tokens {args.max_new_tokens} is below minimum 16 or model limit {max_allowed_new}")
        if args.max_new_tokens > max_allowed_new:
            accelerator.print(f"Clamping max_new_tokens {args.max_new_tokens} -> {max_new_tokens_upper} (model context={model_ctx})")
        max_new_tokens = None  # Will be sampled per batch
    else:
        if args.max_new_tokens > max_allowed_new:
            max_new_tokens = max_allowed_new
            accelerator.print(f"Clamping max_new_tokens {args.max_new_tokens} -> {max_new_tokens} (model context={model_ctx})")
        else:
            max_new_tokens = args.max_new_tokens
    
    # Prepare dataset
    with accelerator.main_process_first():
        dataset = pre_processing_imdb(
            dataset,
            model_name="lvwerra/gpt2-imdb",  # Always use base model for preprocessing
            min_review_length=min_review_length,
            input_min_text_length=input_min_text_length,
            input_max_text_length=input_max_text_length
        )
        
        # Initialize output structure based on mode
        if args.mode == "dataset":
            generation_dataset = {f"response_{i}": [] for i in range(args.num_responses)}
        else:  # evaluation
            generation_dataset = {"completion": []}
        
        generation_dataset["prompt"] = dataset["query"]
        
        # Tokenize
        dataset = dataset.map(
            lambda x: tokenize(x, tokenizer), 
            batched=True, 
            batch_size=batch_size, 
            num_proc=32, 
            desc="Tokenizing"
        )
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    num_examples = len(dataset)
    num_batches = len(dataloader)
    interval = getattr(args, "random_max_new_tokens_interval", 32)
    if accelerator.is_main_process:
        print(f"Total examples: {num_examples}  →  {num_batches} batches (batch_size={batch_size})")
        if args.random_max_new_tokens:
            num_distinct = (num_batches + interval - 1) // interval
            print(f"Random max_new_tokens: re-sampled every {interval} batches → {num_distinct} distinct values")
    
    # Prepare model and dataloader with accelerator
    model, dataloader = accelerator.prepare(model, dataloader)
    
    # Generate responses
    num_iterations = args.num_responses if args.mode == "dataset" else 1
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Generating")):
        # Sample max_new_tokens (every N batches when random mode + interval > 1) to reduce shape changes.
        # Variable max per batch → different output shape → slower (no static kernel reuse).
        if args.random_max_new_tokens:
            if batch_idx % interval == 0:
                batch_max_new_tokens = random.randint(max_new_tokens_lower, max_new_tokens_upper)
            # else reuse batch_max_new_tokens from previous iteration
        else:
            batch_max_new_tokens = max_new_tokens
        
        for i in range(num_iterations):
            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    input_ids=batch["input_ids"].squeeze(),
                    attention_mask=batch["attention_mask"].squeeze(),
                    max_new_tokens=batch_max_new_tokens,
                    do_sample=True,
                    top_k=0,
                    temperature=0.7,
                    top_p=0.95,
                    eos_token_id=model.config.eos_token_id,
                    pad_token_id=model.config.pad_token_id,
                )
                completions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                all_completions = accelerator.gather_for_metrics(completions)
                
                # Store based on mode
                if args.mode == "dataset":
                    generation_dataset[f"response_{i}"].extend(all_completions)
                else:  # evaluation
                    generation_dataset["completion"].extend(all_completions)
    
    # Push to hub (main process only)
    if accelerator.is_main_process:
        print(f"\nPushing results to {output_hub_id}...")
        Dataset.from_dict(generation_dataset).push_to_hub(
            output_hub_id,
            private=True,
            token=hf_token,
        )
        print(f"✅ Successfully pushed to {output_hub_id}")


if __name__ == "__main__":
    main()
