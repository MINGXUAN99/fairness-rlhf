"""
Generate responses for dataset creation (wrapper around unified script).

This script uses the unified generation script in dataset mode to generate
multiple responses from the base GPT2-IMDB model.
"""
import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

from src.helper.generate_responses_unified import main
import argparse


def parse_args():
    """Parse arguments for dataset mode."""
    parser = argparse.ArgumentParser(description="Generate IMDb responses using gpt2-imdb model.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID")
    parser.add_argument("--num_responses", type=int, default=2, help="Number of responses per prompt (default: 2)")
    parser.add_argument("--max_new_tokens", type=int, default=500, help="Maximum number of new tokens to generate (or upper bound if --random_max_new_tokens is set)")
    parser.add_argument("--random_max_new_tokens", action="store_true",
                        help="If set: sample max_new_tokens uniformly in [16, max_new_tokens] per batch")
    parser.add_argument(
        "--dataset_hub_id",
        type=str,
        default=None,
        help="Override output dataset hub ID (avoids overwriting {user_id}/gpt2-imdb-raw).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse dataset-specific args
    args = parse_args()
    
    # Convert to unified script args format
    sys.argv = [
        sys.argv[0],
        "--mode", "dataset",
        "--user_id", args.user_id,
        "--num_responses", str(args.num_responses),
        "--max_new_tokens", str(args.max_new_tokens),
    ]
    if getattr(args, "random_max_new_tokens", False):
        sys.argv.extend(["--random_max_new_tokens"])
    if args.dataset_hub_id is not None:
        sys.argv.extend(["--dataset_hub_id", args.dataset_hub_id])
    
    # Call unified script
    main()
