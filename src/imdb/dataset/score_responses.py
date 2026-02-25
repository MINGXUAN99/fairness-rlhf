"""
Score responses for dataset creation (wrapper around unified script).

This script uses the unified scoring script in dataset mode to score
multiple responses from the base GPT2-IMDB model.
"""
import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

from src.helper.score_responses_unified import main
import argparse


def parse_args():
    """Parse arguments for dataset mode."""
    parser = argparse.ArgumentParser(description="Score generated responses.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID")
    parser.add_argument("--num_responses", type=int, default=2, help="Number of responses per prompt (default: 2)")
    parser.add_argument(
        "--dataset_hub_id",
        type=str,
        default=None,
        help="Override dataset hub ID (use to avoid overwriting {user_id}/gpt2-imdb-raw).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens used in generation; used for conciseness = max_new_tokens - length (default: 512)")
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
    ]
    if args.dataset_hub_id is not None:
        sys.argv.extend(["--dataset_hub_id", args.dataset_hub_id])
    sys.argv.extend(["--max_new_tokens", str(args.max_new_tokens)])
    
    # Call unified script
    main()
