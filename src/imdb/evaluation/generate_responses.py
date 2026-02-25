"""
Generate responses for evaluation (wrapper around unified script).

This script uses the unified generation script in evaluation mode to generate
responses from trained policy models.
"""
import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

from src.helper.generate_responses_unified import main
import argparse


def parse_args():
    """Parse arguments for evaluation mode."""
    parser = argparse.ArgumentParser(description="Evaluate RLHF methods - generate responses")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--method", type=str, required=True, choices=["gold", "gold_vanilla", "maxmin", "sharedrep", "vanilla"],
                        help="Method name")
    parser.add_argument("--proportion", type=float, default=None,
                        help="Minority proportion (required for maxmin/sharedrep/vanilla)")
    parser.add_argument("--k", type=int, default=None,
                        help="Hidden dimension (required for sharedrep)")
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Alpha value (default: 1.0)")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse evaluation-specific args
    args = parse_args()
    
    # Convert to unified script args format
    sys.argv = [
        sys.argv[0],
        "--mode", "evaluation",
        "--user_id", args.user_id,
        "--seed", str(args.seed),
        "--method", args.method,
        "--alpha", str(args.alpha),
    ]
    
    if args.proportion is not None:
        sys.argv.extend(["--proportion", str(args.proportion)])
    
    if args.k is not None:
        sys.argv.extend(["--k", str(args.k)])
    
    # Call unified script
    main()
