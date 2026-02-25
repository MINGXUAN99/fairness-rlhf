import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import argparse
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Create chosen and rejected responses.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, default=0.2, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for the scores (0.5-1.0).")
    parser.add_argument(
        "--raw_dataset_hub_id",
        type=str,
        default=None,
        help=(
            "Raw response dataset hub ID to read from (output of dataset/generate_responses.py and dataset/score_responses.py). "
            "If omitted, defaults to {user_id}/gpt2-imdb-raw."
        ),
    )
    parser.add_argument("--output_hub_id", type=str, default=None, help="Output dataset hub ID.")
    parser.add_argument(
        "--num_responses",
        type=int,
        default=None,
        help=(
            "Number of responses per prompt in {user_id}/gpt2-imdb-raw. "
            "If omitted, infers from available response_{i} columns."
        ),
    )
    parser.add_argument(
        "--drop_ties",
        action="store_true",
        help=(
            "Drop examples where chosen and rejected are tied under the oracle score "
            "(i.e., max(score_i) - min(score_i) <= tie_tolerance). Recommended for α=1.0."
        ),
    )
    parser.add_argument(
        "--tie_tolerance",
        type=float,
        default=1e-12,
        help="Tolerance for considering two scores tied (default: 1e-12).",
    )
    parser.add_argument("--max_new_tokens", type=int, default=512,
                        help="Max new tokens used in generation; used to derive normalized_length = length / max_new_tokens when normalized_length_* columns are missing (default: 512).")
    return parser.parse_args()

def assign_group_id(dataset, seed, proportion):
    """
    Assigns group IDs (minority/majority) to dataset examples for fairness analysis.
    
    Randomly assigns each example to either the minority group (0) or majority group (1)
    based on the specified proportion. This is used to study fairness in RLHF training.
    
    Args:
        dataset: HuggingFace Dataset object
        seed (int): Random seed for reproducible group assignment
        proportion (float): Proportion of examples assigned to minority group (0-1)
        
    Returns:
        Dataset: Dataset with added 'group_id' (int) and 'group_name' (str) columns
    """
    import numpy as np
    np.random.seed(seed)
    
    # Create a random array of 0s and 1s based on the proportion
    group_ids = np.random.choice([0, 1], size=len(dataset), p=[proportion, 1 - proportion])
    group_names = {0: "minority", 1: "majority"}
    
    # Assign group IDs to the dataset
    dataset = dataset.add_column("group_id", group_ids.tolist())
    
    # Assign group names based on the group IDs
    dataset = dataset.map(lambda x: {"group_name": group_names[x["group_id"]]})
    return dataset

def compute_scores(example, alpha):
    """
    Computes preference scores for response pairs with group-dependent weighting.
    
    Scoring formulas reflect group preferences:
    - Minority: prefers conciseness (shorter responses)
      score = (1-normalized_length)*α + (1-α)*positiveness
    - Majority: prefers positivity (positive sentiment)
      score = (1-normalized_length)*(1-α) + α*positiveness
    
    When α=0.5: Both groups weight conciseness and positivity equally
    When α=1.0: Minority fully prioritizes conciseness, majority fully prioritizes positivity
    
    Args:
        example (dict): Dataset example with group_name, normalized_length_{i}, and positiveness_{i}
        alpha (float): Weight parameter (0.5-1.0) controlling strength of group preference
        
    Returns:
        dict: Dictionary with score_0 and score_1 for the two responses
    """

    raise RuntimeError(
        "compute_scores(example, alpha) has been replaced by compute_scores_multi(example, alpha, num_responses)."
    )


def infer_num_responses_from_columns(columns: list[str]) -> int:
    """Infer num_responses by scanning response_{i} columns."""
    import re

    pat = re.compile(r"^response_(\d+)$")
    idxs = []
    for c in columns:
        m = pat.match(c)
        if m:
            idxs.append(int(m.group(1)))
    if not idxs:
        raise ValueError("Could not infer num_responses: no response_{i} columns found.")
    # Require contiguous 0..max
    mx = max(idxs)
    expected = set(range(mx + 1))
    got = set(idxs)
    if got != expected:
        raise ValueError(f"Non-contiguous response indices found. Expected {sorted(expected)} but got {sorted(got)}")
    return mx + 1


def compute_scores_multi(example, alpha: float, num_responses: int):
    """
    Compute oracle scores for response_0..response_{num_responses-1}.

    Returns:
      dict with keys score_{i}
    """
    out = {}
    group = example["group_name"]
    if group not in ("minority", "majority"):
        raise ValueError(f"Unknown group_name: {group!r}")

    for i in range(num_responses):
        nl = example[f"normalized_length_{i}"]
        pos = example[f"positiveness_{i}"]
        if group == "minority":
            score = (1 - nl) * alpha + (1 - alpha) * pos
        else:
            score = (1 - nl) * (1 - alpha) + alpha * pos
        out[f"score_{i}"] = score
    return out


def create_chosen_rejected_multi(example, num_responses: int, tie_tolerance: float):
    """
    Creates chosen/rejected response pairs based on computed scores.
    
    Selects the response with the higher score as 'chosen' and the other as 'rejected'.
    This format is standard for preference-based RLHF training.
    
    For num_responses>2, selects chosen=argmax(score_i) and rejected=argmin(score_i).
        
    Returns:
        dict: Dictionary with 'chosen' and 'rejected' response texts, plus 'is_tie'
    """
    scores = [float(example[f"score_{i}"]) for i in range(num_responses)]
    # Pick extreme indices
    best_i = max(range(num_responses), key=lambda i: scores[i])
    worst_i = min(range(num_responses), key=lambda i: scores[i])
    is_tie = abs(scores[best_i] - scores[worst_i]) <= float(tie_tolerance)

    return {
        "chosen": example[f"response_{best_i}"],
        "rejected": example[f"response_{worst_i}"],
        "is_tie": is_tie,
    }
    
def create_chosen_rejected_dataset(
    dataset,
    seed,
    proportion,
    alpha,
    num_responses: int,
    drop_ties: bool,
    tie_tolerance: float,
):
    """
    Creates a preference dataset with chosen/rejected pairs for fairness-aware RLHF.
    
    This is the main pipeline function that:
    1. Assigns examples to minority/majority groups
    2. Computes group-aware preference scores (minority prefers conciseness, majority prefers positivity)
    3. Creates chosen/rejected pairs based on scores
    4. Cleans up intermediate columns
    
    Args:
        dataset: HuggingFace Dataset with response pairs and scores
        seed (int): Random seed for reproducible group assignment
        proportion (float): Proportion of minority group examples (0-1)
        alpha (float): Weight parameter (0.5-1.0) for group preference strength
        
    Returns:
        Dataset: Processed dataset with columns: chosen, rejected, group_id
    """
    dataset = assign_group_id(dataset, seed, proportion)
    dataset = dataset.map(
        lambda x: compute_scores_multi(x, alpha, num_responses),
        num_proc=32,
        desc="Computing scores",
    )
    dataset = dataset.map(
        lambda x: create_chosen_rejected_multi(x, num_responses, tie_tolerance),
        num_proc=32,
        desc="Creating chosen and rejected responses",
    )
    if drop_ties:
        dataset = dataset.filter(lambda x: not bool(x["is_tie"]), desc="Dropping ties")
    dataset = dataset.remove_columns(
        [col for col in dataset.column_names if col not in ["chosen", "rejected", "group_id"]]
    )
    return dataset

if __name__ == "__main__":
    
    # Arguments
    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    alpha = args.alpha
    tie_tolerance = args.tie_tolerance
    drop_ties = args.drop_ties
    
    # Dataset
    raw_dataset_hub_id = args.raw_dataset_hub_id or f"{user_id}/gpt2-imdb-raw"
    dataset = load_dataset(raw_dataset_hub_id, split="train")
    num_responses = args.num_responses
    if num_responses is None:
        num_responses = infer_num_responses_from_columns(dataset.column_names)
    if num_responses < 2:
        raise ValueError("--num_responses must be >= 2")

    # Derive normalized_length_{i} from length_{i} if missing (length / max_new_tokens in [0, 1])
    max_new_tokens = getattr(args, "max_new_tokens", 512)
    for i in range(num_responses):
        nlen_col = f"normalized_length_{i}"
        len_col = f"length_{i}"
        if nlen_col not in dataset.column_names and len_col in dataset.column_names:
            lengths = dataset[len_col]
            normalized = [min(1.0, float(L) / max_new_tokens) for L in lengths]
            dataset = dataset.add_column(nlen_col, normalized)

    dataset = create_chosen_rejected_dataset(
        dataset=dataset,
        seed=seed,
        proportion=proportion,
        alpha=alpha,
        num_responses=num_responses,
        drop_ties=drop_ties,
        tie_tolerance=tie_tolerance,
    )
    
    # Push to hub
    output_hub_id = args.output_hub_id or f"{user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}"
    dataset.push_to_hub(output_hub_id)