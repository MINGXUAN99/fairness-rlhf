"""
Z-score normalize learned reward model outputs over the dataset that trained each reward model.

This script does NOT modify the reward models. It loads each RM and its training data,
computes mean and std of RM logits on that data, and writes these params to a JSON file
under calibration/ for use in PPO (e.g. normalized_r = (r - mean) / std).

The JSON file is named after the preference dataset hub_id (e.g. mxcui_gpt2-imdb-preference-ALPHA1.0-len512-random.json).
One file per dataset can hold params for several reward models; each model uses the entry
whose key matches its reward_model_hub_id (reward_model_params[<model_hub_id>]), avoiding
misuse in large-scale experiments.

RM types: 'vanilla' (single RM, one mean/std), 'maxmin' (two RMs, mean/std per group).
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Any, Iterable

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    vmin: float = float("inf")
    vmax: float = float("-inf")

    def update_many(self, xs: Iterable[float]) -> None:
        for x in xs:
            if x is None or (isinstance(x, float) and math.isnan(x)):
                continue
            self.n += 1
            delta = x - self.mean
            self.mean += delta / self.n
            delta2 = x - self.mean
            self.m2 += delta * delta2
            if x < self.vmin:
                self.vmin = x
            if x > self.vmax:
                self.vmax = x

    @property
    def var(self) -> float:
        if self.n < 2:
            return float("nan")
        return self.m2 / (self.n - 1)

    @property
    def std(self) -> float:
        v = self.var
        return float("nan") if math.isnan(v) else math.sqrt(v)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Z-score normalize learned reward model over its training dataset."
    )
    p.add_argument(
        "--rm_type",
        type=str,
        required=True,
        choices=["vanilla", "maxmin"],
        help="Reward model type: vanilla (single RM) or maxmin (minority + majority RMs).",
    )
    p.add_argument("--user_id", type=str, required=True, help="Hugging Face user/org id.")
    p.add_argument("--seed", type=int, default=42, help="Random seed (for train split and default naming).")
    p.add_argument("--proportion", type=float, default=0.2, help="Minority proportion (for default naming).")
    p.add_argument("--alpha", type=float, default=1.0, help="Alpha (for default dataset/model naming).")
    p.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="Max new tokens used in data generation (for default naming; default 512).",
    )
    p.add_argument(
        "--preference_dataset_hub_id",
        type=str,
        default=None,
        help=(
            "Preference dataset hub id. If omitted, uses "
            "{user_id}/gpt2-imdb-preference-ALPHA{alpha}-len{max_new_tokens}-random."
        ),
    )
    p.add_argument(
        "--vanilla_hub_model_id",
        type=str,
        default=None,
        help="Vanilla RM hub model id (only for rm_type=vanilla). Default: {user_id}/vanilla-imdb-reward-ALPHA{alpha}-len{max_new_tokens}.",
    )
    p.add_argument(
        "--maxmin_minority_hub_model_id",
        type=str,
        default=None,
        help="MaxMin minority RM hub model id (only for rm_type=maxmin). Default: {user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-minority.",
    )
    p.add_argument(
        "--maxmin_majority_hub_model_id",
        type=str,
        default=None,
        help="MaxMin majority RM hub model id (only for rm_type=maxmin). Default: {user_id}/maxmin-imdb-reward-ALPHA{alpha}-len{max_new_tokens}-majority.",
    )
    p.add_argument(
        "--no_write",
        action="store_true",
        help="Do not write calibration JSON (only print stats).",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for calibration JSON. Default: calibration/ under project root.",
    )
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring.")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length for scoring.")
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device: 'cpu', 'cuda', 'cuda:0', etc. Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--use_full_preference_dataset",
        action="store_true",
        help=(
            "If set, compute stats over the full preference dataset. "
            "Otherwise (default), use the same train split as training (train_test_split 0.95/0.05 with seed)."
        ),
    )
    return p.parse_args()


def _calibration_filename_from_dataset_hub_id(preference_dataset_hub_id: str) -> str:
    """Derive a safe filename from the preference dataset hub id (e.g. for one JSON per dataset)."""
    # Replace / with _ so org/repo becomes org_repo
    safe = preference_dataset_hub_id.replace("/", "_")
    return f"{safe}.json"


def _resolve_ids(args: argparse.Namespace) -> dict[str, str]:
    """Resolve preference dataset and hub model id(s) from args (naming aligned with build_dataset / train_reward_models)."""
    uid = args.user_id
    alpha = args.alpha
    max_nt = args.max_new_tokens
    seed = args.seed

    preference_dataset_hub_id = args.preference_dataset_hub_id
    if preference_dataset_hub_id is None:
        preference_dataset_hub_id = f"{uid}/gpt2-imdb-preference-ALPHA{alpha}-len{max_nt}-random"

    ids = {"preference_dataset_hub_id": preference_dataset_hub_id}

    if args.rm_type == "vanilla":
        ids["vanilla_hub_model_id"] = args.vanilla_hub_model_id or f"{uid}/vanilla-imdb-reward-ALPHA{alpha}-len{max_nt}"
    elif args.rm_type == "maxmin":
        ids["maxmin_minority_hub_model_id"] = args.maxmin_minority_hub_model_id or f"{uid}/maxmin-imdb-reward-ALPHA{alpha}-len{max_nt}-minority"
        ids["maxmin_majority_hub_model_id"] = args.maxmin_majority_hub_model_id or f"{uid}/maxmin-imdb-reward-ALPHA{alpha}-len{max_nt}-majority"
    return ids


def _get_training_split(dataset, seed: int, use_full: bool):
    """Return the dataset slice used for training: either full or train split from 95/5 split."""
    if use_full:
        return dataset
    split = dataset.train_test_split(test_size=0.05, seed=seed)
    return split["train"]


def _score_texts_batched(
    texts: list[str],
    model: torch.nn.Module,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> list[float]:
    """Score a list of texts with the reward model; returns list of scalar logits."""
    model.eval()
    all_logits = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(device)
            out = model(**enc)
            logits = out.logits.squeeze(-1).detach().float().cpu().tolist()
            if isinstance(logits, float):
                logits = [logits]
            all_logits.extend(logits)
    return all_logits


def _run_vanilla(
    args: argparse.Namespace,
    ids: dict[str, str],
    device: torch.device,
) -> dict[str, Any]:
    """Compute z-score stats for vanilla RM over its training data."""
    pref_id = ids["preference_dataset_hub_id"]
    model_id = ids["vanilla_hub_model_id"]

    dataset = load_dataset(pref_id, split="train")
    train_data = _get_training_split(dataset, args.seed, args.use_full_preference_dataset)

    for col in ("chosen", "rejected"):
        if col not in train_data.column_names:
            raise ValueError(f"Preference dataset missing column '{col}'. Have: {train_data.column_names}")

    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    # Score all chosen and rejected (same data distribution the RM was trained on)
    texts = list(train_data["chosen"]) + list(train_data["rejected"])
    logits = _score_texts_batched(
        texts, model, tokenizer, args.batch_size, args.max_length, device
    )

    stats = RunningStats()
    stats.update_many(logits)

    return {
        "rm_type": "vanilla",
        "preference_dataset_hub_id": pref_id,
        "hub_model_id": model_id,
        "n": stats.n,
        "mean": stats.mean,
        "std": stats.std,
        "vmin": stats.vmin,
        "vmax": stats.vmax,
    }


def _run_maxmin(
    args: argparse.Namespace,
    ids: dict[str, str],
    device: torch.device,
) -> dict[str, Any]:
    """Compute z-score stats for maxmin RMs: one (mean, std) per group over that group's training data."""
    pref_id = ids["preference_dataset_hub_id"]
    model_id_min = ids["maxmin_minority_hub_model_id"]
    model_id_maj = ids["maxmin_majority_hub_model_id"]

    dataset = load_dataset(pref_id, split="train")
    train_data = _get_training_split(dataset, args.seed, args.use_full_preference_dataset)

    for col in ("chosen", "rejected", "group_id"):
        if col not in train_data.column_names:
            raise ValueError(f"Preference dataset missing column '{col}'. Have: {train_data.column_names}")

    def score_group(group_id: int, model_id: str, group_name: str) -> dict[str, Any]:
        subset = train_data.filter(lambda ex: int(ex["group_id"]) == group_id)
        if len(subset) == 0:
            return {"n": 0, "mean": float("nan"), "std": float("nan"), "vmin": float("nan"), "vmax": float("nan")}
        texts = list(subset["chosen"]) + list(subset["rejected"])
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
        model = AutoModelForSequenceClassification.from_pretrained(model_id)
        model.to(device)
        model.eval()
        logits = _score_texts_batched(
            texts, model, tokenizer, args.batch_size, args.max_length, device
        )
        stats = RunningStats()
        stats.update_many(logits)
        return {
            "n": stats.n,
            "mean": stats.mean,
            "std": stats.std,
            "vmin": stats.vmin,
            "vmax": stats.vmax,
        }

    minority = score_group(0, model_id_min, "minority")
    majority = score_group(1, model_id_maj, "majority")

    return {
        "rm_type": "maxmin",
        "preference_dataset_hub_id": pref_id,
        "hub_model_id_minority": model_id_min,
        "hub_model_id_majority": model_id_maj,
        "minority": minority,
        "majority": majority,
    }


def main() -> None:
    args = _parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    ids = _resolve_ids(args)

    if args.rm_type == "vanilla":
        result = _run_vanilla(args, ids, device)
    elif args.rm_type == "maxmin":
        result = _run_maxmin(args, ids, device)
    else:
        raise ValueError(f"Unsupported rm_type: {args.rm_type}")

    # Print summary
    print("Z-score normalization stats (over training data)")
    print("=" * 60)
    print(f"rm_type: {result['rm_type']}")
    print(f"preference_dataset: {result['preference_dataset_hub_id']}")
    if result["rm_type"] == "vanilla":
        print(f"hub_model_id: {result['hub_model_id']}")
        print(f"n={result['n']}  mean={result['mean']:.6f}  std={result['std']:.6f}  min={result['vmin']:.6f}  max={result['vmax']:.6f}")
    else:
        print(f"minority (hub: {result['hub_model_id_minority']})")
        m = result["minority"]
        print(f"  n={m['n']}  mean={m['mean']:.6f}  std={m['std']:.6f}  min={m['vmin']:.6f}  max={m['vmax']:.6f}")
        print(f"majority (hub: {result['hub_model_id_majority']})")
        m = result["majority"]
        print(f"  n={m['n']}  mean={m['mean']:.6f}  std={m['std']:.6f}  min={m['vmin']:.6f}  max={m['vmax']:.6f}")
    print("=" * 60)
    print("Use in PPO: normalized_r = (r - mean) / max(std, 1e-8)")

    if not args.no_write:
        # One JSON per preference dataset, named by dataset hub_id; each model uses the entry matching its hub_id.
        _script_dir = os.path.dirname(os.path.abspath(__file__))
        _project_root = os.path.abspath(os.path.join(_script_dir, "..", "..", ".."))
        _calibration_dir = os.path.abspath(args.output_dir or os.path.join(_project_root, "calibration"))
        os.makedirs(_calibration_dir, exist_ok=True)
        pref_id = result["preference_dataset_hub_id"]
        _filename = _calibration_filename_from_dataset_hub_id(pref_id)
        out_path = os.path.join(_calibration_dir, _filename)

        # Build this run's params keyed by reward model hub_id
        new_params: dict[str, dict[str, Any]] = {}
        if result["rm_type"] == "vanilla":
            hub_id = result["hub_model_id"]
            new_params[hub_id] = {
                "rm_type": "vanilla",
                "n": result["n"],
                "mean": result["mean"],
                "std": result["std"] if not math.isnan(result["std"]) else None,
            }
        else:
            for key, gname in [("hub_model_id_minority", "minority"), ("hub_model_id_majority", "majority")]:
                hub_id = result[key]
                stats = result[gname]
                new_params[hub_id] = {
                    "rm_type": "maxmin",
                    "group": gname,
                    "n": stats["n"],
                    "mean": stats["mean"],
                    "std": stats["std"] if not math.isnan(stats["std"]) else None,
                }

        # Load existing file if present and merge
        if os.path.isfile(out_path):
            with open(out_path, "r", encoding="utf-8") as f:
                out = json.load(f)
            if "reward_model_params" not in out:
                out["reward_model_params"] = {}
        else:
            out = {
                "preference_dataset_hub_id": pref_id,
                "reward_model_params": {},
            }
        out["reward_model_params"].update(new_params)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {out_path} (preference_dataset_hub_id={pref_id})")
        print("  Look up params by reward model hub_id: calibration[reward_model_params][<your_rm_hub_id>]")


if __name__ == "__main__":
    main()
