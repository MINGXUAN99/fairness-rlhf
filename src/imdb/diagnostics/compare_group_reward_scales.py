"""
Compare output scales of MaxMin group reward models on the same texts.

This script is meant to diagnose whether the two separately trained MaxMin
reward models (minority vs majority) are on comparable scales. MaxMin group
selection in PPO is sensitive to even small systematic shifts in reward scale.

Default input dataset: the maxmin PPO evaluation generations on the Hub:
  {user_id}/maxmin-imdb-ppo-eval-prop{proportion}-alpha{alpha}-seed{seed}

Expected columns (at least one of these):
  - prompt + completion   (preferred; matches PPO-style "query+response")
  - completion            (fallback)
  - query / response      (fallback)
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Iterable, Optional

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare MaxMin group RM output scales.")
    p.add_argument("--user_id", type=str, required=True, help="Hugging Face user/org id")
    p.add_argument("--seed", type=int, default=42, help="Seed (used for default dataset + shuffling)")
    p.add_argument("--proportion", type=float, default=0.2, help="Minority proportion (used for default dataset/model ids)")
    p.add_argument("--alpha", type=float, default=1.0, help="Alpha (used for default dataset/model ids)")

    p.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "Dataset to score. If omitted, uses "
            "{user_id}/maxmin-imdb-ppo-eval-prop{proportion}-alpha{alpha}-seed{seed}."
        ),
    )
    p.add_argument("--split", type=str, default=None, help="Dataset split (default: first available)")
    p.add_argument("--max_samples", type=int, default=5000, help="Max samples to score (after shuffling)")
    p.add_argument("--batch_size", type=int, default=32, help="Batch size for scoring")
    p.add_argument("--max_length", type=int, default=256, help="Tokenizer max_length (truncates longer texts)")
    p.add_argument(
        "--text_mode",
        type=str,
        default="prompt_completion",
        choices=["prompt_completion", "completion_only", "auto"],
        help=(
            "Which text to score. 'prompt_completion' scores prompt+completion when available. "
            "'completion_only' uses completion/response only. 'auto' tries several fallbacks."
        ),
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override: 'cpu', 'cuda', 'cuda:0', etc. Default: cuda if available else cpu.",
    )
    p.add_argument(
        "--save_calibration_json",
        type=str,
        default=None,
        help=(
            "Optional path to write a JSON file containing mean/std for each group RM. "
            "This can be used to z-score normalize group rewards before MaxMin selection in PPO."
        ),
    )
    p.add_argument(
        "--include_gold",
        action="store_true",
        help=(
            "Also compute oracle ('gold') group scores from positiveness and normalized_length "
            "and compare learned logits vs gold."
        ),
    )
    p.add_argument(
        "--gold_model_name",
        type=str,
        default="lvwerra/distilbert-imdb",
        help="Sentiment model for gold positiveness if dataset lacks 'positiveness' column.",
    )
    p.add_argument(
        "--gold_max_length",
        type=int,
        default=100,
        help="Tokenization max_length for gold scoring (matches dataset pipeline default 100).",
    )
    p.add_argument(
        "--gold_use_dataset_columns",
        action="store_true",
        help=(
            "If set, use dataset columns ('positiveness' and 'normalized_length') when available; "
            "otherwise compute them with the gold model/tokenizer."
        ),
    )
    p.add_argument(
        "--eval_preference_dataset",
        action="store_true",
        help=(
            "Also evaluate each group RM on the underlying preference dataset "
            "({user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}) by checking "
            "whether RM(chosen) > RM(rejected). This helps distinguish scale issues from "
            "a genuinely mislearned minority RM."
        ),
    )
    p.add_argument(
        "--preference_dataset_name",
        type=str,
        default=None,
        help=(
            "Preference dataset name. If omitted, uses "
            "{user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}."
        ),
    )
    p.add_argument(
        "--pref_max_samples",
        type=int,
        default=5000,
        help="Max samples to evaluate from the preference dataset (after shuffling).",
    )
    p.add_argument(
        "--pref_batch_size",
        type=int,
        default=32,
        help="Batch size for preference-dataset evaluation.",
    )
    return p.parse_args()


def _resolve_dataset_name(args: argparse.Namespace) -> str:
    if args.dataset_name:
        return args.dataset_name
    return f"{args.user_id}/maxmin-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"

def _resolve_preference_dataset_name(args: argparse.Namespace) -> str:
    if args.preference_dataset_name:
        return args.preference_dataset_name
    return f"{args.user_id}/gpt2-imdb-seed{args.seed}-prop{args.proportion}-alpha{args.alpha}"


def _pick_split(loaded, requested: Optional[str]) -> str:
    if requested:
        return requested
    if hasattr(loaded, "keys"):
        keys = list(loaded.keys())
        if not keys:
            raise ValueError("Loaded DatasetDict has no splits.")
        return keys[0]
    # plain Dataset
    return ""


def _get_text(example: dict, mode: str) -> str:
    # Common columns in this repo:
    # - evaluation datasets: prompt, completion
    # - training/preproc: query, response or query, completion
    prompt = example.get("prompt")
    completion = example.get("completion")
    query = example.get("query")
    response = example.get("response")

    if mode == "completion_only":
        return completion or response or ""

    if mode == "prompt_completion":
        if prompt is not None and completion is not None:
            # Many generation pipelines decode the *full* generated sequence, which can
            # already include the prompt prefix. Avoid accidentally duplicating the prompt.
            if isinstance(prompt, str) and isinstance(completion, str) and completion.startswith(prompt):
                return completion
            return f"{prompt}{completion}"
        if query is not None and completion is not None:
            if isinstance(query, str) and isinstance(completion, str) and completion.startswith(query):
                return completion
            return f"{query}{completion}"
        if query is not None and response is not None:
            if isinstance(query, str) and isinstance(response, str) and response.startswith(query):
                return response
            return f"{query}{response}"
        # Fallback to completion
        return completion or response or ""

    # auto
    if prompt is not None and completion is not None:
        return f"{prompt}{completion}"
    if query is not None and completion is not None:
        return f"{query}{completion}"
    if query is not None and response is not None:
        return f"{query}{response}"
    if completion is not None:
        return completion
    if response is not None:
        return response
    if query is not None:
        return query
    return ""


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

def _pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(ys) != len(xs):
        return float("nan")
    xm = sum(xs) / len(xs)
    ym = sum(ys) / len(ys)
    cov = sum((x - xm) * (y - ym) for x, y in zip(xs, ys))
    v1 = sum((x - xm) ** 2 for x in xs)
    v2 = sum((y - ym) ** 2 for y in ys)
    return cov / math.sqrt(v1 * v2) if v1 > 0 and v2 > 0 else float("nan")


def _quantiles(xs: list[float], ps=(0.0, 0.01, 0.05, 0.5, 0.95, 0.99, 1.0)) -> dict[float, float]:
    if not xs:
        return {p: float("nan") for p in ps}
    ys = sorted(xs)
    n = len(ys)
    out = {}
    for p in ps:
        if n == 1:
            out[p] = ys[0]
            continue
        # linear interpolation between closest ranks
        idx = p * (n - 1)
        lo = int(math.floor(idx))
        hi = int(math.ceil(idx))
        if lo == hi:
            out[p] = ys[lo]
        else:
            w = idx - lo
            out[p] = (1 - w) * ys[lo] + w * ys[hi]
    return out


def main() -> None:
    args = parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset_name = _resolve_dataset_name(args)
    print(f"Dataset: {dataset_name}")

    loaded = load_dataset(dataset_name)
    split = _pick_split(loaded, args.split)
    ds = loaded if split == "" else loaded[split]
    print(f"Split:   {split or '(Dataset)'}  rows={len(ds)}")

    # Shuffle then take max_samples (deterministic)
    if args.max_samples is not None and args.max_samples > 0:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))
    print(f"Using:   rows={len(ds)}  batch_size={args.batch_size}  max_length={args.max_length}")

    # Resolve model ids
    rm_majority_id = f"{args.user_id}/maxmin-imdb-reward-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-majority"
    rm_minority_id = f"{args.user_id}/maxmin-imdb-reward-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-minority"
    print(f"RM maj:  {rm_majority_id}")
    print(f"RM min:  {rm_minority_id}")

    # Tokenizer: both models should share tokenizer; use majority by convention
    tok = AutoTokenizer.from_pretrained(rm_majority_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    rm_maj = AutoModelForSequenceClassification.from_pretrained(rm_majority_id).to(device)
    rm_min = AutoModelForSequenceClassification.from_pretrained(rm_minority_id).to(device)
    rm_maj.eval()
    rm_min.eval()

    stats_maj = RunningStats()
    stats_min = RunningStats()
    # Keep a sample list for quantiles/correlation (bounded by max_samples anyway)
    logits_maj: list[float] = []
    logits_min: list[float] = []

    # Optional oracle ("gold") group scores, computed on the same texts.
    gold_min_scores: list[float] = []
    gold_maj_scores: list[float] = []
    stats_gold_min = RunningStats()
    stats_gold_maj = RunningStats()

    gold_tok = None
    gold_model = None
    pos_id = None

    # Batch-wise "worst group" counts under batch mean rule
    worst_by_batch = {"minority": 0, "majority": 0, "tie": 0}
    # Per-sample min counts (not your Algorithm 1, but still informative)
    worst_by_sample = {"minority": 0, "majority": 0, "tie": 0}

    worst_by_batch_gold = {"minority": 0, "majority": 0, "tie": 0}

    def batches(n: int, bs: int) -> Iterable[range]:
        for i in range(0, n, bs):
            yield range(i, min(i + bs, n))

    prompt_dup_avoided = 0
    prompt_dup_checked = 0

    with torch.no_grad():
        for idxs in batches(len(ds), args.batch_size):
            exs = [ds[int(i)] for i in idxs]
            texts = [_get_text(e, args.text_mode) for e in exs]
            # Drop empty texts (rare, but can happen if dataset schema differs)
            keep = [(t, i) for i, t in enumerate(texts) if isinstance(t, str) and t.strip()]
            if not keep:
                continue
            texts_kept = [t for t, _ in keep]

            # Roughly estimate whether completions already contain prompts (informational only)
            # Only counts when both fields exist.
            for _, orig_i in keep:
                ex = exs[orig_i]
                p = ex.get("prompt")
                c = ex.get("completion")
                if isinstance(p, str) and isinstance(c, str):
                    prompt_dup_checked += 1
                    if c.startswith(p):
                        prompt_dup_avoided += 1

            enc = tok(
                texts_kept,
                padding=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
            ).to(device)

            out_maj = rm_maj(**enc).logits.squeeze(-1).detach().float().cpu().tolist()
            out_min = rm_min(**enc).logits.squeeze(-1).detach().float().cpu().tolist()

            stats_maj.update_many(out_maj)
            stats_min.update_many(out_min)
            logits_maj.extend(out_maj)
            logits_min.extend(out_min)

            # Batch-mean worst group (your PPO group selection criterion)
            m_maj = float(sum(out_maj) / len(out_maj))
            m_min = float(sum(out_min) / len(out_min))
            if abs(m_maj - m_min) < 1e-12:
                worst_by_batch["tie"] += 1
            elif m_maj < m_min:
                worst_by_batch["majority"] += 1
            else:
                worst_by_batch["minority"] += 1

            # Per-sample worst group
            for a, b in zip(out_min, out_maj):
                if abs(a - b) < 1e-12:
                    worst_by_sample["tie"] += 1
                elif a < b:
                    worst_by_sample["minority"] += 1
                else:
                    worst_by_sample["majority"] += 1

            # --- Optional oracle ("gold") scores ---
            if args.include_gold:
                # Try to use dataset columns first (if requested and present).
                # Note: columns may be missing if you did not run score_responses in evaluation mode.
                use_cols = args.gold_use_dataset_columns and ("positiveness" in ds.column_names) and ("normalized_length" in ds.column_names)

                if use_cols:
                    # Map back to original examples indices within this batch
                    # keep = [(text, original_idx_in_texts)]
                    pos_vals = []
                    norm_vals = []
                    for _, orig_i in keep:
                        ex = exs[orig_i]
                        pos_vals.append(float(ex["positiveness"]))
                        norm_vals.append(float(ex["normalized_length"]))
                    positiveness = pos_vals
                    normalized_length = norm_vals
                else:
                    if gold_model is None:
                        gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_model_name).to(device)
                        gold_model.eval()
                        gold_tok = AutoTokenizer.from_pretrained(args.gold_model_name)
                        # Some tokenizers don't set pad_token by default.
                        if gold_tok.pad_token is None:
                            gold_tok.pad_token = gold_tok.eos_token
                        pos_id = gold_model.config.label2id.get("POSITIVE", 1)

                    gold_enc = gold_tok(
                        texts_kept,
                        padding=True,
                        truncation=True,
                        max_length=args.gold_max_length,
                        return_tensors="pt",
                    ).to(device)

                    gold_out = gold_model(
                        input_ids=gold_enc["input_ids"],
                        attention_mask=gold_enc["attention_mask"],
                    )
                    probs = torch.softmax(gold_out.logits, dim=-1)
                    pos_tensor = probs[:, pos_id].detach().float().cpu()
                    positiveness = pos_tensor.tolist()

                    # normalized_length mirrors score_responses_unified: attention_mask.sum()/100
                    lengths = gold_enc["attention_mask"].sum(dim=-1).detach().float().cpu()
                    normalized_length = (lengths / 100.0).tolist()

                # Compute group-specific oracle scores using the same formulas as create_chosen_rejected.py
                a = float(args.alpha)
                gmin = [float((1 - nl) * a + (1 - a) * p) for p, nl in zip(positiveness, normalized_length)]
                gmaj = [float((1 - nl) * (1 - a) + a * p) for p, nl in zip(positiveness, normalized_length)]

                gold_min_scores.extend(gmin)
                gold_maj_scores.extend(gmaj)
                stats_gold_min.update_many(gmin)
                stats_gold_maj.update_many(gmaj)

                # Batch-mean oracle worst group
                bm_min = sum(gmin) / len(gmin)
                bm_maj = sum(gmaj) / len(gmaj)
                if abs(bm_min - bm_maj) < 1e-12:
                    worst_by_batch_gold["tie"] += 1
                elif bm_min < bm_maj:
                    worst_by_batch_gold["minority"] += 1
                else:
                    worst_by_batch_gold["majority"] += 1

    n = min(stats_maj.n, stats_min.n)
    print("\n=== Raw RM output scale summary (logits) ===")
    print(f"Scored samples: {n}")
    print(f"Majority RM: mean={stats_maj.mean:.6f} std={stats_maj.std:.6f} min={stats_maj.vmin:.6f} max={stats_maj.vmax:.6f}")
    print(f"Minority RM: mean={stats_min.mean:.6f} std={stats_min.std:.6f} min={stats_min.vmin:.6f} max={stats_min.vmax:.6f}")

    qmaj = _quantiles(logits_maj)
    qmin = _quantiles(logits_min)
    print("\nQuantiles (p -> value):")
    print("  Majority:", {p: round(v, 6) for p, v in qmaj.items()})
    print("  Minority:", {p: round(v, 6) for p, v in qmin.items()})

    # Pearson correlation between the two logits on same texts
    corr = _pearson_corr(logits_maj, logits_min)
    if not math.isnan(corr):
        print(f"\nCorr(majority_logit, minority_logit) = {corr:.6f}")

    nb = sum(worst_by_batch.values())
    ns = sum(worst_by_sample.values())
    print("\n=== Worst-group frequencies ===")
    print("Batch-mean rule (matches your PPO group selection):")
    for k, v in worst_by_batch.items():
        pct = 100.0 * v / nb if nb else 0.0
        print(f"  {k:8s}: {v:6d} ({pct:6.2f}%)")
    print("Per-sample min (not Algorithm 1, just diagnostic):")
    for k, v in worst_by_sample.items():
        pct = 100.0 * v / ns if ns else 0.0
        print(f"  {k:8s}: {v:6d} ({pct:6.2f}%)")

    if prompt_dup_checked:
        print(
            f"\nPrompt prefix check (completion startswith prompt): "
            f"{prompt_dup_avoided}/{prompt_dup_checked} ({100.0*prompt_dup_avoided/prompt_dup_checked:.2f}%)"
        )

    if args.include_gold:
        ng = min(stats_gold_min.n, stats_gold_maj.n)
        print("\n=== Oracle ('gold') group score summary ===")
        print(f"Scored samples: {ng}")
        print(
            f"Gold minority: mean={stats_gold_min.mean:.6f} std={stats_gold_min.std:.6f} "
            f"min={stats_gold_min.vmin:.6f} max={stats_gold_min.vmax:.6f}"
        )
        print(
            f"Gold majority: mean={stats_gold_maj.mean:.6f} std={stats_gold_maj.std:.6f} "
            f"min={stats_gold_maj.vmin:.6f} max={stats_gold_maj.vmax:.6f}"
        )
        qgmin = _quantiles(gold_min_scores)
        qgmaj = _quantiles(gold_maj_scores)
        print("\nGold quantiles (p -> value):")
        print("  Gold minority:", {p: round(v, 6) for p, v in qgmin.items()})
        print("  Gold majority:", {p: round(v, 6) for p, v in qgmaj.items()})

        c_maj = _pearson_corr(logits_maj, gold_maj_scores)
        c_min = _pearson_corr(logits_min, gold_min_scores)
        print("\nCorrelations vs gold:")
        print(f"  Corr(majority_logit, gold_majority) = {c_maj:.6f}")
        print(f"  Corr(minority_logit, gold_minority) = {c_min:.6f}")

        # Compare which group should be worst under oracle vs learned selection
        nbg = sum(worst_by_batch_gold.values())
        print("\nOracle batch-mean worst group (same selection rule, but using gold scores):")
        for k, v in worst_by_batch_gold.items():
            pct = 100.0 * v / nbg if nbg else 0.0
            print(f"  {k:8s}: {v:6d} ({pct:6.2f}%)")

    if args.save_calibration_json:
        payload = {
            "schema_version": 1,
            "dataset": {
                "name": dataset_name,
                "split": split or None,
                "max_samples": int(args.max_samples) if args.max_samples is not None else None,
                "text_mode": args.text_mode,
                "max_length": int(args.max_length),
            },
            "models": {
                "majority": rm_majority_id,
                "minority": rm_minority_id,
            },
            "stats": {
                "majority": {"n": int(stats_maj.n), "mean": float(stats_maj.mean), "std": float(stats_maj.std)},
                "minority": {"n": int(stats_min.n), "mean": float(stats_min.mean), "std": float(stats_min.std)},
            },
        }
        if args.include_gold:
            payload["gold"] = {
                "alpha": float(args.alpha),
                "model_name": args.gold_model_name,
                "max_length": int(args.gold_max_length),
                "used_dataset_columns": bool(args.gold_use_dataset_columns and ("positiveness" in ds.column_names) and ("normalized_length" in ds.column_names)),
                "stats": {
                    "majority": {"n": int(stats_gold_maj.n), "mean": float(stats_gold_maj.mean), "std": float(stats_gold_maj.std)},
                    "minority": {"n": int(stats_gold_min.n), "mean": float(stats_gold_min.mean), "std": float(stats_gold_min.std)},
                },
            }
        with open(args.save_calibration_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
        print(f"\nWrote calibration JSON to: {args.save_calibration_json}")

    # ------------------------------------------------------------------
    # Preference dataset evaluation (RM(chosen) > RM(rejected))
    # ------------------------------------------------------------------
    if args.eval_preference_dataset:
        pref_name = _resolve_preference_dataset_name(args)
        print("\n" + "=" * 72)
        print("Preference dataset RM sanity check")
        print("=" * 72)
        print(f"Preference dataset: {pref_name}")

        pref_loaded = load_dataset(pref_name, split="train")
        if args.pref_max_samples is not None and args.pref_max_samples > 0:
            pref_loaded = pref_loaded.shuffle(seed=args.seed).select(
                range(min(args.pref_max_samples, len(pref_loaded)))
            )
        print(f"Using: rows={len(pref_loaded)}  batch_size={args.pref_batch_size}")

        # Required columns
        for col in ("chosen", "rejected", "group_id"):
            if col not in pref_loaded.column_names:
                raise ValueError(
                    f"Preference dataset missing column '{col}'. Have: {pref_loaded.column_names}"
                )

        def _eval_rm_on_group(rm, group_id: int, group_name: str) -> None:
            ds_g = pref_loaded.filter(lambda ex: int(ex["group_id"]) == int(group_id))
            if len(ds_g) == 0:
                print(f"Group {group_name}: 0 rows (skipping)")
                return
            # Keep size reasonable
            if args.pref_max_samples is not None and args.pref_max_samples > 0:
                ds_g = ds_g.select(range(min(len(ds_g), args.pref_max_samples)))

            correct = 0
            total = 0
            ties = 0
            rm_stats_ch = RunningStats()
            rm_stats_rj = RunningStats()

            # Optional: compute oracle consistency for this group
            gold_correct = 0
            gold_total = 0
            gold_ties = 0
            gold_trunc_both = 0
            gold_trunc_any = 0
            gold_stats_ch = RunningStats()
            gold_stats_rj = RunningStats()
            gold_margin_stats = RunningStats()

            # If include_gold, prepare gold model/tokenizer if needed
            local_gold_model = gold_model
            local_gold_tok = gold_tok
            local_pos_id = pos_id

            def _gold_scores(texts: list[str]) -> tuple[list[float], list[float]]:
                # Returns (gold_minority_scores, gold_majority_scores) for each text
                nonlocal local_gold_model, local_gold_tok, local_pos_id
                if local_gold_model is None:
                    local_gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_model_name).to(device)
                    local_gold_model.eval()
                    local_gold_tok = AutoTokenizer.from_pretrained(args.gold_model_name)
                    if local_gold_tok.pad_token is None:
                        local_gold_tok.pad_token = local_gold_tok.eos_token
                    local_pos_id = local_gold_model.config.label2id.get("POSITIVE", 1)

                enc2 = local_gold_tok(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=args.gold_max_length,
                    return_tensors="pt",
                ).to(device)
                out2 = local_gold_model(
                    input_ids=enc2["input_ids"],
                    attention_mask=enc2["attention_mask"],
                )
                probs2 = torch.softmax(out2.logits, dim=-1)
                pos = probs2[:, local_pos_id].detach().float().cpu().tolist()
                lengths = enc2["attention_mask"].sum(dim=-1).detach().float().cpu().tolist()
                norm_len = [l / 100.0 for l in lengths]
                a = float(args.alpha)
                gmin = [(1 - nl) * a + (1 - a) * p for p, nl in zip(pos, norm_len)]
                gmaj = [(1 - nl) * (1 - a) + a * p for p, nl in zip(pos, norm_len)]
                return gmin, gmaj

            for idxs2 in batches(len(ds_g), args.pref_batch_size):
                batch = [ds_g[int(i)] for i in idxs2]
                chosen_texts = [b["chosen"] for b in batch]
                rejected_texts = [b["rejected"] for b in batch]

                enc_ch = tok(
                    chosen_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                enc_rj = tok(
                    rejected_texts,
                    padding=True,
                    truncation=True,
                    max_length=args.max_length,
                    return_tensors="pt",
                ).to(device)
                s_ch = rm(**enc_ch).logits.squeeze(-1).detach().float().cpu().tolist()
                s_rj = rm(**enc_rj).logits.squeeze(-1).detach().float().cpu().tolist()

                rm_stats_ch.update_many(s_ch)
                rm_stats_rj.update_many(s_rj)

                for a, b in zip(s_ch, s_rj):
                    total += 1
                    if abs(a - b) < 1e-12:
                        ties += 1
                    if a > b:
                        correct += 1

                if args.include_gold:
                    # Compute gold scores + also track truncation/ties to quantify label noise.
                    gmin_ch, gmaj_ch = _gold_scores(chosen_texts)
                    gmin_rj, gmaj_rj = _gold_scores(rejected_texts)
                    # Determine oracle for this group
                    if group_id == 0:
                        gch = gmin_ch
                        grj = gmin_rj
                    else:
                        gch = gmaj_ch
                        grj = gmaj_rj
                    gold_stats_ch.update_many(gch)
                    gold_stats_rj.update_many(grj)
                    gold_margin_stats.update_many([a - b for a, b in zip(gch, grj)])
                    for a, b in zip(gch, grj):
                        gold_total += 1
                        # create_chosen_rejected.py uses >= when picking chosen
                        if abs(a - b) < 1e-12:
                            gold_ties += 1
                        if a >= b:
                            gold_correct += 1

            acc = 100.0 * correct / total if total else 0.0
            tie_pct = 100.0 * ties / total if total else 0.0
            print(f"\nGroup {group_name} (group_id={group_id})")
            print(f"  RM preference acc: {acc:6.2f}%   ties: {tie_pct:6.2f}%   n={total}")
            print(
                f"  RM(chosen):  mean={rm_stats_ch.mean:.6f} std={rm_stats_ch.std:.6f} "
                f"min={rm_stats_ch.vmin:.6f} max={rm_stats_ch.vmax:.6f}"
            )
            print(
                f"  RM(rejected):mean={rm_stats_rj.mean:.6f} std={rm_stats_rj.std:.6f} "
                f"min={rm_stats_rj.vmin:.6f} max={rm_stats_rj.vmax:.6f}"
            )
            if args.include_gold and gold_total:
                gold_acc = 100.0 * gold_correct / gold_total
                tie_pct = 100.0 * gold_ties / gold_total
                print(f"  Gold consistency (chosen >= rejected): {gold_acc:6.2f}%   ties: {tie_pct:6.2f}%   n={gold_total}")
                print(
                    f"  Gold(chosen):     mean={gold_stats_ch.mean:.6f} std={gold_stats_ch.std:.6f} "
                    f"min={gold_stats_ch.vmin:.6f} max={gold_stats_ch.vmax:.6f}"
                )
                print(
                    f"  Gold(rejected):   mean={gold_stats_rj.mean:.6f} std={gold_stats_rj.std:.6f} "
                    f"min={gold_stats_rj.vmin:.6f} max={gold_stats_rj.vmax:.6f}"
                )
                print(
                    f"  Gold margin(ch-rj): mean={gold_margin_stats.mean:.6f} std={gold_margin_stats.std:.6f} "
                    f"min={gold_margin_stats.vmin:.6f} max={gold_margin_stats.vmax:.6f}"
                )

        # Evaluate majority RM on majority group, minority RM on minority group
        _eval_rm_on_group(rm_min, group_id=0, group_name="minority")
        _eval_rm_on_group(rm_maj, group_id=1, group_name="majority")


if __name__ == "__main__":
    main()

