"""
Fit per-group calibration for MaxMin reward model outputs to match oracle ("gold") scores in [0, 1].

Why:
- MaxMin group selection compares mean rewards across group reward models.
- Separately trained reward models can have different output scales/offsets.
- We calibrate each group RM output r(x) to a comparable score \u0127(x) \u2248 gold(x) \u2208 [0,1].

Method:
- Collect a calibration set from the preference dataset:
    {user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}
  using both `chosen` and `rejected` texts for each group.
- For each group g in {minority, majority}, compute:
    - RM logit r_g(x) (scalar)
    - gold score gold_g(x) in [0,1] using distilbert-imdb positiveness + length/100
- Fit parameters (a_g, b_g) to minimize MSE:
    sigmoid(a_g * r_g(x) + b_g) \u2248 gold_g(x)

Outputs a JSON file usable by PPO training to calibrate group means for selection (and optionally rewards).
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fit MaxMin RM calibration to gold scores.")
    p.add_argument("--user_id", type=str, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--proportion", type=float, required=True)
    p.add_argument("--alpha", type=float, required=True)
    p.add_argument(
        "--preference_dataset_name",
        type=str,
        default=None,
        help="Override preference dataset hub id. Default: {user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}",
    )
    p.add_argument("--max_samples", type=int, default=20000, help="Max preference rows to use (after shuffle)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--gold_model_name", type=str, default="lvwerra/distilbert-imdb")
    p.add_argument("--gold_max_length", type=int, default=512)
    p.add_argument("--rm_max_length", type=int, default=256)
    p.add_argument(
        "--output_json",
        type=str,
        required=True,
        help="Where to write calibration JSON (e.g., calibration_seed43_alpha1.json).",
    )
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def _pref_dataset_id(args: argparse.Namespace) -> str:
    if args.preference_dataset_name:
        return args.preference_dataset_name
    return f"{args.user_id}/gpt2-imdb-seed{args.seed}-prop{args.proportion}-alpha{args.alpha}"


def _rm_ids(args: argparse.Namespace) -> tuple[str, str]:
    maj = f"{args.user_id}/maxmin-imdb-reward-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-majority"
    min_ = f"{args.user_id}/maxmin-imdb-reward-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}-minority"
    return min_, maj


@dataclass
class RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0
    vmin: float = float("inf")
    vmax: float = float("-inf")

    def update(self, x: float) -> None:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        self.vmin = min(self.vmin, x)
        self.vmax = max(self.vmax, x)

    def update_many(self, xs) -> None:
        for x in xs:
            self.update(float(x))

    @property
    def std(self) -> float:
        if self.n < 2:
            return float("nan")
        return math.sqrt(self.m2 / (self.n - 1))


def _gold_scores(
    texts: list[str],
    gold_model,
    gold_tok,
    pos_id: int,
    alpha: float,
    group_id: int,
    max_length: int,
    device: str,
) -> list[float]:
    enc = gold_tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = gold_model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
    probs = torch.softmax(out.logits, dim=-1)
    pos = probs[:, pos_id]
    lengths = enc["attention_mask"].sum(dim=-1).float()
    norm_len = torch.clamp(lengths / 250.0, min=0.0, max=1.0)

    a = float(alpha)
    if group_id == 0:
        score = (1 - norm_len) * a + (1 - a) * pos
    else:
        score = (1 - norm_len) * (1 - a) + a * pos
    return score.detach().float().cpu().tolist()


def _rm_logits(texts: list[str], rm_model, rm_tok, max_length: int, device: str) -> list[float]:
    enc = rm_tok(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    ).to(device)
    out = rm_model(**enc).logits.squeeze(-1)
    return out.detach().float().cpu().tolist()


def _fit_sigmoid_affine(r: torch.Tensor, g: torch.Tensor) -> tuple[float, float, float]:
    """
    Fit a,b to minimize MSE(sigmoid(a*r + b), g).
    Returns (a, b, final_loss).
    """
    r = r.detach()
    g = g.detach()

    a = torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=r.device))
    b = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float32, device=r.device))

    opt = torch.optim.LBFGS([a, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")
    mse = torch.nn.MSELoss()

    def closure():
        opt.zero_grad()
        pred = torch.sigmoid(a * r + b)
        loss = mse(pred, g)
        loss.backward()
        return loss

    loss = opt.step(closure)
    return float(a.detach().cpu().item()), float(b.detach().cpu().item()), float(loss.detach().cpu().item())


def main() -> None:
    args = parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    pref_id = _pref_dataset_id(args)
    rm_min_id, rm_maj_id = _rm_ids(args)
    out_path = Path(args.output_json)

    print("Preference dataset:", pref_id)
    print("RM minority:", rm_min_id)
    print("RM majority:", rm_maj_id)
    print("Device:", device)

    ds = load_dataset(pref_id, split="train")
    if args.max_samples and args.max_samples > 0:
        ds = ds.shuffle(seed=args.seed).select(range(min(args.max_samples, len(ds))))
    for col in ("chosen", "rejected", "group_id"):
        if col not in ds.column_names:
            raise ValueError(f"Preference dataset missing '{col}'. Have: {ds.column_names}")

    # Load models/tokenizers
    rm_min = AutoModelForSequenceClassification.from_pretrained(rm_min_id).to(device)
    rm_maj = AutoModelForSequenceClassification.from_pretrained(rm_maj_id).to(device)
    rm_min.eval()
    rm_maj.eval()
    rm_tok = AutoTokenizer.from_pretrained(rm_maj_id)
    if rm_tok.pad_token is None:
        rm_tok.pad_token = rm_tok.eos_token

    gold_model = AutoModelForSequenceClassification.from_pretrained(args.gold_model_name).to(device)
    gold_model.eval()
    gold_tok = AutoTokenizer.from_pretrained(args.gold_model_name)
    if gold_tok.pad_token is None:
        gold_tok.pad_token = gold_tok.eos_token
    pos_id = gold_model.config.label2id.get("POSITIVE", 1)

    # Collect calibration data per group from both chosen and rejected texts
    group_data = {0: {"texts": []}, 1: {"texts": []}}
    for ex in ds:
        gid = int(ex["group_id"])
        if gid not in (0, 1):
            continue
        group_data[gid]["texts"].append(ex["chosen"])
        group_data[gid]["texts"].append(ex["rejected"])

    results = {}
    for gid, name, rm in [(0, "minority", rm_min), (1, "majority", rm_maj)]:
        texts = group_data[gid]["texts"]
        if not texts:
            raise RuntimeError(f"No texts collected for group_id={gid}")

        # Batch scoring
        r_all: list[float] = []
        g_all: list[float] = []
        r_stats = RunningStats()
        g_stats = RunningStats()

        with torch.no_grad():
            for i in range(0, len(texts), args.batch_size):
                batch = texts[i : i + args.batch_size]
                r = _rm_logits(batch, rm, rm_tok, args.rm_max_length, device)
                g = _gold_scores(batch, gold_model, gold_tok, pos_id, args.alpha, gid, args.gold_max_length, device)
                r_all.extend(r)
                g_all.extend(g)
                r_stats.update_many(r)
                g_stats.update_many(g)

        r_t = torch.tensor(r_all, dtype=torch.float32, device=device)
        g_t = torch.tensor(g_all, dtype=torch.float32, device=device)

        a, b, loss = _fit_sigmoid_affine(r_t, g_t)
        print(f"\nGroup {name} (gid={gid})")
        print(f"  raw RM logits: mean={r_stats.mean:.6f} std={r_stats.std:.6f} min={r_stats.vmin:.6f} max={r_stats.vmax:.6f}")
        print(f"  gold score:    mean={g_stats.mean:.6f} std={g_stats.std:.6f} min={g_stats.vmin:.6f} max={g_stats.vmax:.6f}")
        print(f"  fit: a={a:.6f} b={b:.6f}  mse={loss:.8f}")

        results[name] = {
            "group_id": gid,
            "a": a,
            "b": b,
            "mse": loss,
            "raw_logit_stats": {"n": int(r_stats.n), "mean": float(r_stats.mean), "std": float(r_stats.std)},
            "gold_stats": {"n": int(g_stats.n), "mean": float(g_stats.mean), "std": float(g_stats.std)},
        }

    payload = {
        "schema_version": 1,
        "type": "sigmoid_affine_to_gold",
        "user_id": args.user_id,
        "seed": int(args.seed),
        "proportion": float(args.proportion),
        "alpha": float(args.alpha),
        "preference_dataset": pref_id,
        "gold_model_name": args.gold_model_name,
        "gold_max_length": int(args.gold_max_length),
        "rm_max_length": int(args.rm_max_length),
        "groups": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print("\nWrote calibration JSON:", str(out_path))


if __name__ == "__main__":
    main()

