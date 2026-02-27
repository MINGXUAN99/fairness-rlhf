#!/usr/bin/env python3
"""
Inspect Vanilla RLHF policy responses (conciseness issue).

Problem: Despite ~0 Pearson between positiveness and conciseness in training data,
vanilla RLHF policy learns to generate very concise responses (~15 tokens avg).

Usage:
    python scripts/inspect_vanilla_conciseness.py
    python scripts/inspect_vanilla_conciseness.py --user_id YOUR_ID
"""
import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import argparse
import pandas as pd
from datasets import load_dataset

def load_eval_dataset(dataset_name):
    try:
        loaded = load_dataset(dataset_name)
        if hasattr(loaded, "keys") and len(loaded.keys()) > 0:
            split = list(loaded.keys())[0]
            ds = loaded[split]
        else:
            ds = loaded
        return ds.to_pandas()
    except Exception as e:
        print(f"Failed to load {dataset_name}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--user_id", type=str, default="mxcui")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proportion", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--n_show", type=int, default=20)
    args = parser.parse_args()

    EVAL_DATASET = f"{args.user_id}/vanilla-imdb-ppo-eval-prop{args.proportion}-alpha{args.alpha}-seed{args.seed}"
    print(f"Loading: {EVAL_DATASET}")

    df = load_eval_dataset(EVAL_DATASET)
    if df is None:
        return
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")

    text_col = "completion" if "completion" in df.columns else ("response" if "response" in df.columns else None)
    if text_col is None:
        print("No completion/response column found")
        return

    if "length" in df.columns:
        df["token_length"] = df["length"].astype(int)
    else:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained("gpt2")
            df["token_length"] = df[text_col].apply(lambda x: len(tok.encode(str(x))))
        except Exception:
            df["token_length"] = (df[text_col].str.split().str.len() * 1.3).astype(int)

    print(f"\nToken length stats:\n{df['token_length'].describe()}")
    print(f"\nMedian: {df['token_length'].median():.0f} tokens")

    n = args.n_show
    shortest = df.nsmallest(n, "token_length")
    print("\n" + "=" * 80)
    print(f"SHORTEST {n} RESPONSES")
    print("=" * 80)
    for i, row in shortest.iterrows():
        text = row[text_col]
        n_tok = row["token_length"]
        pos = row.get("positiveness", "N/A")
        prompt = str(row.get("prompt", ""))[:60] + "..." if len(str(row.get("prompt", ""))) > 60 else row.get("prompt", "")
        print(f"\n--- {n_tok} tokens | positiveness={pos} ---")
        print(f"Prompt: {prompt}")
        print(f"Response: {repr(text)}")

    print("\n" + "=" * 80)
    print("LONGEST RESPONSES (for comparison)")
    print("=" * 80)
    longest = df.nlargest(n, "token_length")
    for i, row in longest.iterrows():
        text = row[text_col]
        n_tok = row["token_length"]
        pos = row.get("positiveness", "N/A")
        print(f"\n--- {n_tok} tokens | positiveness={pos} ---")
        print(f"Response: {repr(text)[:200]}{'...' if len(repr(text)) > 200 else ''}")

    if "positiveness" in df.columns:
        short_cutoff = df["token_length"].quantile(0.1)
        short_df = df[df["token_length"] <= short_cutoff]
        rest_df = df[df["token_length"] > short_cutoff]
        print("\n" + "=" * 80)
        print("Positiveness: Short (bottom 10%) vs Rest")
        print(f"Short mean: {short_df['positiveness'].mean():.4f}, Rest mean: {rest_df['positiveness'].mean():.4f}")


if __name__ == "__main__":
    main()
