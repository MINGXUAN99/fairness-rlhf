"""Export diagnostic JSONL to CSV and draw per-run mechanism figures.

python -m src.imdb.diagnostics.analyze_attribute_gradients RUN_DIR [--csv-only]
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


TABLES = ("rollout_summary", "gradient_summary", "update_summary", "counterfactual_alpha", "training_controls", "evaluation")


def load_rows(path):
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def export_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def analyze(directory, csv_only=False):
    directory = Path(directory)
    if not (directory/"manifest.json").is_file():
        raise ValueError("Expected a diagnostic run directory with manifest.json")
    tables = {name: load_rows(directory/f"{name}.jsonl") for name in TABLES}
    if not tables["rollout_summary"]:
        raise ValueError("No measurements found")
    for name, rows in tables.items():
        export_csv(directory/f"{name}.csv", rows)
    if csv_only:
        return tables
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    output = directory/"figures"
    output.mkdir(exist_ok=True)
    for stream in sorted({r["stream"] for r in tables["rollout_summary"]}):
        stats = sorted([r for r in tables["rollout_summary"] if r["stream"] == stream], key=lambda r: r["ppo_update"])
        gradients = sorted([r for r in tables["gradient_summary"] if r["stream"] == stream and r.get("block") == "all"], key=lambda r: r["ppo_update"])
        updates = sorted([r for r in tables["update_summary"] if r["stream"] == stream], key=lambda r: r["ppo_update"])
        def column(rows, key):
            return np.array([np.nan if r.get(key) is None else r[key] for r in rows], dtype=float)
        def line(ax, rows, key, label, **kwargs):
            if rows and any(r.get(key) is not None for r in rows):
                ax.plot(column(rows, "ppo_update"), column(rows, key), label=label, **kwargs)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
        line(axes[0,0], stats, "policy_length_mean", "Mean length")
        line(axes[0,0], stats, "policy_length_std", "Length standard deviation", linestyle="--")
        axes[0,0].set(title="Rollout length", ylabel="Policy tokens (excluding EOS)")
        line(axes[0,1], gradients, "weighted_norm_conciseness", "Weighted conciseness")
        line(axes[0,1], gradients, "weighted_norm_sentiment", "Weighted sentiment")
        axes[0,1].set(title="Attribute gradient magnitude", ylabel="Ascent-gradient norm")
        if updates:
            for key, label in [("conciseness", "Conciseness"), ("sentiment", "Sentiment"),
                               ("minority_oracle", "Minority oracle"), ("minority_rm", "Minority RM")]:
                line(axes[1,0], updates, f"cos_update__{key}", label)
            axes[1,0].set_title("Alignment with actual optimizer displacement")
        else:
            for key, label in [("conciseness__sentiment", "Conciseness vs sentiment"),
                               ("sentiment__minority_oracle", "Sentiment vs minority oracle"),
                               ("minority_rm__minority_oracle", "Minority RM vs oracle")]:
                line(axes[1,0], gradients, f"cos_{key}", label)
            axes[1,0].set_title("Gradient alignment (no actual-update measurements)")
        axes[1,0].axhline(0, color="grey", linewidth=.7)
        axes[1,0].set(ylabel="Cosine", ylim=(-1.05, 1.05))
        for key, label in [("conciseness_mean", "Conciseness"), ("sentiment_mean", "Sentiment"),
                           ("minority_oracle_mean", "Minority oracle")]:
            line(axes[1,1], stats, key, label)
        axes[1,1].set(title="Attribute and group outcomes", ylabel="Reward")
        for ax in axes.flat:
            ax.set_xlabel("PPO rollout update")
            ax.grid(alpha=.2)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=8)
        manifest = json.loads((directory/"manifest.json").read_text())
        cfg = manifest["config"]
        fig.suptitle(f"{cfg['run_id']} | alpha={cfg['alpha']} | seed={cfg['seed']} | {stream}")
        for suffix in ("png", "pdf"):
            fig.savefig(output/f"mechanism_{stream}.{suffix}", dpi=160)
        plt.close(fig)
        if updates:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            line(ax, updates, "contribution_conciseness", "alpha × g_conciseness · update")
            line(ax, updates, "contribution_sentiment", "(1-alpha) × g_sentiment · update")
            ax.axhline(0, color="grey", linewidth=.7)
            ax.set(xlabel="PPO rollout update", ylabel="Signed first-order contribution",
                   title="Which attribute contributes to minority improvement?")
            ax.legend()
            fig.savefig(output/f"update_contributions_{stream}.png", dpi=160)
            plt.close(fig)
        counterfactual = [r for r in tables["counterfactual_alpha"] if r["stream"] == stream]
        if counterfactual:
            fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
            endpoints = sorted({min(r["ppo_update"] for r in counterfactual), max(r["ppo_update"] for r in counterfactual)})
            key = "cos_update__minority_oracle" if updates else "cos_sentiment__minority_oracle"
            for update in endpoints:
                rows = sorted([r for r in counterfactual if r["ppo_update"] == update], key=lambda r: r["evaluation_alpha"])
                ax.plot(column(rows, "evaluation_alpha"), column(rows, key), marker="o", label=f"Update {update}")
            ax.set(xlabel="Counterfactual minority alpha (same frozen policy)", ylabel="Cosine",
                   ylim=(-1.05, 1.05), title="Mixture alignment with " + ("actual update" if updates else "sentiment"))
            ax.legend()
            fig.savefig(output/f"counterfactual_alpha_{stream}.png", dpi=160)
            plt.close(fig)
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir")
    parser.add_argument("--csv-only", action="store_true")
    args = parser.parse_args()
    analyze(args.run_dir, args.csv_only)
    print(f"Exported analysis under {Path(args.run_dir).resolve()}")


if __name__ == "__main__":
    main()
