"""Summarize helpfulness and harmfulness separately, with paired prompt intervals."""
import argparse
import csv
from pathlib import Path

import numpy as np

METRICS = ("helpfulness_reward", "harmfulness_cost")


def estimate(values, samples=2000, seed=42):
    values = np.asarray(values, dtype=float)
    if values.size == 0 or not np.isfinite(values).all():
        raise ValueError("Scores must be nonempty and finite")
    rng = np.random.default_rng(seed)
    # Batch bootstrap draws to avoid allocating samples * all prompts at once.
    means = []
    for start in range(0, samples, 64):
        indices = rng.integers(0, len(values), size=(min(64, samples-start), len(values)))
        means.extend(values[indices].mean(axis=1).tolist())
    lo, hi = np.quantile(means, [0.025, 0.975])
    return {"mean": float(values.mean()), "ci_low": float(lo), "ci_high": float(hi)}


def summarize(records, manifest, reference_run=None, samples=2000, seed=42):
    from src.pku_saferlhf.evaluation_utils import validate_records
    validate_records(records, manifest)
    if samples < 1:
        raise ValueError("Bootstrap samples must be positive")
    by_run = {}
    for row in records:
        by_run.setdefault(row["run_id"], {})[row["prompt_id"]] = row
    if reference_run is not None and reference_run not in by_run:
        raise ValueError("Reference run is absent from the scored data")
    prompt_ids = manifest["prompt_ids"]
    run_rows, delta_rows, method_rows = [], [], []
    for run in manifest["runs"]:
        run_id = run["run_id"]
        row = {"run_id": run_id, "method": run["method"], "training_seed": run["training_seed"], "n_prompts": len(prompt_ids)}
        delta = {"run_id": run_id, "reference_run": reference_run, "n_prompts": len(prompt_ids)}
        for metric in METRICS:
            values = [by_run[run_id][pid][metric] for pid in prompt_ids]
            for key, value in estimate(values, samples, seed).items():
                row[f"{metric}_{key}"] = value
            row[f"{metric}_truncated_count"] = sum(by_run[run_id][pid].get(f"{metric}_truncated", False) for pid in prompt_ids)
            if reference_run is not None:
                differences = [by_run[run_id][pid][metric] - by_run[reference_run][pid][metric] for pid in prompt_ids]
                for key, value in estimate(differences, samples, seed).items():
                    delta[f"{metric}_delta_{key}"] = value
        row["mean_generated_tokens"] = float(np.mean([by_run[run_id][pid]["generated_tokens"] for pid in prompt_ids]))
        run_rows.append(row)
        if reference_run is not None:
            delta_rows.append(delta)
    for method in sorted({row["method"] for row in run_rows}):
        runs = [row for row in run_rows if row["method"] == method]
        seeds = [row["training_seed"] for row in runs]
        if len(set(seeds)) != len(seeds):
            raise ValueError(f"Duplicate training seeds for {method}; do not treat repeat generations as independent training runs")
        row = {"method": method, "n_training_runs": len(runs)}
        for metric in METRICS:
            means = [run[f"{metric}_mean"] for run in runs]
            row[f"{metric}_mean"] = float(np.mean(means))
            row[f"{metric}_std_across_runs"] = float(np.std(means, ddof=1)) if len(means) > 1 else None
        method_rows.append(row)
    return {"per_run": run_rows, "paired_differences": delta_rows, "per_method": method_rows,
            "uncertainty": "Per-run and paired intervals: 95% percentile bootstrap over shared prompts. "
                           "Per-method standard deviation: variability across training runs; absent for a single run.",
            "directions": {"helpfulness_reward": "higher is better", "harmfulness_cost": "lower is better"}}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scored_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--reference_run", default=None, help="run_id of initial policy for paired differences")
    p.add_argument("--bootstrap_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    from src.pku_saferlhf.evaluation_utils import load_records, new_output, write_json
    records, manifest = load_records(args.scored_dir, "scored_responses.jsonl")
    result = summarize(records, manifest, args.reference_run, args.bootstrap_samples, args.seed)
    output = new_output(args.output_dir)
    result["provenance"] = {"manifest": manifest, "bootstrap_samples": args.bootstrap_samples, "seed": args.seed}
    write_json(output / "summary.json", result)
    for name in ("per_run", "paired_differences", "per_method"):
        write_csv(output / f"{name}.csv", result[name])
    print("run_id\thelpfulness_reward (higher better)\tharmfulness_cost (lower better)")
    for row in result["per_run"]:
        print(f"{row['run_id']}\t{row['helpfulness_reward_mean']:.4f}\t{row['harmfulness_cost_mean']:.4f}")
    print(f"Saved summaries to {output}")


if __name__ == "__main__":
    main()
