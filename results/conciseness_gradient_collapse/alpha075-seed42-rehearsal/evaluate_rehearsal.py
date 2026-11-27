"""Summarize the completed rehearsal using its recorded diagnostics."""
from collections import defaultdict
from pathlib import Path
import json
import numpy as np

root = Path(__file__).resolve().parent
diagnostics = root / "diagnostics"
def read(name):
    return [json.loads(line) for line in (diagnostics / (name + ".jsonl")).read_text().splitlines()]

def fields(row, keys):
    return {k: row[k] for k in keys.split()}

def main():
    audit = json.loads((root/"audit.json").read_text())
    assert audit["updates"] == 26 and audit["episodes"] == 1664
    gradients = read("gradient_summary")
    panels = [r for r in gradients if r["block"] == "all"]
    ratios = []
    for r in panels:
        blocks = [b["rho"] for b in gradients if b["ppo_update"] == r["ppo_update"] and
                  b["stream"] == r["stream"] and b["block"] != "all"]
        row = fields(r, "ppo_update stream phase norm_conciseness norm_sentiment weighted_norm_conciseness weighted_norm_sentiment rho cos_conciseness__minority_rm cos_sentiment__minority_rm cos_sentiment__minority_oracle mixture_error_relative")
        row["block_ratios"] = blocks
        # Each of four prompt-disjoint blocks has 128 responses. The full
        # leave-one-out gradient is the mean of those block gradients.
        if len(blocks) == 4:
            unit_blocks = [b for b in gradients if b["stream"] == r["stream"] and
                           b["ppo_update"] == r["ppo_update"] and b["block"] != "all"]
            row["mean_cross_block_inner_product"] = {
                name: (16*r[name]**2-sum(b[name]**2 for b in unit_blocks))/12
                for name in ("norm_conciseness","norm_sentiment")
            }
        ratios.append(row)
    updates = [fields(r, "ppo_update norm_update cos_update__conciseness cos_update__sentiment cos_update__minority_rm cos_update__minority_oracle contribution_conciseness contribution_sentiment dot_update__minority_oracle learning_rate") for r in read("update_summary")]
    outcomes = [fields(r, "ppo_update policy_length_mean policy_length_std policy_length_within_prompt_variance scorer_truncated_rate cap_hit_rate eos_rate conciseness_mean conciseness_within_prompt_variance sentiment_mean sentiment_within_prompt_variance minority_oracle_mean minority_rm_mean") for r in read("evaluation")]
    observations = defaultdict(lambda: defaultdict(list))
    for r in read("evaluation_rollouts"):
        observations[r["ppo_update"]][r["prompt_id"]].append(r)
    ids = sorted(observations[0])
    assert len(ids) == 128 and set(ids) == set(observations[26])
    rng = np.random.default_rng(42)
    samples = rng.integers(len(ids), size=(10000,len(ids)))
    changes = {}
    for metric in ["policy_length", "conciseness", "sentiment", "minority_oracle", "minority_rm", "cap_hit"]:
        differences = []
        for i in ids:
            def mean_at(u):
                responses = observations[u][i]
                assert len(responses)==4
                return np.mean([r[metric] for r in responses])
            differences.append(mean_at(26)-mean_at(0))
        means = np.array(differences)[samples].mean(axis=1)
        changes[metric] = dict(change=float(np.mean(differences)), interval=np.quantile(means,[.025,.975]).tolist())
    controls = read("training_controls")
    train_stats = [r for r in read("rollout_summary") if r["stream"] == "training_rollout"]
    assert len(train_stats) == 26
    data = dict(audit=audit, gradients=ratios, optimizer_steps=updates, evaluation=outcomes,
                paired_prompt_bootstrap_26_minus_0=changes,
                counterfactual_update_alignment=[fields(r, "ppo_update evaluation_alpha cos_update__minority_oracle") for r in read("counterfactual_alpha") if r["stream"]=="training_rollout"],
                training=dict(mean_length=np.mean([r["policy_length_mean"] for r in train_stats]),
                              cap_hit_rate=np.mean([r["cap_hit_rate"] for r in train_stats]),
                              final_kl=controls[-1]["objective/kl"],final_clip_fraction=controls[-1]["policy/clipfrac_avg"]))
    (root/"hypothesis_metrics.json").write_text(json.dumps(data, indent=2, allow_nan=False)+"\n")
if __name__ == "__main__":
    main()
