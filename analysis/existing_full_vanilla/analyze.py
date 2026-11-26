"""Offline analysis of archived Full/vanilla outputs; no new training or generation.

Run: venv/bin/python analysis/existing_full_vanilla/analyze.py
Inputs and provenance are stored beside this script. Endpoint lengths use the
sentiment scorer tokenizer and include prompts; panel lengths are retokenized
continuations through first EOS. The two lengths must not be equated.
"""
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent


def rows(name):
    with (OUT / name).open() as f:
        return list(csv.DictReader(f))


def main():
    panels = rows('panel_responses.csv')
    metrics = rows('ppo_metrics.csv')
    final = rows('final_evaluation.csv')
    result = {'selection': 'Exploratory analysis of identified existing runs, including contrary examples.',
              'uncertainty': 'Single trained policy per run; no training-seed or causal timing inference.',
              'panel_snapshots': {}, 'endpoints': {}, 'training_endpoints': {}}
    panel_arrays = {}
    for arm in sorted({r['run'] for r in panels}):
        pp = [r for r in panels if r['run'] == arm]
        updates = sorted({int(r['update']) for r in pp})
        summaries = []
        for u in updates:
            batch = [r for r in pp if int(r['update']) == u]
            assert len(batch) == 8
            lengths = np.array([int(r['retokenized_length_including_eos']) for r in batch])
            summaries.append({'update': u, 'count': len(batch), 'length_mean': float(lengths.mean()),
                              'eos_count': sum(int(r['eos']) for r in batch),
                              'short_le20_count': int((lengths <= 20).sum())})
        result['panel_snapshots'][arm] = summaries
        panel_arrays[arm] = summaries
        mm = [r for r in metrics if r['run'] == arm]
        assert len(mm) == 782
        result['training_endpoints'][arm] = {
            'first_update': {k: float(mm[0][k]) for k in ('train/objective/scores', 'train/policy/entropy_avg')},
            'last_update': {k: float(mm[-1][k]) for k in ('train/objective/scores', 'train/policy/entropy_avg')}}
    vectors, grouped = {}, {}
    for arm in ('Full', 'Vanilla_long'):
        ff = [r for r in final if r['run'] == arm]
        assert len(ff) == 24872
        length = np.array([float(r['scoring_length']) for r in ff])
        sentiment = np.array([float(r['sentiment']) for r in ff])
        c = 1 - np.clip(length / 256, 0, 1)
        minority, majority = .8 * c + .2 * sentiment, .2 * c + .8 * sentiment
        result['endpoints'][arm] = {
            'count': len(ff), 'length_mean': float(length.mean()), 'length_median': float(np.median(length)),
            'short_le20_count': int((length <= 20).sum()), 'short_le53_count': int((length <= 53).sum()),
            'zero_conciseness_count': int((length >= 256).sum()), 'conciseness_mean': float(c.mean()),
            'sentiment_mean': float(sentiment.mean()), 'minority_mean': float(minority.mean()),
            'majority_mean': float(majority.mean()),
            'minority_conciseness_contribution': float(.8 * c.mean()),
            'minority_sentiment_contribution': float(.2 * sentiment.mean())}
        vectors[arm] = length
        group = defaultdict(list)
        for r, l, cc, ss, m in zip(ff, length, c, sentiment, minority):
            group[r['prompt_hash']].append([l, cc, ss, m, float(l <= 53)])
        grouped[arm] = {k: np.mean(v, axis=0) for k, v in group.items()}
    common = sorted(set(grouped['Full']) & set(grouped['Vanilla_long']))
    paired = {a: np.array([g[p] for p in common]) for a, g in grouped.items()}
    fields = ['length', 'conciseness', 'sentiment', 'minority', 'short_le53_rate']
    result['common_prompt_comparison'] = {'unique_prompt_count': len(common),
        'weighting': 'Each unique matching prompt has equal weight; multiple rows per prompt averaged first.',
        'means': {a: dict(zip(fields, x.mean(0).tolist())) for a, x in paired.items()}}
    diff = paired['Full'] - paired['Vanilla_long']
    rng = np.random.default_rng(20261013)
    samples = []
    for _ in range(10):
        indices = rng.integers(0, len(common), (100, len(common)))
        samples.append(diff[indices].mean(1))
    boot = np.concatenate(samples)
    result['common_prompt_comparison']['full_minus_vanilla'] = {
        k: {'difference': float(diff[:, j].mean()),
            'descriptive_prompt_bootstrap_95_interval': np.quantile(boot[:, j], [.025, .975]).tolist()}
        for j, k in enumerate(fields)}
    full, vanilla = result['endpoints']['Full'], result['endpoints']['Vanilla_long']
    result['minority_gap_decomposition'] = {
        'Full_minus_Vanilla': full['minority_mean'] - vanilla['minority_mean'],
        'conciseness_component': .8 * (full['conciseness_mean'] - vanilla['conciseness_mean']),
        'sentiment_component': .2 * (full['sentiment_mean'] - vanilla['sentiment_mean'])}
    assert np.isclose(result['minority_gap_decomposition']['Full_minus_Vanilla'],
                      result['minority_gap_decomposition']['conciseness_component'] +
                      result['minority_gap_decomposition']['sentiment_component'])
    (OUT / 'findings.json').write_text(json.dumps(result, indent=2))
    with (OUT / 'snapshot_summary.csv').open('w') as f:
        items = [{'run': a, **r} for a, rr in panel_arrays.items() for r in rr]
        writer = csv.DictWriter(f, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    labels = {'Full': 'Full PCGrad', 'Vanilla_long': 'Vanilla: long-response run',
              'Vanilla_diagnostic': 'Vanilla: diagnostic-launched run'}
    colors = {'Full': '#2563eb', 'Vanilla_long': '#d97706', 'Vanilla_diagnostic': '#64748b'}
    for a in labels:
        rr = panel_arrays[a]
        axes[0, 0].plot([r['update'] for r in rr], [r['eos_count']/8 for r in rr],
                        'o-', label=labels[a], color=colors[a], ms=4)
        axes[0, 1].plot([r['update'] for r in rr], [r['length_mean'] for r in rr],
                        'o-', label=labels[a], color=colors[a], ms=4)
    axes[0, 0].set_title('Saved completion-panel EOS fraction (8 prompts/snapshot)')
    axes[0, 1].set_title('Saved panel continuation length (cap 53)')
    for ax in axes[0]:
        ax.set_xlabel('Completed PPO update')
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    for a in ('Full', 'Vanilla_long'):
        xx = np.sort(vectors[a])
        axes[1, 0].plot(xx, np.arange(1, len(xx)+1)/len(xx), label=labels[a], color=colors[a])
    axes[1, 0].axvline(53, color='gray', linestyle=':', label='53 scoring tokens')
    axes[1, 0].set_title('Final evaluation length CDF (24,872 outputs/policy)')
    axes[1, 0].set_xlabel('Scorer tokens, prompt included')
    axes[1, 0].set_xlim(0, 280)
    axes[1, 0].legend(fontsize=8)
    arms = ['Vanilla_long', 'Full']
    cpart = [result['endpoints'][a]['minority_conciseness_contribution'] for a in arms]
    spart = [result['endpoints'][a]['minority_sentiment_contribution'] for a in arms]
    axes[1, 1].bar(['Vanilla', 'Full PCGrad'], cpart, label='0.8 × conciseness', color='#16a34a')
    axes[1, 1].bar(['Vanilla', 'Full PCGrad'], spart, bottom=cpart, label='0.2 × sentiment', color='#9333ea')
    axes[1, 1].set_title('Minority-score components at final evaluation')
    axes[1, 1].legend(fontsize=8)
    for ax in axes[1]:
        ax.grid(alpha=.2)
    fig.suptitle('Existing-output evidence: strong distribution contrast, unresolved causal timing\n'
                 'Panels use temperature 0.01; they are not stochastic PPO training rollouts.', fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / 'existing_evidence.png', dpi=170)
    plt.close(fig)
    print(json.dumps({'endpoints': result['endpoints'],
                      'common_prompts': result['common_prompt_comparison'],
                      'gap': result['minority_gap_decomposition'],
                      'training_endpoints': result['training_endpoints']}, indent=2))


if __name__ == '__main__':
    main()
