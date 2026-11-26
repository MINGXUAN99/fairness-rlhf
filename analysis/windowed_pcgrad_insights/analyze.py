"""Read-only exploratory audit of archived Early/Late PCGrad results.

Writes derived artifacts beside this script; never changes training artifacts.
Run from repository root with venv/bin/python analysis/windowed_pcgrad_insights/analyze.py.
Bootstrap intervals describe the fixed trained policies' sampled prompt panel,
not uncertainty over training seeds. Analyses were selected after seeing results.
"""
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/windowed-len53-batch64-seed43-20261012_195052'
OUT = Path(__file__).resolve().parent
ARMS = {'Early': RUN / 'early-win1-400', 'Late': RUN / 'late-win401-800'}


def read(path):
    return [json.loads(line) for line in path.open() if line.strip()]


def val(row, key):
    x = row[key]
    return x['mean'] if isinstance(x, dict) else x


def bootstrap(x):
    x = np.asarray(x, dtype=float)
    assert x.shape == (256,)
    rng = np.random.default_rng(20261012)
    means = x[rng.integers(0, len(x), (5000, len(x)))].mean(axis=1)
    return {'difference': float(x.mean()), 'prompt_bootstrap_95_interval':
            np.quantile(means, [.025, .975]).tolist()}


def keyed(rows, updates, field):
    by = {(r['update'], r['prompt_id']): float(r[field]) for r in rows}
    ids = sorted({r['prompt_id'] for r in rows})
    assert len(ids) == 256
    return np.array([[by[u, p] for p in ids] for u in updates])


def main():
    data = {}
    results = {'source_run': str(RUN), 'bootstrap_replicates': 5000,
               'uncertainty_scope': 'Prompt sampling conditional on these two trained policies; '
               'one stochastic response per prompt per snapshot; no training-seed inference.',
               'selection': 'Exploratory, after seeing endpoints; no confirmatory significance claims.'}
    manifests = {a: json.loads((p / 'manifest.json').read_text()) for a, p in ARMS.items()}
    assert manifests['Early']['inputs'] == manifests['Late']['inputs']
    assert manifests['Early']['source_hashes'] == manifests['Late']['source_hashes']
    results['arm_audits'] = {}
    block_csv = []
    for arm, p in ARMS.items():
        train = read(p / 'metrics/train_updates.jsonl')
        oracle = read(p / 'metrics/train_oracle_updates.jsonl')
        evaluation = read(p / 'metrics/evaluation_oracle_updates.jsonl')
        response = read(p / 'metrics/evaluation_response_scores.jsonl')
        tscores = read(p / 'metrics/train_response_scores.jsonl')
        gradients = list(csv.DictReader((p / 'diagnostics/gradient_steps.csv').open()))
        assert [r['update'] for r in train] == list(range(1, 801))
        assert len(gradients) == 2400 and len(tscores) == 51200 and len(response) == 4864
        learned_rows = []
        for f in sorted((p / 'rollouts/train').glob('*.jsonl')):
            learned_rows.extend(read(f))
        assert [(r['update'], r['response_index']) for r in learned_rows] == [
            (r['update'], r['response_index']) for r in tscores]
        learned = np.array([r['learned_rewards_raw'] for r in learned_rows])
        corrfields = ['conciseness', 'sentiment', 'reward_min', 'generation_length']
        audit = {'blocks': [], 'learned_minority_reward_correlations': {
            k: float(np.corrcoef(learned[:, 0], [r[k] for r in tscores])[0, 1])
            for k in corrfields}, 'raw_group_reward_correlation': float(np.corrcoef(learned.T)[0, 1]),
            'training_eos_strata': {}}
        for flag in (True, False):
            m = np.array([r['eos'] == flag for r in tscores])
            audit['training_eos_strata'][str(flag)] = {
                'count': int(m.sum()), 'learned_reward_means': learned[m].mean(0).tolist(),
                **{k: float(np.array([r[k] for r in tscores])[m].mean()) for k in corrfields}}
        for lo, hi in [(1, 400), (401, 800)]:
            tr = train[lo - 1:hi]
            g = [r for r in gradients if lo <= int(r['update']) <= hi]
            dot = np.array([float(r['dot']) for r in g])
            n1 = np.array([float(r['norm_g1']) for r in g])
            n2 = np.array([float(r['norm_g2']) for r in g])
            negative = dot < 0
            # Reconstruct the norm of the additive two-gradient projection
            # correction using only saved dots and norms. Excludes KL and Adam.
            correction_norm2 = dot**2 * (1/n1**2 + 1/n2**2 + 2*dot/(n1**2*n2**2))
            sum_norm2 = n1**2 + n2**2 + 2*dot
            correction_ratio = np.sqrt(np.maximum(correction_norm2, 0) / sum_norm2)
            row = {'arm': arm, 'first_update': lo, 'last_update': hi,
                   'train_length': float(np.mean([val(r, 'generation_length') for r in tr])),
                   'train_eos_rate': float(np.mean([r['eos_rate'] for r in tr])),
                   'train_cap_rate': float(np.mean([r['generation_cap_rate'] for r in tr])),
                   'gradient_count': len(g), 'conflict_count': int(negative.sum()),
                   'projection_count': sum(r['projection_applied'] == 'True' for r in g),
                   'mean_cosine': float(np.mean(dot/(n1*n2))),
                   'negative_cosine_mean': float(np.mean((dot/(n1*n2))[negative])),
                   'sum_gradient_opposes_minority_count': int((n1**2 + dot < 0).sum()),
                   'sum_gradient_opposes_majority_count': int((n2**2 + dot < 0).sum()),
                   'conflict_correction_relative_to_sum_mean': float(correction_ratio[negative].mean()),
                   'outer_update_learning_rate_sum': float(sum(r['ppo']['lr'] for r in tr)),
                   'lr_weighted_relative_correction_proxy': float(sum(
                       correction_ratio[i] * train[int(r['update'])-1]['ppo']['lr']
                       for i, r in enumerate(g) if r['projection_applied'] == 'True'))}
            audit['blocks'].append(row)
            block_csv.append(row)
        for update in (0, 400, 550, 800):
            rows = [r for r in response if r['update'] == update]
            short = [r for r in rows if r['generation_length'] <= 53]
            eos = [r for r in rows if r['eos']]
            audit.setdefault('evaluation_strata', {})[update] = {
                'short_le53_count': len(short), 'short_le20_count': sum(r['generation_length'] <= 20 for r in rows),
                'eos_count': len(eos), 'eos_length_mean': float(np.mean([r['generation_length'] for r in eos])),
                'cap_count': sum(r['generation_cap_reached'] for r in rows)}
        results['arm_audits'][arm] = audit
        data[arm] = {'train': train, 'oracle': oracle, 'evaluation': evaluation,
                     'response': response, 'gradients': gradients}
    assert keyed(data['Early']['response'], [0], 'generation_length').tolist() == keyed(
        data['Late']['response'], [0], 'generation_length').tolist()
    fields = ['generation_length', 'conciseness', 'sentiment', 'reward_min', 'reward_maj', 'eos']
    results['paired_early_minus_late'] = {}
    for label, updates in [('update400', [400]), ('update800', [800]),
                           ('late_window_regular_snapshots450_800', list(range(450, 801, 50))),
                           ('last_five_regular_snapshots600_800', list(range(600, 801, 50))),
                           ('early_window_regular_snapshots50_400', list(range(50, 401, 50)))]:
        results['paired_early_minus_late'][label] = {k: bootstrap((
            keyed(data['Early']['response'], updates, k) - keyed(data['Late']['response'], updates, k)
        ).mean(0)) for k in fields}
    results['within_arm_800_minus400'] = {a: {k: bootstrap((
        keyed(d['response'], [800], k) - keyed(d['response'], [400], k))[0])
        for k in fields} for a, d in data.items()}
    rows = []
    for e, l in zip(data['Early']['evaluation'], data['Late']['evaluation']):
        assert e['update'] == l['update']
        row = {'update': e['update']}
        for a, r in [('Early', e), ('Late', l)]:
            for k in fields[:-1] + ['eos_rate', 'generation_cap_reached_rate']:
                row[f'{a}_{k}'] = val(r, k)
        rows.append(row)
    for name, items in [('evaluation_snapshots.csv', rows), ('training_blocks.csv', block_csv)]:
        with (OUT / name).open('w') as f:
            w = csv.DictWriter(f, fieldnames=list(items[0]))
            w.writeheader()
            w.writerows(items)
    (OUT / 'findings.json').write_text(json.dumps(results, indent=2))
    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    colors = {'Early': '#2563eb', 'Late': '#d97706'}
    for a, d in data.items():
        x = [r['update'] for r in d['evaluation']]
        for ax, k in zip(axes[0], ['generation_length', 'reward_min']):
            ax.plot(x, [val(r, k) for r in d['evaluation']], 'o-', label=a, color=colors[a], ms=3)
        axes[1, 0].plot(x, [r['eos_rate'] for r in d['evaluation']], 'o-', label=a, color=colors[a], ms=3)
        # Display smoothing only; all numerical comparisons use raw records.
        tx = [r['update'] for r in d['train']]
        smooth = np.convolve([r['eos_rate'] for r in d['train']], np.ones(40)/40, 'valid')
        axes[1, 1].plot(tx[39:], smooth, label=a, color=colors[a])
        gx = np.array([int(r['update']) for r in d['gradients']])
        for is_projection, style in [(False, '--'), (True, '-')]:
            gy = np.array([r['projection_applied' if is_projection else 'conflict_detected'] == 'True'
                           for r in d['gradients']])
            bins = np.arange(0, 801, 50)
            axes[2, 0].plot(bins[1:]-25, [gy[(gx > lo) & (gx <= hi)].mean()
                                         for lo, hi in zip(bins[:-1], bins[1:])], style,
                           color=colors[a], label=f'{a}: ' + ('projected' if is_projection else 'conflict'))
    axes[2, 1].plot(range(1, 801), [r['ppo']['lr'] for r in data['Early']['train']], color='#475569')
    titles = ['Evaluation response length (cap 256)', 'Evaluation minority oracle score',
              'Evaluation EOS fraction', 'Training EOS fraction (40-update moving mean; cap 53)',
              'Conflicts and applied projections (50-update bins)', 'Learning rate: equal windows, unequal budget']
    for ax, title in zip(axes.flat, titles):
        ax.set_title(title, fontsize=10)
        ax.axvline(400.5, color='gray', linestyle=':')
        ax.grid(alpha=.2)
    for ax in axes.flat:
        if ax is not axes[2, 1]:
            ax.legend(fontsize=8)
    axes[2, 0].set_xlabel('Outer update')
    axes[2, 1].set_xlabel('Outer update')
    fig.suptitle('PCGrad timing audit: one training seed, matched Early/Late; historical Full excluded', fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / 'timing_audit.png', dpi=170)
    plt.close(fig)
    print(json.dumps({'paired': results['paired_early_minus_late'],
                      'arm_audits': results['arm_audits']}, indent=2))


if __name__ == '__main__':
    main()
