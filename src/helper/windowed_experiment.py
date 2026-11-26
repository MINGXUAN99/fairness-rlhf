"""Prepare frozen inputs, then score and check minimal windowed-PCGrad artifacts."""
import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.imdb.ppo_training.windowed_monitor import (digest, read_jsonl, write_jsonl,
                                                   summary, schedule)


def prepare(args):
    from datasets import load_dataset
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    from src.helper.imdb_utils import sample_fragment
    root = Path(args.output)
    if root.exists():
        raise FileExistsError(f'{root}: use the existing frozen inputs, or a new path')
    api = HfApi()
    models = {}
    for key, name in [('policy', 'gpt2'), ('minority', args.minority_rm),
                      ('majority', args.majority_rm), ('scorer', 'lvwerra/distilbert-imdb')]:
        models[key] = dict(id=name, revision=api.model_info(name).sha)
    dataset_id = 'stanfordnlp/imdb'
    dataset_revision = api.dataset_info(dataset_id).sha
    ds = load_dataset(dataset_id, revision=dataset_revision, split='train')
    tokenizer = AutoTokenizer.from_pretrained('gpt2', revision=models['policy']['revision'])
    rng = random.Random(43)
    rows = []
    for idx, item in enumerate(ds):
        if len(item['text']) <= 200:
            continue
        prompt = sample_fragment(item['text'], tokenizer, rng.randint(2, 8))
        ids = tokenizer(prompt, truncation=True, max_length=8)['input_ids']
        rows.append(dict(source_id=f'imdb/train/{idx}', prompt_id=digest(ids),
                         prompt=tokenizer.decode(ids, skip_special_tokens=True), input_ids=ids))
    # Split source examples before choosing unique evaluation prompts. Remove
    # short-prefix duplicates of held-out prompts from the training partition.
    from datasets import Dataset
    split = Dataset.from_list(rows).train_test_split(test_size=.05, seed=43)
    candidates = list(split['test'])
    random.Random(1043).shuffle(candidates)
    panel, seen = [], set()
    for row in candidates:
        if row['prompt_id'] not in seen:
            panel.append(row)
            seen.add(row['prompt_id'])
        if len(panel) == 256:
            break
    if len(panel) != 256:
        raise ValueError('Insufficient unique held-out prompts')
    train = [row for row in split['train'] if row['prompt_id'] not in seen]
    root.mkdir(parents=True)
    write_jsonl(root / 'training.jsonl', train)
    write_jsonl(root / 'evaluation.jsonl', panel)
    import importlib.metadata
    metadata = dict(versions={name: importlib.metadata.version(name) for name in
                              ("torch", "transformers", "trl", "accelerate", "datasets")}, seed=43, models=models, dataset=dict(id=dataset_id, revision=dataset_revision),
                    training_hash=digest(train), evaluation_hash=digest(panel),
                    scoring_text=args.scoring_text, training_count=len(train),
                    evaluation_count=len(panel), qualitative_prompt_ids=[r['prompt_id'] for r in panel[:12]],
                    prompt_preprocessing='seed-43 serial fragment lengths 2..8; first 8 tokens; held-out duplicate prefixes excluded',
                    controls={'none': args.none_artifact, 'full': args.full_artifact},
                    controls_verified=False)
    (root / 'inputs.json').write_text(json.dumps(metadata, indent=2))
    print(f'Frozen {len(train)} training prompts and 256 evaluation prompts in {root}')


def score_values(length, sentiment):
    c = 1 - min(max(length / 256, 0), 1)
    return dict(conciseness=c, sentiment=sentiment,
                reward_min=.8 * c + .2 * sentiment, reward_maj=.2 * c + .8 * sentiment)


def summarize_scores(rows):
    result = {key: summary([r[key] for r in rows]) for key in
              ('generation_length', 'scoring_length', 'conciseness', 'sentiment', 'reward_min', 'reward_maj')}
    rmin, rmaj = result['reward_min']['mean'], result['reward_maj']['mean']
    result.update(F=min(rmin, rmaj), U=.2 * rmin + .8 * rmaj,
                  signed_gap=rmin-rmaj, absolute_gap=abs(rmin-rmaj), count=len(rows))
    for key in ('eos', 'generation_cap_reached', 'scorer_truncated'):
        result[key + '_rate'] = sum(bool(r[key]) for r in rows) / len(rows)
    result['length_gt256_rate'] = sum(r['scoring_length'] > 256 for r in rows) / len(rows)
    return result


def validate_run(root, manifest):
    horizon = manifest['arguments']['expected_updates']
    if (root / 'TRAINING_COMPLETE').read_text().strip() != str(horizon):
        raise ValueError('Training is incomplete')
    if not (root / 'final_policy/config.json').exists():
        raise ValueError('Missing final policy')
    train_logs = read_jsonl(root / 'metrics/train_updates.jsonl')
    if [r['update'] for r in train_logs] != list(range(1, horizon + 1)):
        raise ValueError('Missing/duplicate/out-of-order training metrics')
    panel = read_jsonl(root / 'prompts/evaluation.jsonl')
    panel_ids = [r['prompt_id'] for r in panel]
    if digest(panel) != manifest['inputs']['evaluation_hash'] or len(set(panel_ids)) != 256:
        raise ValueError('Evaluation panel differs from frozen inputs')
    for kind, updates, phase, count in (
        ('train', list(range(1, horizon + 1)), 'pre_update', manifest['resolved_config']['batch_size']),
        ('eval', manifest['evaluation_updates'], 'post_update', 256)):
        paths = sorted((root / 'rollouts' / kind).glob('update_*.jsonl'))
        if [int(p.stem.split('_')[1]) for p in paths] != updates:
            raise ValueError(f'Incomplete {kind} snapshot schedule')
        for path, update in zip(paths, updates):
            rows = read_jsonl(path)
            if len(rows) != count or any(r['update'] != update or r['phase'] != phase for r in rows):
                raise ValueError(f'Invalid snapshot {path}')
            if kind == 'eval' and [r['prompt_id'] for r in rows] != panel_ids:
                raise ValueError(f'Wrong evaluation prompts in {path}')
    with (root / 'diagnostics/gradient_steps.csv').open() as handle:
        gradients = list(csv.DictReader(handle))
    epochs = manifest['resolved_config']['num_ppo_epochs']
    expected_rows = horizon * epochs * manifest['resolved_config']['num_mini_batches']
    if len(gradients) != expected_rows or [int(r['step']) for r in gradients] != list(range(expected_rows)):
        raise ValueError('Missing/duplicate gradient rows')
    per_update = epochs * manifest['resolved_config']['num_mini_batches']
    if [int(r['update']) for r in gradients] != [u for u in range(1, horizon + 1) for _ in range(per_update)]:
        raise ValueError('Gradient rows have wrong outer update identity')
    lo, hi = (manifest['arguments'][k] for k in ('pcgrad_window_start', 'pcgrad_window_end'))
    for r in gradients:
        active = lo <= int(r['update']) <= hi
        if (r['window_active'] == 'True') != active or (r['projection_applied'] == 'True') != (active and float(r['dot']) < 0):
            raise ValueError('Incorrect intervention execution flags')
    return gradients


def analyze(args):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    root = Path(args.run_dir)
    manifest = json.loads((root / 'manifest.json').read_text())
    gradients = validate_run(root, manifest)
    if args.check_only:
        print('Artifact schedules, prompts, row counts and intervention flags passed')
        return
    scorer = manifest['inputs']['models']['scorer']
    tokenizer = AutoTokenizer.from_pretrained(scorer['id'], revision=scorer['revision'])
    model = AutoModelForSequenceClassification.from_pretrained(scorer['id'], revision=scorer['revision']).to(args.device).eval()
    # The established lvwerra scorer uses class 1 as positive.
    if model.config.num_labels != 2:
        raise ValueError('Expected binary sentiment scorer')
    results = {}
    for kind, output_name in [('train', 'train'), ('eval', 'evaluation')]:
        snapshots = []
        scores_path = root / 'metrics' / f'{output_name}_response_scores.jsonl'
        temp_path = scores_path.with_suffix('.jsonl.tmp')
        with temp_path.open('w') as output:
            for path in sorted((root / 'rollouts' / kind).glob('update_*.jsonl')):
                original = read_jsonl(path)
                scored = []
                for offset in range(0, len(original), args.batch_size):
                    batch = original[offset:offset + args.batch_size]
                    texts = [r['scored_text'] for r in batch]
                    raw_lengths = [len(ids) for ids in tokenizer(texts, truncation=False)['input_ids']]
                    encoded = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(args.device)
                    with torch.no_grad():
                        sentiment = model(**encoded).logits.float().softmax(-1)[:, 1].cpu().tolist()
                    lengths = encoded['attention_mask'].sum(1).cpu().tolist()
                    for row, length, raw_length, positive in zip(batch, lengths, raw_lengths, sentiment):
                        result = {k: v for k, v in row.items() if k not in ('prompt', 'continuation', 'scored_text')}
                        result.update(scoring_length=length, scorer_truncated=raw_length > 512,
                                      **score_values(length, positive))
                        scored.append(result)
                        output.write(json.dumps(result, allow_nan=False) + '\n')
                snapshots.append(dict(update=original[0]['update'], phase=original[0]['phase'],
                                      **summarize_scores(scored)))
        temp_path.replace(scores_path)
        summary_path = root / 'metrics' / f'{output_name}_oracle_updates.jsonl'
        with summary_path.open('w') as handle:
            for row in snapshots:
                handle.write(json.dumps(row, allow_nan=False) + '\n')
        results[kind] = snapshots
    # Small diagnostic figures; raw data remains the primary record.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = root / 'figures'
    figures.mkdir(exist_ok=True)
    for kind, fields in [('train', ['generation_length', 'length_gt256_rate', 'conciseness']),
                         ('eval', ['generation_length', 'conciseness', 'sentiment', 'F'])]:
        fig, axes = plt.subplots(len(fields), 1, figsize=(8, 2.5 * len(fields)), sharex=True)
        for axis, field in zip(axes, fields):
            rows = results[kind]
            values = [r[field]['mean'] if isinstance(r[field], dict) else r[field] for r in rows]
            axis.plot([r['update'] for r in rows], values)
            axis.axvline(manifest['arguments']['pcgrad_window_start'] - .5, color='gray', linestyle='--')
            axis.axvline(manifest['arguments']['pcgrad_window_end'] + .5, color='gray', linestyle='--')
            axis.set_ylabel(field)
        axes[-1].set_xlabel('Outer update (pre-update)' if kind == 'train' else 'Completed update (post-update)')
        fig.tight_layout()
        fig.savefig(figures / f'{kind}.png', dpi=160)
        plt.close(fig)
    fig, axis = plt.subplots(figsize=(8, 3))
    for name in ('conflict_detected', 'projection_applied'):
        updates = range(1, manifest['arguments']['expected_updates'] + 1)
        axis.plot(list(updates), [sum(r[name] == 'True' for r in gradients if int(r['update']) == u) /
                                 sum(int(r['update']) == u for r in gradients) for u in updates], label=name)
    axis.legend()
    axis.set_xlabel('Outer update')
    fig.tight_layout()
    fig.savefig(figures / 'intervention.png', dpi=160)
    plt.close(fig)
    checksums = {}
    for path in sorted(root.rglob('*')):
        if path.is_file() and path.name != 'checksums.json':
            h = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b''):
                    h.update(chunk)
            checksums[str(path.relative_to(root))] = h.hexdigest()
    (root / 'checksums.json').write_text(json.dumps(checksums, indent=2))
    print(f'Scored all snapshots, generated three figures and checksums in {root}')


def compare(args):
    roots = {'Early': Path(args.early), 'Late': Path(args.late)}
    manifests = {arm: json.loads((root / 'manifest.json').read_text()) for arm, root in roots.items()}
    early, late = manifests.values()
    if (early['inputs'] != late['inputs'] or early['evaluation_updates'] != late['evaluation_updates']
            or early['source_hashes'] != late['source_hashes']
            or early.get('model_parameter_dtypes') != late.get('model_parameter_dtypes')):
        raise ValueError('Arms differ in frozen inputs, evaluation schedule, source code or parameter dtypes')
    for key in ('seed', 'response_length', 'batch_size', 'num_total_batches', 'learning_rate',
                'num_ppo_epochs', 'gradient_accumulation_steps', 'kl_coef', 'whiten_rewards', 'temperature'):
        if early['resolved_config'][key] != late['resolved_config'][key]:
            raise ValueError(f'Arms differ in {key}')
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    curves = {arm: {kind: read_jsonl(root / f'metrics/{kind}_oracle_updates.jsonl')
                    for kind in ('train', 'evaluation')} for arm, root in roots.items()}
    horizon = early['arguments']['expected_updates']
    boundary = early['arguments']['pcgrad_window_end']
    if (early['arguments']['pcgrad_window_start'] != 1
            or late['arguments']['pcgrad_window_start'] != boundary + 1
            or late['arguments']['pcgrad_window_end'] != horizon):
        raise ValueError('Expected disjoint Early/Late windows covering the horizon')
    fields = ('generation_length', 'scoring_length', 'conciseness', 'sentiment', 'reward_min', 'reward_maj', 'F', 'U')
    def value(row, key):
        return row[key]['mean'] if isinstance(row[key], dict) else row[key]
    finals = {arm: next(r for r in data['evaluation'] if r['update'] == horizon) for arm, data in curves.items()}
    late_eval = {r['update']: r for r in curves['Late']['evaluation']}
    report = {'final': {arm: {key: value(row, key) for key in fields} for arm, row in finals.items()},
              'early_minus_late': {key: value(finals['Early'], key) - value(finals['Late'], key) for key in fields},
              'late_final_minus_initial': {key: value(finals['Late'], key) - value(late_eval[0], key) for key in fields}}
    if boundary in late_eval:
        report['late_final_minus_switch'] = {key: value(finals['Late'], key) - value(late_eval[boundary], key) for key in fields}
    (output / 'comparison.json').write_text(json.dumps(report, indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for kind, plot_fields in [('train', ('generation_length', 'length_gt256_rate', 'conciseness')),
                              ('evaluation', ('generation_length', 'conciseness', 'sentiment', 'F'))]:
        fig, axes = plt.subplots(len(plot_fields), 1, figsize=(8, 2.5 * len(plot_fields)), sharex=True)
        for axis, key in zip(axes, plot_fields):
            for arm, data in curves.items():
                axis.plot([r['update'] for r in data[kind]], [value(r, key) for r in data[kind]], label=arm)
            axis.axvline(boundary + .5, color='gray', linestyle='--')
            axis.set_ylabel(key)
            axis.legend()
        axes[-1].set_xlabel('Outer update (pre-update)' if kind == 'train' else 'Completed update (post-update)')
        fig.tight_layout()
        fig.savefig(output / f'{kind}.png', dpi=160)
        plt.close(fig)
    print(f'Early/Late comparison saved in {output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('--output', required=True)
    prep.add_argument('--minority_rm', required=True)
    prep.add_argument('--majority_rm', required=True)
    prep.add_argument('--scoring_text', choices=['full', 'continuation'], default='full',
                      help='Existing evaluation decodes/scorers use prompt plus continuation')
    prep.add_argument('--none_artifact')
    prep.add_argument('--full_artifact')
    score = subs.add_parser('analyze')
    score.add_argument('--run_dir', required=True)
    score.add_argument('--device', default='cpu')
    score.add_argument('--batch_size', type=int, default=32)
    score.add_argument('--check_only', action='store_true')
    comparison = subs.add_parser('compare')
    comparison.add_argument('--early', required=True)
    comparison.add_argument('--late', required=True)
    comparison.add_argument('--output', required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'analyze': analyze, 'compare': compare}[args.command](args)


if __name__ == '__main__':
    main()
