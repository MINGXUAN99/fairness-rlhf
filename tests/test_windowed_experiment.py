import copy
import csv
import json
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification, PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from datasets import Dataset

from src.imdb.ppo_training import unified_ppo_training as training
from src.imdb.ppo_training.windowed_monitor import (
    WindowedMonitor, isolated_evaluation, response_record, schedule, write_jsonl, digest,
)
from src.helper.windowed_experiment import score_values, summarize_scores


class WindowedTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        training._sumgrad_diag.update(csv_path=None, step=0, buffer=[])
        training._windowed_monitor = None

    def tokenizer(self):
        backend = Tokenizer(WordLevel({'<pad>': 0, '<eos>': 1, '<unk>': 2,
                                      'a': 3, 'b': 4, 'c': 5, 'd': 6, 'e': 7}, unk_token='<unk>'))
        return PreTrainedTokenizerFast(tokenizer_object=backend, pad_token='<pad>',
                                      eos_token='<eos>', unk_token='<unk>', padding_side='left')

    def test_scoring_and_schedule(self):
        self.assertEqual(len(schedule()), 19)
        self.assertEqual(schedule(2), [0, 2])
        self.assertEqual(score_values(0, 1)['conciseness'], 1)
        self.assertEqual(score_values(256, 1)['conciseness'], 0)
        self.assertEqual(score_values(500, 1)['conciseness'], 0)
        rows = [dict(generation_length=1, scoring_length=2, eos=True,
                     generation_cap_reached=False, scorer_truncated=False,
                     **score_values(length, sentiment)) for length, sentiment in [(0, 0), (256, 1)]]
        result = summarize_scores(rows)
        self.assertAlmostEqual(result['F'], .5)
        self.assertNotAlmostEqual(result['F'], .2)  # mean of response-wise minima

    def test_evaluation_restores_rng_modes_and_gradients_on_failure(self):
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
        model.train()
        model[0].eval()
        for p in model.parameters():
            p.grad = torch.ones_like(p)
        random.seed(4); np.random.seed(4); torch.manual_seed(4)
        states = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with self.assertRaises(RuntimeError):
            with isolated_evaluation(model):
                random.random(); np.random.rand(); torch.rand(3)
                raise RuntimeError('generation interrupted')
        self.assertEqual(random.getstate(), states[0])
        self.assertTrue(np.array_equal(np.random.get_state()[1], states[1][1]))
        self.assertTrue(torch.equal(torch.get_rng_state(), states[2]))
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        for p in model.parameters():
            self.assertTrue(torch.equal(p.grad, torch.ones_like(p)))

    def test_eos_and_cap(self):
        tok = self.tokenizer()
        row = response_record(tok, [0, 3], [4, 1, 0, 0], 4, 'full')
        self.assertEqual(row['generation_length'], 2)
        self.assertTrue(row['eos'])
        self.assertFalse(row['generation_cap_reached'])
        row = response_record(tok, [3], [4] * 4, 4, 'continuation')
        self.assertTrue(row['generation_truncated'])
        self.assertEqual(row['scored_text'], row['continuation'])

    def test_window_boundaries_and_projection(self):
        accelerator = SimpleNamespace(unwrap_model=lambda x: x, backward=lambda loss, **kw: loss.backward(**kw),
                                      is_main_process=True)
        g1, g2 = torch.tensor([1., 0.]), torch.tensor([-1., 1.])
        for lo, hi in ((1, 400), (401, 800)):
            training._pcgrad_window.update(start=lo, end=hi)
            for update in (399, 400, 401, 800, 801):
                policy = torch.nn.Linear(2, 1, bias=False)
                policy.weight.data.zero_()
                model = SimpleNamespace(policy=policy)
                opt = torch.optim.SGD(policy.parameters(), lr=.1)
                p = policy.weight.flatten()
                training._pcgrad_state['update'] = update
                training._windowed_pcgrad_compute_and_apply(model, accelerator, opt,
                    p.sum() * 0, p @ g1, p @ g2, p.sum() * 0,
                    SimpleNamespace(gradient_accumulation_steps=1, vf_coef=.1))
                projected = training.pcgrad_project_2(g1, g2)
                expected = sum(projected) if lo <= update <= hi else g1 + g2
                torch.testing.assert_close(policy.weight.flatten(), -.1 * expected)
        for left, right in ((g1, g2), (g1, g1), (g1, torch.zeros(2))):
            p1, p2 = training.pcgrad_project_2(left, right)
            self.assertGreaterEqual(torch.dot(p1, p2).item(), -1e-6)
            self.assertTrue(torch.isfinite(p1).all())
        training._sumgrad_diag['csv_path'] = str(self.root / 'gradients.csv')
        training._sumgrad_record_cosine(g1, torch.zeros(2), window_active=True)
        training._sumgrad_flush_cosine()
        with (self.root / 'gradients.csv').open() as f:
            row = next(csv.DictReader(f))
        self.assertEqual(row['valid_cosine'], 'False')
        self.assertEqual(row['cos_g1_g2'], '')
        self.assertEqual(row['projection_applied'], 'False')

    def test_patch_fails_on_version_mismatch(self):
        with patch.object(training.inspect, 'getsource', return_value='def train(self): pass'):
            with self.assertRaises(RuntimeError):
                training._patch_ppo_for_pcgrad()

    def test_offline_scoring_and_artifact_validation(self):
        from src.helper.windowed_experiment import analyze, validate_run
        tok = self.tokenizer()
        scorer_dir = self.root / 'scorer'
        tok.save_pretrained(scorer_dir)
        model = GPT2ForSequenceClassification(GPT2Config(vocab_size=8, n_positions=32,
            n_embd=8, n_layer=1, n_head=1, num_labels=2, pad_token_id=0))
        model.save_pretrained(scorer_dir)
        root = self.root / 'artifacts'
        panel = [dict(prompt_id=f'p{i}', input_ids=[3], prompt='a') for i in range(256)]
        write_jsonl(root / 'prompts/evaluation.jsonl', panel)
        write_jsonl(root / 'metrics/train_updates.jsonl', [{'update': 1}, {'update': 2}])
        for kind, updates, count in [('train', [1, 2], 64), ('eval', [0, 2], 256)]:
            for update in updates:
                write_jsonl(root / f'rollouts/{kind}/update_{update:04d}.jsonl',
                    [dict(prompt_id=f'p{i}', update=update, phase='pre_update' if kind == 'train' else 'post_update',
                          generation_length=2, eos=True, generation_cap_reached=False,
                          scored_text='a b', prompt='a', continuation='b') for i in range(count)])
        (root / 'TRAINING_COMPLETE').write_text('2')
        (root / 'final_policy').mkdir()
        (root / 'final_policy/config.json').write_text('{}')
        (root / 'diagnostics').mkdir()
        with (root / 'diagnostics/gradient_steps.csv').open('w') as f:
            writer = csv.writer(f)
            writer.writerow(['step', 'update', 'dot', 'window_active', 'conflict_detected', 'projection_applied'])
            for step in range(6):
                update = step // 3 + 1
                writer.writerow([step, update, -1, update == 2, True, update == 2])
        manifest = dict(arguments=dict(expected_updates=2, pcgrad_window_start=2, pcgrad_window_end=2),
                        inputs=dict(evaluation_hash=digest(panel), models={'scorer': {'id': str(scorer_dir), 'revision': None}}),
                        resolved_config=dict(batch_size=64, num_ppo_epochs=3, num_mini_batches=1),
                        evaluation_updates=[0, 2])
        (root / 'manifest.json').write_text(json.dumps(manifest))
        analyze(SimpleNamespace(run_dir=str(root), check_only=False, device='cpu', batch_size=64))
        self.assertTrue((root / 'figures/train.png').exists())
        self.assertTrue((root / 'checksums.json').exists())
        from src.imdb.ppo_training.windowed_monitor import read_jsonl
        self.assertEqual(len(read_jsonl(root / 'metrics/evaluation_response_scores.jsonl')), 512)
        (root / 'rollouts/eval/update_0002.jsonl').write_text('')
        with self.assertRaises(ValueError):
            validate_run(root, manifest)

    def test_comparison_checks_matching_configuration(self):
        from src.helper.windowed_experiment import compare
        roots = {}
        config = {key: 1 for key in ('seed', 'response_length', 'batch_size', 'num_total_batches',
                  'learning_rate', 'num_ppo_epochs', 'gradient_accumulation_steps', 'kl_coef',
                  'whiten_rewards', 'temperature')}
        for arm, start, end in [('early', 1, 400), ('late', 401, 800)]:
            root = self.root / arm
            roots[arm] = root
            root.mkdir()
            manifest = dict(inputs={'shared': True}, evaluation_updates=[0, 400, 800],
                            source_hashes={'code': 'same'}, resolved_config=config,
                            arguments=dict(expected_updates=800, pcgrad_window_start=start, pcgrad_window_end=end))
            (root / 'manifest.json').write_text(json.dumps(manifest))
            rows = []
            for update in (0, 400, 800):
                row = dict(update=update, F=.5, U=.5, length_gt256_rate=.1)
                row.update({key: {'mean': .6 if arm == 'early' else .4} for key in
                            ('generation_length', 'scoring_length', 'conciseness', 'sentiment', 'reward_min', 'reward_maj')})
                rows.append(row)
            for kind in ('train', 'evaluation'):
                write_jsonl(root / f'metrics/{kind}_oracle_updates.jsonl', rows)
        args = SimpleNamespace(early=str(roots['early']), late=str(roots['late']), output=str(self.root / 'comparison'))
        compare(args)
        report = json.loads((self.root / 'comparison/comparison.json').read_text())
        self.assertAlmostEqual(report['early_minus_late']['conciseness'], .2)
        self.assertIn('late_final_minus_switch', report)
        manifest['resolved_config']['seed'] = 44
        (roots['late'] / 'manifest.json').write_text(json.dumps(manifest))
        with self.assertRaises(ValueError):
            compare(args)

    def test_tiny_actual_ppo_monitoring_equivalence(self):
        """Real installed TRL loop, local random models; no downloads or GPU."""
        from transformers import set_seed
        torch.set_num_threads(1)
        original_train = training._experimental_ppo_module.PPOTrainer.train
        original_reward = training._experimental_ppo_module.get_reward
        self.addCleanup(setattr, training._experimental_ppo_module.PPOTrainer, 'train', original_train)
        self.addCleanup(setattr, training._experimental_ppo_module, 'get_reward', original_reward)
        training._patch_ppo_for_pcgrad(apply_fn=training._windowed_pcgrad_compute_and_apply)
        tokenizer = self.tokenizer()
        # Keep real evaluation code but shorten generation for this CPU smoke.
        from src.imdb.ppo_training import windowed_monitor
        real_config = windowed_monitor.GenerationConfig
        def short_config(**kw):
            kw['max_new_tokens'] = 2
            return real_config(**kw)
        data = Dataset.from_dict({'input_ids': [[3, 4]] * 16, 'attention_mask': [[1, 1]] * 16})
        final_states = []
        for monitored in (False, True):
            set_seed(43)
            config = GPT2Config(vocab_size=8, n_positions=16, n_embd=8, n_layer=1, n_head=1,
                                eos_token_id=1, pad_token_id=0, num_labels=1,
                                attn_pdrop=0, resid_pdrop=0, embd_pdrop=0)
            policy = GPT2LMHeadModel(config)
            ref = copy.deepcopy(policy)
            value = GPT2ForSequenceClassification(config)
            reward_base = GPT2ForSequenceClassification(config)
            reward = training.MaxMinRewardWrapper(reward_base, copy.deepcopy(reward_base.score), reward_base.score)
            args = training.PPOConfig(output_dir=str(self.root / str(monitored)), use_cpu=True,
                bf16=False, fp16=False, report_to=[], push_to_hub=False, total_episodes=16,
                per_device_train_batch_size=8, gradient_accumulation_steps=1,
                num_ppo_epochs=3, num_mini_batches=1, response_length=2, seed=43,
                num_sample_generations=0, whiten_rewards=True, stop_token='eos',
                save_strategy='no', disable_tqdm=True, learning_rate=3e-6)
            callbacks = []
            monitor = None
            training._pcgrad_window.update(start=2, end=2)
            if monitored:
                inputs = self.root / 'inputs'
                write_jsonl(inputs / 'evaluation.jsonl', [dict(prompt_id=digest([3, 4]), input_ids=[3, 4])])
                monitor = WindowedMonitor(self.root / 'run', inputs, tokenizer,
                    SimpleNamespace(expected_updates=2, seed=43, training_response_length=2, scoring_text='full'),
                    training._sumgrad_flush_cosine)
                callbacks.append(monitor)
                training._sumgrad_diag.update(csv_path=str(self.root / 'run/diagnostics/gradient_steps.csv'), step=0, buffer=[])
            training._windowed_monitor = monitor
            training._experimental_ppo_module.get_reward = training.get_pcgrad_reward
            trainer = training.PPOTrainer(args=args, model=policy, ref_model=ref,
                processing_class=tokenizer, reward_model=reward, value_model=value,
                train_dataset=data, eval_dataset=data, callbacks=callbacks)
            if monitor:
                monitor.trainer = trainer
            with patch.object(windowed_monitor, 'GenerationConfig', side_effect=short_config):
                trainer.train()
            final_states.append(copy.deepcopy(policy.state_dict()))
        for name in final_states[0]:
            torch.testing.assert_close(final_states[0][name], final_states[1][name], rtol=0, atol=0)
        self.assertTrue((self.root / 'run/rollouts/train/update_0001.jsonl').exists())
        self.assertTrue((self.root / 'run/rollouts/eval/update_0000.jsonl').exists())
        self.assertTrue((self.root / 'run/rollouts/eval/update_0002.jsonl').exists())
        with (self.root / 'run/metrics/train_updates.jsonl').open() as f:
            self.assertEqual([json.loads(line)['update'] for line in f], [1, 2])
        training._windowed_monitor = None
        training._sumgrad_diag['csv_path'] = None


if __name__ == '__main__':
    unittest.main()
