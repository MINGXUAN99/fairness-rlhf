"""CPU-only synthetic correctness checks. No pretrained models or GPU required.

python -m unittest src.imdb.diagnostics.test_attribute_gradient_diagnostic -v
"""
import ast
import copy
import importlib.util
import inspect
import json
import linecache
import random
import tempfile
import textwrap
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from . import attribute_gradient_diagnostic as d
from . import attribute_gradient_runner as runner
from .attribute_training_probe import AttributeTrainingProbe, instrument_source, install_hooks


# Synthetic source fixture for the upstream loop's hook boundaries. Never run.
LOOP_FIXTURE = '''def train(self):
    args = self.args
    optimizer = self.optimizer
    generation_config = self.generation_config
    for update in range(1, args.num_total_batches + 1):
        with torch.no_grad():
            context_length = 8
            query_responses = sample()
            scores = score(query_responses)
            non_score_reward = -args.kl_coef * kl
            rewards = non_score_reward.clone()
            if args.whiten_rewards:
                rewards = masked_whiten(rewards)
            lastgaelam = 0
            advantages = rewards
            advantages = masked_whiten(advantages, ~padding_mask)
            advantages = torch.masked_fill(advantages, padding_mask, 0)
        for ppo_epoch_idx in range(args.num_ppo_epochs):
            for minibatch_idx in batches:
                for micro_batch_start in microbatches:
                    with accelerator.accumulate(model):
                        accelerator.backward(loss)
                        optimizer.step()
                        optimizer.zero_grad()
        mean_kl = kl.sum(1).mean()
        metrics = {}
        self.lr_scheduler.step()
    self.control = self.callback_handler.on_train_end(args, self.state, self.control)
'''

HAS_TORCH = importlib.util.find_spec("torch") is not None
if HAS_TORCH:
    import numpy as np
    import torch

    class CategoricalPolicy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.logits = torch.nn.Parameter(torch.zeros(3))
            self.dropout = torch.nn.Dropout(.5)

        def forward(self, input_ids, attention_mask=None, use_cache=False):
            return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, 3))


class BasicTests(unittest.TestCase):
    def test_first_eos_included_later_nonpadding_excluded(self):
        self.assertEqual(d.response_extent([4, 2, 5, 2], 2, 2), (2, "eos"))
        self.assertEqual(d.response_extent([2, 5, 2], 2, 2), (1, "eos"))
        self.assertEqual(d.response_extent([4, 0, 5], 2, 0), (1, "padding"))
        self.assertEqual(d.response_extent([4, 5], 2, 0), (2, "cap"))

    def test_canonical_lengths_and_termination(self):
        row = d.canonical_sample([0, 3], [4, 2, 5], eos_token_id=2, pad_token_id=0,
                                 prompt_id="a", sample_id="a:0", max_new_tokens=3)
        self.assertEqual(row["policy_length"], 1)
        self.assertEqual(row["action_count"], 2)
        self.assertEqual(row["response_tokens"], [4, 2])
        self.assertEqual(row["cap_hit"], 0)
        sampled_pad = d.canonical_sample([3], [0, 4, 2], eos_token_id=2, pad_token_id=0,
                                         prompt_id="a", sample_id="a:1", max_new_tokens=3)
        self.assertEqual(sampled_pad["response_tokens"], [0, 4, 2])
        self.assertEqual(sampled_pad["action_count"], 3)

    def test_loo_and_constant_reward(self):
        self.assertEqual(d.advantages([1, 3, 5, 9], ["a", "a", "b", "b"]), [-2, 2, -4, 4])
        self.assertEqual(d.advantages([2, 2], ["a", "a"]), [0, 0])
        with self.assertRaisesRegex(ValueError, "two"):
            d.advantages([1, 2], ["a", "b"])

    def test_variance_is_within_source_prompt(self):
        self.assertIsNone(d.within_prompt_variance([1, 2], ["a", "b"]))
        self.assertEqual(d.within_prompt_variance([1, 3, 10, 12], ["a", "a", "b", "b"]), 2.)

    def test_schedule(self):
        cfg = {"diagnostic": {"early_every": 25, "early_until": 300, "late_every": 100}}
        self.assertTrue(runner.due(0, cfg))
        self.assertTrue(runner.due(275, cfg))
        self.assertFalse(runner.due(325, cfg))
        self.assertTrue(runner.due(781, cfg, final=True))

    def test_example_schema_without_model_download(self):
        path = Path(__file__).resolve().parents[3]/"experiments/attribute_gradient_config.example.json"
        cfg = runner.load_config(path)
        self.assertEqual(len(runner.load_prompts(cfg["prompts_file"])), 4)

    def test_unsupported_trainer_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "Unsupported TRL"):
            instrument_source("def train(self):\n    pass\n")

    def test_hooks_leave_training_math_unchanged(self):
        class RemoveHooks(ast.NodeTransformer):
            def visit_Expr(self, node):
                if isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Attribute):
                    owner = node.value.func.value
                    if isinstance(owner, ast.Attribute) and owner.attr == "attribute_probe":
                        return None
                return node
        patched = ast.parse(instrument_source(LOOP_FIXTURE))
        restored = RemoveHooks().visit(patched)
        self.assertEqual(ast.dump(restored), ast.dump(ast.parse(LOOP_FIXTURE)))
        with self.assertRaisesRegex(RuntimeError, "exactly one"):
            instrument_source(LOOP_FIXTURE.replace('                        optimizer.step()', '                        optimizer.step()\n                        optimizer.step()'))

    def test_hooks_compose_with_existing_mean_kl_patch_and_live_globals(self):
        filename = "<attribute-test-original-train>"
        linecache.cache[filename] = (len(LOOP_FIXTURE), None, LOOP_FIXTURE.splitlines(True), filename)
        module = types.ModuleType("fixture")
        exec(compile(LOOP_FIXTURE, filename, "exec"), module.__dict__)
        module.PPOTrainer = type("PPOTrainer", (), {"train": module.train})
        install_hooks(module)
        path = Path(__file__).resolve().parents[1]/"ppo_training/unified_ppo_training.py"
        source = path.read_text()
        function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "_patch_ppo_kl_to_mean_per_token")
        namespace = dict(inspect=inspect, textwrap=textwrap, _experimental_ppo_module=module)
        exec(ast.get_source_segment(source, function), namespace)
        namespace["_patch_ppo_kl_to_mean_per_token"]()
        self.assertIs(module.PPOTrainer.train.__globals__, module.__dict__)
        self.assertIn("kl_mean_per_seq", module.PPOTrainer.train.__code__.co_varnames)
        module.get_reward = object()
        self.assertIs(module.PPOTrainer.train.__globals__["get_reward"], module.get_reward)


@unittest.skipUnless(HAS_TORCH, "Install diagnostic dependencies for CPU tensor tests")
class GradientTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.policy = CategoricalPolicy()
        self.samples = [dict(prompt_id="p", sample_id=str(i), prompt_tokens=[0], response_tokens=[i]) for i in range(3)]
        self.c = [1-510/512, 1-509/512, 1-511/512]
        self.s = [.1, .5, .9]

    def gradients(self, rewards=None, **kwargs):
        return d.attribute_gradients(self.policy, self.samples,
            rewards or {"conciseness": self.c, "sentiment": self.s}, pad_token_id=0,
            alpha=.75, baseline="zero", **kwargs)

    def test_analytic_policy_gradient_and_dominance(self):
        vectors = self.gradients(batch_size=1)
        expected = torch.tensor(self.s)/3 - sum(self.s)/9
        torch.testing.assert_close(vectors["sentiment"], expected)
        metrics = d.gradient_metrics(vectors, .75, vectors["sentiment"])
        self.assertLess(metrics["rho"], .02)
        self.assertGreater(metrics["cos_update__minority_oracle"], .999)
        self.assertLess(metrics["dot_update__conciseness"], 0)
        self.assertGreater(metrics["dot_update__minority_oracle"], 0)

    def test_mixture_reconstruction_and_microbatch_invariance(self):
        one, all_at_once = self.gradients(batch_size=1), self.gradients(batch_size=3)
        for name in one:
            torch.testing.assert_close(one[name], all_at_once[name])
        torch.testing.assert_close(one["minority_oracle"], .75*one["conciseness"]+.25*one["sentiment"])

    def test_shared_scale_preserves_ratio(self):
        raw, scaled = self.gradients(), self.gradients(shared_scale=5.)
        for name in raw:
            torch.testing.assert_close(raw[name]/5, scaled[name])
        self.assertAlmostEqual(d.gradient_metrics(raw, .75)["rho"], d.gradient_metrics(scaled, .75)["rho"], places=6)

    def test_constant_reward_zero_after_centering(self):
        vectors = d.attribute_gradients(self.policy, self.samples,
            {"conciseness": [2]*3, "sentiment": self.s}, pad_token_id=0, alpha=.75, baseline="loo")
        self.assertEqual(d.vector_norm(vectors["conciseness"]), 0)
        self.assertIsNone(d.cosine(vectors["conciseness"], vectors["sentiment"]))

    def test_temperature_matches_sampling_policy(self):
        with torch.no_grad():
            self.policy.logits.copy_(torch.tensor([.2, -.3, .5]))
        # Exact expectation by giving each enumerated action 3*p_i reward weight.
        temp = .6
        p = (self.policy.logits.detach()/temp).softmax(0)
        weighted = (3*p*torch.tensor(self.s)).tolist()
        vectors = self.gradients({"conciseness": weighted, "sentiment": weighted}, temperature=temp)
        expected = p*(torch.tensor(self.s)-(p*torch.tensor(self.s)).sum())/temp
        torch.testing.assert_close(vectors["sentiment"], expected)

    def test_diagnostic_preserves_modes_rng_and_training_gradients(self):
        self.policy.train()
        self.policy.dropout.eval()
        self.policy.logits.grad = torch.tensor([1., 2., 3.])
        grad_ref = self.policy.logits.grad
        torch_state, py_state, np_state = torch.get_rng_state(), random.getstate(), np.random.get_state()
        with d.isolated_diagnostic(self.policy, 123):
            torch.rand(4)
            random.random()
            np.random.rand()
            self.gradients()
        self.assertTrue(self.policy.training)
        self.assertFalse(self.policy.dropout.training)
        self.assertIs(self.policy.logits.grad, grad_ref)
        torch.testing.assert_close(grad_ref, torch.tensor([1., 2., 3.]))
        self.assertTrue(torch.equal(torch_state, torch.get_rng_state()))
        self.assertEqual(py_state, random.getstate())
        self.assertTrue(np.array_equal(np_state[1], np.random.get_state()[1]))

    def test_update_hooks_wait_for_accumulation_and_measure_actual_step(self):
        with tempfile.TemporaryDirectory() as temp:
            probe = AttributeTrainingProbe.__new__(AttributeTrainingProbe)
            probe.policy, probe.output = self.policy, Path(temp)
            probe.cfg = {"alpha": .75, "diagnostic": {"cosine_floor": 1e-10}}
            probe.before = None
            probe.pending = dict(vectors=self.gradients(), metadata={"stream": "synthetic_test"})
            optimizer = torch.optim.SGD(self.policy.parameters(), lr=.1)
            trainer = SimpleNamespace(accelerator=SimpleNamespace(sync_gradients=False))
            local = dict(optimizer=optimizer, ppo_epoch_idx=0, minibatch_idx=0)
            self.policy.logits.grad = -probe.pending["vectors"]["sentiment"].clone()
            probe.before_optimizer(trainer, local)
            self.assertIsNone(probe.before)
            trainer.accelerator.sync_gradients = True
            probe.before_optimizer(trainer, local)
            optimizer.step()
            probe.after_optimizer(trainer, local)
            row = json.loads((Path(temp)/"update_summary.jsonl").read_text())
            self.assertGreater(row["dot_update__minority_oracle"], 0)
            self.assertLess(row["dot_update__conciseness"], 0)
            self.assertIsNone(probe.pending)

    def test_actual_parameter_displacement(self):
        before = d.snapshot_parameters(self.policy)
        with torch.no_grad():
            self.policy.logits.add_(torch.tensor([.1, -.2, .3]))
        torch.testing.assert_close(d.parameter_displacement(self.policy, before), torch.tensor([.1, -.2, .3]))


@unittest.skipUnless(HAS_TORCH and importlib.util.find_spec("transformers"), "Requires offline diagnostic dependencies")
class OfflineRunnerTests(unittest.TestCase):
    def test_tiny_local_checkpoints_end_to_end(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification, PreTrainedTokenizerFast
        from .analyze_attribute_gradients import analyze
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            backend = Tokenizer(models.WordLevel({"[PAD]": 0, "[EOS]": 1, "[UNK]": 2,
                "movie": 3, "good": 4, "bad": 5}, unk_token="[UNK]"))
            backend.pre_tokenizer = pre_tokenizers.Whitespace()
            tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, pad_token="[PAD]", eos_token="[EOS]", unk_token="[UNK]")
            for name, labels in [("sentiment", 2), ("minority", 1), ("policy", None)]:
                config = GPT2Config(vocab_size=6, n_positions=16, n_layer=1, n_head=1, n_embd=8,
                                    pad_token_id=0, eos_token_id=1, bos_token_id=1, num_labels=labels or 1)
                model = GPT2LMHeadModel(config) if labels is None else GPT2ForSequenceClassification(config)
                model.save_pretrained(root/name)
                tokenizer.save_pretrained(root/name)
            source = Path(__file__).resolve().parents[3]/"experiments/attribute_gradient_config.example.json"
            cfg = json.loads(source.read_text())
            for key, name in [("sentiment_model", "sentiment"), ("minority_model", "minority"), ("policy_tokenizer", "policy")]:
                cfg[key]["path"] = str(root/name)
                cfg[key]["revision"] = "local"
            (root/"prompts.jsonl").write_text('{"id":"review1","text":"good movie"}\n{"id":"review2","text":"bad movie"}\n')
            cfg.update(device="cpu", output_dir=str(root/"result"), prompts_file=str(root/"prompts.jsonl"),
                       checkpoints=[dict(path=str(root/"policy"), revision="local", update=0)])
            cfg["sampling"].update(max_new_tokens=3, max_prompt_tokens=3, responses_per_prompt=2)
            cfg["diagnostic"].update(batch_size=2, blocks=2)
            cfg["reward"]["scorer_max_length"] = 8
            (root/"config.json").write_text(json.dumps(cfg))
            cfg = runner.load_config(root/"config.json")
            runner.run(cfg)
            tables = analyze(root/"result", csv_only=True)
            self.assertEqual(len(tables["gradient_summary"]), 3)
            self.assertTrue((root/"result/completed.json").is_file())
            self.assertTrue((root/"result/gradient_summary.csv").is_file())
            for row in tables["gradient_summary"]:
                self.assertLess(row["mixture_error_relative"], .03)
            with self.assertRaisesRegex(ValueError, "fresh"):
                runner.run(cfg)


if __name__ == "__main__":
    unittest.main()
