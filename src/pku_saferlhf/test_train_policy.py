"""Offline PPO integration tests, using tiny randomly initialized models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, GPT2ForSequenceClassification, PreTrainedTokenizerFast

from src.pku_saferlhf.policy_utils import RewardBundle, prepare_prompts, select_rewards
from src.pku_saferlhf.reward_utils import format_response


class PolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        vocab = {s: i for i, s in enumerate(["[UNK]", "[PAD]", "[EOS]", "User", ":", "Assistant",
            "question", "answer"] + [str(i) for i in range(12)])}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
            pad_token="[PAD]", eos_token="[EOS]", padding_side="left")
        cls.config = GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=1, n_embd=8,
            n_positions=64, num_labels=1, eos_token_id=2, pad_token_id=1, bos_token_id=2)
        cls.policy_path = cls.root / "policy"
        cls.tokenizer.save_pretrained(cls.policy_path)
        torch.manual_seed(42)
        policy = GPT2LMHeadModel(cls.config)
        policy.save_pretrained(cls.policy_path)
        cls.initial = {k: v.clone() for k, v in policy.state_dict().items()}
        cls.paths = {}
        for i, name in enumerate(("harmlessness", "helpfulness", "pooled")):
            torch.manual_seed(i + 50)
            model = GPT2ForSequenceClassification(cls.config)
            path = cls.root / name
            cls.paths[name] = str(path)
            model.save_pretrained(path)
            cls.tokenizer.save_pretrained(path)
            (path / "run_config.json").write_text(json.dumps({"group": name, "format": "plain", "max_length": 48}))
        cls.rows = [{"prompt": f"question {i}"} for i in range(8)]
        DatasetDict({"train": Dataset.from_list(cls.rows),
                     "test": Dataset.from_list([{"not_a_prompt": "test must never be used"}])}).save_to_disk(str(cls.root / "data"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prompt_selection(self):
        data, lookup, stats = prepare_prompts(self.rows + self.rows[:1], self.tokenizer, 24, "plain")
        self.assertEqual(len(data), 8)
        self.assertEqual(stats["unique_prompts"], 8)
        self.assertEqual(set(lookup.values()), {r["prompt"] for r in self.rows})
        with self.assertRaises(ValueError):
            prepare_prompts(self.rows, self.tokenizer, 1, "plain")

    def test_complete_independent_reward_models(self):
        paths = {k: self.paths[k] for k in ("harmlessness", "helpfulness")}
        bundle = RewardBundle(paths, torch.float32, "cpu")
        actual = bundle.score(["question 1", "question 2"], ["answer", "answer"], 1)
        for i, path in enumerate(paths.values()):
            model = GPT2ForSequenceClassification.from_pretrained(path).eval()
            strings = [format_response(self.tokenizer, f"question {n}", "answer", "plain")[0] for n in (1, 2)]
            inputs = self.tokenizer(strings, padding=True, return_tensors="pt")
            with torch.no_grad():
                expected = model(**inputs).logits.flatten()
            self.assertTrue(torch.allclose(actual[:, i], expected, atol=1e-6))
        self.assertFalse(any(p.requires_grad for p in bundle.parameters()))
        self.assertFalse(torch.equal(bundle.models[0].transformer.wte.weight,
                                     bundle.models[1].transformer.wte.weight))

    def test_batch_maxmin_not_elementwise_min(self):
        scores = torch.tensor([[0., 3.], [10., 4.]])
        chosen, group = select_rewards(scores, "maxmin")
        self.assertEqual(group, 1)
        self.assertTrue(torch.equal(chosen, torch.tensor([3., 4.])))

    def test_patch_rejects_unknown_trl(self):
        from src.imdb.ppo_training import unified_ppo_training as shared
        with patch.object(shared.inspect, "getsource", return_value="def train(self):\n    pass\n"):
            with self.assertRaises(RuntimeError):
                shared._patch_ppo_for_pcgrad(strict=True)

    def test_four_methods_and_lora_export(self):
        for method, mode in [("vanilla", "full"), ("maxmin", "full"),
                             ("sumgrad", "full"), ("pcgrad", "full"), ("pcgrad", "lora")]:
            with self.subTest(method=method, mode=mode):
                output = self.root / f"run-{method}-{mode}"
                cmd = [sys.executable, "-m", "src.pku_saferlhf.train_policy",
                       "--method", method, "--training_mode", mode,
                       "--policy_model", str(self.policy_path), "--dataset", str(self.root / "data"),
                       "--output_dir", str(output), "--pooled_rm", self.paths["pooled"],
                       "--harmlessness_rm", self.paths["harmlessness"], "--helpfulness_rm", self.paths["helpfulness"],
                       "--cpu", "--precision", "fp32", "--batch_size", "8", "--total_episodes", "8",
                       "--ppo_epochs", "1", "--max_prompt_length", "24", "--response_length", "4"]
                run = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                         "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"})
                self.assertEqual(run.returncode, 0, run.stdout + run.stderr)
                model = GPT2LMHeadModel.from_pretrained(output)
                self.assertTrue(any(not torch.equal(v, self.initial[k]) for k, v in model.state_dict().items()))
                self.assertTrue(all(torch.isfinite(p).all() for p in model.parameters()))
                self.assertFalse((output / "adapter_config.json").exists())
                state = json.loads((output / "trainer_state.json").read_text())
                self.assertEqual(state["global_step"], 1)
                trace = [json.loads(line) for line in (output / "reward_trace.jsonl").read_text().splitlines()]
                self.assertEqual(len(trace), 1)  # One full-rollout selection, not per scoring chunk.
                if method == "maxmin":
                    self.assertIn(trace[0]["selected_group"], ["harmlessness", "helpfulness"])
                if method in ("pcgrad", "sumgrad"):
                    self.assertGreater(len((output / "gradient_cosine.csv").read_text().splitlines()), 1)


if __name__ == "__main__":
    unittest.main()
