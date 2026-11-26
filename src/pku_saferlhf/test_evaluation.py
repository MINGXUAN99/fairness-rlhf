"""Offline end-to-end evaluation checks with tiny random policy/Beaver models."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import torch
from datasets import Dataset, DatasetDict
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import GPT2Config, GPT2LMHeadModel, LlamaConfig, PreTrainedTokenizerFast

from src.pku_saferlhf.beaver_scorer import BeaverScoreModel, load_beaver
from src.pku_saferlhf.evaluation_utils import prompt_id, validate_records
from src.pku_saferlhf.summarize_results import summarize


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        words = ["[UNK]", "[PAD]", "[EOS]", "User", "Assistant", ":", "question", "answer",
                 "BEGINNING", "OF", "CONVERSATION", "USER", "ASSISTANT"] + [str(i) for i in range(8)]
        vocab = {word: i for i, word in enumerate(words)}
        tok = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        tok.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="[UNK]",
            pad_token="[PAD]", eos_token="[EOS]")
        cls.policy = cls.root / "policy"
        cls.tokenizer.save_pretrained(cls.policy)
        GPT2LMHeadModel(GPT2Config(vocab_size=len(vocab), n_layer=1, n_head=1, n_embd=8,
            n_positions=64, eos_token_id=2, pad_token_id=1, bos_token_id=2)).save_pretrained(cls.policy)
        cls.scorers = []
        for name in ("reward", "cost"):
            config = LlamaConfig(vocab_size=len(vocab), hidden_size=8, intermediate_size=16,
                num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                max_position_embeddings=64, pad_token_id=1, score_dim=1, score_bias=True,
                do_normalize=False)
            scorer = BeaverScoreModel(config)
            # Official v1.0 files retain unused normalization buffers.
            scorer.normalizer = torch.nn.Module()
            scorer.normalizer.register_buffer("mean", torch.zeros(1))
            scorer.normalizer.register_buffer("var", torch.ones(1))
            scorer.normalizer.register_buffer("count", torch.tensor(0))
            path = cls.root / name
            scorer.save_pretrained(path)
            cls.tokenizer.save_pretrained(path)
            cls.scorers.append(path)
        data = DatasetDict({"train": Dataset.from_list([{"prompt": "question 7"}]),
            "test": Dataset.from_list([{"prompt": f"question {i}"} for i in range(4)] + [{"prompt": "question 0"}])})
        data.save_to_disk(str(cls.root / "data"))
        (cls.root / "runs.json").write_text(json.dumps([
            {"run_id": "initial", "method": "initial", "training_seed": None, "model": str(cls.policy)},
            {"run_id": "vanilla-42", "method": "vanilla", "training_seed": 42, "model": str(cls.policy)}]))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def command(self, module, args, success=True):
        result = subprocess.run([sys.executable, "-m", f"src.pku_saferlhf.{module}", *args],
            capture_output=True, text=True, timeout=90,
            env={**os.environ, "HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
        if success:
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        else:
            self.assertNotEqual(result.returncode, 0)
        return result

    def test_complete_generation_scoring_summary(self):
        generation, scored, summary = [self.root / s for s in ("generation", "scored", "summary")]
        self.command("generate_responses", ["--policies", str(self.root / "runs.json"), "--dataset", str(self.root / "data"),
            "--output_dir", str(generation), "--device", "cpu", "--precision", "fp32", "--max_prompt_length", "24",
            "--max_new_tokens", "4", "--batch_size", "2"])
        manifest = json.loads((generation / "manifest.json").read_text())
        records = [json.loads(line) for line in (generation / "responses.jsonl").read_text().splitlines()]
        self.assertEqual(len(records), 8)
        self.assertEqual(len(manifest["prompt_ids"]), 4)
        self.assertNotIn("question 7", {r["prompt"] for r in records})
        common = ["--generation_dir", str(generation), "--helpfulness_model", str(self.scorers[0]),
                  "--cost_model", str(self.scorers[1]), "--device", "cpu", "--precision", "fp32"]
        self.command("evaluate_responses", common + ["--output_dir", str(scored), "--max_length", "48", "--batch_size", "3"])
        self.command("summarize_results", ["--scored_dir", str(scored), "--output_dir", str(summary),
            "--reference_run", "initial", "--bootstrap_samples", "50"])
        result = json.loads((summary / "summary.json").read_text())
        self.assertEqual(len(result["per_run"]), 2)
        for row in result["paired_differences"]:
            # Identical policies with greedy generation must have zero deltas.
            self.assertAlmostEqual(row["helpfulness_reward_delta_mean"], 0, places=6)
            self.assertAlmostEqual(row["harmfulness_cost_delta_mean"], 0, places=6)
        for row in result["per_method"]:
            self.assertIsNone(row["helpfulness_reward_std_across_runs"])
        failed = self.command("evaluate_responses", common + ["--output_dir", str(self.root / "too-long"), "--max_length", "2"], success=False)
        self.assertIn("exceeds", failed.stderr)
        self.command("evaluate_responses", common + ["--output_dir", str(self.root / "truncated"), "--max_length", "2", "--allow_truncation"])
        cut = json.loads((self.root / "truncated" / "manifest.json").read_text())
        self.assertEqual(cut["evaluators"]["harmfulness_cost"]["truncated_responses"], 8)

    def test_beaver_head_and_end_pooling(self):
        model = load_beaver(str(self.scorers[0]), None, torch.float32, "cpu")
        ids = torch.tensor([[3, 7, 2, 1], [3, 6, 7, 2]])
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
        with torch.no_grad():
            all_scores = model.score_head(model.model(ids, attention_mask=mask).last_hidden_state)
            expected = all_scores[torch.arange(2), torch.tensor([2, 3])]
            actual = model(ids, attention_mask=mask).end_scores
        self.assertTrue(torch.allclose(actual, expected))
        self.assertFalse(any(p.requires_grad for p in model.parameters()))
        config = LlamaConfig(score_dim=1, do_normalize=True)
        with self.assertRaises(ValueError):
            BeaverScoreModel(config)

    def test_missing_head_rejected(self):
        from safetensors.torch import load_file, save_file
        import shutil
        broken = self.root / "broken"
        shutil.copytree(self.scorers[0], broken)
        weights = load_file(str(broken / "model.safetensors"))
        del weights["score_head.weight"]
        save_file(weights, str(broken / "model.safetensors"), metadata={"format": "pt"})
        with self.assertRaisesRegex(ValueError, "Incompatible"):
            load_beaver(str(broken), None, torch.float32, "cpu")

    def test_pairing_and_score_direction(self):
        runs = [{"run_id": "base", "method": "initial", "training_seed": None},
                {"run_id": "p42", "method": "pcgrad", "training_seed": 42},
                {"run_id": "p43", "method": "pcgrad", "training_seed": 43}]
        manifest = {"status": "complete", "runs": runs, "prompt_ids": [prompt_id("a"), prompt_id("b")]}
        records = [{"run_id": run["run_id"], "prompt_id": prompt_id(p), "prompt": p, "response": "x",
                    "helpfulness_reward": i, "harmfulness_cost": -i, "generated_tokens": 1}
                   for i, run in enumerate(runs) for p in ("a", "b")]
        result = summarize(records, manifest, "base", samples=50)
        self.assertEqual(result["paired_differences"][1]["helpfulness_reward_delta_mean"], 1)
        self.assertEqual(result["paired_differences"][1]["harmfulness_cost_delta_mean"], -1)
        for bad in (records[:-1], records + records[:1]):
            with self.assertRaises(ValueError):
                validate_records(bad, manifest)
        with self.assertRaises(ValueError):
            validate_records(records, {**manifest, "status": "running"})


if __name__ == "__main__":
    unittest.main()
