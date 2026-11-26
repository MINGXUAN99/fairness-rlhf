"""Offline CPU tests using a tiny random model, never a downloaded checkpoint."""
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
from transformers import GPT2Config, GPT2ForSequenceClassification, PreTrainedTokenizerFast

from src.pku_saferlhf.reward_utils import tokenize_pair
from src.pku_saferlhf.reward_trainer import PairCollator, PairwiseRewardTrainer


class RewardTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        vocab = {t: i for i, t in enumerate(
            ["[UNK]", "[PAD]", "[EOS]", "User", ":", "Assistant", "question", "good", "bad"])}
        backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
        backend.pre_tokenizer = Whitespace()
        cls.tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend,
            unk_token="[UNK]", pad_token="[PAD]", eos_token="[EOS]", padding_side="left")
        cls.model_path = cls.root / "base"
        cls.tokenizer.save_pretrained(cls.model_path)
        model = GPT2ForSequenceClassification(GPT2Config(vocab_size=len(vocab), n_layer=1,
            n_head=1, n_embd=8, n_positions=64, num_labels=1, pad_token_id=1))
        model.save_pretrained(cls.model_path)
        cls.initial_weights = {k: v.clone() for k, v in model.state_dict().items()}
        cls.row = {"prompt": "question", "chosen": "good", "rejected": "bad", "group_id": 1}
        train = [dict(cls.row, group_id=i % 2) for i in range(8)]
        test = [dict(cls.row, group_id=i % 2) for i in range(5)]
        DatasetDict({"train": Dataset.from_list(train), "test": Dataset.from_list(test)}).save_to_disk(str(cls.root / "data"))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_prompt_formatting_and_truncation(self):
        pair = tokenize_pair(self.row, self.tokenizer, 32, "plain")
        self.assertTrue(pair["usable"])
        self.assertIn("question", self.tokenizer.decode(pair["chosen_ids"]))
        self.assertFalse(tokenize_pair(self.row, self.tokenizer, 2, "plain")["usable"])
        with self.assertRaises(ValueError):
            tokenize_pair(self.row, self.tokenizer, 32, "chat")
        old = self.tokenizer.chat_template
        self.tokenizer.chat_template = "{% for m in messages %}{{m['role'] + ': ' + m['content'] + ' '}}{% endfor %}"
        try:
            chat = tokenize_pair(self.row, self.tokenizer, 32, "chat")
            self.assertIn("question", self.tokenizer.decode(chat["chosen_ids"]))
            self.assertTrue(chat["usable"])
        finally:
            self.tokenizer.chat_template = old

    def test_loss_direction_and_collator(self):
        pair = tokenize_pair(self.row, self.tokenizer, 32, "plain")
        batch = PairCollator(self.tokenizer)([pair, pair])
        self.assertEqual(batch["input_ids"].shape[0], 4)
        self.assertEqual(batch["labels"].shape[0], 2)

        class Scores:
            def __call__(self, **inputs):
                from types import SimpleNamespace
                return SimpleNamespace(logits=torch.tensor([[2.], [2.], [0.], [0.]], requires_grad=True))
        loss = PairwiseRewardTrainer.compute_loss(None, Scores(), batch)
        self.assertAlmostEqual(loss.item(), torch.nn.functional.softplus(torch.tensor(-2.)).item())

    def test_all_groups_and_exports(self):
        for group, mode in [("helpfulness", "head"), ("harmlessness", "head"),
                            ("pooled", "head"), ("pooled", "lora"), ("pooled", "full")]:
            with self.subTest(group=group, mode=mode):
                output = self.root / f"{group}-{mode}"
                cmd = [sys.executable, "-m", "src.pku_saferlhf.train_reward",
                       "--dataset", str(self.root / "data"), "--model", str(self.model_path),
                       "--group", group, "--training_mode", mode, "--output_dir", str(output),
                       "--precision", "fp32", "--cpu", "--max_steps", "2", "--max_length", "32",
                       "--batch_size", "2", "--eval_batch_size", "2", "--gradient_accumulation_steps", "2"]
                result = subprocess.run(cmd, text=True, capture_output=True, timeout=90,
                    env={**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false",
                         "HF_HUB_OFFLINE": "1", "WANDB_MODE": "disabled"})
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                metrics = json.loads((output / "test_results.json").read_text())
                self.assertIn("test_accuracy", metrics)
                if group == "pooled":
                    self.assertIn("test_helpfulness_accuracy", metrics)
                    self.assertIn("test_harmlessness_accuracy", metrics)
                config = json.loads((output / "run_config.json").read_text())
                self.assertEqual(config["counts"]["train"]["usable_rows"], 8 if group == "pooled" else 4)
                loaded = GPT2ForSequenceClassification.from_pretrained(output)
                self.assertFalse((output / "adapter_config.json").exists())
                if mode == "head":
                    for key, value in loaded.state_dict().items():
                        if key.startswith("transformer."):
                            self.assertTrue(torch.equal(value, self.initial_weights[key]), key)
                    self.assertFalse(torch.equal(loaded.score.weight, self.initial_weights["score.weight"]))


if __name__ == "__main__":
    unittest.main()
