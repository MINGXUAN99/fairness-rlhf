import unittest
import tempfile

from src.pku_saferlhf.prepare_dataset import prepare_split, remove_train_test_overlap


def example(i, safe0=False, safe1=False):
    return {"prompt": f"prompt {i}", "response_0": f"answer {i} A",
            "response_1": f"answer {i} B", "better_response_id": 0,
            "safer_response_id": 0 if safe0 and not safe1 else 1,
            "is_response_0_safe": safe0, "is_response_1_safe": safe1}


class PreparationTests(unittest.TestCase):
    def test_assignment_and_preference_direction(self):
        rows = [example(i, i < 5, True) for i in range(10)]
        prepared, report = prepare_split(rows, 0.2, 42)
        self.assertEqual(len(prepared), len(rows))
        self.assertEqual(report["harmlessness"], 2)
        self.assertEqual(prepare_split(rows, 0.2, 42)[0], prepared)
        for row, result in zip(rows, prepared):
            if row["is_response_0_safe"] and row["is_response_1_safe"]:
                self.assertEqual(result["group_id"], 1)
            expected = row["safer_response_id"] if result["group_id"] == 0 else row["better_response_id"]
            self.assertEqual(result["chosen"], row[f"response_{expected}"])
            self.assertEqual(result["rejected"], row[f"response_{1 - expected}"])

    def test_both_unsafe_uses_safety_label(self):
        prepared, _ = prepare_split([example(0)], 1)
        self.assertEqual(prepared[0]["chosen"], "answer 0 B")

    def test_insufficient_eligible_pairs(self):
        with self.assertRaisesRegex(ValueError, "eligible"):
            prepare_split([example(0, True, True)], 1)

    def test_bad_labels(self):
        row = example(0)
        row["better_response_id"] = 2
        with self.assertRaises(ValueError):
            prepare_split([row])
        row = example(0, True, False)
        row["safer_response_id"] = 1
        with self.assertRaisesRegex(ValueError, "unsafe"):
            prepare_split([row])

    def test_overlap_removal(self):
        train = [example(0), example(1)]
        test = [dict(example(0), prompt=" prompt  0 ")]
        self.assertEqual(remove_train_test_overlap(train, test), [example(1)])

    def test_proportion_and_rounding(self):
        with self.assertRaises(ValueError):
            prepare_split([], -0.1)
        _, counts = prepare_split([example(i) for i in range(7)], 0.2)
        self.assertEqual(counts["harmlessness"], 1)

    def test_huggingface_disk_roundtrip(self):
        try:
            from datasets import Dataset, DatasetDict, load_from_disk
        except ImportError:
            self.skipTest("Install datasets to check disk serialization")
        records, _ = prepare_split([example(i) for i in range(10)])
        dataset = DatasetDict({"train": Dataset.from_list(records),
                               "test": Dataset.from_list(records[:2])})
        with tempfile.TemporaryDirectory() as path:
            dataset.save_to_disk(path)
            restored = load_from_disk(path)
            self.assertEqual(set(restored), {"train", "test"})
            self.assertEqual(restored["train"].to_list(), records)
            self.assertEqual(restored["train"].features["group_id"].dtype, "int64")


if __name__ == "__main__":
    unittest.main()
