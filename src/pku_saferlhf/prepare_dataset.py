"""Prepare one group-labeled PKU preference dataset without model inference."""

import argparse
import json
import random
from pathlib import Path


def prepare_split(rows, proportion=0.2, seed=42):
    """Assign each pair once: 0=harmlessness, 1=helpfulness.

    Sample group 0 only from pairs that are not both safe. Assign the remainder
    to group 1, preserving all prompts and the requested ratio (rounded down).
    Group assignment is therefore conditional on safety.
    """
    if not 0 <= proportion <= 1:
        raise ValueError("proportion must be between 0 and 1")
    rows = list(rows)
    eligible = []
    for i, row in enumerate(rows):
        for field in ("prompt", "response_0", "response_1"):
            if not isinstance(row.get(field), str):
                raise ValueError(f"Row {i}: {field} must be a string")
        for field in ("better_response_id", "safer_response_id"):
            if type(row.get(field)) is not int or row[field] not in (0, 1):
                raise ValueError(f"Row {i}: {field} must be 0 or 1")
        for field in ("is_response_0_safe", "is_response_1_safe"):
            if type(row.get(field)) is not bool:
                raise ValueError(f"Row {i}: {field} must be boolean")
        safe0, safe1 = row["is_response_0_safe"], row["is_response_1_safe"]
        if safe0 != safe1 and row["safer_response_id"] != (0 if safe0 else 1):
            raise ValueError(f"Row {i}: safer_response_id prefers the unsafe response")
        if not (safe0 and safe1):
            eligible.append(i)
    n_safety = int(len(rows) * proportion)
    if n_safety > len(eligible):
        raise ValueError(f"Requested {n_safety} harmlessness rows but only {len(eligible)} "
                         "are eligible; reduce --proportion. No oversampling is performed.")
    safety_indices = set(random.Random(seed).sample(eligible, n_safety))
    output = []
    for i, row in enumerate(rows):
        group = 0 if i in safety_indices else 1
        chosen_id = row["safer_response_id" if group == 0 else "better_response_id"]
        output.append({"prompt": row["prompt"], "chosen": row[f"response_{chosen_id}"],
                       "rejected": row[f"response_{1 - chosen_id}"], "group_id": group})
    return output, {
        "rows": len(rows), "harmlessness": n_safety, "helpfulness": len(rows) - n_safety,
        "both_safe": len(rows) - len(eligible), "safety_eligible": len(eligible),
        "actual_harmlessness_proportion": n_safety / len(rows) if rows else None,
        "label_disagreements": sum(r["better_response_id"] != r["safer_response_id"] for r in rows),
    }


def remove_train_test_overlap(train, test):
    """Keep official test unchanged; remove matching prompts from train."""
    test_prompts = {" ".join(row["prompt"].split()) for row in test}
    return [row for row in train if " ".join(row["prompt"].split()) not in test_prompts]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="PKU-Alignment/PKU-SafeRLHF")
    parser.add_argument("--config", default="default")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--proportion", type=float, default=0.2,
                        help="Harmlessness fraction; helpfulness gets the remainder.")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--cache_dir", default="tmp/hf-datasets")
    parser.add_argument("--output_hub_id", help="Also upload to this single HF dataset repository.")
    parser.add_argument("--public", action="store_true", help="Create a public HF repository (default: private).")
    args = parser.parse_args()
    if not 0 <= args.proportion <= 1:
        parser.error("--proportion must be between 0 and 1")

    from datasets import Dataset, DatasetDict, Features, Value, load_dataset
    from huggingface_hub import DatasetCard, HfApi

    output_dir = Path(args.output_dir or f"output/pku-preferences-seed{args.seed}-prop{args.proportion}")
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error(f"Output directory is not empty: {output_dir}; use a new directory")
    revision = HfApi().dataset_info(args.source, revision=args.revision).sha
    source = load_dataset(args.source, name=args.config, revision=revision, cache_dir=args.cache_dir)
    if not {"train", "test"}.issubset(source):
        raise ValueError("Source must have official train and test splits")
    train_rows = remove_train_test_overlap(source["train"], source["test"])
    features = Features({"prompt": Value("string"), "chosen": Value("string"),
                         "rejected": Value("string"), "group_id": Value("int64")})
    prepared = DatasetDict()
    report = {"source": args.source, "config": args.config, "revision": revision,
              "seed": args.seed, "requested_harmlessness_proportion": args.proportion,
              "group_mapping": {"0": "harmlessness", "1": "helpfulness"},
              "train_rows_removed_for_test_prompt_overlap": len(source["train"]) - len(train_rows),
              "splits": {}}
    for split, rows in (("train", train_rows), ("test", source["test"])):
        records, counts = prepare_split(rows, args.proportion, args.seed)
        if not records:
            raise ValueError(f"{split} has no rows")
        prepared[split] = Dataset.from_list(records, features=features)
        report["splits"][split] = counts
    prepared.save_to_disk(str(output_dir))
    (output_dir / "preparation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    card = DatasetCard(f"""---
license: cc-by-nc-4.0
language:
- en
task_categories:
- text-generation
---
# PKU SafeRLHF grouped preferences

Derived from [{args.source}](https://huggingface.co/datasets/{args.source}),
configuration `{args.config}`, revision `{revision}`.

One dataset with train/test splits and prompt, chosen, rejected, group_id columns.
Chosen/rejected are response text only: training must include the prompt.
Group 0 follows harmlessness; group 1 follows helpfulness. Each pair is assigned
once. Both-safe pairs go to helpfulness and provide no safety ranking supervision.
Group 0 is sampled without replacement from eligible pairs; all remaining pairs
become group 1. The group prompt distributions consequently differ. Group 0 counts
are rounded down. Official test is preserved; train rows with whitespace-normalized
prompts appearing in test are removed. No validation split or reward calibration.

```json
{json.dumps(report, indent=2)}
```
""")
    card.save(str(output_dir / "README.md"))
    print(json.dumps(report, indent=2))
    print(f"Saved dataset to {output_dir}")
    if args.output_hub_id:
        prepared.push_to_hub(args.output_hub_id, private=not args.public)
        # Preserve split/schema metadata generated by datasets.push_to_hub.
        hub_card = DatasetCard.load(args.output_hub_id)
        hub_card.text = card.text
        for key, value in card.data.to_dict().items():
            hub_card.data[key] = value
        hub_card.push_to_hub(args.output_hub_id)
        print(f"Uploaded to https://huggingface.co/datasets/{args.output_hub_id}")


if __name__ == "__main__":
    main()
