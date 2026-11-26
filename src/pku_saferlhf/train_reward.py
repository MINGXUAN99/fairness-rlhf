"""Train a helpfulness, harmlessness, or pooled PKU reward model."""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="mxcui/pku-saferlhf-preferences")
    p.add_argument("--dataset_revision", default="4730d5ec7e69f2d42ab3ffc14dab4ea9eae4119f")
    p.add_argument("--group", required=True, choices=["helpfulness", "harmlessness", "pooled"])
    p.add_argument("--model", required=True, help="Base model Hub ID or local directory")
    p.add_argument("--model_revision", default="main")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--training_mode", choices=["head", "lora", "full"], default="head")
    p.add_argument("--format", choices=["auto", "chat", "plain"], default="auto")
    p.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--eval_batch_size", type=int, default=8)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--epochs", type=float, default=4)
    p.add_argument("--max_steps", type=int, default=-1, help="Override epochs; useful for smoke tests")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--report_to", default="none", choices=["none", "wandb"])
    args = p.parse_args()
    if args.cpu and args.precision != "fp32":
        p.error("Use --precision fp32 with --cpu")
    if min(args.max_length, args.batch_size, args.eval_batch_size, args.gradient_accumulation_steps) < 1:
        p.error("Lengths and batch sizes must be positive")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        p.error("Output directory must be empty; use a new directory for each run")
    return args


def main():
    args = parse_args()
    import torch
    import transformers
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, set_seed
    from src.helper.model_utils import configure_tokenizer, freeze_backbone, build_lora_config, default_lora_target_modules
    from src.pku_saferlhf.reward_utils import select_group, tokenize_pair
    from src.pku_saferlhf.reward_trainer import PairCollator, PairwiseRewardTrainer, pairwise_accuracy

    set_seed(args.seed)
    if Path(args.dataset).is_dir():
        dataset = load_from_disk(args.dataset)
        dataset_revision = "local"
    else:
        from huggingface_hub import HfApi
        dataset_revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
        dataset = load_dataset(args.dataset, revision=dataset_revision)
    if not {"train", "test"}.issubset(dataset):
        raise ValueError("Dataset must contain train and test splits")
    for split in ("train", "test"):
        required = {"prompt", "chosen", "rejected", "group_id"}
        if not required.issubset(dataset[split].column_names):
            raise ValueError(f"Missing columns in {split}: {required - set(dataset[split].column_names)}")
        if not set(dataset[split]["group_id"]).issubset({0, 1}):
            raise ValueError("Expected group_id 0 (harmlessness) or 1 (helpfulness)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=args.model_revision)
    configure_tokenizer(tokenizer, padding_side="left")
    tokenizer.truncation_side = "right"
    processed, counts = {}, {}
    for split in ("train", "test"):
        selected = select_group(dataset[split], args.group)
        tokens = selected.map(lambda row: tokenize_pair(row, tokenizer, args.max_length, args.format),
                              remove_columns=selected.column_names, load_from_cache_file=False)
        usable = tokens.filter(lambda row: row["usable"])
        counts[split] = {"source_rows": len(selected), "usable_rows": len(usable),
                         "indistinguishable_or_empty_pairs_removed": len(tokens) - len(usable),
                         "group_counts": {str(g): sum(x == g for x in usable["group_id"]) for g in (0, 1)}}
        if not len(usable):
            raise ValueError(f"No usable {args.group} pairs in {split}")
        processed[split] = usable
    print(json.dumps(counts, indent=2))
    dtype = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[args.precision]
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model, revision=args.model_revision, num_labels=1, torch_dtype=dtype,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model_revision = getattr(model.config, "_commit_hash", None)
    if args.training_mode == "head":
        freeze_backbone(model)
    elif args.training_mode == "lora":
        from peft import get_peft_model
        if not hasattr(model, "score"):
            raise ValueError("LoRA currently requires a decoder classifier with a score head")
        model = get_peft_model(model, build_lora_config(
            "SEQ_CLS", r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout, target_modules=default_lora_target_modules(model),
            modules_to_save=["score"],
        ))
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    run = {**vars(args), "resolved_dataset_revision": dataset_revision,
           "resolved_model_revision": model_revision, "counts": counts,
           "transformers_version": transformers.__version__, "torch_version": torch.__version__,
           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)}
    (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    tokenizer.save_pretrained(output)
    trainer = PairwiseRewardTrainer(
        model=model, processing_class=tokenizer, data_collator=PairCollator(tokenizer),
        train_dataset=processed["train"], compute_metrics=pairwise_accuracy,
        args=TrainingArguments(
            output_dir=str(output), seed=args.seed, data_seed=args.seed,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.eval_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate, num_train_epochs=args.epochs,
            max_steps=args.max_steps, bf16=args.precision == "bf16", fp16=args.precision == "fp16",
            use_cpu=args.cpu, eval_strategy="no", save_strategy="no", logging_steps=10,
            report_to=args.report_to, remove_unused_columns=False,
            dataloader_pin_memory=not args.cpu,
        ),
    )
    result = trainer.train()
    trainer.save_metrics("train", result.metrics)
    metrics = trainer.evaluate(processed["test"], metric_key_prefix="test")
    if args.group == "pooled":
        for group in ("helpfulness", "harmlessness"):
            subset = select_group(processed["test"], group)
            if len(subset):
                metrics.update(trainer.evaluate(subset, metric_key_prefix=f"test_{group}"))
    trainer.save_metrics("test", metrics)
    trainer.log_metrics("test", metrics)
    # Export complete models so downstream inference does not depend on adapters.
    if args.training_mode == "lora":
        model = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
        if trainer.is_world_process_zero():
            model.save_pretrained(output, safe_serialization=True)
    else:
        trainer.save_model(output)
    print(f"Saved final {args.group} reward model and test metrics to {output}")


if __name__ == "__main__":
    main()
