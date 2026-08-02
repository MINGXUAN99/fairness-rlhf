import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

import argparse
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.helper.model_utils import (
    configure_tokenizer,
    freeze_backbone as freeze_backbone_fn,
    build_lora_config,
    default_lora_target_modules,
    add_lora_cli_args,
    resolve_use_lora,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Maxmin Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, required=True, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for the scores (0.5-1.0).")
    parser.add_argument("--group", required=True, type=int, help="Group ID.")
    parser.add_argument("--preference_dataset_hub_id", type=str, default=None,
                        help="Hugging Face preference dataset hub id (e.g. from build_dataset.sh). If not set, uses {user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}.")
    parser.add_argument("--hub_model_id", type=str, required=True,
                        help="Hugging Face hub model id for pushing the trained reward model (e.g. {user_id}/maxmin-imdb-reward-ALPHA1.0-len512-minority-gpt2).")
    parser.add_argument("--model_hub_id", type=str, default="gpt2",
                        help="Hugging Face model id for the backbone (e.g. gpt2, lvwerra/gpt2-imdb, EleutherAI/pythia-160m, meta-llama/Llama-3.2-1B, Qwen/Qwen3-4B-Base). Default: gpt2.")
    add_lora_cli_args(parser)
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    alpha = args.alpha
    group_id = args.group

    # Dataset
    dataset_name = args.preference_dataset_hub_id or f"{user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}"
    try:
        dataset = load_dataset(dataset_name, split="train")
    except Exception as e:
        print(f"❌ Error loading dataset {dataset_name}: {e}")
        raise
    
    dataset = dataset.filter(lambda example: example["group_id"] == group_id)
    
    # Check if filtered dataset is empty
    if len(dataset) == 0:
        raise ValueError(f"No data found for group_id={group_id} in dataset {dataset_name}")
    
    dataset = dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Reward Params
    num_labels = 1
    backbone_model_id = args.model_hub_id
    output_hub_id = args.hub_model_id
    use_lora = resolve_use_lora(args.use_lora, backbone_model_id)
    # HF repo privacy: HF_PRIVATE=0 pushes public (needed when the private
    # storage quota is full). Defaults to private to preserve prior behavior.
    _hf_private = os.environ.get("HF_PRIVATE", "1") != "0"

    # Reward Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone_model_id)
    configure_tokenizer(tokenizer)

    # Reward Model. bf16 keeps a 4B backbone light enough for a single 32GB GPU.
    import torch
    model = AutoModelForSequenceClassification.from_pretrained(
        backbone_model_id, num_labels=num_labels,
        torch_dtype=torch.bfloat16, trust_remote_code=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    if use_lora:
        from peft import get_peft_model
        lora_config = build_lora_config(
            task_type="SEQ_CLS",
            r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=default_lora_target_modules(model),
            modules_to_save=["score"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    else:
        # Linear-probe reward model: freeze backbone, train only the score head.
        freeze_backbone_fn(model)

    # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=16,
        bf16=True,
        learning_rate=1e-4,
        num_train_epochs=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,  # Reduced evaluation frequency to save time (was 10)
        save_strategy="no",
        report_to=["wandb"],
        run_name=output_hub_id,
        push_to_hub=True,
        hub_model_id=output_hub_id,
        hub_private_repo=_hf_private,
        ddp_find_unused_parameters=False,
    )

    # Reward trainer
    trainer = RewardTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # Push to hub. For LoRA, merge adapters so downstream loaders (PPO MaxMin
    # wrapper, normalization) can use a plain AutoModelForSequenceClassification.
    if use_lora:
        merged = trainer.accelerator.unwrap_model(trainer.model).merge_and_unload()
        merged.config.pad_token_id = tokenizer.pad_token_id
        merged.push_to_hub(output_hub_id, private=_hf_private)
        tokenizer.push_to_hub(output_hub_id, private=_hf_private)
    else:
        trainer.push_to_hub(
            dataset_name=dataset_name,
        )

    # Both paths push a plain AutoModel; remove any stale PEFT adapter left over
    # from a prior run reusing this repo name (else it hijacks AutoModel loading).
    from src.helper.model_utils import cleanup_stale_hub_files
    cleanup_stale_hub_files(output_hub_id, ["adapter_config.json", "adapter_model.safetensors"])