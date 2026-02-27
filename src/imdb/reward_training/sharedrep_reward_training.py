import sys, os
FILE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(FILE_DIR, "../../.."))
sys.path.insert(0, ROOT)

from trl import RewardConfig
from accelerate import Accelerator
from transformers import AutoTokenizer
from datasets import load_dataset
from src.models.sr_gpt2.sr_gpt2_modeling import SharedRepGPT2RM
from src.helper.sharedrep_reward_trainer import SharedRepRMTrainer, SharedRepRewardDataCollatorWithPadding

import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Train SharedRep Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, required=True, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for the scores (0.5-1.0).")
    parser.add_argument("--k", type=int, required=True, help="Inner hidden dimension size.")
    parser.add_argument("--hub_model_id", type=str, required=True,
                        help="Hugging Face hub model id for pushing the trained reward model (e.g. {user_id}/sharedrep-imdb-reward-ALPHA1.0-len512-k64).")
    parser.add_argument("--model_hub_id", type=str, default="gpt2",
                        help="Hugging Face model id for the backbone (e.g. gpt2, lvwerra/gpt2-imdb). Default: gpt2.")
    return parser.parse_args()


if __name__ == "__main__":

    accelerator = Accelerator()
    
    # Arguments
    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    alpha = args.alpha
    k = args.k

    # Dataset
    dataset_num_proc = 8
    dataset_batch_size = 128
    dataset_name = f"{user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}"
    with accelerator.main_process_first():
        dataset = load_dataset(dataset_name, split="train")
        dataset = dataset.train_test_split(test_size=0.05, seed=seed)

    # Reward Params
    n_groups=2
    use_tanh = True
    backbone_model_id = args.model_hub_id
    freeze_backbone = True
    output_hub_id = args.hub_model_id
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone_model_id)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token  

    # Reward Model
    model = SharedRepGPT2RM.from_pretrained(backbone_model_id, k=k, n_heads=n_groups)
    model.disable_maxmin()
    if freeze_backbone: model.freeze_backbone()
    model.unfreeze_heads()
    model.config.pad_token_id = tokenizer.pad_token_id

    # Create the data collator
    data_collator = SharedRepRewardDataCollatorWithPadding(
        tokenizer=tokenizer,
        padding=True,
        return_tensors="pt",
    )

        # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=16,
        learning_rate=5e-4,
        num_train_epochs=4,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,  # Reduced evaluation frequency to save time (was missing, default was too frequent)
        save_strategy="no",
        report_to=["wandb"],
        run_name=output_hub_id,
        push_to_hub=True,
        hub_model_id=output_hub_id,
        hub_private_repo=True,
    )

    # Trainer
    trainer = SharedRepRMTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        use_tanh=use_tanh,
        data_collator=data_collator,
    )
    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    trainer.push_to_hub(
        dataset_name=dataset_name,
    )





