import argparse
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    """
    Parses command line arguments for vanilla (baseline) reward model training.
    
    Returns:
        argparse.Namespace: Parsed arguments for training a single reward model on all data
    """
    parser = argparse.ArgumentParser(description="Train Vanilla (Baseline) Reward Model on all data without group considerations.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, default=0.2, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--alpha", type=float, default=1.0, help="Alpha value for the scores (0.5-1.0).")
    parser.add_argument("--preference_dataset_hub_id", type=str, default=None,
                        help="Hugging Face preference dataset hub id (e.g. from build_dataset.sh). If not set, uses {user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hugging Face hub model id for pushing the reward model. If not set, uses {user_id}/vanilla-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}.")
    return parser.parse_args()

if __name__ == "__main__":

    args = parse_args()
    user_id = args.user_id
    seed = args.seed
    proportion = args.proportion
    alpha = args.alpha

    # Dataset - Load all data without filtering by group
    dataset_name = args.preference_dataset_hub_id or f"{user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}"
    dataset = load_dataset(dataset_name, split="train")
    # No group filtering - train on all data
    dataset = dataset.train_test_split(test_size=0.05, seed=seed)
    train_dataset = dataset["train"]
    eval_dataset = dataset["test"]

    # Reward Params
    num_labels = 1
    model_name = "lvwerra/gpt2-imdb"
    freeze_backbone = True
    model_hub_id = args.hub_model_id or f"{user_id}/vanilla-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}"

    # Reward Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Freeze Backbone
    if freeze_backbone:
        for param in model.transformer.parameters():
            param.requires_grad = False

    # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=16,
        learning_rate=5e-5,
        num_train_epochs=3,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,  # Reduced evaluation frequency to save time (was 10)
        save_strategy="no",
        report_to=["wandb"],
        run_name=model_hub_id,
        push_to_hub=True,
        hub_model_id=model_hub_id,
        hub_private_repo=True,
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

    # Train
    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # Push to hub
    trainer.push_to_hub(
        dataset_name=dataset_name,
    )

