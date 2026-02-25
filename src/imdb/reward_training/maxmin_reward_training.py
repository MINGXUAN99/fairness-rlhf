import argparse
from trl import RewardConfig, RewardTrainer
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification


def parse_args():
    parser = argparse.ArgumentParser(description="Train Maxmin Reward Model with configurable parameters.")
    parser.add_argument("--user_id", type=str, required=True, help="Hugging Face user ID.")
    parser.add_argument("--seed", type=int, required=True, help="Random seed to use.")
    parser.add_argument("--proportion", type=float, required=True, help="Proportions of helpful and harmless responses.")
    parser.add_argument("--alpha", type=float, required=True, help="Alpha value for the scores (0.5-1.0).")
    parser.add_argument("--group", required=True, type=int, help="Group ID.")
    parser.add_argument("--preference_dataset_hub_id", type=str, default=None,
                        help="Hugging Face preference dataset hub id (e.g. from build_dataset.sh). If not set, uses {user_id}/gpt2-imdb-seed{seed}-prop{proportion}-alpha{alpha}.")
    parser.add_argument("--hub_model_id", type=str, default=None,
                        help="Hugging Face hub model id for pushing the reward model. If not set, uses {user_id}/maxmin-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}-{group_name}.")
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
    model_name = "lvwerra/gpt2-imdb"
    freeze_backbone = True
    group_name = "majority" if group_id == 1 else "minority"
    model_hub_id = args.hub_model_id or f"{user_id}/maxmin-imdb-reward-prop{proportion}-alpha{alpha}-seed{seed}-{group_name}"

    # Reward Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward Model
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Freeze Backbone
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

    trainer.train()

    # Evaluation
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)

    # Push to hub
    trainer.push_to_hub(
        dataset_name=dataset_name,
    )