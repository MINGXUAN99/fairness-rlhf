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
    parser.add_argument("--hub_model_id", type=str, required=True,
                        help="Hugging Face hub model id for pushing the trained reward model (e.g. {user_id}/maxmin-imdb-reward-ALPHA1.0-len512-minority-gpt2).")
    parser.add_argument("--model_hub_id", type=str, default="gpt2",
                        help="Hugging Face model id for the backbone (e.g. gpt2, lvwerra/gpt2-imdb). Default: gpt2.")
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
    freeze_backbone = True
    output_hub_id = args.hub_model_id

    # Reward Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(backbone_model_id)
    tokenizer.pad_token = tokenizer.eos_token

    # Reward Model
    model = AutoModelForSequenceClassification.from_pretrained(backbone_model_id, num_labels=num_labels)
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # Freeze Backbone (GPT-2 uses .transformer, Qwen/LLaMA use .model)
    # #region agent log
    import json as _dbg_json; _lp = "/home/mx/fairness-rlhf/.cursor/debug-d71a21.log"; _h1=hasattr(model,"model"); _h2=hasattr(model,"transformer"); open(_lp,"a").write(_dbg_json.dumps({"hypothesisId":"H1","message":"backbone attrs","data":{"has_model":_h1,"has_transformer":_h2,"model_type":type(model).__name__},"location":"maxmin_reward_training.py:64"})+"\n")
    # #endregion
    if hasattr(model, "model"):
        backbone = model.model
    elif hasattr(model, "transformer"):
        backbone = model.transformer
    else:
        raise AttributeError(f"{type(model).__name__} has no 'model' or 'transformer' attribute for freezing")
    for param in backbone.parameters():
        param.requires_grad = False

    # Training config
    training_args = RewardConfig(
        per_device_train_batch_size=16,
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