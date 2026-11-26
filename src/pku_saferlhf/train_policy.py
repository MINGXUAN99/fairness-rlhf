"""PKU PPO entry point reusing the existing gradient-combination algorithms."""
import argparse
import json
import os
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True, choices=["vanilla", "maxmin", "sumgrad", "pcgrad"])
    p.add_argument("--policy_model", required=True)
    p.add_argument("--policy_revision", default="main")
    p.add_argument("--pooled_rm")
    p.add_argument("--helpfulness_rm")
    p.add_argument("--harmlessness_rm")
    p.add_argument("--dataset", default="mxcui/pku-saferlhf-preferences")
    p.add_argument("--dataset_revision", default="4730d5ec7e69f2d42ab3ffc14dab4ea9eae4119f")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--training_mode", choices=["full", "lora"], default="full")
    p.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--batch_size", type=int, default=16, help="Full rollout and PPO minibatch size; at least 8")
    p.add_argument("--reward_batch_size", type=int, default=4)
    p.add_argument("--total_episodes", type=int, default=50000)
    p.add_argument("--ppo_epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=3e-6)
    p.add_argument("--kl_coef", type=float, default=0.1)
    p.add_argument("--kl_div", choices=["mean_kl", "total"], default="mean_kl")
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--response_length", type=int, default=256)
    p.add_argument("--prompt_format", choices=["auto", "chat", "plain"], default="auto")
    p.add_argument("--reward_format", choices=["auto", "chat", "plain"], default="auto",
                   help="Fallback when RM run_config.json is unavailable")
    p.add_argument("--reward_max_length", type=int, default=1024, help="Fallback without RM metadata")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--gradient_checkpointing", action="store_true")
    p.add_argument("--report_to", choices=["none", "wandb"], default="none")
    args = p.parse_args()
    if args.method == "vanilla" and not args.pooled_rm:
        p.error("Vanilla requires --pooled_rm")
    if args.method != "vanilla" and not (args.helpfulness_rm and args.harmlessness_rm):
        p.error("Group-aware methods require --helpfulness_rm and --harmlessness_rm")
    if args.batch_size < 8 or min(args.reward_batch_size, args.total_episodes, args.ppo_epochs,
                                 args.max_prompt_length, args.response_length, args.reward_max_length) < 1:
        p.error("Batch size must be >=8 for PPO whitening; all lengths/counts must be positive")
    if args.cpu and args.precision != "fp32":
        p.error("Use --precision fp32 with --cpu")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()):
        p.error("Output directory must be empty")
    return args


def main():
    args = parse_args()
    if args.cpu:
        os.environ["ACCELERATE_USE_CPU"] = "true"
    import copy
    import torch
    import transformers
    import trl
    from accelerate import Accelerator
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, set_seed
    from src.helper.model_utils import configure_tokenizer, build_lora_config, default_lora_target_modules
    from src.pku_saferlhf.policy_utils import prepare_prompts, RewardBundle, make_reward_function
    from src.imdb.ppo_training import unified_ppo_training as shared

    if trl.__version__ != "0.24.0":
        raise RuntimeError("Use the tested policy requirements (TRL 0.24.0)")
    accelerator = Accelerator()
    if accelerator.num_processes != 1:
        raise ValueError("This entry point currently supports single-process PPO only")
    set_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if args.method in ("pcgrad", "sumgrad"):
        apply_fn = shared._pcgrad_compute_and_apply if args.method == "pcgrad" else shared._sumgrad_compute_and_apply
        shared._patch_ppo_for_pcgrad(args.kl_div == "mean_kl", apply_fn=apply_fn, strict=True)
        shared._sumgrad_diag["csv_path"] = str(output / "gradient_cosine.csv")
    elif args.kl_div == "mean_kl":
        shared._patch_ppo_kl_to_mean_per_token()
        if "kl_mean_per_seq" not in shared.PPOTrainer.train.__code__.co_varnames:
            raise RuntimeError("Failed to apply mean-per-token KL patch")

    policy_revision = args.policy_revision
    if not Path(args.policy_model).is_dir():
        from huggingface_hub import HfApi
        policy_revision = HfApi().model_info(args.policy_model, revision=policy_revision).sha
    tokenizer = AutoTokenizer.from_pretrained(args.policy_model, revision=policy_revision, padding_side="left")
    configure_tokenizer(tokenizer, padding_side="left")
    # PPO masks padding by token identity; EOS must not be mistaken for padding.
    if tokenizer.pad_token_id == tokenizer.eos_token_id:
        tokenizer.add_special_tokens({"pad_token": "<|pku_pad|>"})
    if tokenizer.eos_token_id is None:
        raise ValueError("Policy tokenizer must define EOS")
    if Path(args.dataset).is_dir():
        rows = load_from_disk(args.dataset)["train"]
        dataset_revision = "local"
    else:
        from huggingface_hub import HfApi
        dataset_revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision).sha
        rows = load_dataset(args.dataset, revision=dataset_revision, split="train")
    prompts, lookup, prompt_stats = prepare_prompts(rows, tokenizer, args.max_prompt_length, args.prompt_format)
    if len(prompts) < args.batch_size:
        raise ValueError("Need at least one complete rollout batch of unique training prompts")
    dtype = torch.float32 if args.precision == "fp32" else torch.bfloat16
    policy = AutoModelForCausalLM.from_pretrained(args.policy_model, revision=policy_revision, torch_dtype=dtype)
    if len(tokenizer) != policy.get_input_embeddings().num_embeddings:
        policy.resize_token_embeddings(len(tokenizer))
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.generation_config.pad_token_id = tokenizer.pad_token_id
    context_limit = getattr(policy.config, "max_position_embeddings", None)
    if context_limit and args.max_prompt_length + args.response_length > context_limit:
        raise ValueError("Prompt plus response budget exceeds policy context length")
    # Same initialization, including any added pad embedding, for the reference.
    reference = copy.deepcopy(policy) if args.training_mode == "full" else None
    if reference is not None:
        reference.requires_grad_(False).eval()
    if args.training_mode == "lora":
        from peft import get_peft_model
        policy = get_peft_model(policy, build_lora_config("CAUSAL_LM", r=args.lora_r,
            lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
            target_modules=default_lora_target_modules(policy)))
        if args.gradient_checkpointing:
            policy.enable_input_require_grads()
    value = AutoModelForSequenceClassification.from_pretrained(args.policy_model, revision=policy_revision,
                                                              num_labels=1, torch_dtype=dtype)
    if len(tokenizer) != value.get_input_embeddings().num_embeddings:
        value.resize_token_embeddings(len(tokenizer))
    value.config.pad_token_id = tokenizer.pad_token_id
    paths = {"pooled": args.pooled_rm} if args.method == "vanilla" else {
        "harmlessness": args.harmlessness_rm, "helpfulness": args.helpfulness_rm}
    rewards = RewardBundle(paths, dtype, accelerator.device, args.reward_format, args.reward_max_length)
    custom_reward = make_reward_function(rewards, tokenizer, lookup, args.method, shared,
        args.reward_batch_size, args.batch_size, output / "reward_trace.jsonl")
    shared._experimental_ppo_module.get_reward = custom_reward
    config = shared.PPOConfig(
        output_dir=str(output), seed=args.seed, data_seed=args.seed,
        per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=1,
        local_rollout_forward_batch_size=args.batch_size, num_mini_batches=1,
        num_ppo_epochs=args.ppo_epochs, total_episodes=args.total_episodes,
        response_length=args.response_length, learning_rate=args.learning_rate,
        kl_coef=args.kl_coef, kl_estimator="k1", whiten_rewards=True,
        missing_eos_penalty=0.0, stop_token="eos", lr_scheduler_type="cosine",
        bf16=args.precision == "bf16", use_cpu=args.cpu,
        gradient_checkpointing=args.gradient_checkpointing,
        save_strategy="no", eval_strategy="no", num_sample_generations=0,
        report_to=args.report_to, logging_steps=1, push_to_hub=False,
    )
    metadata = {**vars(args), "prompt_stats": prompt_stats, "reward_specs": rewards.specs,
                "resolved_policy_revision": policy_revision, "resolved_dataset_revision": dataset_revision,
                "group_order": rewards.names, "trl_version": trl.__version__,
                "transformers_version": transformers.__version__, "reward_calibration": False}
    (output / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    trainer = shared.PPOTrainer(args=config, processing_class=tokenizer, model=policy,
        ref_model=reference, reward_model=rewards, value_model=value, train_dataset=prompts,
        # TRL requires an eval dataloader. Sampling/evaluation is disabled; this
        # placeholder uses training prompts and never loads the held-out test set.
        eval_dataset=prompts.select(range(args.batch_size)))
    trainer.train()
    shared._sumgrad_flush_cosine()
    final_policy = trainer.accelerator.unwrap_model(trainer.model).policy
    if args.training_mode == "lora":
        final_policy = final_policy.merge_and_unload()
    final_policy.save_pretrained(output, safe_serialization=True)
    tokenizer.save_pretrained(output)
    trainer.state.save_to_json(str(output / "trainer_state.json"))
    print(f"Saved {args.method} policy to {output}")


if __name__ == "__main__":
    main()
