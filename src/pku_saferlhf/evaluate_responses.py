"""Score generated responses with frozen external Beaver reward and cost models."""
import argparse
import gc
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generation_dir", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--helpfulness_model", default="PKU-Alignment/beaver-7b-v1.0-reward")
    p.add_argument("--cost_model", default="PKU-Alignment/beaver-7b-v1.0-cost")
    p.add_argument("--helpfulness_revision", default="main")
    p.add_argument("--cost_revision", default="main")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--allow_truncation", action="store_true",
                   help="Explicitly allow right truncation; affected scores are flagged")
    p.add_argument("--device", choices=["cuda", "cpu", "mps"], default="cuda")
    p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
    args = p.parse_args()
    if min(args.batch_size, args.max_length) < 1:
        p.error("Lengths and counts must be positive")
    if args.device == "cpu" and args.precision != "fp32":
        p.error("Use --precision fp32 on CPU")
    import torch
    import transformers
    from transformers import AutoTokenizer
    from src.helper.model_utils import configure_tokenizer
    from src.pku_saferlhf.beaver_scorer import load_beaver, beaver_text
    from src.pku_saferlhf.evaluation_utils import load_records, new_output, resolve_revision, write_json

    records, generation = load_records(args.generation_dir, "responses.jsonl")
    output = new_output(args.output_dir)
    manifest = {**generation, "status": "running", "scoring_settings": vars(args),
                "evaluators": {}, "transformers_version": transformers.__version__}
    write_json(output / "manifest.json", manifest)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    for metric, path, revision in (("helpfulness_reward", args.helpfulness_model, args.helpfulness_revision),
                                   ("harmfulness_cost", args.cost_model, args.cost_revision)):
        revision = resolve_revision(path, revision)
        tokenizer = AutoTokenizer.from_pretrained(path, revision=revision)
        configure_tokenizer(tokenizer, padding_side="right")
        tokenizer.truncation_side = "right"
        model = load_beaver(path, revision, dtype, args.device)
        limit = min(args.max_length, model.config.max_position_embeddings)
        truncated_count = 0
        for start in range(0, len(records), args.batch_size):
            batch = records[start:start+args.batch_size]
            ids = [tokenizer(beaver_text(row["prompt"], row["response"], tokenizer.eos_token),
                             add_special_tokens=True, truncation=False)["input_ids"] for row in batch]
            truncated = [len(seq) > limit for seq in ids]
            if any(truncated) and not args.allow_truncation:
                raise ValueError(f"{metric}: prompt+response exceeds {limit} tokens. "
                                 "Use shorter generation or explicitly allow truncation; no scores were silently truncated.")
            inputs = tokenizer.pad([{"input_ids": seq[:limit], "attention_mask": [1] * min(len(seq), limit)}
                                    for seq in ids], padding=True, return_tensors="pt").to(args.device)
            with torch.inference_mode():
                scores = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"]).end_scores.flatten()
            if not torch.isfinite(scores).all():
                raise ValueError("Evaluator produced nonfinite scores")
            for row, score, cut in zip(batch, scores.tolist(), truncated):
                row[metric] = score
                row[f"{metric}_truncated"] = cut
            truncated_count += sum(truncated)
        manifest["evaluators"][metric] = {"model": path, "revision": revision, "max_length": limit,
            "truncated_responses": truncated_count, "higher_is_better": metric == "helpfulness_reward",
            "normalization": False}
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"Scored {len(records)} responses for {metric}")
    with (output / "scored_responses.jsonl").open("x") as handle:
        for row in records:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(f"Saved external evaluator scores to {output}")


if __name__ == "__main__":
    main()
