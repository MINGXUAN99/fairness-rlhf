"""Generate all policies on a shared set of unique PKU test prompts."""
import argparse
import gc
import json
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policies", required=True, help="JSON list of run_id, method, training_seed, model")
    p.add_argument("--dataset", default="mxcui/pku-saferlhf-preferences")
    p.add_argument("--dataset_revision", default="4730d5ec7e69f2d42ab3ffc14dab4ea9eae4119f")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--seed", type=int, default=42, help="Prompt sampling and generation seed")
    p.add_argument("--max_prompts", type=int, default=None)
    p.add_argument("--max_prompt_length", type=int, default=512)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--sample", action="store_true", help="Default is deterministic greedy decoding")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu", "mps"])
    p.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    args = p.parse_args()
    if min(args.max_prompt_length, args.max_new_tokens, args.batch_size) < 1 or (args.max_prompts is not None and args.max_prompts < 1):
        p.error("Lengths and counts must be positive")
    if args.temperature <= 0 or not 0 < args.top_p <= 1:
        p.error("Temperature must be positive and top_p must be in (0, 1]")
    if args.device == "cpu" and args.precision != "fp32":
        p.error("Use --precision fp32 on CPU")
    import torch
    from datasets import load_dataset, load_from_disk
    from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM, GenerationConfig, set_seed
    from src.helper.model_utils import configure_tokenizer
    from src.pku_saferlhf.policy_utils import prepare_prompts
    from src.pku_saferlhf.evaluation_utils import new_output, resolve_revision, prompt_id, write_json

    runs = json.loads(Path(args.policies).read_text())
    if not isinstance(runs, list) or not runs:
        raise ValueError("Policies file must be a nonempty list")
    for run in runs:
        if not all(key in run for key in ("run_id", "method", "training_seed", "model")):
            raise ValueError("Each run needs run_id, method, training_seed (null for initial), and model")
    if len({r["run_id"] for r in runs}) != len(runs):
        raise ValueError("run_id values must be unique")
    output = new_output(args.output_dir)
    dataset_revision = resolve_revision(args.dataset, args.dataset_revision, "dataset")
    data = (load_from_disk(args.dataset)["test"] if dataset_revision is None else
            load_dataset(args.dataset, revision=dataset_revision, split="test"))
    unique = {}
    for text in data["prompt"]:
        unique.setdefault(" ".join(text.split()), text)
    texts = list(unique.values())
    random.Random(args.seed).shuffle(texts)
    tokenizers, mappings = [], []
    for run in runs:
        revision = resolve_revision(run["model"], run.get("revision", "main"))
        run["resolved_revision"] = revision
        tokenizer = AutoTokenizer.from_pretrained(run["model"], revision=revision, padding_side="left")
        configure_tokenizer(tokenizer, padding_side="left")
        config = AutoConfig.from_pretrained(run["model"], revision=revision)
        budget = min(args.max_prompt_length, getattr(config, "max_position_embeddings", args.max_prompt_length + args.max_new_tokens) - args.max_new_tokens)
        local_config = Path(run["model"]) / "run_config.json"
        saved = json.loads(local_config.read_text()) if local_config.is_file() else {}
        mode = run.get("prompt_format", saved.get("prompt_format", "auto"))
        _, lookup, stats = prepare_prompts([{"prompt": text} for text in texts], tokenizer, budget, mode,
                                           reject_pad_in_prompt=False)
        run["prompt_format"] = mode
        run["prompt_filter_stats"] = stats
        tokenizers.append(tokenizer)
        mappings.append({text: list(ids) for ids, text in lookup.items()})
    shared = [text for text in texts if all(text in mapping for mapping in mappings)]
    if args.max_prompts is not None:
        shared = shared[:args.max_prompts]
    if not shared:
        raise ValueError("No prompts fit all policies")
    manifest = {"status": "running", "dataset": args.dataset, "dataset_revision": dataset_revision,
        "split": "test", "runs": runs, "settings": vars(args),
        "source_unique_prompts": len(texts), "prompt_ids": [prompt_id(text) for text in shared]}
    write_json(output / "manifest.json", manifest)
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    with (output / "responses.jsonl").open("x") as handle:
        for run, tokenizer, mapping in zip(runs, tokenizers, mappings):
            set_seed(args.seed)
            model = AutoModelForCausalLM.from_pretrained(run["model"], revision=run["resolved_revision"], torch_dtype=dtype).to(args.device).eval()
            decoding = GenerationConfig(max_new_tokens=args.max_new_tokens, do_sample=args.sample,
                num_beams=1, num_return_sequences=1, top_k=0,
                temperature=args.temperature if args.sample else 1.0,
                top_p=args.top_p if args.sample else 1.0,
                eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id)
            for start in range(0, len(shared), args.batch_size):
                batch_texts = shared[start:start+args.batch_size]
                inputs = tokenizer.pad([{"input_ids": mapping[text], "attention_mask": [1] * len(mapping[text])} for text in batch_texts],
                                       padding=True, return_tensors="pt").to(args.device)
                with torch.inference_mode():
                    generated = model.generate(**inputs, generation_config=decoding)
                completions = generated[:, inputs["input_ids"].shape[1]:]
                for text, ids in zip(batch_texts, completions.tolist()):
                    eos = ids.index(tokenizer.eos_token_id) if tokenizer.eos_token_id in ids else None
                    used = ids[:eos+1] if eos is not None else ids
                    record = {"run_id": run["run_id"], "prompt_id": prompt_id(text), "prompt": text,
                        "response": tokenizer.decode(used, skip_special_tokens=True),
                        "generated_tokens": len(used), "ended_with_eos": eos is not None}
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"Generated {len(shared)} responses for {run['run_id']}")
    manifest["status"] = "complete"
    write_json(output / "manifest.json", manifest)
    print(f"Saved responses and shared prompt manifest to {output}")


if __name__ == "__main__":
    main()
