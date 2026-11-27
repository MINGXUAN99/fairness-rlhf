"""Offline checkpoint/panel runner; --validate-only imports no ML packages.

Usage: python -m src.imdb.diagnostics.attribute_gradient_runner --config plan.json
"""
from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import math
import subprocess
from pathlib import Path

from . import attribute_gradient_diagnostic as d


def load_config(path):
    path = Path(path).resolve()
    cfg = json.loads(path.read_text())
    required = {"run_id", "seed", "alpha", "output_dir", "prompts_file", "policy_tokenizer",
                "sentiment_model", "minority_model", "reward", "sampling", "diagnostic"}
    missing = required - cfg.keys()
    if missing:
        raise ValueError(f"Missing configuration keys: {sorted(missing)}")
    if not 0 <= cfg["alpha"] <= 1 or not isinstance(cfg["seed"], int):
        raise ValueError("alpha must be in [0,1], seed must be an integer")
    for key in ("output_dir", "prompts_file"):
        cfg[key] = str((path.parent / cfg[key]).resolve())
    if cfg.get("evaluation_prompts_file"):
        cfg["evaluation_prompts_file"] = str((path.parent / cfg["evaluation_prompts_file"]).resolve())
    for key in ("policy_tokenizer", "sentiment_model", "minority_model", "pooled_model"):
        spec = cfg.get(key)
        if key == "pooled_model" and spec is None:
            continue
        if not isinstance(spec, dict) or not spec.get("path") or not spec.get("revision"):
            raise ValueError(f"{key} requires explicit path and revision")
        if spec["path"].startswith(("./", "../", "/")):
            spec["path"] = str((path.parent / spec["path"]).resolve())
        if key.endswith("model") and spec.get("padding_side") not in ("left", "right"):
            raise ValueError(f"{key} needs an explicit padding_side")
    if not isinstance(cfg["sentiment_model"].get("positive_label_index"), int):
        raise ValueError("Specify sentiment_model.positive_label_index explicitly")
    reward = cfg["reward"]
    if reward.get("length_source") not in ("policy", "scorer", "scorer_untruncated"):
        raise ValueError("reward.length_source must be policy/scorer/scorer_untruncated")
    if reward.get("text_scope") not in ("response", "prompt_response"):
        raise ValueError("reward.text_scope must be response/prompt_response")
    if reward.get("length_denominator", 0) <= 0 or reward.get("scorer_max_length", 0) < 2:
        raise ValueError("Explicit positive length denominator and scorer maximum required")
    if not isinstance(reward.get("clip_conciseness"), bool):
        raise ValueError("Explicit reward.clip_conciseness boolean required")
    sampling = cfg["sampling"]
    for key in ("max_new_tokens", "max_prompt_tokens", "responses_per_prompt"):
        if not isinstance(sampling.get(key), int) or sampling[key] < 1:
            raise ValueError(f"sampling.{key} must be a positive integer")
    if sampling["responses_per_prompt"] < 2:
        raise ValueError("At least two responses per prompt are required for LOO")
    if not math.isfinite(sampling.get("temperature", 0)) or sampling["temperature"] <= 0:
        raise ValueError("sampling.temperature must be finite and positive")
    if sampling.get("top_p", 1) != 1 or sampling.get("top_k", 0) != 0:
        raise ValueError("Clean panel requires unfiltered sampling (top_p=1, top_k=0)")
    diag = cfg["diagnostic"]
    for key in ("gradient_accumulation_steps", "rollout_chunk_size"):
        if key in cfg and (type(cfg[key]) is not int or cfg[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    for key in ("batch_size", "blocks", "early_every", "late_every"):
        if not isinstance(diag.get(key), int) or diag[key] < 1:
            raise ValueError(f"diagnostic.{key} must be a positive integer")
    if diag.get("early_until", -1) < 0:
        raise ValueError("diagnostic.early_until must be nonnegative")
    for key in ("shared_scale", "cosine_floor"):
        if not math.isfinite(diag.get(key, 0)) or diag[key] <= 0:
            raise ValueError(f"diagnostic.{key} must be finite and positive")
    seen = set()
    for spec in cfg.get("checkpoints", []):
        if not spec.get("path") or not spec.get("revision") or not isinstance(spec.get("update"), int):
            raise ValueError("Each checkpoint requires path, revision, integer update")
        if spec["update"] < 0 or spec["update"] in seen:
            raise ValueError("Checkpoint updates must be unique and nonnegative")
        seen.add(spec["update"])
        if spec["path"].startswith(("./", "../", "/")):
            spec["path"] = str((path.parent / spec["path"]).resolve())
    cfg["config_path"] = str(path)
    return cfg


def load_prompts(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or any(not row.get("id") or not isinstance(row.get("text"), str) or not row["text"].strip() for row in rows):
        raise ValueError("prompts_file requires JSONL records with nonempty source-review id and text")
    if len({str(r["id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate source-review IDs in fixed prompt panel")
    return rows


def evaluation_prompts(cfg, diagnostic_prompts):
    if not cfg.get("evaluation_prompts_file"):
        return []
    rows = load_prompts(cfg["evaluation_prompts_file"])
    if {str(r["id"]) for r in rows} & {str(r["id"]) for r in diagnostic_prompts}:
        raise ValueError("Evaluation and diagnostic source-review IDs must be disjoint")
    return rows


def due(update, cfg, final=False):
    diag = cfg["diagnostic"]
    every = diag["early_every"] if update <= diag["early_until"] else diag["late_every"]
    return update == 0 or final or update % every == 0


def parameter_manifest(model):
    return [{"name": n, "shape": list(p.shape), "dtype": str(p.dtype)} for n, p in d.trainable_parameters(model)]


def environment_manifest(cfg):
    versions = {}
    for package in ("torch", "transformers", "trl", "accelerate", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    sources = {p.name: __import__("hashlib").sha256(p.read_bytes()).hexdigest()
               for p in Path(__file__).parent.glob("attribute_*.py")}
    training_source = Path(__file__).parent.parent/"ppo_training/unified_ppo_training.py"
    sources[training_source.name] = __import__("hashlib").sha256(training_source.read_bytes()).hexdigest()
    return dict(config=cfg, source_commit=commit, diagnostic_source_hashes=sources, versions=versions,
                gradient_convention="ascent", policy_length_includes_eos=False,
                action_log_probability_includes_eos=True,
                notes="Training seeds, generation samples and prompt blocks are different replication units.")


class RewardScorers:
    def __init__(self, cfg, device):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.cfg, self.device, self.loaded, self.identities = cfg, device, {}, {}
        for key in ("sentiment_model", "minority_model", "pooled_model"):
            spec = cfg.get(key)
            if spec is None:
                continue
            model, loading = AutoModelForSequenceClassification.from_pretrained(
                spec["path"], revision=spec["revision"], torch_dtype=torch.float32,
                trust_remote_code=False, output_loading_info=True)
            if loading.get("missing_keys") or loading.get("mismatched_keys"):
                raise ValueError(f"Incomplete trained reward checkpoint: {key}: {loading}")
            if key != "sentiment_model" and model.config.num_labels != 1:
                raise ValueError(f"{key} must have a trained scalar head")
            if key == "sentiment_model" and not 0 <= spec["positive_label_index"] < model.config.num_labels:
                raise ValueError("Invalid sentiment positive_label_index")
            tokenizer = AutoTokenizer.from_pretrained(spec["path"], revision=spec["revision"], trust_remote_code=False)
            if tokenizer.pad_token_id is None:
                if tokenizer.eos_token_id is None:
                    raise ValueError(f"No pad or EOS token in {key}")
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = spec["padding_side"]
            tokenizer.truncation_side = "right"
            model.config.pad_token_id = tokenizer.pad_token_id
            model.to(device).eval().requires_grad_(False)
            self.loaded[key] = (model, tokenizer)
            self.identities[key] = dict(spec, resolved_revision=getattr(model.config, "_commit_hash", None),
                                         id2label=model.config.id2label)

    def score(self, samples, policy_tokenizer, alpha):
        import torch
        cfg = self.cfg
        scope = cfg["reward"]["text_scope"]
        texts = [policy_tokenizer.decode((s["prompt_tokens"] if scope == "prompt_response" else []) +
                                         s["response_tokens"], skip_special_tokens=True) for s in samples]
        maximum = cfg["reward"]["scorer_max_length"]
        batch_size = cfg["diagnostic"]["batch_size"]
        rows = [dict(s, text=text) for s, text in zip(samples, texts)]
        with torch.no_grad():
            for key, (model, tokenizer) in self.loaded.items():
                for start in range(0, len(rows), batch_size):
                    batch = texts[start:start+batch_size]
                    full_lengths = [len(ids) for ids in tokenizer(batch, truncation=False)["input_ids"]]
                    encoded = tokenizer(batch, truncation=True, max_length=maximum, padding=True, return_tensors="pt")
                    lengths = encoded["attention_mask"].sum(1).tolist()
                    logits = model(**{k: v.to(self.device) for k, v in encoded.items()}).logits.float()
                    if not torch.isfinite(logits).all():
                        raise ValueError(f"Non-finite reward scores: {key}")
                    if key == "sentiment_model":
                        values = logits.softmax(-1)[:, cfg[key]["positive_label_index"]].cpu().tolist()
                        name = "sentiment"
                    else:
                        values = logits[:, 0].cpu().tolist()
                        name = "minority_rm" if key == "minority_model" else "pooled_rm"
                    for j, value in enumerate(values):
                        row = rows[start+j]
                        row[name] = value
                        if key == "sentiment_model":
                            row.update(scorer_length=lengths[j], scorer_length_untruncated=full_lengths[j],
                                       scorer_truncated=int(full_lengths[j] > maximum))
        length_key = {"policy": "policy_length", "scorer": "scorer_length",
                      "scorer_untruncated": "scorer_length_untruncated"}[cfg["reward"]["length_source"]]
        for row in rows:
            c = 1 - row[length_key]/cfg["reward"]["length_denominator"]
            row["conciseness"] = max(0., min(1., c)) if cfg["reward"]["clip_conciseness"] else c
            row["minority_oracle"] = alpha*row["conciseness"] + (1-alpha)*row["sentiment"]
        return rows


def generate_panel(policy, tokenizer, prompts, cfg, sample_seed):
    import torch
    from transformers import GenerationConfig
    s = cfg["sampling"]
    encoded = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt["text"], add_special_tokens=False)
        if not tokens or len(tokens) > s["max_prompt_tokens"]:
            raise ValueError(f"Prompt {prompt['id']} is empty or exceeds max_prompt_tokens; prepare prompts explicitly")
        for repeat in range(s["responses_per_prompt"]):
            encoded.append((prompt["id"], repeat, tokens))
    device = next(policy.parameters()).device
    context_limit = getattr(policy.config, "max_position_embeddings", None) or getattr(policy.config, "n_positions", None)
    if context_limit and max(len(t) for _, _, t in encoded) + s["max_new_tokens"] > context_limit:
        raise ValueError("Prompt plus generation cap exceeds policy context")
    result = []
    with d.isolated_diagnostic(policy, sample_seed), torch.no_grad():
        for start in range(0, len(encoded), cfg["diagnostic"]["batch_size"]):
            batch = encoded[start:start+cfg["diagnostic"]["batch_size"]]
            width = max(len(tokens) for _, _, tokens in batch)
            ids = torch.full((len(batch), width), tokenizer.pad_token_id, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids)
            for i, (_, _, tokens) in enumerate(batch):
                ids[i, -len(tokens):] = torch.tensor(tokens, device=device)
                mask[i, -len(tokens):] = 1
            config = GenerationConfig(do_sample=True, temperature=s["temperature"], top_p=1., top_k=0,
                                      max_new_tokens=s["max_new_tokens"], eos_token_id=tokenizer.eos_token_id,
                                      pad_token_id=tokenizer.pad_token_id, use_cache=True)
            output = policy.generate(input_ids=ids, attention_mask=mask, generation_config=config)
            for (prompt_id, repeat, tokens), generated in zip(batch, output[:, width:].cpu().tolist()):
                result.append(d.canonical_sample(tokens, generated, eos_token_id=tokenizer.eos_token_id,
                    pad_token_id=tokenizer.pad_token_id, prompt_id=prompt_id, sample_id=f"{prompt_id}:{repeat}",
                    max_new_tokens=s["max_new_tokens"]))
    return result


def measure(policy, tokenizer, rows, cfg, *, baseline, temperature, displacement=None):
    rewards = {name: [row[name] for row in rows] for name in
               ("conciseness", "sentiment", "minority_rm", "pooled_rm") if all(name in r for r in rows)}
    diag = cfg["diagnostic"]
    vectors = d.attribute_gradients(policy, rows, rewards, pad_token_id=tokenizer.pad_token_id,
        alpha=cfg["alpha"], batch_size=diag["batch_size"], baseline=baseline,
        shared_scale=diag["shared_scale"], temperature=temperature)
    metrics = d.gradient_metrics(vectors, cfg["alpha"], displacement, diag["cosine_floor"])
    # FP32 full policies should be much tighter; bf16 accumulation needs looser tolerance.
    if metrics["mixture_error_absolute"] > 1e-6 and metrics["mixture_error_relative"] > .03:
        raise ValueError("Oracle mixture reconstruction failed; inspect precision/estimator before interpreting norms")
    return vectors, metrics


def record_panel(policy, tokenizer, scorers, prompts, cfg, update, *, phase="post_update"):
    output = Path(cfg["output_dir"])
    sample_seed = cfg["seed"] + 1000003 + update
    samples = generate_panel(policy, tokenizer, prompts, cfg, sample_seed)
    rows = scorers.score(samples, tokenizer, cfg["alpha"])
    meta = dict(run_id=cfg["run_id"], seed=cfg["seed"], alpha=cfg["alpha"], ppo_update=update,
                stream="clean_panel", phase=phase, generation_seed=sample_seed,
                temperature=cfg["sampling"]["temperature"], baseline="loo",
                shared_scale=cfg["diagnostic"]["shared_scale"])
    d.append_jsonl(output/"rollouts.jsonl", [dict(meta, **row) for row in rows])
    d.append_jsonl(output/"rollout_summary.jsonl", [dict(meta, **d.summarize_samples(rows))])
    ids = [str(p["id"]) for p in prompts]
    block_count = min(cfg["diagnostic"]["blocks"], len(ids))
    assignments = {key: i % block_count for i, key in enumerate(ids)}
    # All-sample gradient plus prompt-disjoint blocks. No per-sample Jacobians.
    for block in ["all"] + list(range(block_count)):
        selected = rows if block == "all" else [r for r in rows if assignments[r["prompt_id"]] == block]
        vectors, metrics = measure(policy, tokenizer, selected, cfg, baseline="loo", temperature=meta["temperature"])
        d.append_jsonl(output/"gradient_summary.jsonl", [dict(meta, block=block, **metrics)])
        if block == "all":
            d.append_jsonl(output/"counterfactual_alpha.jsonl", [dict(meta, **m) for m in d.counterfactual_metrics(vectors)])
        del vectors
    held_out = evaluation_prompts(cfg, prompts)
    if held_out:
        samples = generate_panel(policy, tokenizer, held_out, cfg, sample_seed + 500009)
        evaluated = scorers.score(samples, tokenizer, cfg["alpha"])
        evaluation_meta = dict(meta, stream="heldout_evaluation", generation_seed=sample_seed + 500009)
        d.append_jsonl(output/"evaluation.jsonl", [dict(evaluation_meta, **d.summarize_samples(evaluated))])
        d.append_jsonl(output/"evaluation_rollouts.jsonl", [dict(evaluation_meta, **row) for row in evaluated])


def run(cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not cfg.get("checkpoints"):
        raise ValueError("Offline mode requires explicit checkpoints")
    output = Path(cfg["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use a fresh output_dir; refusing to mix/overwrite experiment runs")
    device = cfg.get("device", "cuda")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. No models have been loaded.")
    prompts = load_prompts(cfg["prompts_file"])
    evaluation_prompts(cfg, prompts)
    token_spec = cfg["policy_tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(token_spec["path"], revision=token_spec["revision"], trust_remote_code=False)
    if tokenizer.eos_token_id is None:
        raise ValueError("Explicit EOS token required")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    scorer = RewardScorers(cfg, device)
    manifest = environment_manifest(cfg)
    manifest.update(reward_models=scorer.identities, prompts_sha256=d.digest(prompts), checkpoints=[])
    for spec in sorted(cfg["checkpoints"], key=lambda s: s["update"]):
        policy, loading = AutoModelForCausalLM.from_pretrained(spec["path"], revision=spec["revision"],
            torch_dtype=torch.float32, trust_remote_code=False, output_loading_info=True)
        if loading.get("missing_keys") or loading.get("mismatched_keys"):
            raise ValueError(f"Incomplete policy checkpoint: {spec['path']}")
        policy.to(device)
        manifest["checkpoints"].append(dict(spec, resolved_revision=getattr(policy.config, "_commit_hash", None),
                                             parameters=parameter_manifest(policy)))
        d.write_json(output/"manifest.json", manifest)
        record_panel(policy, tokenizer, scorer, prompts, cfg, spec["update"], phase="checkpoint")
        del policy
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    d.write_json(output/"completed.json", {"checkpoints": len(cfg["checkpoints"]), "manifest_sha256": d.digest(manifest)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--validate-only", action="store_true", help="Validate schema and prompt IDs; no ML imports/downloads")
    args = parser.parse_args()
    cfg = load_config(args.config)
    prompts = load_prompts(cfg["prompts_file"])
    evaluation_prompts(cfg, prompts)
    if args.validate_only:
        print(f"Valid configuration: {cfg['run_id']}; {len(prompts)} prompts; {len(cfg.get('checkpoints', []))} checkpoints. Model paths/access not checked.")
        return
    run(cfg)


if __name__ == "__main__":
    main()
