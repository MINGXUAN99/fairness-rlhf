"""Opt-in, single-process vanilla PPO instrumentation.

TRL's PPO train loop is not a public callback interface for rollouts or optimizer
steps. Install checked source hooks BEFORE the repository's mean-KL patch. An
unsupported loop fails before training; nothing is patched by importing this file.
"""
from __future__ import annotations

import ast
import inspect
import linecache
import textwrap
from pathlib import Path

from . import attribute_gradient_diagnostic as d
from . import attribute_gradient_runner as runner


def instrument_source(source):
    source = textwrap.dedent(source)
    edits = [
        ("    for update in range(1, args.num_total_batches + 1):",
         "    self.attribute_probe.begin(self)\n    for update in range(1, args.num_total_batches + 1):"),
        ("            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)",
         "            self.attribute_probe.capture_logprobs(logprobs, update, args.num_total_batches)\n"
         "            logprobs = torch.masked_fill(logprobs, padding_mask, INVALID_LOGPROB)"),
        ("            if args.whiten_rewards:",
         "            self.attribute_probe.observe('rewards_pre_whiten', rewards, ~padding_mask_p1)\n            if args.whiten_rewards:"),
        ("            lastgaelam = 0",
         "            self.attribute_probe.observe('rewards_post_whiten', rewards, ~padding_mask_p1)\n            lastgaelam = 0"),
        ("            advantages = masked_whiten(advantages, ~padding_mask)",
         "            self.attribute_probe.observe('advantages_pre_whiten', advantages, ~padding_mask)\n            advantages = masked_whiten(advantages, ~padding_mask)"),
        ("            advantages = torch.masked_fill(advantages, padding_mask, 0)",
         "            advantages = torch.masked_fill(advantages, padding_mask, 0)\n            self.attribute_probe.observe('advantages_post_whiten', advantages, ~padding_mask)"),
        ("        for ppo_epoch_idx in range(args.num_ppo_epochs):",
         "        self.attribute_probe.capture_rollout(self, locals())\n        for ppo_epoch_idx in range(args.num_ppo_epochs):"),
        ("                        optimizer.step()",
         "                        self.attribute_probe.before_optimizer(self, locals())\n                        optimizer.step()\n                        self.attribute_probe.after_optimizer(self, locals())"),
        ("        self.lr_scheduler.step()",
         "        self.lr_scheduler.step()\n        self.attribute_probe.finish_rollout(self, update, metrics)"),
        ("    self.control = self.callback_handler.on_train_end(args, self.state, self.control)",
         "    self.attribute_probe.complete(self)\n    self.control = self.callback_handler.on_train_end(args, self.state, self.control)"),
    ]
    for old, new in edits:
        count = source.count(old)
        if count != 1:
            raise RuntimeError(f"Unsupported TRL PPO loop: expected exactly one hook {old.strip()!r}, found {count}. No patch installed.")
        source = source.replace(old, new, 1)
    # Ensure variables used by capture still exist in this trainer version.
    tree = ast.parse(source)
    assigned = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    required = {"query_responses", "context_length", "scores", "generation_config", "optimizer", "update",
                "metrics", "padding_mask", "padding_mask_p1"}
    if not required <= assigned:
        raise RuntimeError(f"Unsupported PPO loop, missing locals: {required-assigned}")
    return source


def _generate_rollout_logprobs(model, queries, chunk_size, pad_token_id, generation_config):
    """Reduce each TRL generation chunk before starting the next one."""
    import math
    import torch
    from torch.nn.functional import pad
    from trl.experimental.ppo.ppo_trainer import generate, selective_log_softmax

    tokens, probabilities = [], []
    for chunk in queries.split(chunk_size):
        sequence, logits = generate(model, chunk, pad_token_id, generation_config)
        tokens.append(sequence)
        probabilities.append(selective_log_softmax(logits, sequence[:, queries.size(1):]))
        # TRL pads missing score vectors with zeros (a uniform distribution).
        uniform_logprob = -math.log(logits.size(-1))
        del logits
    width = max(t.size(1) for t in tokens)
    padded_tokens = [pad(t, (0, width - t.size(1)), value=pad_token_id) for t in tokens]
    padded_probs = [pad(p, (0, width - t.size(1)), value=uniform_logprob) for t, p in zip(tokens, probabilities)]
    return torch.cat(padded_tokens), torch.cat(padded_probs)


def install_hooks(ppo_module, *, compact_rollouts=False):
    original = ppo_module.PPOTrainer.train
    if getattr(original, "_attribute_probe", False):
        raise RuntimeError("Attribute hooks already installed")
    source = instrument_source(inspect.getsource(original))
    if compact_rollouts:
        replacements = {
            "query_responses, logitss = batch_generation(": "query_responses, rollout_logp = _generate_rollout_logprobs(",
            "logits = logitss[i : i + args.local_rollout_forward_batch_size]": "logprob = rollout_logp[i : i + args.local_rollout_forward_batch_size]",
            "logprob = selective_log_softmax(logits, response)\n                del logits": "# Log probabilities were collected during generation.",
        }
        for old, new in replacements.items():
            if source.count(old) != 1:
                raise RuntimeError("Unsupported TRL rollout generation loop")
            source = source.replace(old, new)
    filename = "<ppo_train_with_attribute_probe>"
    # Let the existing mean-KL patch inspect this function in its next pass.
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    # Share the live module globals: setup_reward_function changes get_reward
    # AFTER hook installation and must remain visible to the patched loop.
    namespace = original.__globals__
    if compact_rollouts:
        namespace["_generate_rollout_logprobs"] = _generate_rollout_logprobs
    exec(compile(source, filename, "exec"), namespace)
    patched = namespace[original.__name__]
    patched._attribute_probe = True
    ppo_module.PPOTrainer.train = patched


class AttributeTrainingProbe:
    def __init__(self, cfg, policy, tokenizer, training_args):
        import torch
        self.cfg, self.policy, self.tokenizer = cfg, policy, tokenizer
        self.output = Path(cfg["output_dir"])
        self.output.mkdir(parents=True, exist_ok=True)
        if any(self.output.iterdir()):
            raise ValueError("Attribute diagnostics require a fresh output_dir")
        self.prompts = runner.load_prompts(cfg["prompts_file"])
        runner.evaluation_prompts(cfg, self.prompts)
        with d.isolated_diagnostic(policy):
            self.scorers = runner.RewardScorers(cfg, next(policy.parameters()).device)
        self.manifest = runner.environment_manifest(cfg)
        self.manifest.update(reward_models=self.scorers.identities, prompts_sha256=d.digest(self.prompts),
                             parameters=runner.parameter_manifest(policy),
                             training_args=training_args.to_dict(),
                             training_baseline="batch_mean (descriptive surrogate)",
                             actual_update="first non-skipped synchronized optimizer step per selected rollout")
        d.write_json(self.output/"manifest.json", self.manifest)
        self.controls, self.pending, self.before = {}, None, None
        self.update = 0
        self.last_panel = None
        if next(policy.parameters()).dtype == torch.float16:
            raise ValueError("Probe currently supports FP32/bf16, not scaled FP16 gradients")

    def metadata(self, phase="pre_update"):
        return dict(run_id=self.cfg["run_id"], seed=self.cfg["seed"], alpha=self.cfg["alpha"],
                    ppo_update=self.update, stream="training_rollout", phase=phase,
                    baseline="batch_mean", shared_scale=self.cfg["diagnostic"]["shared_scale"])

    def begin(self, trainer):
        if trainer.accelerator.num_processes != 1:
            raise ValueError("Attribute probe supports single-process training only")
        # Policy gradient buffers then contain only policy loss if parameters are disjoint.
        wrapper = trainer.accelerator.unwrap_model(trainer.model)
        policy_ids = {id(p) for p in self.policy.parameters()}
        if policy_ids.intersection(id(p) for p in wrapper.value_model.parameters()):
            raise ValueError("Policy/value shared parameters are unsupported for PPO-gradient attribution")
        runner.record_panel(self.policy, self.tokenizer, self.scorers, self.prompts, self.cfg, 0, phase="initial")
        self.last_panel = 0
        self.save_checkpoint(trainer, 0)

    def observe(self, name, tensor, mask):
        values = tensor.detach().float()[mask]
        self.controls[f"{name}_std"] = float(values.std(unbiased=False)) if values.numel() else None
        self.controls[f"{name}_mean"] = float(values.mean()) if values.numel() else None

    def capture_logprobs(self, logprobs, update, total_updates):
        # PPO replaces masked entries with a sentinel immediately after this hook.
        save = self.cfg.get("save_all_rollouts", False) or runner.due(update, self.cfg, final=update == total_updates)
        self.rollout_logprobs = logprobs.detach().cpu() if save else None

    def capture_rollout(self, trainer, local):
        import torch
        self.update = int(local["update"])
        self.pending, self.before = None, None
        context = int(local["context_length"])
        qr = local["query_responses"].detach().cpu().tolist()
        generation = local["generation_config"]
        if generation.top_p != 1 or generation.top_k not in (0, 0.0, None):
            raise ValueError("Training probe requires unfiltered TRL sampling for matched temperature log probabilities")
        self.temperature = float(generation.temperature)
        samples = []
        for i, tokens in enumerate(qr):
            prompt = tokens[:context]
            # This is a token-prefix identity, NOT a recovered source-review ID.
            # Single training responses do not support within-review variance.
            prompt_key = d.digest(prompt)
            sample = d.canonical_sample(prompt, tokens[context:], eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id, prompt_id=f"update-{self.update}-row-{i}",
                sample_id=f"{self.update}:{i}", max_new_tokens=trainer.args.response_length)
            sample["prompt_token_hash"] = prompt_key
            samples.append(sample)
        with d.isolated_diagnostic(self.policy):
            rows = self.scorers.score(samples, self.tokenizer, self.cfg["alpha"])
        # Use the reward actually passed into vanilla PPO, including configured transformations.
        for row, value in zip(rows, local["scores"].detach().float().cpu().tolist()):
            row["pooled_rm"] = value
        meta = dict(self.metadata(), temperature=self.temperature, episode=int(trainer.state.episode),
                    generation_cap=trainer.args.response_length, source_review_ids_available=False)
        d.append_jsonl(self.output/"rollout_summary.jsonl", [dict(meta, **d.summarize_samples(rows), **self.controls)])
        selected = runner.due(self.update, self.cfg, final=self.update == trainer.args.num_total_batches)
        if self.rollout_logprobs is not None:
            policy_masks = local["padding_mask"].detach().cpu().tolist()
            value_masks = local["padding_mask_p1"].detach().cpu().tolist()
            for row, tokens, logprobs, policy_mask, value_mask in zip(
                rows, qr, self.rollout_logprobs.tolist(), policy_masks, value_masks, strict=True
            ):
                row.update(rollout_response_tokens=tokens[context:], rollout_logprobs=logprobs,
                           padding_mask=policy_mask, padding_mask_p1=value_mask)
            d.append_jsonl(self.output/"rollouts.jsonl", [dict(meta, **r) for r in rows])
            self.rollout_logprobs = None
        if selected:
            vectors, metrics = runner.measure(self.policy, self.tokenizer, rows, self.cfg,
                                               baseline="batch_mean", temperature=self.temperature)
            self.pending = dict(vectors=vectors, metadata=meta, metrics=metrics)
            d.append_jsonl(self.output/"gradient_summary.jsonl", [dict(meta, block="all", **metrics)])
        self.controls = {}

    def before_optimizer(self, trainer, local):
        if self.pending is None or not trainer.accelerator.sync_gradients:
            return
        self.before = d.snapshot_parameters(self.policy)
        self.pending["vectors"]["ppo_ascent"] = d.accumulated_ascent_gradient(self.policy)
        self.pending["step_metadata"] = dict(ppo_epoch=int(local["ppo_epoch_idx"]),
            minibatch=int(local["minibatch_idx"]), learning_rate=float(local["optimizer"].param_groups[0]["lr"]))

    def after_optimizer(self, trainer, local):
        if self.before is None:
            return
        if getattr(local["optimizer"], "step_was_skipped", False):
            self.before = None
            return
        displacement = d.parameter_displacement(self.policy, self.before)
        vectors = self.pending["vectors"]
        metrics = d.gradient_metrics(vectors, self.cfg["alpha"], displacement,
                                     self.cfg["diagnostic"]["cosine_floor"])
        meta = dict(self.pending["metadata"], phase="first_optimizer_step", **self.pending["step_metadata"])
        d.append_jsonl(self.output/"update_summary.jsonl", [dict(meta, **metrics)])
        d.append_jsonl(self.output/"counterfactual_alpha.jsonl", [dict(meta, **r) for r in
                       d.counterfactual_metrics(vectors, displacement=displacement)])
        self.before, self.pending = None, None

    def finish_rollout(self, trainer, update, metrics):
        clean = {str(k): float(v) for k, v in metrics.items() if isinstance(v, (int, float))}
        d.append_jsonl(self.output/"training_controls.jsonl", [dict(self.metadata("post_update"), **clean)])
        final = update == trainer.args.num_total_batches
        if runner.due(update, self.cfg, final=final):
            if self.pending is not None:
                d.append_jsonl(self.output/"events.jsonl", [dict(self.metadata(), event="no_non_skipped_optimizer_update")])
                self.pending = None
            runner.record_panel(self.policy, self.tokenizer, self.scorers, self.prompts, self.cfg, update)
            self.last_panel = update
        self.save_checkpoint(trainer, update, final)

    def save_checkpoint(self, trainer, update, final=False):
        import random
        import numpy as np
        import torch
        if update not in self.cfg.get("save_checkpoint_updates", []) and not (final and self.cfg.get("save_final_checkpoint", False)):
            return
        path = self.output/"checkpoints"/f"update-{update:06d}"
        path.mkdir(parents=True, exist_ok=False)
        with d.isolated_diagnostic(self.policy):
            self.policy.save_pretrained(path/"policy", safe_serialization=True)
            self.tokenizer.save_pretrained(path/"policy")
            wrapper = trainer.accelerator.unwrap_model(trainer.model)
            torch.save(dict(value_model=wrapper.value_model.state_dict(), optimizer=trainer.optimizer.state_dict(),
                scheduler=trainer.lr_scheduler.state_dict(), torch_rng=torch.random.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                numpy_rng=np.random.get_state(), python_rng=random.getstate(), update=update,
                episode=int(trainer.state.episode)), path/"training_state.pt")
        d.write_json(path/"metadata.json", dict(update=update, phase="post_update" if update else "initial",
                     continuation_state=True, exact_dataloader_resume=False,
                     note="Includes optimizer/value/scheduler/RNG for explicit continuation; existing PPOTrainer.train resets its loop. Not an automatic exact-resume API."))

    def complete(self, trainer):
        d.write_json(self.output/"completed.json", dict(final_update=int(trainer.state.global_step),
            final_panel_update=self.last_panel, manifest_sha256=d.digest(self.manifest)))
