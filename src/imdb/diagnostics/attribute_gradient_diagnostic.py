"""Attribute-level policy gradients. No model loading or training on import.

All vectors are ASCENT directions. Rewards are detached and never independently
whitened. Diagnostic autograd does not populate/clear training .grad buffers.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import math
import random
import statistics
from collections import defaultdict
from pathlib import Path


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        for row in records:
            handle.write(json.dumps(row, allow_nan=False) + "\n")


def describe(values):
    values = [float(v) for v in values]
    if not values:
        return {k: None for k in ("mean", "std", "variance", "min", "max", "q25", "median", "q75")}
    if not all(math.isfinite(v) for v in values):
        raise ValueError("Non-finite diagnostic value")
    ordered = sorted(values)
    def quantile(p):
        pos = (len(ordered) - 1) * p
        lo = int(pos)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (pos - lo)
    return dict(mean=statistics.mean(values), std=statistics.pstdev(values),
                variance=statistics.pvariance(values), min=min(values), max=max(values),
                q25=quantile(.25), median=quantile(.5), q75=quantile(.75))


def within_prompt_variance(values, prompt_ids):
    groups = defaultdict(list)
    for key, value in zip(prompt_ids, values):
        groups[key].append(float(value))
    variances = [statistics.variance(v) for v in groups.values() if len(v) > 1]
    return statistics.mean(variances) if variances else None


def advantages(rewards, prompt_ids, baseline="loo", shared_scale=1.0):
    """LOO uses other independent responses to the SAME source prompt.

    batch_mean is a descriptive, finite-batch-centered surrogate (not an
    unbiased baseline). zero is useful for analytic tests/external baselines.
    """
    if not math.isfinite(shared_scale) or shared_scale <= 0:
        raise ValueError("shared_scale must be finite and positive")
    values = [float(x) for x in rewards]
    if not values or len(values) != len(prompt_ids) or not all(map(math.isfinite, values)):
        raise ValueError("Rewards must be finite, nonempty, and match prompt IDs")
    if baseline == "zero":
        bases = [0.0] * len(values)
    elif baseline == "batch_mean":
        bases = [statistics.mean(values)] * len(values)
    elif baseline == "loo":
        groups = defaultdict(list)
        for i, key in enumerate(prompt_ids):
            groups[key].append(i)
        if any(len(indices) < 2 for indices in groups.values()):
            raise ValueError("LOO requires at least two independent responses per prompt")
        bases = [0.0] * len(values)
        for indices in groups.values():
            total = math.fsum(values[i] for i in indices)
            for i in indices:
                bases[i] = (total - values[i]) / (len(indices) - 1)
    else:
        raise ValueError(f"Unknown baseline: {baseline}")
    return [(v - b) / shared_scale for v, b in zip(values, bases)]


def response_extent(tokens, eos_token_id, pad_token_id=None):
    """Return valid action count (INCLUDING first EOS) and termination reason.

    A distinct pad token is padding; when pad==EOS, the first such token is the
    EOS action. Tokens after the first EOS never contribute, even if non-pad.
    """
    for i, token in enumerate(tokens):
        if token == eos_token_id:
            return i + 1, "eos"
        if pad_token_id is not None and pad_token_id != eos_token_id and token == pad_token_id:
            return i, "padding"
    return len(tokens), "cap"


def canonical_sample(prompt_tokens, response_tokens, *, eos_token_id, pad_token_id,
                     prompt_id, sample_id, max_new_tokens):
    prompt_tokens = [int(x) for x in prompt_tokens]
    while prompt_tokens and prompt_tokens[0] == pad_token_id:
        prompt_tokens.pop(0)
    if not prompt_tokens:
        raise ValueError("Empty prompt after removing left padding")
    # These are raw generator actions. A distinct PAD ID can itself be sampled;
    # it is not padding until generation has ended. EOS identifies that boundary.
    extent, reason = response_extent(response_tokens, eos_token_id)
    if extent == 0:
        raise ValueError("No response actions to differentiate")
    response = [int(x) for x in response_tokens[:extent]]
    # Exclude EOS from behavioral token length; include it in action_count.
    return dict(prompt_id=str(prompt_id), sample_id=str(sample_id),
                prompt_tokens=prompt_tokens, response_tokens=response,
                action_count=extent, policy_length=extent - int(reason == "eos"),
                termination=reason, eos=int(reason == "eos"),
                cap_hit=int(reason == "cap" and extent >= max_new_tokens))


@contextlib.contextmanager
def isolated_diagnostic(model, seed=None):
    """Restore RNG and every submodule's mode; leave .grad/optimizer untouched."""
    import numpy as np
    import torch
    py_state, np_state = random.getstate(), np.random.get_state()
    cpu_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed % (2**32))
            torch.manual_seed(seed)
        with torch.enable_grad():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(py_state)
        np.random.set_state(np_state)
        torch.random.set_rng_state(cpu_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def trainable_parameters(model):
    return [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]


def pack_samples(samples, pad_token_id, device):
    import torch
    width = max(len(s["prompt_tokens"]) + len(s["response_tokens"]) for s in samples)
    ids = torch.full((len(samples), width), pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(ids)
    actions = torch.zeros_like(ids, dtype=torch.bool)
    for i, sample in enumerate(samples):
        p, r = sample["prompt_tokens"], sample["response_tokens"]
        ids[i, :len(p) + len(r)] = torch.tensor(p + r, device=device)
        attention[i, :len(p) + len(r)] = 1
        actions[i, len(p):len(p) + len(r)] = True
    return ids, attention, actions


def response_log_prob(model, samples, pad_token_id, temperature=1.0):
    import torch
    if temperature <= 0 or not math.isfinite(temperature):
        raise ValueError("temperature must be finite and positive")
    device = next(model.parameters()).device
    ids, attention, actions = pack_samples(samples, pad_token_id, device)
    logits = model(input_ids=ids, attention_mask=attention, use_cache=False).logits[:, :-1].float()
    logits = logits / temperature
    # cross_entropy avoids materializing a second full log-softmax tensor.
    token_logp = -torch.nn.functional.cross_entropy(
        logits.transpose(1, 2), ids[:, 1:], reduction="none")
    return (token_logp * actions[:, 1:]).sum(1)


def attribute_gradients(model, samples, rewards, *, pad_token_id, alpha,
                        batch_size=4, baseline="loo", shared_scale=1.0, temperature=1.0):
    """CPU FP32 vectors, no per-response Jacobian, one forward per microbatch.

    The direct mixture is differentiated independently to verify reconstruction.
    Computing raw gradients before any whitening is the point of this diagnostic.
    """
    import torch
    if not samples or batch_size < 1 or not 0 <= alpha <= 1:
        raise ValueError("Nonempty samples, positive batch size, alpha in [0,1] required")
    if not {"conciseness", "sentiment"}.issubset(rewards):
        raise ValueError("Both oracle attributes are required")
    rewards = dict(rewards)
    rewards["minority_oracle"] = [alpha*c + (1-alpha)*s for c, s in
                                   zip(rewards["conciseness"], rewards["sentiment"])]
    prompt_ids = [s["prompt_id"] for s in samples]
    adv = {name: advantages(values, prompt_ids, baseline, shared_scale)
           for name, values in rewards.items()}
    named = trainable_parameters(model)
    if not named:
        raise ValueError("Policy has no trainable parameters")
    params = [p for _, p in named]
    sizes = [p.numel() for p in params]
    vectors = {name: torch.zeros(sum(sizes), dtype=torch.float32) for name in adv}
    with isolated_diagnostic(model):
        for start in range(0, len(samples), batch_size):
            batch = samples[start:start+batch_size]
            logp = response_log_prob(model, batch, pad_token_id, temperature)
            for j, (name, values) in enumerate(adv.items()):
                weights = torch.tensor(values[start:start+len(batch)], device=logp.device, dtype=logp.dtype)
                objective = (logp * weights).sum() / len(samples)
                grads = torch.autograd.grad(objective, params, allow_unused=True,
                                            retain_graph=j < len(adv)-1)
                offset = 0
                for size, grad in zip(sizes, grads):
                    if grad is not None:
                        vectors[name][offset:offset+size].add_(grad.detach().float().reshape(-1).cpu())
                    offset += size
    for name, vector in vectors.items():
        if not torch.isfinite(vector).all():
            raise ValueError(f"Non-finite gradient: {name}")
    return vectors


def vector_dot(a, b):
    # Double-precision reductions in chunks, without copying whole vectors to FP64.
    return sum(float((a[i:i+65536].double() * b[i:i+65536].double()).sum())
               for i in range(0, a.numel(), 65536))


def vector_norm(a):
    return math.sqrt(max(0.0, vector_dot(a, a)))


def cosine(a, b, floor=1e-10):
    na, nb = vector_norm(a), vector_norm(b)
    return None if min(na, nb) <= floor else max(-1.0, min(1.0, vector_dot(a, b)/(na*nb)))


def gradient_metrics(vectors, alpha, displacement=None, floor=1e-10):
    c, s = vectors["conciseness"], vectors["sentiment"]
    result = {f"norm_{name}": vector_norm(v) for name, v in vectors.items()}
    wc, ws = alpha*vector_norm(c), (1-alpha)*vector_norm(s)
    result.update(weighted_norm_conciseness=wc, weighted_norm_sentiment=ws,
                  rho=wc/ws if ws > floor else None)
    names = list(vectors)
    for i, left in enumerate(names):
        for right in names[i+1:]:
            result[f"cos_{left}__{right}"] = cosine(vectors[left], vectors[right], floor)
    reconstructed = alpha*c + (1-alpha)*s
    direct = vectors["minority_oracle"]
    error = vector_norm(direct-reconstructed)
    result["mixture_error_absolute"] = error
    result["mixture_error_relative"] = error/max(vector_norm(direct), vector_norm(reconstructed), floor)
    if displacement is not None:
        result["norm_update"] = vector_norm(displacement)
        for name, v in vectors.items():
            result[f"cos_update__{name}"] = cosine(displacement, v, floor)
            result[f"dot_update__{name}"] = vector_dot(displacement, v)
        result["contribution_conciseness"] = alpha*result["dot_update__conciseness"]
        result["contribution_sentiment"] = (1-alpha)*result["dot_update__sentiment"]
    return result


def counterfactual_metrics(vectors, weights=(.5, .75, .9, 1.), displacement=None):
    rows = []
    for alpha in weights:
        v = dict(vectors, minority_oracle=alpha*vectors["conciseness"] + (1-alpha)*vectors["sentiment"])
        rows.append(dict(evaluation_alpha=alpha, **gradient_metrics(v, alpha, displacement)))
    return rows


def summarize_samples(samples):
    keys = ("policy_length", "scorer_length", "scorer_length_untruncated", "conciseness",
            "sentiment", "minority_oracle", "minority_rm", "pooled_rm")
    summary = {"n": len(samples), "n_prompts": len({s["prompt_id"] for s in samples})}
    for key in keys:
        if not samples or any(key not in s for s in samples):
            continue
        values = [s[key] for s in samples]
        summary.update({f"{key}_{stat}": value for stat, value in describe(values).items()})
        summary[f"{key}_within_prompt_variance"] = within_prompt_variance(values, [s["prompt_id"] for s in samples])
    for key in ("cap_hit", "eos", "scorer_truncated"):
        if samples and all(key in s for s in samples):
            summary[f"{key}_rate"] = statistics.mean(s[key] for s in samples)
    if samples:
        summary["unique_policy_lengths"] = len({s["policy_length"] for s in samples})
    if samples and all("conciseness" in s and "sentiment" in s for s in samples):
        cm, sm = summary["conciseness_mean"], summary["sentiment_mean"]
        summary["attribute_covariance"] = statistics.mean((s["conciseness"]-cm)*(s["sentiment"]-sm) for s in samples)
    return summary


def snapshot_parameters(model):
    import torch
    return torch.cat([p.detach().float().reshape(-1).cpu() for _, p in trainable_parameters(model)])


def parameter_displacement(model, before):
    """Subtract CPU FP32 snapshots even if the policy is trained in bf16."""
    return snapshot_parameters(model) - before


def accumulated_ascent_gradient(model):
    import torch
    return torch.cat([(-p.grad.detach().float().reshape(-1).cpu() if p.grad is not None
                       else torch.zeros(p.numel())) for _, p in trainable_parameters(model)])
