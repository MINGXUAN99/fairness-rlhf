"""Minimal, opt-in artifacts for the single-GPU windowed PCGrad experiment."""
import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from transformers import GenerationConfig, TrainerCallback


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def read_jsonl(path):
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        for row in rows:
            handle.write(json.dumps(row, allow_nan=False) + '\n')


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as handle:
        handle.write(json.dumps(row, allow_nan=False) + '\n')


def schedule(horizon=800):
    return sorted({0, horizon, *range(50, horizon + 1, 50),
                   *(x for x in (399, 400, 401) if x <= horizon)})


def summary(values):
    a = np.asarray(values, dtype=float)
    return dict(mean=float(a.mean()), median=float(np.median(a)),
                p10=float(np.quantile(a, .1)), p90=float(np.quantile(a, .9)),
                sd=float(a.std(ddof=1)) if len(a) > 1 else None)


@contextmanager
def isolated_evaluation(model):
    """Restore every module's mode and all RNG streams even when generation fails."""
    py_state, np_state = random.getstate(), np.random.get_state()
    modes = [(m, m.training) for m in model.modules()]
    with torch.random.fork_rng():
        try:
            model.eval()
            with torch.no_grad():
                yield
        finally:
            for module, mode in modes:
                module.training = mode
            random.setstate(py_state)
            np.random.set_state(np_state)


def response_record(tokenizer, query, response, cap, scoring_text):
    query = [int(x) for x in query if int(x) != tokenizer.pad_token_id]
    response = list(map(int, response))
    eos = tokenizer.eos_token_id in response
    if eos:
        response = response[:response.index(tokenizer.eos_token_id) + 1]
    elif tokenizer.pad_token_id in response:
        response = response[:response.index(tokenizer.pad_token_id)]
    prompt = tokenizer.decode(query, skip_special_tokens=True)
    continuation = tokenizer.decode(response, skip_special_tokens=True)
    full = tokenizer.decode(query + response, skip_special_tokens=True)
    return dict(prompt_id=digest(query), prompt=prompt, continuation=continuation,
                scored_text=full if scoring_text == 'full' else continuation,
                generation_length=len(response), eos=eos,
                generation_cap_reached=len(response) >= cap,
                generation_truncated=not eos and len(response) >= cap)


class WindowedMonitor(TrainerCallback):
    def __init__(self, run_dir, inputs, tokenizer, args, flush):
        self.root = Path(run_dir)
        self.tokenizer, self.args, self.flush = tokenizer, args, flush
        self.panel = read_jsonl(Path(inputs) / 'evaluation.jsonl')
        self.updates = schedule(args.expected_updates)
        self.pending = {}
        self.trainer = None
        self.root.mkdir(parents=True, exist_ok=False)
        write_jsonl(self.root / 'prompts/evaluation.jsonl', self.panel)

    def capture(self, update, queries, responses, raw, normalized, kl):
        rows = []
        for i, (q, r) in enumerate(zip(queries.tolist(), responses.tolist())):
            row = response_record(self.tokenizer, q, r, self.args.training_response_length,
                                  self.args.scoring_text)
            row.update(run=self.root.name, seed=self.args.seed, update=update,
                       phase='pre_update', response_index=i,
                       learned_rewards_raw=raw[i].float().tolist(),
                       learned_rewards_normalized=normalized[i].float().tolist())
            rows.append(row)
        write_jsonl(self.root / f'rollouts/train/update_{update:04d}.jsonl', rows)
        lengths = torch.tensor([r['generation_length'] for r in rows], device=kl.device).clamp(min=1)
        self.pending = dict(update=update, phase='pre_update', count=len(rows),
                            generation_length=summary([r['generation_length'] for r in rows]),
                            eos_rate=float(np.mean([r['eos'] for r in rows])),
                            generation_cap_rate=float(np.mean([r['generation_cap_reached'] for r in rows])),
                            kl_total=float(kl.sum(1).mean()),
                            kl_mean_per_token=float((kl.sum(1) / lengths).mean()))

    def on_log(self, args, state, control, logs=None, **kwargs):
        if self.pending:
            if self.pending['update'] != state.global_step:
                raise RuntimeError('Rollout/log update mismatch')
            append_jsonl(self.root / 'metrics/train_updates.jsonl',
                         {**self.pending, 'ppo': logs or {}})
            self.pending = {}

    def on_train_begin(self, args, state, control, **kwargs):
        self.evaluate(0)

    def on_step_end(self, args, state, control, **kwargs):
        self.flush()
        if state.global_step in self.updates:
            self.evaluate(state.global_step)

    def evaluate(self, update):
        model = self.trainer.accelerator.unwrap_model(self.trainer.model).policy
        device = next(model.parameters()).device
        config = GenerationConfig(max_new_tokens=256, do_sample=True, temperature=.7,
                                  top_p=.95, top_k=0, eos_token_id=self.tokenizer.eos_token_id,
                                  pad_token_id=self.tokenizer.pad_token_id)
        rows = []
        with isolated_evaluation(model):
            for offset in range(0, len(self.panel), 16):
                batch = self.panel[offset:offset + 16]
                # Frozen grouping/order gives each prompt the same batch RNG
                # mapping in both arms while avoiding 256 sequential generations.
                eval_seed = int(digest([[item['prompt_id'] for item in batch], update])[:8], 16)
                random.seed(eval_seed)
                np.random.seed(eval_seed)
                torch.manual_seed(eval_seed)
                encoded = self.tokenizer.pad(
                    [{'input_ids': item['input_ids'], 'attention_mask': [1] * len(item['input_ids'])}
                     for item in batch], padding=True, return_tensors='pt').to(device)
                output = model.generate(**encoded, generation_config=config)
                width = encoded['input_ids'].shape[1]
                for index, item in enumerate(batch):
                    row = response_record(self.tokenizer, item['input_ids'],
                                          output[index, width:].tolist(), 256, self.args.scoring_text)
                    row.update(run=self.root.name, seed=self.args.seed, update=update,
                               phase='post_update', evaluation_seed=eval_seed,
                               evaluation_batch=offset // 16, evaluation_batch_position=index)
                    rows.append(row)
        write_jsonl(self.root / f'rollouts/eval/update_{update:04d}.jsonl', rows)
