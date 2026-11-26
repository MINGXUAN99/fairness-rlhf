"""Compare saved windowed policies with rounded initialization; probe Adam.

No training models or existing artifacts are modified. Uses the local GPT-2
cache and writes precision_audit.json beside this script.
"""
import json
from pathlib import Path

import torch
from safetensors import safe_open

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / 'runs/windowed-len53-batch64-seed43-20261012_195052'
BASE = Path('/home/mx/.cache/huggingface/hub/models--gpt2/snapshots/'
            '607a30d783dfa663caf39e06633721c8d4cfcd7e/model.safetensors')


def main():
    torch.set_num_threads(4)
    result = {'base_revision': BASE.parent.name, 'arms': {}, 'adam_probes': {}}
    with safe_open(BASE, framework='pt', device='cpu') as base:
        keys = set(base.keys())
        for arm in ('early-win1-400', 'late-win401-800'):
            tensors = []
            with safe_open(RUN / arm / 'final_policy/model.safetensors', framework='pt', device='cpu') as final:
                for key in final.keys():
                    base_key = key if key in keys else key.removeprefix('transformer.')
                    assert base_key in keys, key
                    current = final.get_tensor(key)
                    original = base.get_tensor(base_key).to(current.dtype)
                    delta = current.float() - original.float()
                    tensors.append({'tensor': key, 'count': current.numel(),
                                    'changed': int((current != original).sum()),
                                    'delta_sq': float(delta.square().sum()),
                                    'base_sq': float(original.float().square().sum()),
                                    'max_abs_delta': float(delta.abs().max()),
                                    'dtype': str(current.dtype)})
            n = sum(t['count'] for t in tensors)
            changed = sum(t['changed'] for t in tensors)
            result['arms'][arm] = {'count': n, 'changed': changed,
                                   'changed_fraction': changed / n,
                                   'relative_l2': (sum(t['delta_sq'] for t in tensors) /
                                                   sum(t['base_sq'] for t in tensors))**.5,
                                   'tensors': tensors}
        for dtype in (torch.float32, torch.bfloat16):
            p = torch.nn.Parameter(torch.full((4,), .03, dtype=dtype))
            initial = p.detach().float().tolist()
            opt = torch.optim.AdamW([p], lr=3e-6, weight_decay=0, fused=True)
            for _ in range(100):
                p.grad = torch.ones_like(p)
                opt.step()
                opt.zero_grad()
            result['adam_probes'][str(dtype)] = {
                'steps': 100, 'learning_rate': 3e-6, 'initial': initial,
                'final': p.detach().float().tolist(),
                'moment_dtype': str(opt.state[p]['exp_avg'].dtype)}
    out = Path(__file__).with_name('precision_audit.json')
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps({a: {k: v for k, v in r.items() if k != 'tensors'}
                      for a, r in result['arms'].items()}, indent=2))
    print(json.dumps(result['adam_probes'], indent=2))


if __name__ == '__main__':
    main()
