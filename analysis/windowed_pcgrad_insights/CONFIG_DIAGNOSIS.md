**A major precision mismatch is the strongest identified explanation for Full
versus Early/Late.** Historical Full stored FP32 parameters with BF16 mixed
precision. Current Early/Late load the trainable policy and value model directly
as BF16. At learning rate 3e−6, direct BF16 parameter storage can round small Adam
updates away. The archived windowed checkpoints show very little parameter
movement, and a CPU Adam probe reproduces this rounding behavior. This is a
strong diagnosis to test, not proof that precision explains the entire gap.

The historical model was inspected at pinned revision
`d6a83a8a9cb58794a9c7bc4640ed0cee5869e6b6` in
[the Full model repository](https://huggingface.co/mxcui/pcgrad-imdb-ppo-prop0.2-alpha0.8-seed43/tree/d6a83a8a9cb58794a9c7bc4640ed0cee5869e6b6).
Its card links to [W&B run 541g6ruj](https://wandb.ai/mxcui-none/huggingface/runs/541g6ruj).
That run's launch arguments, configuration, metric history and sampled completion
tables were read. Its `training_args.bin` was inspected statically with
`pickletools`, without deserializing executable objects. The checkpoint's tensor
header was read separately without downloading its weights.

| Setting | Historical Full | Current Early/Late |
| --- | --- | --- |
| Actual PPO seed | 42 | 43 |
| Outer updates | 782 | 800 |
| Resolved episodes | 50,048 (requested 50,000) | 51,200 |
| Training rollout cap | 53 | 53 |
| Rollout batch / accumulation | 64 / 1 | 64 / 1 |
| PPO epochs per rollout | 3 | 3 |
| Learning rate / schedule | 3e−6 / cosine | 3e−6 / cosine |
| KL coefficient | 0.1 | 0.1 |
| Requested KL convention | mean_kl | mean_kl |
| Reward whitening | true | true |
| Reward-model normalization | disabled | disabled |
| Missing-EOS penalty | 0 | 0 |
| Training temperature | 0.7 | 0.7 |
| BF16 mixed-precision flag | true | true |
| Saved policy parameter dtype | FP32 | BF16 |
| TRL | 1.4.0 | 0.27.2 |
| Transformers | 5.9.0 | 5.1.0 |
| PyTorch | 2.11.0+cu128 | 2.10.0 |
| Accelerate | 1.13.0 | 1.12.0 |
| Datasets | 4.8.6.dev0 | 4.5.0 |
| Reward model IDs | same minority/majority IDs | pinned revisions of those IDs |
| Prompt preprocessing | historical pipeline | frozen serial seed-43 preprocessing |

The name `seed43` is misleading for the Full artifact: saved training arguments,
W&B launch arguments and model card all identify seed42. This is a provenance
mismatch, not evidence that the checkpoint or evaluation is otherwise invalid.
The reward model IDs match, but the historical run did not record immutable RM
revisions in the inspected metadata. Matching names alone cannot establish
identical reward weights.

The saved parameter dtype matters more than the shared `bf16=True` flag. With
FP32 parameters plus BF16 autocast, small optimizer updates can accumulate in
FP32. Loading parameters as BF16 instead also makes this Adam implementation's
moment buffers BF16: `torch.optim.Adam` initializes them with `zeros_like(p)`.
There is no FP32 master-parameter mechanism in the inspected ordinary fused
AdamW path. Current policy, reference, value and reward loads explicitly request
BF16; the policy/value precision change is especially consequential for training.

Directly comparing the final windowed checkpoint with initial GPT-2 rounded to
the checkpoint's BF16 dtype gives:

| Policy | Parameters compared | Changed after 800 updates | Unchanged | Relative L2 displacement |
| --- | ---: | ---: | ---: | ---: |
| Early | 124,439,808 | 1,156,466 (0.9293%) | 99.0707% | 0.00010378 |
| Late | 124,439,808 | 1,156,405 (0.9293%) | 99.0707% | 0.00010374 |

Both final layer-normalization scale tensors are unchanged. This is not a claim
that a small parameter displacement cannot affect outputs; it is direct evidence
that most stored parameters did not move at all from rounded initialization.
Unchanged final weights could in principle include parameters that moved and
returned; intermediate optimizer states/checkpoints were not available.

The probe uses a parameter near 0.03, constant gradient 1, ordinary fused AdamW,
learning rate 3e−6, no weight decay, and 100 updates:

| Parameter storage | Initial | Final | Adam moment storage |
| --- | ---: | ---: | --- |
| FP32 | 0.0299999993 | 0.0296999272 | FP32 |
| BF16 | 0.0300292969 | 0.0300292969 | BF16 |

For parameters in this magnitude range, a BF16 spacing is about 0.000122,
roughly 41 times the nominal 3e−6 Adam step. Repeated sub-resolution steps can
be discarded instead of accumulating. This probe is a numerical demonstration,
not a replacement for rerunning PPO with matched precision.

The training histories are also consistent with Full learning and the windowed
policies largely stalling:

| PPO metric | Full first update | Full final update | Early first update | Early final update |
| --- | ---: | ---: | ---: | ---: |
| Learned objective score | 0.0459 | 2.4347 | 0.3027 | 0.3457 |
| Value loss | 5.7127 | 0.1643 | 7.2324 | 6.9059 |
| Approximate PPO update KL | 0.000781 | 4.03e−10 | 0.06084 | 0.06606 |

Historical Full's sampled completion tables show long responses initially and
short positive responses in later snapshots. Thus its short outputs are also
visible in the training run; they are not solely a consequence of the separate
evaluation prompt set. These small qualitative tables are not an aggregate
EOS-rate estimate. In particular, the logged `val/num_eos_tokens` counts tokens
and can count padding when PAD equals EOS; it is not a response-level rate.

The approximately 0.06 update-KL values persisting in current runs even at a
near-zero final learning rate deserve a separate audit of rollout versus
optimization log-probability computation. Full's corresponding value tends to
zero. This is an anomaly, not a verified second bug; dtype, generation/forward
paths and upstream versions may affect these measurements.

The TRL 1.4.0 wheel was downloaded for inspection without installing it. Its PPO
training loop is largely the same as installed 0.27.2. One relevant difference is
that 1.4.0 omits the standard gradient-clipping line; the current PCGrad patch
explicitly removes that line from 0.27.2. Therefore the inspected clipping
difference is already accounted for and should not be presented as the root
cause. Broader Transformers/PyTorch numerical differences remain uncontrolled.

Exact historical custom source is unresolved. W&B records Git commit `ca7a7cf`,
but that committed training file has no PCGrad implementation. The run therefore
used local changes beyond its recorded commit. We cannot reconstruct its exact
gradient surgery from that commit alone or assume the newer committed code is
identical. Its reward IDs and other launch settings are established by the run
metadata, but its complete custom implementation is not.

The first corrective experiment should restore FP32 storage for trainable
policy/value parameters while retaining `bf16=True` for mixed-precision forward
passes. To match historical Full most closely, also check frozen reference/RM
precision and avoid changing unrelated objectives or schedules. Do not use
learning-rate inflation as a substitute for fixing precision. Before a long run,
verify that parameters and Adam moments remain FP32, small steps accumulate,
the value loss can decrease, and rollout/optimization log probabilities agree
when the policy is held fixed. Then compare Full, Early, Late and SumGrad under
the same implementation and environment. The existing windowed results should
not be used to support irreversibility until this issue is resolved.

Reproduce the weight audit and CPU probe with
`venv/bin/python analysis/windowed_pcgrad_insights/compare_precision.py`.
Results are in [precision_audit.json](precision_audit.json). No training code,
packages or original run artifacts were changed by this investigation.
