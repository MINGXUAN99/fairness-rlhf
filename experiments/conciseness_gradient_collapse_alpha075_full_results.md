# Full α = 0.75 conciseness-gradient experiment

Vanilla GPT-2, seed 42. All 782 PPO updates completed, covering 50,048 responses in batches of 64. Training allowed 53 new tokens. Each independent evaluation panel used 128 fixed IMDb prompts, four responses per prompt, and a 256-token cap. The local policy passed the reload-and-generate check.

## Finding

The 53-token training batches consistently supplied a smaller weighted conciseness gradient than sentiment gradient. This supports a weak conciseness signal under the training sampler. The 256-token diagnostic panel showed the opposite ordering from update 75 onward. The two streams also differ in sampling temperature and baseline, so this comparison cannot isolate the causal effect of the cap.

Training responses averaged 52.36 tokens excluding EOS. Across the run, 97.6% reached 53 tokens, and every update had a median length of 53. The median ratio of weighted conciseness to weighted sentiment gradient norms was 0.193 across 17 measured training batches. The ratio ranged from 0.089 to 0.986. A value of 1 means equal weighted magnitudes.

## Held-out outcomes

| Update | Mean response length | Conciseness | Sentiment | Oracle minority reward | Learned minority score |
|---:|---:|---:|---:|---:|---:|
| 0 | 227.070 | 0.541 | 0.510 | 0.533 | 0.010 |
| 100 | 205.510 | 0.585 | 0.870 | 0.656 | 0.480 |
| 300 | 237.402 | 0.517 | 0.985 | 0.634 | 0.584 |
| 600 | 234.381 | 0.527 | 0.982 | 0.640 | 0.587 |
| 782 | 237.936 | 0.520 | 0.983 | 0.636 | 0.619 |

Final minus initial changes, with 95% paired prompt-bootstrap intervals. Each of 10,000 resamples retains all four responses per prompt. These intervals describe prompt uncertainty within this seed.

| Outcome | Change [95% interval] |
|---|---:|
| Length | +10.865 [+4.088, +17.592] |
| Conciseness | -0.021 [-0.034, -0.008] |
| Sentiment | +0.473 [+0.431, +0.513] |
| Oracle minority reward | +0.102 [+0.087, +0.117] |
| Learned minority score | +0.609 [+0.516, +0.700] |

## Gradient and optimizer evidence

The ratio below is (0.75 × conciseness gradient norm) / (0.25 × sentiment gradient norm). Training gradients use the actual pre-update batch. Panel gradients use fresh post-update responses. Signed contributions are the weighted attribute gradients dotted with the first synchronized optimizer displacement of each measured update.

| Update | Training ratio | Panel ratio | Cosine with learned minority gradient | Conciseness contribution | Sentiment contribution |
|---:|---:|---:|---:|---:|---:|
| 25 | 0.097 | 0.883 | +0.00377 | -8.36e-06 | +0.000146 |
| 50 | 0.089 | 0.939 | +0.00273 | -9.68e-06 | +0.00035 |
| 75 | 0.145 | 1.663 | +0.00696 | +2.73e-05 | +0.000546 |
| 100 | 0.124 | 1.868 | +0.01069 | +2.85e-05 | +0.00085 |
| 125 | 0.112 | 2.873 | +0.00599 | +3.75e-05 | +0.000223 |
| 150 | 0.146 | 2.880 | +0.00559 | +1.72e-05 | -0.000393 |
| 175 | 0.193 | 4.750 | +0.00379 | +4.57e-05 | +0.00022 |
| 200 | 0.181 | 6.010 | +0.00773 | -1.45e-05 | +0.000397 |
| 225 | 0.237 | 5.438 | +0.00182 | +4.27e-05 | +1.37e-05 |
| 250 | 0.115 | 9.211 | -0.00419 | +5.49e-05 | +3.4e-05 |
| 275 | 0.348 | 7.807 | +0.01112 | -5.6e-06 | +0.000372 |
| 300 | 0.209 | 6.820 | +0.00748 | +0.000115 | -0.000119 |
| 400 | 0.237 | 7.039 | +0.00914 | +9.36e-06 | +4.84e-05 |
| 500 | 0.508 | 5.787 | +0.00231 | +5.78e-06 | -1.79e-05 |
| 600 | 0.986 | 5.759 | -0.00018 | +2.47e-06 | +3.8e-07 |
| 700 | 0.433 | 3.725 | +0.01726 | +1.01e-05 | -1.65e-06 |
| 782 | 0.201 | 4.370 | +0.00019 | -2.38e-13 | -2.34e-12 |

The learned minority gradient had positive cosine with the measured optimizer step at 15 of 17 checkpoints. Sentiment supplied the larger positive first-order contribution at 9 of 17, including 7 of the first 10. The signed contributions and small cosines give mixed evidence for sentiment driving individual updates. At step 600 the training gradient ratio nearly reached parity (0.986). At step 782 the optimizer displacement norm was only 1.41 × 10⁻⁹ after learning-rate decay, limiting interpretation of its direction.

**Rebuttal scope.** Long, cap-limited training rollouts retain little conciseness variation, and the estimated weighted gradient favors sentiment on every measured training batch in this seed. These measurements do not establish causality, replication across seeds, or a sentiment-driven explanation for every positive minority-aligned update.

[Completion audit](../results/conciseness_gradient_collapse/alpha075-seed42-full/audit.json) · [Held-out evaluations](../results/conciseness_gradient_collapse/alpha075-seed42-full/diagnostics/evaluation.csv) · [Optimizer measurements](../results/conciseness_gradient_collapse/alpha075-seed42-full/diagnostics/update_summary.csv) · [Training trajectory figure](../results/conciseness_gradient_collapse/alpha075-seed42-full/diagnostics/figures/mechanism_training_rollout.png)
