# Conciseness-gradient hypothesis in the α = 0.75 rehearsal

The measured gradients argue against sentiment dominance during this rehearsal. At updates 25 and 26 the raw conciseness gradient was smaller than the raw sentiment gradient. The 0.75 conciseness weight nevertheless made conciseness the larger component of the oracle minority gradient on both actual training batches and at every fixed-panel checkpoint.

This run trained vanilla GPT-2 for 26 PPO updates on 1,664 IMDb episodes with seed 42 and a 512-token generation cap. The rebuttal target used a 53-token cap and a 782-update schedule. This rehearsal covers one seed and only two measured optimizer steps, both near the end of its own shortened learning-rate schedule. Evidence for the historical late-training mechanism requires measurements under the historical settings.

## Attribute-gradient magnitudes

C denotes the conciseness score and S the sentiment score. The pre-specified ratio is 0.75 × ‖gC‖ / (0.25 × ‖gS‖). A ratio below 1 favors sentiment, and below 0.1 indicates a tenfold imbalance. At this alpha, the raw sentiment gradient must exceed three times the raw conciseness gradient for sentiment to dominate. Raw gradient magnitudes on the training batches and on the diagnostic panel use different samplers and baselines. Compare the attributes within each stream.

| Sample | Update | Raw C norm | Raw S norm | 0.75 × C norm | 0.25 × S norm | Ratio | Four-block ratio range |
|---|---|---|---|---|---|---|---|
| clean_panel | 0 | 67.4 | 78.4 | 50.6 | 19.6 | 2.58 | 2.35–2.58 |
| clean_panel | 25 | 67.5 | 79.9 | 50.6 | 20 | 2.53 | 2.07–2.52 |
| clean_panel | 26 | 70.9 | 93.5 | 53.2 | 23.4 | 2.28 | 1.96–2.82 |
| training_rollout | 25 | 155 | 192 | 116 | 48.1 | 2.42 | — |
| training_rollout | 26 | 155 | 213 | 116 | 53.2 | 2.18 | — |

The independent-generation panel used 128 fixed prompts with four responses per prompt and a leave-one-out baseline. The training stream used the 64 responses actually sampled by PPO at each measured update. The four blocks partition the panel by prompt and show sampling sensitivity within this seed. They do not supply four independent training replications.

## Where the measured optimizer steps went

Each row measures the first synchronized optimizer displacement at that update. The two signed contributions are each attribute's gradient dotted with that displacement and multiplied by its objective weight. Their sum predicts the first-order change in the oracle minority score. The learned minority reward model has its own gradient.

| Update | Step norm | Cosine with C | Cosine with S | Cosine with learned minority | C contribution | S contribution | Oracle minority dot step |
|---|---|---|---|---|---|---|---|
| 25 | 0.00016 | 0.00272 | 0.00074 | 0.0053 | 5.07e-05 | 5.71e-06 | 5.64e-05 |
| 26 | 3.13e-05 | 0.000964 | 0.00149 | 0.00134 | 3.5e-06 | 2.48e-06 | 5.98e-06 |

Both measured displacements had positive, very small cosines with the oracle and learned minority gradients. Conciseness supplied the larger positive contribution on both steps. At evaluation weight 1, removing sentiment from the oracle mixture left the cosine positive at both updates (0.00272 and 0.000964). The learning rates had fallen to 4.36 × 10⁻⁸ and 1.09 × 10⁻⁸. Small realized steps and weak directional alignment are compatible with limited progress even with a sizable conciseness gradient. This rehearsal does not identify which factor determines the paper's outcome.

## Length of actual training rollouts

The median response was 512 policy tokens at every one of the 26 updates, excluding EOS. The per-update mean ranged from 387 to 442 tokens. Of 1,664 responses, 1,073 reached the 512-token cap (64.5%). The other 591 averaged 235.8 tokens. Training responses were long from the start and retained a substantial shorter tail.

| PPO updates | Responses | Mean length | Length SD | At cap |
|---|---:|---:|---:|---:|
| 1–13 | 832 | 420.5 | 153.5 | 66.3% |
| 14–26 | 832 | 407.3 | 158.8 | 62.6% |
| All | 1,664 | 413.9 | 156.3 | 64.5% |

The equal halves give a descriptive time comparison. Training samples usually have one response per prompt, so the repeated-response panel supplies the within-prompt variance check.

## Held-out outcomes

The held-out panel contained a separate set of 128 source-review prompts with four responses each. Length excludes EOS. The benchmark conciseness reward uses the scorer's capped token count on the prompt and response. The diagnostic panel's within-prompt conciseness variance was 0.0996 initially and 0.0999 at update 26, despite frequent generation-cap hits.

| Update | Length mean | Length SD | At generation cap | Scorer truncated | Within-prompt C variance | Within-prompt S variance | C reward | S reward |
|---|---|---|---|---|---|---|---|---|
| 0 | 420 | 155 | 68.6% | 51.8% | 0.0871 | 0.0974 | 0.179 | 0.533 |
| 25 | 401 | 169 | 62.3% | 47.5% | 0.106 | 0.101 | 0.214 | 0.579 |
| 26 | 405 | 164 | 63.1% | 47.1% | 0.0924 | 0.0977 | 0.209 | 0.586 |

| Held-out outcome | Final minus initial | 95% prompt-bootstrap interval |
|---|---|---|
| policy_length | -15.5 | [-35.8, 4.43] |
| conciseness | 0.0301 | [-0.00767, 0.0681] |
| sentiment | 0.0521 | [0.0117, 0.0923] |
| minority_oracle | 0.0356 | [0.00488, 0.0672] |
| minority_rm | 0.0844 | [0.00807, 0.16] |
| cap_hit | -0.0547 | [-0.109, 0.00195] |

Intervals resample the 128 paired prompt IDs 10,000 times, keeping the four responses together within each prompt and checkpoint. They describe uncertainty across held-out prompts for this run. Gradient and optimizer results use the separately recorded diagnostic streams.

Held-out sentiment rose by 0.052. The estimated conciseness change was +0.030, with a prompt-bootstrap interval spanning zero. The rewards alone cannot identify which gradient dominated. The directly measured norms and signed contributions favor conciseness at both sampled training updates.

[Training length by update](../results/conciseness_gradient_collapse/alpha075-seed42-rehearsal/training_rollout_lengths.png) · [Hypothesis and decision criteria](conciseness_gradient_collapse.md) · [Run audit](../results/conciseness_gradient_collapse/alpha075-seed42-rehearsal/audit.json) · [Detailed numerical results](../results/conciseness_gradient_collapse/alpha075-seed42-rehearsal/hypothesis_metrics.json) · [Training rollout figure](../results/conciseness_gradient_collapse/alpha075-seed42-rehearsal/diagnostics/figures/mechanism_training_rollout.png)
