# Windowed gradient correction: experiment plan

Status: planning only. No training launched and no training code changed.
Prepared 2026-10-12 for implementation and execution on a GPU machine.

## 1. Question and scope

At preference divergence alpha = 0.8, does removing the component of the
pooled PPO gradient that conflicts with the minority objective early in
training change the subsequent reward trajectory and final worst-group reward?
Compare correction during updates 1–400 against updates 401–800, with
no-correction and full-correction controls.

This is a diagnostic intervention, not an optimizer ranking. Do not assume
early correction will win. The previous alpha = 0.7 datasets yielded nearly
identical early and late results; do not reuse the rebuttal's unverified
0.736/0.307 results or label these new runs alpha = 0.75.

## 2. Use the vanilla deconfliction intervention

Use the existing `--method vanilla --vanilla_deconflict` path. It computes
the pooled, KL-inclusive clipped PPO loss gradient g_pool and a separate
minority reward PPO loss gradient g_min on the same training minibatch.
Within the specified window, when dot(g_pool, g_min) < 0:

    g_corrected = g_pool - dot(g_pool, g_min) / ||g_min||² * g_min

Otherwise g_corrected = g_pool. Use the implementation's numerical safeguard
for a near-zero denominator. The minority gradient acts only as a guard;
it is not added as a second objective. The value update remains unchanged.

Name this experiment “windowed correction of vanilla PPO,” not “early-only
PCGrad-PPO.” The separate `--method pcgrad` window mode combines two group
objectives and uses SumGrad outside the window; it answers a different question.

## 3. Run matrix and exact meaning of a step

| Arm | Correction window (inclusive) | Outside window |
| --- | --- | --- |
| None | Never | Pooled vanilla PPO gradient |
| Early | 1–400 | Pooled vanilla PPO gradient |
| Late | 401–800 | Pooled vanilla PPO gradient |
| Full | 1–800 | Not applicable |

Use alpha = 0.8, minority proportion = 0.2, GPT-2, and paired training seeds
42, 43, 44: 12 main runs. Hold policy initialization and the pooled/minority
reward checkpoints fixed across arms; each seed block shares prompt order
and training settings. Use seed-specific initial policies identically within
a block if initialization is stochastic. Report that seeds vary PPO training,
not reward-model training, unless separate RM seeds are actually used.

“Step” means one outer rollout/PPO update, not a minibatch optimizer step,
episode, or PPO epoch. All inner optimizer steps associated with an outer
update use its intervention-window decision. Retain 3 PPO epochs per rollout.
Use disjoint windows: step 400 belongs only to Early; step 401 only to Late.

Target exactly 800 outer updates, with a resolved rollout batch of 64 responses:
51,200 episodes on one GPU if the pinned TRL version resolves batch size to 64.
Inspect and assert `num_total_batches == 800` and the resolved batch/minibatch
sizes at startup; do not rely on the arithmetic without checking TRL's config.
The current 50,000-episode default does not give exactly 800 batches of 64.

Route all four arms through the same instrumented deconfliction code with
gradient accumulation = 1, including None with projection disabled. This avoids
the ordinary vanilla path's different accumulation setup. The existing code
can represent None with a window outside the horizon (801–801), but an explicit
`projection_enabled=false` option is preferable and must be logged.

Keep the established settings initially: learning rate 3e-6, cosine scheduler,
mean per-token KL with coefficient 0.1, reward whitening, bf16, training generation
cap 512 and evaluation cap 256. Verify these against the intended baseline.
Record actual resolved settings, including sampling temperature/top-p/top-k,
reward calibration, tokenizers, starting policy, reference and value models.
Do not silently replace unavailable checkpoints or change normalization by arm.

## 4. Gradient measurements

### Every inner optimizer step: training-gradient trace

Instrument `_deconflict_compute_and_apply` to record these quantities on the
same minibatch and parameters before `optimizer.step()`:

| Field | Meaning |
| --- | --- |
| outer_update, ppo_epoch, minibatch, optimizer_step | Unambiguous timing |
| window_active, conflict_detected, projection_applied | Separate scheduled and actual intervention |
| cosine_before | cos(g_pool, g_min), before correction |
| cosine_after | cos(g_corrected, g_min), before optimizer transformation |
| norm_pool, norm_min, norm_corrected | Gradient magnitudes |
| dot_before, dot_after | Unnormalized alignment |
| removed_norm, removed_fraction | Magnitude removed; fraction relative to norm_pool |
| valid_cosine | False for zero/near-zero norms; use missing values, not artificial zero |

Both g_pool and g_min are **loss gradients** (descent convention). Negative
cosine means their descent directions conflict. Do not mix a loss gradient
with a reward-ascent gradient without reversing one sign.

The current `--sumgrad_cosine_csv` records only the before-correction cosine
in this path. Its `cos_g1_g2` name is misleading here: the vectors are the pooled
and minority gradients, not the two group gradients. Give the new fields explicit
names. Compute dot products/norms in float32 or higher, flush each outer update,
and flush/close on interruption. Log scalars, not complete gradient vectors.

### Actual optimizer movement

Adam's momentum and preconditioning mean g_corrected is not the actual parameter
displacement. At outer updates 1, 50, 100, ..., 800 and 399, 400, 401, measure
each inner step's delta_theta = theta_after - theta_before using a temporary
copy of trainable policy parameters. Record:

    cosine_actual = cos(delta_theta, -g_min_before_step)
    minority_first_order_change = -dot(g_min_before_step, delta_theta)

Positive values indicate alignment with the local minority-loss descent
direction; they do not guarantee higher evaluated reward after a finite step.
Also record norm_delta and any skipped optimizer step. If memory/time permits,
collect these for every inner step. Do not call the raw-gradient trace the
“actual PPO update” without this distinction.

### Aggregation

Keep every raw row. Summarize each outer update with mean/median cosine,
fraction of negative cosines, projection frequency, and gradient-norm summaries.
An average of minibatch cosines is not the cosine of an aggregated gradient.
Plot unsmoothed traces faintly with a fixed smoothing window for readability;
perform quantitative comparisons on unsmoothed data.

The existing fixed-prompt `--diag_every` callback computes separate surrogate
gradients on fresh generations. It is not the exact clipped-PPO training-gradient
trace above. Retain it only as an optional, separately labelled diagnostic.
Its default `diag_beta=0` also differs from the KL-inclusive target here.

## 5. Rollout and evaluation logging every 50 updates

Save two distinct sources; do not merge their trajectories without labels.

### A. Actual training rollouts

At rollout batches 50, 100, ..., 800 (plus 1, 399, 400, 401), preserve every
response in that rollout batch, not only successful examples. Record that these
were generated **before** the corresponding outer update's optimization.
Use the current training generation settings, without an extra training rollout.

For every saved response store run/seed/update, immutable prompt ID and text,
prompt and continuation token IDs, generated continuation, the exact text
passed to scoring, EOS/cap flags, generated-token count, scoring-token count,
and original learned pooled/minority rewards (before and after normalization).
Store per-response KL information if available. Offline scoring can add oracle
metrics without loading another classifier alongside PPO.

Log available PPO statistics every outer update: KL (mean and total, labelled),
policy/value loss, entropy, clip fraction, learning rate, learned reward, and
response-length summaries. Setting `logging_steps=50` alone does not save texts
or provide these new diagnostics.

### B. A fixed evaluation panel

Before running the arms, save one manifest of 256 held-out prompt IDs/texts
(disjoint from training) and reuse it across all arms, checkpoints, and seeds.
Sample it independently of outcomes. Evaluate at completed updates
0, 50, 100, ..., 800, plus 399 and 401. Update 400 is already in this schedule.
Thus there are 19 evaluation snapshots per run. Label these **post-update**.

Generate one response per prompt with frozen evaluation decoding settings and
the 256-token cap used by the paper's evaluation. Fix evaluation RNG seeds
by prompt/checkpoint and use the same mapping across arms and training seeds.
Use isolated RNG state, restore Python/NumPy/PyTorch CPU/CUDA RNG and model
mode afterwards, and run only after the current optimization has completed.
Evaluation must not alter training gradients, optimizer state, or data-loader
order. Alternatively, evaluate saved checkpoints offline on the GPU server.

Predetermine 12 of these prompts as the qualitative panel. Show the same prompts
at updates 0, 200, 400, 600, 800 across arms. Archive all 256 responses per snapshot
so later examples are auditable. Do not rewrite generated text; mark display
truncation and compute scores on full scored text.

### Quantitative scoring convention

Use the existing `lvwerra/distilbert-imdb` scorer at a pinned revision.
To match the alpha-sweep notebook, define:

    c = 1 - clip(scoring_length / 256, 0, 1)
    r_min = 0.8*c + 0.2*positiveness
    r_maj = 0.2*c + 0.8*positiveness
    F = min(mean(r_min), mean(r_maj))
    U = 0.2*mean(r_min) + 0.8*mean(r_maj)

F is the minimum of the two group means, not the mean of per-response minima.
Scoring length counts non-padding DistilBERT tokens including special tokens,
with scorer truncation at 512. Save generation length separately. Confirm whether
the historical scorer sees prompt-plus-response or continuation, freeze that
choice, and record it. Do not substitute the stored unnormalized `conciseness`
column (256 minus length) directly into these formulas.

For each snapshot report minority/majority mean and sample SD, F, U, signed
and absolute group gap, sentiment, conciseness, length mean/median/quantiles,
EOS/cap rates, scorer-truncation rate and clipping rate (length > 256).
For 512-token training rollouts, clipping may be common: report it and use the
fixed 256-token evaluation panel for comparisons to final evaluation scores.
Keep unclipped conciseness only as an explicitly labelled secondary diagnostic.
Do not mix conventions between arms or qualitative tables.

## 6. Checkpoints, provenance, and recovery

Current training uses `save_strategy="no"`; periodic durable saves need to be
implemented. Save evaluation-loadable policy checkpoints at 0, 50, ..., 800
and 399, 401. Save resumable state at least every 100 updates and at 399,
400, 401, 800: policy, value model, optimizer, scheduler, RNG, rollout/update
counters, and data-loader/sampler position. Resume only at outer-update boundaries.
Verify an interrupted/resumed short run against an uninterrupted run before
claiming exact resume. In particular, restore the minority guard and the window
configuration; a policy-only checkpoint cannot resume PPO faithfully.

Give each run a unique directory containing:

    manifest.json                  # IDs/revisions, code commit, seed, settings, versions
    prompts/evaluation.jsonl       # shared immutable prompt manifest
    diagnostics/gradient_steps.csv
    diagnostics/gradient_updates.csv
    rollouts/train/update_*.jsonl
    rollouts/eval/update_*.jsonl
    metrics/train_updates.jsonl
    metrics/evaluation.jsonl
    checkpoints/update_*/
    final_evaluation/              # complete held-out set, per-response scores
    logs/train.log

Include alpha, arm, window bounds, seed and a new run tag in model/dataset names;
do not overwrite old alpha = 0.7 repositories. Keep local files as the primary
record and W&B as a mirror. Back up manifests, diagnostics, texts and checkpoints
to persistent storage during the run, with checksums and completion markers.
Save exact Hub revisions; never store authentication tokens in manifests/logs.

## 7. Implementation work before the GPU run

1. In `src/imdb/ppo_training/unified_ppo_training.py`, add a projection-disabled
   arm through the same deconfliction path, explicit before/after gradient fields,
   outer/inner counters, optional actual-displacement measurements, and safe flushes.
2. Add rollout capture before optimization and fixed-panel evaluation/checkpoint
   hooks after complete outer updates. Reuse the scorer or score archived text
   offline using `src/helper/score_responses_unified.py` conventions.
3. Add a dedicated experiment launcher with an explicit four-arm matrix,
   alpha = 0.8, pinned checkpoint IDs, per-run paths and resolved-config assertions.
   The current shell launcher's `RECORD_COSINE` branch only covers SumGrad/MGDA;
   it is not a complete launcher for this vanilla-deconfliction experiment.
4. Add an analysis script that joins traces and snapshots by run and outer update,
   verifies complete schedules, computes final metrics, and builds paired text tables.

These are proposed additions, not currently available logging switches. Do not
start the full sweep with only the existing flags and expect all artifacts above.

## 8. Validation and execution order

### CPU checks on this machine

- Synthetic conflicting/aligned/zero gradients: projection only in active windows;
  corrected dot product nonnegative within numerical tolerance; finite output.
- Boundaries: 399/400/401/800/801; Early and Late never overlap.
- Adam check: record displacement separately; do not assert it must align just
  because the corrected raw gradient does.
- Toy logged/unlogged run: monitoring preserves RNG, model mode and gradients.
- Scoring tests: lengths 0, 256 and >256; F computed from group means; consistent
  prompt handling, alpha and sign conventions.
- Read-back checks for JSONL/CSV, snapshot labels and resume append behavior.

### GPU smoke test

Use a tiny horizon with correspondingly shortened disjoint windows, preserving
the full experiment's other settings. Run all four arms; ensure every scheduled
artifact appears. Compare the instrumented no-correction path against an
otherwise identical pooled PPO implementation using identical batches and
optimizer settings. Test monitoring on/off equivalence and one resume boundary.
The dynamic TRL source patches must match the pinned library version and fail
loudly if any expected replacement is missing.

### Full run

1. Run one 800-update seed block (all four arms) to validate the full schedule,
   baseline behavior, runtime, disk use, and absence of missing observations.
2. If it is technically valid, run the other two prespecified seed blocks
   regardless of whether the preliminary outcome favors early correction.
3. Run final evaluation on the complete frozen held-out prompt set for every arm.
4. Verify backup integrity before releasing the GPU machine.

Use one GPU per run, sequentially if only one is available. Estimate the total
budget from measured smoke/pilot throughput plus evaluation and checkpoint time;
do not assume the old one-hour estimate includes the new instrumentation.
If vanilla does not collapse at alpha = 0.8, inspect configuration consistency
and report the failure to reproduce; do not silently search seeds for collapse.

## 9. Analysis and paper claims

Primary outcome: final fixed-evaluation worst-group reward F at update 800.
Report Early minus Late within each matched seed, alongside None and Full.
Show individual seeds and mean/SD across seeds. Per-response SD is a different
quantity and must not be labelled training-seed uncertainty. Three seeds provide
limited precision; do not make a significance claim from one favorable run.

Secondary outcomes: fixed-panel reward trajectories and their area under the
curve, reward at update 400, minimum observed F, conflict frequency in each
half, correction frequency/magnitude, and response-length/sentiment trajectories.
Do not select a “best” intermediate checkpoint as the final result.

Suggested figures: (1) raw pooled/minority cosine over updates, with correction
windows and optional actual-movement alignment in a separate panel;
(2) fixed-panel minority/majority or F trajectories for all four arms;
(3) length and sentiment trajectories. Shade windows 1–400 and 401–800.
Appendix: per-seed final table and matched-prompt generations over training.

Post-projection nonnegative cosine is largely a consequence of the rule; it is
an implementation check, not independent evidence of better long-term behavior.
Early > Late across seeds, accompanied by better subsequent reward trajectories,
would support a timing effect. Similar outcomes, improvements from Late, or
failure of Full must also be reported. Changes in on-policy sampling are a
plausible explanation, not established solely by these traces. Do not claim
irreversibility, universal Pareto improvement, or optimizer superiority.
