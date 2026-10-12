# Windowed PCGrad: experiment plan

Status: windowed PCGrad already exists; launch scripts now expose Early/Late.
No training launched. Extended instrumentation below remains planned.
Updated 2026-10-12. Current scope: seed 43 only, Early and Late (two new runs).
Reuse existing None and Full results; do not retrain them.

## 1. Question and scope

At alpha = 0.8, does applying full two-objective PCGrad early versus late
change subsequent reward trajectories and final worst-group reward?
Compare updates 1-400 versus 401-800, with SumGrad outside each window,
and compare against existing Full PCGrad. Existing vanilla None is an
additional baseline, not a matched no-surgery (SumGrad) control.
Do not assume Early will win, reuse unverified rebuttal results, or label
this experiment alpha = 0.75.

## 2. Full PCGrad inside the intervention window

Use `--method pcgrad` with `--pcgrad_window_start/--pcgrad_window_end`.
This uses the same two objectives and projection function as Full PCGrad.
Let g_min/g_maj be minority/majority reward PPO loss gradients and g_kl the
separate KL loss gradient. Set d = dot(g_min, g_maj). Inside the window if d < 0:

    g_min_proj = g_min - d / max(||g_maj||^2, 1e-12) * g_maj
    g_maj_proj = g_maj - d / max(||g_min||^2, 1e-12) * g_min
    g_update = g_min_proj + g_maj_proj + g_kl

Both projections use the original other gradient. Outside the window, or if d >= 0:

    g_update = g_min + g_maj + g_kl

Surgery does not force a direction change on aligned gradients. KL and value
updates remain as in Full PCGrad. Adam displacement can differ from the raw
gradient direction. This experiment does not use `--vanilla_deconflict`.

    Early: --method pcgrad --seed 43 --pcgrad_window_start 1 --pcgrad_window_end 400
    Late:  --method pcgrad --seed 43 --pcgrad_window_start 401 --pcgrad_window_end 800

Bounds are inclusive outer PPO updates shared by all inner steps. Omit both
bounds for Full PCGrad. On the configured GPU machine run:

    bash scripts/train_windowed_pcgrad.sh        # both, sequentially
    bash scripts/train_windowed_pcgrad.sh early  # Early only
    bash scripts/train_windowed_pcgrad.sh late   # Late only

The wrapper fixes seed 43, alpha 0.8, GPT-2, proportion 0.2 and 51,200 episodes.
Other settings come from `train_ppo_policy.sh`, including disabled reward
normalization by default; match them to the existing Full run. Model and cosine
CSV names include window bounds and a run tag. Only the existing pre-projection
cosine logging is implemented; extended diagnostics below remain planned.

## 3. Run matrix and exact meaning of a step

| Arm | Correction window (inclusive) | Outside window | Execution |
| --- | --- | --- | --- |
| None | Never | Verify existing method: vanilla or SumGrad | Reuse existing result; verify configuration |
| Early | 1–400 | SumGrad + KL | New run, seed 43 |
| Late | 401–800 | SumGrad + KL | New run, seed 43 |
| Full | 1–800 | Not applicable | Reuse existing Full PCGrad |

Use alpha = 0.8, minority proportion = 0.2, GPT-2, and training seed 43:
two new main runs (Early and Late). Hold policy initialization, prompt order,
training settings, and minority/majority reward checkpoints fixed between them.
Seed 43 controls PPO training; do not imply independently seeded reward-model
training unless that is actually used. Additional seeds are outside this scope.

“Step” means one outer rollout/PPO update, not a minibatch optimizer step,
episode, or PPO epoch. All inner optimizer steps associated with an outer
update use its intervention-window decision. Retain 3 PPO epochs per rollout.
Use disjoint windows: step 400 belongs only to Early; step 401 only to Late.

Target exactly 800 outer updates, with a resolved rollout batch of 64 responses:
51,200 episodes on one GPU if the pinned TRL version resolves batch size to 64.
Inspect and assert `num_total_batches == 800` and the resolved batch/minibatch
sizes at startup; do not rely on the arithmetic without checking TRL's config.
The current 50,000-episode default does not give exactly 800 batches of 64.

Route both new arms through the same windowed PCGrad code with
gradient accumulation = 1. Reuse existing None and Full artifacts without
launching those arms again. Record their paths/revisions and verify alpha,
seed, initialization, reward models, horizon, resolved batch/accumulation,
decoding and scoring conventions before treating them as matched controls.
Full must use full-horizon `--method pcgrad` to be a matched control. A vanilla
None differs in objectives as well as surgery. Label any mismatches
and missing diagnostics explicitly; do not assume old runs used the new hooks.

Keep the established settings initially: learning rate 3e-6, cosine scheduler,
mean per-token KL with coefficient 0.1, reward whitening, bf16, training generation
cap 512 and evaluation cap 256. Verify these against the intended baseline.
Record actual resolved settings, including sampling temperature/top-p/top-k,
reward calibration, tokenizers, starting policy, reference and value models.
Do not silently replace unavailable checkpoints or change normalization by arm.

## 4. Gradient measurements

### Every inner optimizer step: training-gradient trace

Instrument `_windowed_pcgrad_compute_and_apply` to record these quantities on the
same minibatch and parameters before `optimizer.step()`:

| Field | Meaning |
| --- | --- |
| outer_update, ppo_epoch, minibatch, optimizer_step | Unambiguous timing |
| window_active, conflict_detected, projection_applied | Separate scheduled and actual intervention |
| cosine_before | cos(g_min, g_maj), before projection |
| cosine_after | cos(g_min_proj, g_maj_proj), unchanged gradients outside the window |
| norm_min, norm_maj, norm_sum, norm_combined, norm_kl, norm_update | Gradient magnitudes |
| dot_before, dot_after | Group-gradient dot products before/after projection |
| correction_norm, correction_fraction | Norm of combined reward gradient minus original sum; fraction relative to norm_sum |
| valid_cosine | False for zero/near-zero norms; use missing values, not artificial zero |

Both g_min and g_maj are **loss gradients** (descent convention). Negative
cosine means their descent directions conflict. Do not mix a loss gradient
with a reward-ascent gradient without reversing one sign.

The current `--sumgrad_cosine_csv` records only the before-correction cosine
in this path. Its `cos_g1_g2` field refers to minority/majority reward gradients,
excluding KL. Give new fields explicit names and separately record total-gradient
(including KL) alignment with each group gradient. Compute dot products/norms in float32 or higher, flush each outer update,
and flush/close on interruption. Log scalars, not complete gradient vectors.

### Actual optimizer movement

Adam's momentum and preconditioning mean g_update is not the actual parameter
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
Record its `diag_beta` separately from training KL.

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
and original learned minority/majority rewards (before and after normalization).
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
claiming exact resume. In particular, restore both group reward models and the window
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

1. In `src/imdb/ppo_training/unified_ppo_training.py`, add explicit before/after gradient fields,
   outer/inner counters, optional actual-displacement measurements, and safe flushes.
2. Add rollout capture before optimization and fixed-panel evaluation/checkpoint
   hooks after complete outer updates. Reuse the scorer or score archived text
   offline using `src/helper/score_responses_unified.py` conventions.
3. Extend the existing Early/Late launcher (seed = 43) as needed with
   alpha = 0.8, pinned checkpoint IDs, per-run paths and resolved-config assertions.
   Accept existing None/Full artifact references for analysis, not new training jobs.
   The launcher now supports window bounds and PCGrad cosine CSVs; additional
   resolved-config assertions and provenance capture remain to be implemented.
4. Add an analysis script that joins traces and snapshots by run and outer update,
   verifies complete schedules, computes final metrics, and builds paired text tables.

These are proposed additions, not currently available logging switches. Do not
start the full sweep with only the existing flags and expect all artifacts above.

## 8. Validation and execution order

### CPU checks on this machine

- Synthetic conflicting/aligned/zero gradients: projection only in active windows;
  projected group-gradient dot product nonnegative within numerical tolerance;
  finite output. Full-window PCGrad must match Full PCGrad; outside-window steps
  must match SumGrad for identical inputs.
- Boundaries: 399/400/401/800/801; Early and Late never overlap.
- Adam check: record displacement separately; do not assert it must align just
  because the corrected raw gradient does.
- Toy logged/unlogged run: monitoring preserves RNG, model mode and gradients.
- Scoring tests: lengths 0, 256 and >256; F computed from group means; consistent
  prompt handling, alpha and sign conventions.
- Read-back checks for JSONL/CSV, snapshot labels and resume append behavior.

### GPU smoke test

Use a tiny horizon with correspondingly shortened disjoint windows, preserving
the full experiment's other settings. Run Early and Late; ensure every scheduled
artifact appears. Compare the instrumented no-correction path against an
otherwise identical SumGrad implementation using identical batches and
optimizer settings. Test monitoring on/off equivalence and one resume boundary.
The dynamic TRL source patches must match the pinned library version and fail
loudly if any expected replacement is missing.

### Full run

1. Verify provenance and comparability of the existing None and Full results.
2. Run Early and Late for seed 43, 800 updates each, checking the full schedule,
   runtime, disk use, and absence of missing observations. Complete both arms
   regardless of which result is favorable; do not launch seeds 42 or 44.
3. Run final evaluation on the complete frozen held-out prompt set for both new
   arms. Reuse comparable existing control evaluations; if checkpoints are
   available, they can be evaluated with the same convention without retraining.
4. Verify backup integrity before releasing the GPU machine.

Use one GPU per run, sequentially if only one is available. Estimate the total
budget from measured smoke/pilot throughput plus evaluation and checkpoint time;
do not assume the old one-hour estimate includes the new instrumentation.
If the baseline does not collapse at alpha = 0.8, inspect configuration consistency
and report the failure to reproduce; do not silently search seeds for collapse.

## 9. Analysis and paper claims

Primary outcome: final fixed-evaluation worst-group reward F at update 800.
Report Early minus Late for seed 43, alongside comparable existing None and Full
results. This is a single-seed diagnostic: do not report across-seed SD or claim
statistical significance or robustness across seeds. Per-response SD describes
response variation and must not be labelled training-seed uncertainty.

Secondary outcomes: fixed-panel reward trajectories and their area under the
curve, reward at update 400, minimum observed F, conflict frequency in each
half, correction frequency/magnitude, and response-length/sentiment trajectories.
Do not select a “best” intermediate checkpoint as the final result.

Suggested figures: (1) raw minority/majority reward-gradient cosine over updates, with correction
windows and optional actual-movement alignment in a separate panel;
(2) fixed-panel minority/majority or F trajectories for Early and Late, adding
existing None/Full trajectories only where compatible observations exist;
(3) length and sentiment trajectories. Shade windows 1–400 and 401–800.
Appendix: per-seed final table and matched-prompt generations over training.

Post-projection nonnegative cosine is largely a consequence of the rule; it is
an implementation check, not independent evidence of better long-term behavior.
Early > Late for seed 43, accompanied by better subsequent reward trajectories,
would be preliminary evidence of a timing effect for this seed, requiring
additional seeds for broader claims. Similar outcomes, improvements from Late,
or failure of a comparable existing Full control must also be reported. Changes in on-policy sampling are a
plausible explanation, not established solely by these traces. Do not claim
irreversibility, universal Pareto improvement, or optimizer superiority.
