# Windowed PCGrad: minimum experiment plan

Precision correction (2026-10-13): new runs explicitly load policy, value,
reference and unquantized reward parameters in FP32 (`--model_parameter_dtype
float32`), retaining `bf16=True` for PPO mixed-precision computation. The previous
length-53 Early/Late results used BF16 parameter storage and showed little weight
movement; they must not be treated as FP32 timing ablations. Restart from the
original policy in fresh run directories and perform a new GPU smoke test before
long runs. Manifests now record actual model parameter dtypes. Historical Full
also stored FP32 weights, but its seed, versions and exact custom source remain
unmatched; see `analysis/windowed_pcgrad_insights/CONFIG_DIAGNOSIS.md`.

Updated 2026-10-12. Scope: alpha = 0.8, training seed 43, Early and Late
(two new runs). Reuse existing None and Full results; do not retrain them.
Current configuration: PPO training response length 53, rollout batch 64,
seed 43 (unchanged), 800 outer updates and 51,200 episodes per arm. Early uses
updates 1–400 and Late 401–800. Evaluation cap stays 256. Existing reward
checkpoints retain their `len512` names; reward-model naming is independent of
PPO response length through `--training_response_length`.

The previous 512-token/batch-8 experiments were stopped by the user. Their
smoke test passed, but the new 53-token/batch-64 configuration still needs a GPU
smoke test. Existing artifacts remain separate; restart both arms from the
original policy in fresh run directories. Historical Full used PPO seed 42 and
782 updates, so these remain explicit differences from the current experiment.
Must-have monitoring, scoring and comparison are implemented; Adam displacement,
post-projection diagnostics, periodic checkpoints and exact resume are deferred.

Entry points:

    bash scripts/train_windowed_pcgrad.sh smoke  # 4 updates per arm, windows 1–2 / 3–4
    bash scripts/train_windowed_pcgrad.sh        # Early + Late, then offline scoring/comparison
    bash scripts/train_windowed_pcgrad.sh early
    bash scripts/train_windowed_pcgrad.sh late

The wrapper defaults to rollout batch 64 (ROLLOUT_BATCH_SIZE overrides it,
minimum 8 for reward whitening). It computes episodes as updates times batch:
51,200 for a main run and 256 for a 4-update smoke run. Clear or update any old
`ROLLOUT_BATCH_SIZE=8` environment variable before launching.

The wrapper uses `venv/bin/python` (override with PYTHON), a new directory under
`runs/`, and local final-model saves without Hub pushes. Set EXPERIMENT_ROOT to a
stable shared directory when launching arms separately; existing arm directories
are rejected. EXPERIMENT_INPUTS can select an already prepared frozen input set.
MINORITY_RM/MAJORITY_RM override reward IDs before preparation; preparation resolves
immutable revisions for policy, reward models, scorer and dataset. NONE_ARTIFACT
and FULL_ARTIFACT record existing control references without claiming they are
verified. SCORING_DEVICE=cuda scores on the GPU after training; default is CPU.

Manual preparation/scoring/checking:

    python src/helper/windowed_experiment.py prepare --output RUN/inputs \
        --minority_rm MINORITY_ID --majority_rm MAJORITY_ID
    python src/helper/windowed_experiment.py analyze --run_dir RUN/early-win1-400 --check_only
    python src/helper/windowed_experiment.py analyze --run_dir RUN/early-win1-400 --device cuda
    python src/helper/windowed_experiment.py compare --early EARLY_DIR --late LATE_DIR --output COMPARISON_DIR

Preparation freezes seed-43 serial prompt sampling, the train/held-out split and
256 unique evaluation prompts, removing matching short prompt prefixes from the
training split. Both arms use these exact files and package versions. This explicit
preprocessing may differ from historical multiprocessing preprocessing; record that
when judging old controls. Existing generation/scoring code confirms prompt-plus-
continuation scoring, which is the default here.

Experiment mode explicitly sets PPO seed 43 and response length 53. Previously
these were not forwarded to PPOConfig; the installed TRL defaults are seed 42 and
response length 53. Historical runs need their resolved configs checked before
being called matched controls. TRL source replacements now fail on mismatches;
the installed version's optional standard clipping line is replaced by the
existing custom multi-gradient update, retaining that update's clipping behavior.

## 1. Hypothesis and limits

Test the proposed chain:

    no early PCGrad
        -> longer, less concise on-policy training rollouts
        -> late PCGrad does not restore conciseness within the remaining budget

Compare Early PCGrad at updates 1–400 against Late PCGrad at updates 401–800,
using SumGrad outside each window. Examine whether Late deteriorates before
update 400 and whether it recovers after PCGrad starts. Do not assume Early wins.

This comparison can provide preliminary evidence of a timing effect for seed 43.
It cannot isolate longer rollouts as the cause: policy parameters and optimizer
state also evolve differently. Failure to recover by update 800 means failure
within this configuration and the remaining 400-update budget, not impossibility
of correction or irreversibility. Do not claim across-seed robustness or
statistical significance from this single-seed experiment.

## 2. Intervention and run configuration

Use the existing full two-objective PCGrad projection, with separate KL/value
updates unchanged. For minority/majority reward loss gradients g_min and g_maj,
let d = dot(g_min, g_maj). Inside the active window, if d < 0:

    g_min_proj = g_min - d / max(||g_maj||^2, 1e-12) * g_maj
    g_maj_proj = g_maj - d / max(||g_min||^2, 1e-12) * g_min
    g_update = g_min_proj + g_maj_proj + g_kl

Both projections use the original other gradient. Outside the window, or if
d >= 0, use g_update = g_min + g_maj + g_kl. Do not use
`--vanilla_deconflict`. All gradient signs use the loss/descent convention.

| Arm | Inclusive correction window | Outside window | Execution |
| --- | --- | --- | --- |
| None | Never | Verify existing vanilla/SumGrad method | Reuse; label objective differences |
| Early | 1–400 | SumGrad + KL | New run, seed 43 |
| Late | 401–800 | SumGrad + KL | New run, seed 43 |
| Full | Entire horizon | Not applicable | Reuse existing Full PCGrad |

    Early: --method pcgrad --seed 43 --pcgrad_window_start 1 --pcgrad_window_end 400
    Late:  --method pcgrad --seed 43 --pcgrad_window_start 401 --pcgrad_window_end 800

A step is one outer rollout/PPO update. Its window decision applies to every
inner optimizer step. Update 400 belongs only to Early; 401 only to Late.
Omit both bounds for Full PCGrad.

Fix alpha = 0.8, minority proportion = 0.2, GPT-2, seed 43, identical starting
policy, prompt ordering, reward checkpoints and training settings. Use gradient
accumulation = 1 and 3 PPO epochs per rollout. Target exactly 800 outer updates:
51,200 episodes with the resolved rollout batch of 64. Assert
`num_total_batches == 800` and record resolved batch/minibatch sizes at startup;
do not rely only on episode arithmetic. Do not launch seeds 42 or 44.

Retain the intended baseline settings: learning rate 3e-6, cosine scheduler,
mean per-token KL coefficient 0.1, reward whitening, bf16, training generation cap
53 and evaluation cap 256. Distinguish reward whitening from reward-model
normalization, which the launcher disables by default. Verify actual settings
against the intended Full baseline, including decoding and model initialization.
Do not silently substitute checkpoints or change normalization between arms.

## 3. Must-have data: training rollouts

The hypothesis concerns on-policy rollouts, so record statistics for every outer
update using the existing rollout, without generating an extra training batch.
Label them pre-update: rollout k is generated before optimization at update k.

For the entire rollout batch record:

- Generation length mean, median and quantiles; EOS and generation-cap rates.
- Scoring length mean, median and quantiles; fraction with scoring length > 256
  and scorer-truncation rate.
- Oracle conciseness and sentiment means, minority/majority reward means and F,
  once offline scoring is complete.
- Available learned rewards before/after normalization and existing PPO metrics
  (KL with convention labelled, policy/value loss, entropy, clip fraction,
  learning rate). Do not build new intrusive hooks solely for optional PPO metrics.

Archive every response temporarily for offline scoring if the oracle scorer is
not loaded during training. After scoring, retain per-response scores and the
per-update summaries for all updates. Permanently retain complete text batches
at updates 1, 50, 100, ..., 800, plus 399, 400 and 401. Do not discard temporary
text until scoring and completeness checks succeed.

Each archived response needs run/seed/update, prompt ID and text, continuation,
the exact text passed to scoring, generation/scoring lengths, EOS/cap flags and
available learned rewards. Token IDs and per-response KL are optional for this
experiment; do not delay execution to implement them.

Training rollouts use changing prompts and policies. Their curves describe the
training distribution and must remain separate from held-out evaluation curves.

## 4. Must-have data: fixed held-out evaluation

Freeze one manifest of 256 held-out prompt IDs/texts, disjoint from training,
before running either arm. Use the same prompts, evaluation decoding settings,
256-token cap and evaluation RNG mapping across arms and snapshots. Generation
uses fixed ordered batches of 16; the seed hashes the batch prompt IDs and update.
Each response records its batch, position and seed. This mapping is independent
of training seed and identical across arms.

Preferred post-update schedule:

    0, 50, 100, ..., 800, plus 399 and 401 (19 snapshots)

Measure evaluation cost in the smoke test. If this schedule is too expensive,
freeze the reduced schedule before the main runs:

    0, 100, 200, 300, 400, 450, 500, 600, 700, 800 (10 snapshots)

Use one schedule for both arms. Updates 450 and 500 in the reduced schedule
capture the initial response to late intervention. Do not choose the schedule
based on observed main-run outcomes.

Generate one response per prompt after the entire outer update finishes. Save
all responses and score offline to avoid loading an additional classifier during
PPO. Evaluation must preserve Python/NumPy/PyTorch CPU/CUDA RNG, model mode,
training gradients, optimizer state and data-loader ordering.

Use `lvwerra/distilbert-imdb` at a pinned revision and the alpha-sweep convention:

    c = 1 - clip(scoring_length / 256, 0, 1)
    r_min = 0.8*c + 0.2*positiveness
    r_maj = 0.2*c + 0.8*positiveness
    F = min(mean(r_min), mean(r_maj))
    U = 0.2*mean(r_min) + 0.8*mean(r_maj)

Scoring length counts non-padding DistilBERT tokens including special tokens,
with truncation at 512. Confirm whether historical scoring includes the prompt
or only the continuation; freeze and record that choice. Generated length is a
separate field. Do not use the unnormalized `256 - length` column directly as c.

For each snapshot retain per-response scores and report:

- Generation and scoring length mean, median and quantiles.
- Conciseness, sentiment, minority/majority mean reward, F and U.
- Signed/absolute group gap and per-response sample SD of group rewards.
- EOS/cap rates, scoring length > 256 rate and scorer-truncation rate.

Conciseness saturates at zero for scoring lengths >= 256. For evaluation and the previous 512-token training runs, c alone cannot
distinguish increasingly long responses;
length distributions and the >256 rate are essential.

Predetermine 12 prompts for qualitative comparison at updates 0, 200, 400, 600,
800. Use full archived responses for scoring; mark any display truncation.
A separate complete-held-out-set evaluation is optional for this time-limited
experiment. The frozen 256-prompt panel is the primary final evaluation, and
historical controls must be labelled if they use another prompt set.

## 5. Must-have data: intervention execution

Keep the existing pre-projection CSV fields:

    step, update, cos_g1_g2, dot, norm_g1, norm_g2

At every inner optimization step add the minimum execution fields:

    window_active, conflict_detected, projection_applied, valid_cosine

These distinguish scheduled intervention from actual projection. Missing Late
recovery is difficult to interpret if late gradients rarely conflict and thus
receive little surgery. Use missing cosine for near-zero gradients.

Deferred lightweight additions (not implemented in the must-have scope):

    cosine_after, correction_fraction

Define correction_fraction as the norm of the combined reward-gradient change
divided by the original sum norm, with an explicit near-zero convention. Compute
scalar diagnostics in float32 or higher. Aggregate by outer update into conflict
frequency, projection frequency and mean/median pre-projection cosine, plus
correction summaries if implemented. Preserve raw rows and flush each outer
update and on exit/interruption.

Post-projection cosine verifies the implementation; it is not independent
evidence of better long-term reward. These are raw training-gradient measurements,
not actual Adam parameter displacement.

## 6. Minimal artifacts and provenance

Use unique per-arm run directories and names containing alpha, arm, bounds,
seed and run tag. Do not overwrite old repositories or results.

    manifest.json
    prompts/evaluation.jsonl
    diagnostics/gradient_steps.csv
    rollouts/train/update_*.jsonl       # temporary all updates; retained selected batches
    rollouts/eval/update_*.jsonl        # all scheduled evaluation responses
    metrics/train_updates.jsonl        # all updates, including offline oracle summaries
    metrics/train_response_scores.jsonl
    metrics/train_oracle_updates.jsonl  # offline scoring summaries for every rollout
    metrics/evaluation_oracle_updates.jsonl
    metrics/evaluation_response_scores.jsonl
    final_policy/
    logs/train.log
    TRAINING_COMPLETE
    checksums.json                    # produced after successful scoring/checks
    figures/

The manifest records code commit, model/reward/scorer IDs and revisions, library
versions, seed, initialization, actual resolved configuration, window bounds,
prompt-manifest identity, evaluation schedule, RNG mapping and scoring convention.
Do not include authentication tokens.

Save the final policy and tokenizer. Use existing persistent storage for logs,
responses, scores and final models; check completeness and file integrity before
releasing the GPU. A new automatic backup service is outside scope.

Verify existing None/Full paths, revisions, alpha, seed, horizon, objectives,
initialization, batch/accumulation, decoding and scoring before comparison.
Vanilla None is not a matched SumGrad control. Do not fabricate missing historical
trajectories or diagnostics. Available control checkpoints may be evaluated on
the same panel without retraining.

## 7. Explicitly deferred work

Do not block this experiment on:

- Adam parameter copies/displacement, actual-movement cosine or first-order change.
- Complete gradient-norm inventories, total-gradient/KL alignment, detailed
  PPO-epoch/minibatch counters beyond existing step and outer-update identity.
- Extra fixed-prompt surrogate-gradient callbacks (`--diag_every`).
- Policy checkpoints at every evaluation snapshot.
- A new exact PPO resume system and interrupted/uninterrupted equivalence suite.
  Current training uses `save_strategy="no"`; final-policy saves cannot resume PPO.
  Assess run duration and interruption risk in the smoke test. Use existing
  verified recovery facilities if available; do not claim exact resume otherwise.
- A general analysis framework, automatic qualitative-table generator or backup service.
- Additional seeds or a larger final evaluation set.

## 8. Implemented scope and remaining validation

1. Implemented: the seed-43 Early/Late wrapper with unique output paths, actual resolved
   configuration capture and the 800-update assertion. Verify baseline references.
2. Implemented: training rollout capture and basic per-update length/reward statistics.
3. Implemented: fixed-panel generation after outer updates, preserving training state;
   offline scoring uses the frozen convention.
4. Implemented: intervention execution flags and safe CSV flushing, including
   SIGTERM/finally handling. Optional after-cosine/correction diagnostics are deferred.
5. Implemented: a small analysis script that checks schedules and missing/duplicate rows,
   scores archived text and produces the three core figures below.

Run CPU checks with:

    venv/bin/python -m unittest discover -s tests -p test_windowed_experiment.py -v

CPU checks:

- Conflicting/aligned/zero synthetic gradients; projection only in active windows;
  finite output and nonnegative projected dot product within tolerance.
- Boundaries 399/400/401/800/801; outside-window equals SumGrad and full-window
  equals Full PCGrad for identical inputs.
- Scoring lengths 0/256/>256, F from group means, and consistent prompt handling.
- Evaluation preserves RNG/model mode/gradients; CSV/JSONL can be read back.

GPU smoke test (required for the new length-53/batch-64 configuration): invoke `unified_ppo_training.py` with the same
wrapper arguments and prepared inputs, changing `--total_episodes 256
--expected_updates 4 --experiment_batch_size 64 --training_response_length 53` and windows to 1–2 / 3–4, in new smoke directories. This
keeps the resolved rollout batch at 64 and uses the shortened evaluation schedule
0 and 4. Run the analyzer on each smoke directory before starting full runs. Ensure all
scheduled logs/texts/evaluations and final policy appear; compare monitoring
on/off with identical inputs; measure runtime, evaluation overhead and disk use.
Dynamic TRL patches must match the pinned version and fail loudly if an expected
replacement is absent. Freeze the main-run evaluation schedule after this test.

Then run both arms sequentially on one GPU, regardless of which looks favorable.
Do not launch full runs until must-have logging and scoring are validated.
Check all 800 training updates, chosen evaluation snapshots and final artifacts.

## 9. Analysis focused on the hypothesis

Predetermined main comparisons:

1. Before intervention in Late: compare length/conciseness trajectories through
   update 400, using both training rollouts and the fixed panel.
2. After intervention starts: compare Late at 400 against subsequent snapshots,
   especially the initial post-switch snapshots and update 800. Report improvement,
   continued deterioration or no clear recovery without changing the criterion.
3. Compare Early and Late at update 800 on fixed-panel conciseness, length,
   sentiment, both group rewards and F. Final F remains the main reward outcome;
   conciseness and length directly address the proposed mechanism.
4. Check that Late actually applies projection during 401–800 and report its
   frequency and magnitude where available.

Report recovery as continuous changes, not an outcome-selected binary threshold:
Late's change from update 400 to 800, its difference from its own update 0, and
its final difference from Early. Show intermediate trajectories rather than
selecting a best checkpoint. Optional secondary summaries are F trajectory AUC
and minimum observed F, calculated on the same evaluation schedule.

Produce three core figures:

1. Training-rollout length, >256 rate and conciseness versus outer update.
2. Fixed-panel length, conciseness, sentiment and F versus completed update.
3. Conflict/projection frequency and optional correction fraction versus update.

Mark the 400/401 intervention boundary and label training observations as
pre-update and evaluation as post-update. Use unsmoothed data for comparisons;
fixed smoothing may be added for display. Add historical None/Full curves only
where compatible observations exist.

Conclude whether results are consistent with early omission producing persistent
loss of conciseness within this budget. If Late recovers, Early does not help,
or a comparable Full baseline fails, report that outcome. The rollout-mediated
explanation remains a hypothesis; isolating mediation would require a separate
controlled intervention beyond this time-limited experiment.
