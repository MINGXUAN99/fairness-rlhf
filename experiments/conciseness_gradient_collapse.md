# WFDb question 3: does low rollout-length variation hide conciseness failure behind positive minority alignment?

Date: November 25, 2026.

Branch: `experiment/conciseness-gradient-collapse`.

Parent: `naacl_rebuttal` at `398ac6a` (also the current local `main` commit when this branch was created).

Status: diagnostics implemented; see the shorter [run guide and output-field list](run_attribute_diagnostics.md). No project training, GPU evaluation, or research results have been obtained. This plan addresses WFDb's third weakness in [the reviews](../review/review.md), especially why minority performance remains poor after gradient alignment becomes positive. It refines the temporal experiment proposed in [the rebuttal strategy](../review/rebuttal_strategy.md).

## Measurement checklist: what we will collect

All rows carry `run_id`, `seed`, `alpha`, and `ppo_update`. Per-response records also carry `prompt_id` and `sample_id`. Compute the attribute gradients on the **same responses**, using raw reward scales or one shared scale.

| Measurement | Fields to collect | Why |
|---|---|---|
| Rollout lengths | Mean, standard deviation, within-prompt variance, fraction hitting generation cap, EOS rate | Are responses long with little length variation? |
| Scored length | Scorer-token length, truncation fraction, scored-length standard deviation | Is apparent low variance caused by the scorer's cap? |
| Attribute rewards | Mean and standard deviation of conciseness `c` and sentiment `s`; within-prompt variances | Is conciseness losing variation while sentiment retains it? |
| Group rewards | Mean oracle minority reward and mean learned minority reward | Does minority performance remain poor? |
| Attribute gradient norms | `||g_c||`, `||g_s||`, `alpha ||g_c||`, `(1-alpha) ||g_s||` | Which attribute supplies the larger weighted gradient? |
| Dominance ratio | `rho = alpha ||g_c|| / ((1-alpha) ||g_s||)` | `rho << 1` indicates sentiment dominance in weighted norm. |
| Attribute alignment | `cos(g_c, g_s)` and `cos(g_min_oracle, g_s)` | Does the mixed oracle direction approach sentiment? |
| Learned-RM alignment | `cos(g_min_RM, g_c)`, `cos(g_min_RM, g_s)`, `cos(g_min_RM, g_min_oracle)` | Does the learned minority gradient follow the proposed oracle mechanism? |
| Actual-update alignment | For `d = theta_after - theta_before`: cosine with `g_c`, `g_s`, `g_min_oracle`, and `g_min_RM` | Is a positively aligned minority update failing to improve conciseness? |
| Signed update contributions | `g_c dot d`, `g_s dot d`, and their alpha-weighted values | Which attribute contributes to predicted minority improvement? |
| Update controls | Learning rate, `||d||`, KL, PPO clipping fraction, pre/post-whitening reward and advantage standard deviations | Separate component imbalance from small steps or normalization effects. |

**Collection frequency:** length/reward statistics every rollout update; gradient and actual-update measurements at selected checkpoints (initialization, every 25 updates through 300, then every 100, plus final). Actual-update measurements require a pre-update gradient and the immediately following optimizer displacement. Within-prompt variance uses repeated responses to each prompt on the diagnostic panel; mark it unavailable on training batches with only one response per prompt.

**Minimum figure:** length mean/std → weighted attribute-gradient norms → update alignment with each attribute → conciseness and sentiment reward trajectories. The detailed sections below specify estimators, controls, and interpretation; they are not additional prerequisite experiments.

## 1. Assessment of the hypothesis

**The hypothesis is plausible and worth testing.** A mixed objective can become positively aligned with the training direction because its sentiment component dominates, even while conciseness receives little effective improvement. The proposed mechanism is:

1. Rollouts become long and concentrated in length, potentially at a generation or scoring cap.
2. Conciseness varies little among sampled responses, weakening its covariance with policy score gradients.
3. The weighted sentiment gradient becomes larger than the weighted conciseness gradient.
4. Alignment with the mixed minority gradient becomes positive because of sentiment, while response length and conciseness fail to recover.

This is a sharper explanation of Figure 3 than simply saying that later updates are too small. However, each link requires evidence. Long responses alone do not imply small gradients; low reward variance alone does not prove sentiment dominance. Sentiment may also saturate, and gradient geometry matters.

Use “weak conciseness signal on the sampled rollout distribution” until measurements justify a stronger statement. A constant reward in one finite batch does not establish a globally zero policy gradient: rare short responses can remain possible but unsampled.

Two distinctions are essential to this experiment, rather than broader manuscript issues:

- The exact mixture identity applies to **oracle expected-reward policy gradients**. The separately trained minority reward model is not algebraically constrained to equal that mixture.
- Independently standardizing attribute rewards changes their relative gradient magnitudes. The primary diagnostic must preserve the reward scales used in the hypothesis.

## 2. Mathematical predictions

### 2.1 Exact oracle decomposition

Let `c(x,y)` denote the frozen benchmark conciseness score and `s(x,y)` the frozen sentiment score. With a fixed prompt distribution, define ascent gradients

\[
g_c=\nabla_\theta\mathbb E_{\pi_\theta}[c],\qquad
g_s=\nabla_\theta\mathbb E_{\pi_\theta}[s].
\]

For the oracle minority reward,

\[
g_{\min}^{\mathrm{oracle}}=\alpha g_c+(1-\alpha)g_s.
\]

At alpha 0.75 this is exactly `0.75 g_c + 0.25 g_s`. The relevant magnitude ratio is

\[
\rho=\frac{\alpha\|g_c\|}{(1-\alpha)\|g_s\|}.
\]

Sentiment exceeds conciseness in weighted norm at alpha 0.75 only if `||g_s|| > 3 ||g_c||`; strong dominance requires a substantially larger ratio. Record the two component norms as well as the ratio. Mark the ratio undefined when its denominator is unresolved or zero.

When rho is small and the mixed gradient is not numerically negligible, its direction approaches the sentiment direction. This can explain positive alignment without meaningful conciseness progress.

**Analytic illustration, not an experimental result.** Consider a uniform softmax policy over three responses, with lengths `{510, 509, 511}`, conciseness `1 - length/512`, and sentiment scores `{0.1, 0.5, 0.9}`. For logits, the exact expected-reward gradient is `g_i = p_i (r_i - E[r])`. At alpha 0.75, the weighted conciseness-to-sentiment norm ratio is approximately 0.01465, and `cos(g_min,oracle, g_s)` is approximately 0.999918. Taking an ascent direction `d = g_s` gives `g_min,oracle dot d = 0.008824 > 0`, but `g_c dot d = -0.00008681 < 0`. Thus the proposed combination of positive mixed-objective alignment and worsening conciseness is mathematically possible even with a 75% conciseness coefficient. This example checks logical possibility; it makes no claim about the project's learned reward models or measured rollouts.

### 2.2 Why small length variation could matter

For a matched on-policy score-function estimator, write `z = grad log pi(y|x)`. Conditional on a prompt, a response-independent baseline gives

\[
g_c(x)=\mathbb E[(c-b_c(x))z\mid x].
\]

Taking `b_c(x) = E[c|x]` yields the bound

\[
\|g_c(x)\|\leq \sigma(c\mid x)\sqrt{\mathbb E[\|z\|^2\mid x]}.
\]

Thus low conditional conciseness variance limits this gradient for a bounded score-function second moment. It does not establish a comparison with sentiment unless both gradients are measured. If `c = 1 - L/C` without clipping, `Var(c|x) = Var(L|x)/C^2`. With clipping or truncated scoring, actual length and scored conciseness can behave differently; record both.

The score-function framework follows [Sutton et al., Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://proceedings.neurips.cc/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html). The mixture identity and bound above are direct calculations for this experiment.

### 2.3 Test contribution to the actual update

For a realized policy displacement `d = theta_after - theta_before`, measure

\[
D_c=g_c^\top d,\quad D_s=g_s^\top d,\quad
D_{\min}=\alpha D_c+(1-\alpha)D_s.
\]

The strongest signature is positive mixed-objective alignment and `D_min > 0`, with the positive contribution arising predominantly from sentiment while `D_c` is negligible or negative and observed conciseness stays poor. Report signed contributions, not only cosines or norm ratios. Avoid percentages when the total is close to zero or components cancel.

These are first-order predictions; verify finite changes by fresh rollout evaluation. The actual optimizer displacement includes Adam, clipping, KL, and learning-rate effects. A raw loss gradient is not interchangeable with that displacement.

### 2.4 Normalization and learned rewards

If each attribute is independently standardized on the same fixed batch, the decomposition becomes

\[
g_{\min,\mathrm{std}}=
\frac{\alpha\sigma_c g_{c,\mathrm{std}}+(1-\alpha)\sigma_s g_{s,\mathrm{std}}}
{\sigma_{\min}},
\]

for the linear score-function surrogate with matching centering, detached statistics, and nonzero standard deviations. It is not generally `alpha g_c,std + (1-alpha) g_s,std`. Primary measurements use unscaled rewards with an appropriate baseline, or one shared detached scalar for all components. Independently standardized gradients are a secondary diagnostic only.

Measure `g_min,RM` from the actual minority reward model separately from `g_min,oracle`. Do not assign the learned model's gradient an exact attribute decomposition. Compare its alignment with `g_c`, `g_s`, and `g_min,oracle`; substantial mismatch is evidence that oracle dominance alone cannot explain the Figure 3 diagnostic.

PPO also uses a clipped surrogate and repeated minibatch updates; its gradients need not preserve the simple mixture identity after separate whitening, value-baseline construction, and clipping. Measure those effects rather than imposing the identity on them. See [Schulman et al., Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347).

## 3. Primary study: observe the proposed mechanism over training

### 3.1 Run matrix and budget order

| Phase | Configuration | Purpose |
|---|---|---|
| Instrumentation pilot | Vanilla GPT-2, alpha 0.75, one existing/reproduced seed | Verify measurements and whether the proposed signature is present. |
| Confirmation | Vanilla GPT-2, alpha 0.70, 0.75, 0.80; three matched seeds each | Compare a nearby noncollapsed setting with the two settings discussed in Figure 3. Nine runs if no reusable trajectories exist. |
| Optional extension | One additional backbone at an already observed collapse setting | Broaden scope only after establishing the mechanism in GPT-2. |

Start with available checkpoints and saved rollouts, but do not infer temporal behavior from final checkpoints alone. The current trainer sets `save_strategy="no"`; intermediate checkpoints must be explicitly saved or diagnostics collected during a rerun. Existing final artifacts do not imply that resumable intermediate states exist.

Retain the baseline training configuration: population proportion, reward-model IDs/revisions, initialization, rollout sampler, token caps, PPO epochs, whitening, optimizer, KL, and learning-rate schedule. Validate exact model IDs before use; do not infer availability from a name template. No new reward-model training is required for the primary study.

For a smaller rebuttal budget, complete alpha 0.75 across three seeds and a matched alpha 0.70 control before expanding to alpha 0.80. Describe limited replication honestly.

### 3.2 Time points and sampling

- Save cheap rollout statistics at every PPO rollout update.
- Capture expensive diagnostics at initialization, then every 25 updates through update 300 and every 100 updates thereafter, plus the final update. Use update index and episode count in all files. Freeze this schedule after the instrumentation pilot.
- Use a fixed panel of 128 source-review prompts with four independent responses per prompt per checkpoint; pilot with 32 prompts if necessary. Preserve source-review IDs even if short prompt prefixes coincide. Multiple responses estimate within-prompt variation.
- Divide the panel into four prompt-disjoint blocks and retain block gradient estimates to assess Monte Carlo stability. Repeat generation at selected early and late checkpoints using independent sampling seeds. Do not treat blocks or rollouts as independent training runs.
- Keep a separate, fixed held-out evaluation panel for reward and length trajectories. These are mechanistic diagnostics, not claims of reward-model holdout accuracy.

Primary length statistics come from the **actual training rollouts**, not only freshly generated diagnostic responses. The existing diagnostic defaults to 200 tokens and temperature 0.7/top-p 0.95, whereas the training task uses a longer cap; copying those defaults could create the phenomenon being measured.

### 3.3 Separate sampling distributions explicitly

Maintain two labeled measurement streams:

1. **Training-distribution stream:** capture actual rollout tokens, sampler settings, rewards, log probabilities, masks, and PPO updates. Compute attribute surrogates on these same samples and actual-update directional products. This connects the diagnosis to the training behavior.
2. **Matched score-function stream:** sample an explicitly defined policy and differentiate its matching log probability. Prefer unfiltered categorical sampling at temperature 1 for the clean oracle calculation. If a different temperature is used, differentiate the temperature-transformed policy. This estimates gradients of that explicitly defined sampling policy, not automatically the original training sampler.

Use the same samples for all attributes within each stream. If the training sampler matches the clean stream they can share data. Otherwise report the difference, and require the qualitative signature to survive measurement on the actual rollout stream before using it to explain training. Do not call samples from top-p/temperature-adjusted generation unbiased estimates of an unadjusted policy gradient.

### 3.4 Rollout measurements

For every checkpoint or rollout batch, record:

- Response-only policy-token length: mean, standard deviation, interquartile range, quantiles, unique lengths, and fraction reaching the generation cap.
- Natural EOS frequency and token position; termination reason; padding and valid-token masks. Retain EOS in the action log probability, but exclude padding. State whether EOS counts toward the reported length.
- The benchmark scorer's length before and after truncation, scored conciseness, and scoring-cap frequency. Keep tokenizer differences visible.
- Mean, standard deviation, and within-prompt variance of conciseness and sentiment, plus their covariance.
- Oracle minority reward, learned minority reward, pooled reward, and their variances on identical responses.
- Training reward/advantage standard deviations before and after whitening; learning rate, entropy, clipping fraction, and achieved KL.

Freeze the historical benchmark definition before running: tokenizer, prompt inclusion, denominator, truncation, and clipping. Do not change the conciseness denominator when changing the generation cap in a sensitivity check. Supplement with raw response length rather than redefining the target mid-experiment.

### 3.5 Gradient measurements

On the same frozen policy and response batch, compute ascent directions for:

- Oracle conciseness `g_c`.
- Oracle sentiment `g_s`.
- Direct oracle minority mixture `g_min,oracle`.
- Learned minority reward `g_min,RM`.
- Learned pooled reward `g_pool,RM`.
- The actual PPO policy-loss ascent gradient at selected optimizer steps, and realized parameter displacement `d`.

For the clean oracle estimator use a response-independent baseline. With four responses per prompt, a leave-one-out mean of the other responses is a useful unbiased baseline under independent sampling. Use identical baseline construction for both attributes; the mixture baseline must be the same weighted mixture of attribute baselines. Detach all rewards and baselines. Do not differentiate a token-count function through sampled discrete token IDs.

Record component norms, rho, `cos(g_c,g_s)`, mixture-to-sentiment alignment, and learned-minority alignment with each oracle direction. Report actual-update alignment and signed products with both attributes and both minority definitions. Use ascent conventions throughout; negate gradients of loss functions when needed.

Verify that the direct oracle-mixture gradient reconstructs from weighted attribute gradients on the same samples. Use FP32 diagnostic accumulation where feasible. Disable dropout for paired gradient calculations, reuse masks and tokenization, and treat near-zero gradients as undefined for cosine interpretation rather than forcing a numerical direction.

Gradient norms estimated from noisy batches are upward biased. Require consistency across blocks, repeated samples, and training seeds. Report whether a direction is stable above sampling/numerical noise; a large norm ratio between two unstable estimates is not evidence of dominance. Stream or offload gradients as needed; do not retain a per-response full-parameter Jacobian.

### 3.6 Fixed-policy alpha control

At the same alpha-0.75-trained checkpoint and on the same batch, form counterfactual oracle mixtures with weights 0.5, 0.75, 0.9, and 1.0. This changes only the diagnostic mixture, not the policy or training history.

If the late positive alignment is sentiment-driven, it should weaken or become numerically unresolved as the sentiment weight approaches zero. At alpha 1 the oracle mixture is exactly `g_c`; this is a useful internal control. Label these as **counterfactual evaluation weights**, not independently trained alpha-1 policies. If both attributes are similarly aligned, the conclusion should be small conciseness magnitude rather than sentiment-specific directional reversal.

## 4. Decision criteria and alternative explanations

Report the chain in order; do not infer all links from a single correlation plot.

| Observation | Interpretation |
|---|---|
| Long rollouts, falling within-prompt conciseness variance, declining weighted conciseness norm, stable sentiment direction, and sentiment-dominated positive update contribution | Supports the proposed mechanism. |
| Oracle mixture shows this pattern and learned-minority alignment follows it | Connects the mechanism to the observed minority diagnostic more directly. |
| Oracle mixture is sentiment-dominated but learned minority gradient differs materially | Supports oracle component imbalance; insufficient to explain Figure 3 without accounting for the learned reward model. |
| Length varies, but scored conciseness is flat because of truncation/clipping | A scoring-saturation explanation, rather than uniformly low behavioral length variation. |
| Low length variance but a substantial, stable conciseness gradient | Rejects the proposed small-conciseness-gradient link. |
| Both attribute gradients become negligible | General signal saturation is more plausible than meaningful sentiment dominance. |
| Conciseness gradient remains substantial but its actual-update projection is weak/negative | Update geometry, KL, clipping, or optimizer state is the stronger explanation. |
| Signal appears only with a shorter diagnostic cap or a changed sampler | A diagnostic-distribution effect; do not generalize it to the training rollouts. |
| Positive conciseness projection but tiny realized updates | Learning-rate/step-size limitation remains a candidate. |

Define early and late windows from the existing trajectory/pilot, before viewing confirmation runs. Report every scheduled checkpoint. As an interpretable descriptive threshold, rho below 1 means sentiment has the larger weighted norm; rho below 0.1 means a tenfold weighted-norm imbalance. These are effect-size labels, not statistical significance tests. Also show continuous values and uncertainty.

Primary contrast: late versus early component norms, rollout variance, and signed update contributions within each run, compared across alpha 0.70 and 0.75/0.80. If late mixed alignment is positive but conciseness reward shows no practically meaningful improvement, quantify that change and its interval; absence of statistical significance is not proof of zero improvement. Freeze a practical reward-change tolerance after the pilot and before confirmation.

## 5. Optional interventions after the observational diagnosis

### 5.1 Frozen-batch attenuation control: cheapest mechanism check

On a fixed checkpoint and batch, replace the centered conciseness component by `k * (c - b_c)`, with `k` in `{1, 0.5, 0.1, 0}` and the sentiment component unchanged. Use raw scales or one common normalization. Verify that the oracle mixture rotates toward sentiment as its conciseness component is attenuated when the component directions differ.

This is a controlled numerical demonstration and implementation check. The resulting linearity is expected by construction; it is not independent evidence that natural rollout collapse causes the attenuation. The temporal measurements in Section 3 supply that evidence.

### 5.2 Cap and exploration sensitivity: distinguish saturation from missing variation

At representative early and late frozen checkpoints, compare the baseline generation cap with a longer cap within the model's context limit, keeping the reward definition fixed. Also inspect an explicitly labeled unfiltered sampling condition if baseline sampling filters suppress alternatives. Measure whether raw lengths, scored conciseness variation, and `g_c` recover.

Treat these as sensitivity analyses: changing the cap changes the response distribution, and changing temperature/filtering changes more than variance. If the scorer itself is capped, a longer generation alone may not restore scored variation. An additional untruncated-length objective is diagnostic only and must not silently replace the benchmark reward.

### 5.3 Short continuation: test whether a usable conciseness signal can restore improvement

Run only if the primary study finds a stable but weak conciseness component. From the same late checkpoint and optimizer state, compare:

1. Original vanilla PPO continuation.
2. Vanilla continuation plus a small explicit oracle-conciseness auxiliary ascent term, with its coefficient selected using the pilot and then frozen.
3. A global step-size control matched to the update magnitude or achieved local KL of arm 2, without the attribute-specific term.

Use the same continuation budget and paired seeds; evaluate freshly generated responses under the original sampler and fixed reward definition. If arm 2 improves conciseness more than arm 3, this supports a role for the missing attribute signal beyond merely larger steps. It does not prove that low variance was the unique cause: the auxiliary term adds objective information unavailable to the pooled reward model.

If the measured conciseness gradient is numerically zero, do not normalize or multiply noise into a large step. Stop this intervention and use the sampling/scoring diagnostics above. Recovering signal would require changing coverage, scoring, or the estimator, which is a separate intervention.

## 6. Implementation outline for this branch

The following are planned files and changes, not existing runnable interfaces:

1. Add `src/imdb/diagnostics/attribute_gradient_diagnostic.py`: deterministic token masking, raw/shared-scale score-function gradients, oracle mixture reconstruction, gradient norms/cosines, and signed update contributions. Keep learned-RM gradients separately named.
2. Add a runner accepting an explicit manifest of checkpoint paths/revisions and fixed prompt IDs. Cache tokenized rollouts, attributes, and block summaries. Support a raw on-policy diagnostic sampler and an explicitly labeled training-rollout input mode.
3. Add opt-in rollout capture and diagnostic hooks to `unified_ppo_training.py`; retain its default training behavior. Capture pre-update directions and immediate optimizer displacement over the same parameter set at selected steps. Save resumable policy/value/optimizer/scheduler/RNG state only when requested.
4. Add a notebook producing the four-panel mechanism figure and the fixed-policy alpha control. All results must load from manifests and saved diagnostic outputs.
5. Add focused correctness checks before running on project checkpoints: exact mixture reconstruction, zero centered constant-reward gradient, common-scaling behavior, leave-one-out baseline construction, correct first-EOS/padding masks, and a tiny categorical policy with an analytically known expected-reward gradient. Check the actual-update sign convention using a known ascent step.

Repository details that affect implementation:

- `gradient_cosine_diagnostic.py::compute_grad` currently z-scores each reward separately and puts the policy in training mode. Do not reuse it unchanged for attribute-magnitude attribution.
- Its existing gradient is a surrogate computed on newly generated responses, not the realized PPO optimizer displacement. Preserve this distinction in names and outputs.
- `_pcgrad_gae` whitens rewards and advantages; simply feeding attributes through it does not establish the raw mixture identity.
- The online callback currently generates its own diagnostic rollouts. Add actual-rollout capture rather than assuming that sample matches the training distribution.
- The current `--no_push` flag also skips local final model saving. A local-only experiment with resumable checkpoints needs explicit local save support; do not accidentally publish models to obtain checkpoints.
- Diagnostic calls must save/restore model mode, Python/NumPy/Torch/CUDA RNG state, and avoid altering optimizer state or pending accumulated training gradients. Run hooks at safe boundaries or use isolated checkpoint evaluation.

## 7. Artifacts and rebuttal deliverable

Store large artifacts under the already ignored directory `results/conciseness_gradient_collapse/<run_id>/`:

- `manifest.json`: source commit, run/checkpoint identifiers, model and tokenizer revisions, reward definitions, sampler settings, caps, alpha, seeds, optimizer configuration, and diagnostic normalization.
- `rollouts.parquet`: source-review/prompt IDs, sample IDs, tokens or token artifact references, masks, termination reason, raw/scored lengths, oracle attributes, and learned rewards.
- `rollout_summary.csv`: cheap per-update statistics including within-prompt statistics where repeated responses exist.
- `gradient_summary.csv`: checkpoint/block IDs, component norms, mixture reconstruction error, cosines, weighted ratios, and actual-update contributions.
- `evaluation.csv`: held-out attribute and group rewards with cap frequencies.
- `figures/`: trajectory panels and counterfactual-weight control.

Keep this plan and a compact findings summary in Git. Record artifact locations and manifest hashes in the findings summary.

The minimum useful rebuttal result is one four-panel figure:

1. Mean response length, length variation, and cap/EOS rates over training.
2. Weighted conciseness and sentiment gradient norms over training.
3. Alignment of the update with conciseness, sentiment, oracle minority, and learned minority directions.
4. Actual conciseness/sentiment reward trajectories and signed first-order update contributions.

Add a compact early-versus-late table and the same-policy alpha-weight control. Preserve per-seed outcomes and avoid presenting a single run as replicated evidence.

## 8. Conditional rebuttal wording

Before experiments, the defensible claim is:

> A plausible explanation is that positive alignment with the mixed minority objective is driven by its sentiment component. If rollout lengths become concentrated, the conciseness signal may weaken even though conciseness retains a larger nominal reward coefficient. We will test this explanation by separating attribute-level gradients and measuring their contributions to the realized updates.

Only if the predicted chain is measured, use wording of this form with actual values:

> Our additional diagnostic shows that late positive minority alignment is predominantly associated with sentiment rather than recovery of conciseness. As rollout lengths concentrate, the weighted conciseness gradient [measured change], while the sentiment component [measured result]. The resulting updates remain positively aligned with the mixed minority objective but provide [measured contribution] to conciseness, consistent with its persistently low reward. This explains why recovery of aggregate gradient alignment does not imply recovery of the neglected attribute in these runs.

If the learned reward model does not track the oracle-mixture explanation, explicitly narrow the conclusion to oracle attribute imbalance and report that it does not fully explain the original learned-minority diagnostic.
