With only Full PCGrad and vanilla outputs, an early-negative/later-positive
alignment pattern can support a hypothesis about training history. It cannot
identify the causal importance of early versus late projection. The explanation
below takes the user's reported cosine trajectory as an observation; its raw
online CSV was not found in this workspace during this review. “Collapse” is
assumed to mean minority oracle evaluation performance, rather than pooled
learned training reward. These quantities must be distinguished in the paper.

For an actual parameter displacement u_t = theta_(t+1) - theta_t and a true
minority expected reward J_min, a sufficiently small update gives

    J_min(theta_(t+1)) - J_min(theta_t)
        approximately equals <grad J_min(theta_t), u_t>.

The sign of this inner product measures a local change, not the absolute reward
level. Negative early changes can produce a large deficit; smaller positive
later changes need not repay it. For example, a hypothetical loss of 0.20 followed
by gains totaling 0.02 leaves a policy 0.18 below its original performance.
Cosine alone omits displacement and gradient magnitudes. The cumulative local
first-order change depends on their products as well as on the cosine. This
approximation requires matching reward gradients, actual update directions and
sufficiently small steps; it does not automatically apply to finite PPO/Adam
steps or a differently defined oracle metric.

An on-policy mechanism consistent with the observations is:

    Early minority-opposing updates reduce short/EOS response probability
      -> subsequent rollouts contain fewer concise responses
      -> later measured gradients describe mostly long-response behavior
      -> local alignment may become positive while absolute conciseness stays low.

This is a hypothesis about response availability and gradient estimates. It is
not proof that short responses disappeared or that recovery became impossible.
Rare short responses, low gradient norms, decaying learning rate, PPO clipping,
critic quality and differences between learned and oracle rewards can all affect
the rate of recovery. The data are also compatible with mechanisms that do not
involve rollout-length mediation.

A particularly relevant possibility is that later positive alignment reflects
improving sentiment on long outputs rather than recovering conciseness. Under
the alpha=0.8 oracle used in the windowed analysis,

    J_min = 0.8 * E[conciseness] + 0.2 * E[sentiment].

For responses beyond the scoring-length threshold, clipped conciseness is zero.
If most sampled responses are in this region, the observed length reward has
little variation, while sentiment can still vary. Both pooled and minority
objectives can then favor the same sentiment improvements without producing
short responses. This does not make the true expected conciseness gradient
identically zero: changing probabilities of rare short responses could still
improve it. Check cap fractions, EOS fractions and reward-component variation
before adopting this explanation.

The repository's online cosine implementation needs accurate labeling:

- `compute_grad` in `src/imdb/diagnostics/gradient_cosine_diagnostic.py` computes
  the gradient of `-mean(A.detach() * sum_log_pi)`, using batch-standardized
  rewards and an optional sampled KL-related term.
- It does not reproduce the complete PPO clipped surrogate, GAE/critic baseline,
  or Adam parameter displacement. Calling it the “actual PPO update” overstates
  the measurement.
- `--diag_beta` defaults to zero in the training launcher. Without a nonzero
  matching beta and reference policy, the diagnostic excludes KL.
- Both compared vectors are loss gradients. Their positive cosine indicates
  alignment of the corresponding descent directions. Comparing an actual
  descent displacement with a minority *loss* gradient instead would reverse
  the sign interpretation.
- A gradient for a learned minority RM does not necessarily equal the gradient
  of the oracle minority evaluation metric. Positive learned-reward alignment
  cannot by itself establish oracle recovery.

The most useful analyses with existing outputs are:

1. Plot diagnostic cosine and both gradient norms alongside minority oracle
   reward, sentiment, conciseness, response length, EOS fraction and cap fraction.
   Distinguish pooled learned reward from each group oracle score. Demonstrate
   whether negative alignment precedes minority deterioration rather than merely
   accompanying it. Keep any discovered change-point analysis exploratory.
2. Compute the raw surrogate inner product from the saved cosine and norms.
   Report whether later positive alignment has a meaningful gradient magnitude.
   A learning-rate-weighted sum is only a proxy; Adam and PPO prevent treating
   it as actual cumulative parameter-level harm. Actual displacements require
   corresponding parameter snapshots or online logging.
3. Examine reward-component variation and availability of short responses.
   Check whether later positive alignment occurs when sentiment improves but
   conciseness remains low. Where sufficient response-level artifacts exist,
   compare short versus long/EOS versus non-EOS strata and avoid treating noisy
   estimates from tiny short-response subsets as established mechanisms.
4. Compare Full and vanilla at matched prompts, checkpoints and decoding
   settings, and verify parameter precision and reward-model revisions. Compare
   their response distributions and absolute reward levels, not just cosines
   computed on different policies' generated samples.

Full PCGrad's better result is evidence that the complete group-aware training
procedure can avoid vanilla's failure under its tested configuration. It does
not isolate projection: vanilla uses a pooled reward, whereas Full PCGrad uses
separate group objectives and a different gradient-combination procedure.
Without a matched no-projection SumGrad control, the benefits of retaining both
group objectives and the benefits of gradient surgery remain confounded.
Without credible timing interventions, their temporal causal roles remain
unidentified. The earlier BF16 windowed runs are not valid supporting timing
evidence until the parameter-storage issue is corrected and runs repeated.

Suggested wording:

“Vanilla exhibits minority-opposing surrogate-gradient alignment early in
training, followed by positive alignment after minority performance has
deteriorated. Positive late alignment describes local geometry at the resulting
policy and does not imply recovery of the earlier performance loss. The pattern
is consistent with a history-dependent, on-policy failure mechanism, potentially
involving reduced sampling of concise responses. Full PCGrad avoids the observed
collapse, but these comparisons alone do not establish when projection is
necessary or whether late intervention can recover performance.”

If the *pooled training reward itself* collapses, state that separately. Positive
alignment with a learned minority surrogate does not guarantee improvement of
the pooled PPO objective or oracle score; reward mismatch, finite-step effects,
changing gradient estimates and optimization failures then need separate checks.
