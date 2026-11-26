# PKU SafeRLHF experiment plan

Extend the IMDb experiment to fixed helpfulness and harmlessness preference groups
using [PKU-SafeRLHF](https://huggingface.co/datasets/PKU-Alignment/PKU-SafeRLHF).
There is no preference-divergence sweep. These are constructed objective groups,
not observed demographic groups. Work belongs on `experiment/pku-saferlhf`.

## Data preparation

Save one Hugging Face dataset with official `train` and `test` splits and columns
`prompt`, `chosen`, `rejected`, `group_id`. Do not create a validation split.

- Group 0: harmlessness, 20%; select chosen using `safer_response_id`.
- Group 1: helpfulness, 80%; select chosen using `better_response_id`.
- Both-safe pairs provide no harmlessness ranking supervision.

Within each split, sample the required number of harmlessness records uniformly
without replacement from pairs that are not both safe. Assign all remaining
records to helpfulness. This retains each source pair once and keeps both-safe
prompts, while achieving 80/20 up to integer rounding. It deliberately conditions
group assignment on safety, so group prompt distributions differ. Fail if there
are insufficient eligible pairs; do not silently oversample or change proportions.

Keep official test unchanged and remove train rows with whitespace-normalized
prompts appearing in test. Record the source revision, seed, group counts, pair
categories, label disagreement, and overlap removals. The existing labels supply
all preference supervision; no response generation or attribute scoring is needed.
See [preparation instructions](../src/pku_saferlhf/README.md).

## Reward and policy training

Train three reward models from the single dataset's training split:

| Reward model | Training rows |
|---|---|
| Helpfulness | group_id = 1 |
| Harmlessness | group_id = 0 |
| Pooled | All rows |

Train four policies from the same initialization:

| Method | Reward signal |
|---|---|
| Vanilla | Pooled reward model |
| MaxMin | Lowest batch-mean raw score from the two group reward models |
| SumGrad | Sum both group policy gradients without conflict correction |
| PCGrad | Both group objectives, with conflicting gradients corrected |

Do not add reward calibration or normalization of model scores. Preserve the
existing PPO whitening convention separately and consistently. Log raw reward
scales, since MaxMin selection depends on their relative offsets and scales.
MaxMin, SumGrad, and PCGrad must use the same two group reward models. If the backbones are
adapted separately, load both complete models; a shared backbone with two scoring
heads is valid only when the backbone weights are identical.

Use all unique training prompts for PPO, including both-safe examples. Apply
consistent instruction formatting throughout reward training, PPO, and evaluation.
Freeze model revisions, hyperparameters, generation settings, and training budgets
before main runs. Use final checkpoints rather than selecting on test performance.
Start with three matched policy seeds and report that this measures policy-training
variability conditional on the trained reward models.

## Evaluation and planned results

Evaluate reward-model preference accuracy on test pairs per group. Evaluate policy
responses on unique test prompts. New generated responses do not have dataset
oracle scores. The initial automated evaluation uses frozen external
`PKU-Alignment/beaver-7b-v1.0-reward` and `PKU-Alignment/beaver-7b-v1.0-cost`
checkpoints. Record immutable evaluator revisions and score all methods using
identical input formatting. These models are separate from our training reward
models but were also trained on PKU-SafeRLHF; they are not independent ground truth.

Generate each policy's response to the same held-out prompts, including a frozen
initial-policy baseline. Report mean helpfulness reward (higher is better), mean
harmfulness cost (lower is better), and paired differences from the baseline.
Keep the two scores separate: their scales do not justify a minimum or weighted
average. Cost is not an unsafe probability. Include prompt-bootstrap intervals,
per-seed results, and across-seed standard deviations as distinct uncertainty
summaries. Achieved KL remains a training diagnostic.

As corroboration, use blinded human or separate LLM judgments on a sample with
separate helpfulness and safety criteria and ties allowed. Win rates and an
unsafe-response rate require these additional judgments; the initial scorer
does not infer them from arbitrary raw-score thresholds.

Produce a table comparing the initial policy, Vanilla, MaxMin, SumGrad, and PCGrad; plots
of group outcomes over training; gradient conflict and update-alignment traces;
and a joint helpfulness versus harmlessness plot. Report individual seeds and
paired prompt uncertainty without conflating it with training-seed variability.

Minority improvement with preserved majority performance supports avoidable
baseline loss. Minority improvement with majority degradation shows a different
trade-off. Negative gradient alignment alone does not establish causation, and no
benefit from correction is a valid outcome. No results are assumed in advance.

## Execution order

1. Prepare and inspect the CPU-only group-labeled dataset.
2. Train the helpfulness, harmlessness, and pooled reward models.
3. Adapt prompt loading and complete reward-model loading in PPO.
4. Run Vanilla, MaxMin, SumGrad, and PCGrad and independently evaluate outcomes.

Exact model choices and training budgets remain to be fixed before training.
Reversed population mixtures and original-label safety comparisons are optional
follow-up experiments, outside the initial four-method run.
