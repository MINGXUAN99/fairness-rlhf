# RLHF Fairness Paper — Resubmission Revision Plan

## 1. Overall positioning

The revision should consistently preserve two scope constraints:

1. **This is a diagnostic paper, not a new-method proposal.**
   - PCGrad-PPO, MGDA-PPO, and SumGrad are controlled optimization interventions used to diagnose the failure mechanism.
   - Avoid language such as “our proposed method,” “PCGrad is superior,” or “PCGrad is the solution.”
   - Preferred framing: the experiments test whether **update geometry** contributes to worst-group collapse.

2. **We are not trying to prove Pareto optimality.**
   - Do not claim that MGDA finds a Pareto optimum or that the paper characterizes the Pareto frontier.
   - Use the wording **local joint-improvement test**.
   - Safe claim: at the collapsed vanilla checkpoint, there exists a direction / small update that improves both group rewards, so the observed collapse is **not locally unavoidable** due to value conflict.

A concise central question for the revised paper:

> Is worst-group collapse in pooled RLHF necessarily caused by conflicting group values, or can it instead arise from the geometry of the optimization updates?

---

## 2. New experimental results available from the rebuttal

### 2.1 SumGrad control
**Design**
- Uses the same separate group rewards/objectives as PCGrad.
- Keeps both group objectives active.
- Directly sums the two group gradients without projection.

**Purpose**
- **MaxMin → SumGrad**: tests the effect of keeping both objectives active rather than selecting one worst-group objective.
- **SumGrad → PCGrad**: isolates the contribution of gradient projection / gradient surgery.

**Key result**
- In the more conflicting regime, SumGrad deteriorates while PCGrad remains substantially more robust.
- Averaged over \(\alpha \ge 1.0\):
  - SumGrad improves over MaxMin by about **0.03**.
  - PCGrad improves over SumGrad by about **0.39**.

**Interpretation**
- Keeping both objectives active alone does not explain the robustness of PCGrad.
- The geometry of how group gradients are combined matters.

---

### 2.2 MGDA-PPO
**Design**
- Add MGDA-PPO as a second gradient-based multi-objective intervention.

**Purpose**
- Show that the observed benefit is **not specific to PCGrad**.
- Support the broader conclusion that correcting update geometry matters.

**Interpretation**
- MGDA performs comparably to PCGrad across most \(\alpha\) values and is better at \(\alpha=1.3\).
- Do not frame this as choosing the “best” optimizer.

---

### 2.3 Early-only vs. late-only PCGrad
**Design**
- At \(\alpha=0.75\):
  - Early intervention: apply PCGrad only during steps 1–400.
  - Late intervention: apply PCGrad only during steps 401–800.
  - Compare with Vanilla and full PCGrad.

**Key result**
- Vanilla: 0.246 worst-group reward.
- Full PCGrad: 0.915.
- Early-only: 0.736.
- Late-only: 0.307.

**Purpose**
- Explain why a transient early gradient conflict can lead to persistent collapse.
- Show that correcting the early update direction prevents much of the damage, while correcting later does little.

**Interpretation**
- Because PPO is on-policy, early updates change the future sampling distribution.
- Later positive cosine can occur around an already-collapsed policy.

---

### 2.4 Group-mixture / minority-proportion sweep
**Design**
- Fix \(\alpha=1.0\).
- Sweep minority proportion.

**Purpose**
- Test whether collapse severity changes with population imbalance.

**Interpretation**
- Smaller minority proportions produce more severe minority collapse and larger group imbalance.
- This is consistent with majority domination of the pooled objective.

---

### 2.5 Additional model backbones
**Models**
- Pythia-160M.
- Llama-3.2-1B.

**Purpose**
- Show that the phenomenon is not unique to GPT-2.

**Interpretation**
- Vanilla again gives very low minority reward at \(\alpha=1.0\).
- MaxMin and PCGrad substantially recover minority performance.
- Frame only as **backbone robustness**, not broad generalization to all modern RLHF systems.

---

### 2.6 Local joint-improvement test
**Design**
- Start from the collapsed vanilla checkpoint.
- Use MGDA to identify a direction positively aligned with both group gradients.
- Apply a small update and evaluate both group rewards.

**Purpose**
- Provide stronger evidence than positive Pearson correlation that simultaneous improvement is still possible at collapse.

**Safe claim**
> At the collapsed vanilla checkpoint, simultaneous local improvement of both group rewards remains possible. Therefore, the observed collapse is not locally unavoidable due to conflicting group values.

**Avoid**
- “Pareto optimal.”
- “Pareto frontier.”
- “MGDA finds a better Pareto optimum.”

---

### 2.7 Qualitative generations
**Design**
- Representative outputs for Vanilla, MaxMin, and PCGrad at selected \(\alpha\) values.
- Include sentiment, conciseness, and response length.

**Purpose**
- Show that quantitative reward differences correspond to visible differences in generated text.
- Best placed primarily in the appendix.

---

## 3. Priority: main text vs. appendix

### Main text priorities
1. **SumGrad control** — essential for isolating gradient surgery from simply keeping both objectives active.
2. **Early-vs-late intervention** — directly addresses transient cause vs. persistent effect.
3. **Local joint-improvement test** — strengthens the argument that collapse is not locally forced by value conflict.
4. **MGDA-PPO** — shows the conclusion is not PCGrad-specific.

### Mainly appendix / short main-text mention
5. Group-mixture sweep.
6. Pythia/Llama robustness results.
7. Qualitative generations.
8. Extended multi-group limitations and practical discussion.

---

## 4. Detailed SumGrad integration plan

### Figure
**Add SumGrad directly into the existing Figure 1**, rather than creating a separate figure.

Current Figure 1:
- Vanilla
- MaxMin
- PCGrad

Revised Figure 1:
- Vanilla
- MaxMin
- SumGrad
- PCGrad

Reason:
- SumGrad is not an independent result; it is a **control required to interpret the existing PCGrad comparison**.
- Putting all four curves on the same worst-group-vs.-\(\alpha\) plot makes the causal comparison clear.

### Section title
Rename Section 3.2 from:

> Objective-Level Selection vs. Gradient-Level Repair

to:

> **Objective Selection, Joint Optimization, and Gradient Repair**

### Suggested opening logic for Section 3.2
Introduce the three interventions:

- **MaxMin-RLHF:** selects the currently worst-performing group as the active objective.
- **SumGrad:** keeps both group objectives active and directly sums their gradients.
- **PCGrad:** uses the same group-specific objectives as SumGrad but modifies their composition when the gradients conflict.

Then explain:

- **MaxMin → SumGrad** examines the effect of jointly optimizing both objectives.
- **SumGrad → PCGrad** isolates the contribution of gradient projection.

### Suggested result interpretation
Use wording like:

> As shown in Figure 1, keeping both objectives active alone does not explain the robustness of PCGrad. SumGrad performs similarly to PCGrad when group objectives are relatively compatible, but deteriorates as preference conflict becomes stronger. Averaged over \(\alpha\ge1.0\), SumGrad improves worst-group reward over MaxMin by only 0.03, whereas PCGrad improves over SumGrad by 0.39. This control supports our diagnostic conclusion that the geometry of the joint update matters beyond whether both group objectives are kept active.

### Figure caption direction
Possible revised caption:

> **Figure 1: Worst-group reward under preference heterogeneity.** Vanilla RLHF exhibits a sharp worst-group collapse as preference divergence increases. MaxMin-RLHF changes the active group objective, while SumGrad keeps both group objectives active without modifying their gradients. PCGrad uses the same group objectives as SumGrad but projects conflicting gradient components. The SumGrad–PCGrad comparison isolates the effect of gradient projection.

### MGDA placement
If Figure 1 becomes visually crowded with five curves:
- Keep **Vanilla + MaxMin + SumGrad + PCGrad** in the main Figure 1.
- Put MGDA in a separate robustness figure or appendix.
- MGDA serves a different purpose: showing the conclusion is **not PCGrad-specific**.

---

## 5. Planned manuscript changes

### Abstract
Change the framing away from:
- “PCGrad outperforms MaxMin.”

Toward:
- worst-group collapse can occur before strong value conflict;
- simultaneous local improvement can still be possible;
- gradient-aware interventions such as PCGrad and MGDA can mitigate the failure.

Avoid presenting PCGrad as a proposed method.

---

### Introduction
Clarify:

**Value-level conflict**
- Improving one group’s reward intrinsically harms another.

**Optimization-level interference**
- The actual pooled PPO update harms one group even when simultaneous local improvement is still possible.

Explain why the distinction matters:
- Value conflict motivates changing the objective.
- Optimization interference motivates examining / correcting update geometry.

Change the headline question from “refuting fairness–utility trade-off” to:

> Is the observed worst-group collapse necessarily a value trade-off, or can it instead be an optimization failure?

Also define:
- utility = population-weighted reward;
- fairness = worst-group reward;
- worst-group collapse = sharp deterioration of the minimum group performance while another group remains comparatively high.

---

### Contribution list
Possible revised contributions:

1. Construct a controlled preference-divergence benchmark and identify worst-group collapse before strong value conflict.
2. Show that the collapsed vanilla checkpoint still admits simultaneous local improvement.
3. Diagnose update-level interference near collapse onset and use early-vs-late intervention to show that early conflicting updates are consequential.
4. Use SumGrad to separate keeping both objectives active from gradient surgery, and MGDA to show that the conclusion is not PCGrad-specific.
5. Provide robustness evidence across population mixtures and additional language-model backbones.

---

### Section 2.3
Broaden from a PCGrad-specific presentation to a general **group-aware gradient optimization** discussion.

Introduce:
- SumGrad.
- PCGrad.
- MGDA.

Frame all three as **diagnostic optimization interventions**, not proposed methods.

---

### Section 3.1
After the collapse / Pearson-correlation result:

- Keep Pearson correlation as a global descriptive statistic.
- Add the **local joint-improvement test** as more direct evidence that simultaneous improvement remains possible at the collapsed vanilla checkpoint.

Use “local joint improvement,” not Pareto-optimality language.

---

### After Figure 3
Add the **early-vs-late PCGrad intervention**.

Use it to explain:
- why early negative cosine can have persistent consequences;
- why later positive cosine does not imply the earlier damage has been reversed.

This should be in the main text because it directly closes a causal gap in the original manuscript.

---

### Section 3.2
Rename to:

> **Objective Selection, Joint Optimization, and Gradient Repair**

Integrate SumGrad into Figure 1 and the main comparison.

Core interpretation:
- MaxMin vs. SumGrad: objective selection vs. keeping both objectives active.
- SumGrad vs. PCGrad: no projection vs. gradient projection.
- Add MGDA as evidence that the conclusion is not specific to PCGrad.

Do not conclude that PCGrad is universally superior.

---

### Robustness subsection
Add a short subsection for:
- Pythia-160M.
- Llama-3.2-1B.

Keep the claim modest:
- the observed collapse/recovery pattern is not unique to GPT-2.

Place detailed settings in appendix.

---

### Appendix
Add:
- full group-mixture sweep;
- full Pythia/Llama results and settings;
- qualitative generations;
- expanded MGDA results if not in main Figure 1;
- multi-group scaling / limitation discussion.

---

### Limitations
Update the current GPT-2-only limitation.

New limitations should emphasize:
- controlled synthetic two-attribute benchmark;
- additional backbones improve robustness evidence but do not establish frontier-scale or deployment generalization;
- two-group setting is intentionally diagnostic;
- PCGrad may depend on projection order for multiple groups;
- pairwise projection scales as \(O(K^2)\);
- gradient-magnitude imbalance and unequal group sampling are not solved by PCGrad;
- no claim that PCGrad is the correct optimizer for arbitrary heterogeneous-preference RLHF.

---

### Conclusion
Shift from:
- “PCGrad repairs the failure better than MaxMin.”

Toward:
- worst-group collapse can occur even when simultaneous local improvement remains possible;
- pooled PPO can exhibit early update-level interference;
- early correction matters;
- SumGrad shows that keeping both objectives active alone is insufficient;
- PCGrad and MGDA show that modifying update geometry can mitigate the failure.

Final broader message:

> The contribution is not a new optimizer, but evidence that heterogeneous-preference RLHF should distinguish **which objective is optimized** from **how multiple objectives interact during optimization**.
