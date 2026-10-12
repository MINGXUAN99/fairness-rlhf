# Figure 1 with SumGrad

Run `python3 notebooks/plot_figure1_with_sumgrad.py` from the repository root.
Outputs: `output/pdf/figure1_with_sumgrad.pdf` and a 300 dpi PNG preview.
The CSV and script work offline without the missing evaluation datasets.

## Data provenance

- SumGrad: reported worst-group means from `Rebuttal EMNLP26 RLHF.pdf`,
  page 12, W1 table, repeated across pages 6-7:
  `(0.6, 0.939), (0.8, 0.897), (1.0, 0.822), (1.1, 0.819),
  (1.2, -0.106), (1.3, -0.073)`.
- At the user's request, SumGrad means are linearly interpolated at alpha
  0.7 (0.918, from 0.6 and 0.8) and 0.9 (0.8595, from 0.8 and 1.0). Alpha 0.5 is linearly extrapolated (0.960, from
  0.6 and 0.8), because it is outside the reported range. These are not
  experimental measurements; `mean_kind` distinguishes their provenance.
- SumGrad uncertainty is not reported. Its `std` column remains blank.
  At the user's request, `proxy_std` holds 0.8 times the PCGrad standard
  deviation at the same alpha, used as the SumGrad error-bar half-height.
  These are illustrative proxy bars, not measured SumGrad standard
  deviations or confidence intervals. The suggested caption below discloses this explicitly; the legend uses only method names.
- Vanilla, MaxMin, and PCGrad: means and error-bar half-heights recovered
  from the vector drawing of Figure 1 on page 3 of
  `10802_Not_All_Heterogeneity_Is.pdf`. These are reconstructed plotted
  values, not recovered raw evaluation samples. Six decimal places retain
  extraction precision, not experimental accuracy.
- Recovery used PyMuPDF drawing paths 238, 249, and 260 (respectively)
  and the matching vertical error-bar paths. The union of background
  rectangles 3 and 4 defines the axes bounds in PDF coordinates:
  `(326.4552307, 73.3616486, 521.9119873, 183.3953400)`.
  Map x linearly to `[0.45, 1.35]` and y inversely to `[1.3, -0.45]`.
  All 12 overlapping MaxMin and PCGrad means match the rebuttal table to
  within 0.001.
- Existing error bars represent one standard deviation of the selected
  worst group's evaluation rewards, following cell 5 of
  `notebooks/analyze_ppo_by_alpha.ipynb`; they are not across-seed confidence
  intervals. Their original sizes are preserved.
- Vanilla at alpha 1.1 was interpolated in the original notebook (cell 1).
  Its original plotted value is retained, with blank uncertainty rather
  than presenting the original zero-height bar as a measured zero variance.
- PCGrad has no point at alpha 0.75 in the original Figure 1. It remains
  absent here. The existing notebook references seed 42 for SumGrad and
  mostly seed 43 for the other methods; this recovery does not establish
  a matched-seed or multi-seed comparison.

## Suggested caption

**Worst-group reward under preference heterogeneity.** Vanilla RLHF
exhibits worst-group collapse as preference divergence increases. MaxMin
selects the worst-group objective, while SumGrad keeps both group objectives
active and sums their gradients. PCGrad uses the same group objectives as
SumGrad but projects conflicting gradient components. The SumGrad-PCGrad
comparison tests the contribution of gradient projection. Original-method
error bars denote one standard deviation of evaluation rewards for the
worst group. SumGrad means at alpha 0.7 and 0.9 are linearly interpolated;
alpha 0.5 is linearly extrapolated; other SumGrad means are recovered from
the rebuttal. SumGrad error bars are illustrative proxies equal to
0.8 times PCGrad's standard deviation, not measured SumGrad uncertainty.
Lines are visual guides.
The Vanilla point at alpha 1.1 is interpolated as in the original figure.
