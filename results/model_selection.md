# Model selection record (written BEFORE the test set was used)

Date: 24 Oct 2026.

Primary metric: validation MAE (people). Tie-break: RMSE (large misses
matter most for ventilation sizing), then CV mean MAE.

Validation MAE, top three: RBF SVR 0.3339, decision tree 0.3355, MLP 0.3369.
Pairwise day-block bootstrap (2000 resamples) of MAE differences:

| pair | diff | 95% CI |
|---|---|---|
| MLP - SVR | +0.0030 | [-0.0125, +0.0133] |
| MLP - tree | +0.0014 | [-0.0058, +0.0080] |
| SVR - tree | -0.0016 | [-0.0105, +0.0100] |

All include 0 -> statistical tie on MAE.

Tie-break by RMSE: MLP 0.836 < kNN 0.847 < tree 0.877 < SVR 0.958.
MLP also has the best CV MAE (0.371 +/- 0.074) and best R2 (0.586).
SVR additionally under-predicts (bias -0.065 people) and was trained on a
5,000-row subsample.

**Selected: Neural network (MLP), hidden_layer_sizes=(32,), alpha=0.01.**
Runner-up kept for interpretability discussion: decision tree (depth 14, min leaf 100).

Final test protocol: refit selected model and baselines on train + validation
(Nov 2023 - Oct 2025), evaluate once on test (Nov 2025 - Apr 2026).

## Addendum (25 Oct 2026, after the review)

All five families were later scored on the test period for completeness of
reporting; the selection above was not revisited. On test, RBF SVR had the
lowest MAE (0.2745 vs MLP 0.2833) but higher RMSE (0.861 vs 0.828) and a
negative bias (-0.040, under-predicts); k-NN and ridge did not beat
persistence on test (bootstrap CIs include or exceed zero).
