---
title: "Occupancy Forecasting for HVAC Scheduling"
subtitle: "MMA3001 Numerical Methods and Machine Learning — Project Report (Dataset 2)"
author: "Harvey"
date: "October 2026"
---

A small neural network forecasts next-hour headcount in five Monash spaces with a mean absolute error of 0.28 people, beating the best simple baseline by 5% (MAE) and 15% (RMSE) on six months of withheld data. Used to set back HVAC in timetabled hours, it would skip 1,503 of the 2,285 hours that were actually empty while wrongly setting back only 6% of occupied hours. A CO₂ mass-balance model calibrated from a single room sensor gives physically sensible ventilation rates but is too noisy to replace occupancy sensors.

## 1. Engineering problem

**Context.** Heating, ventilation and air-conditioning (HVAC) is a major energy use in university buildings. Ventilation is sized per occupant, because people exhale CO₂ and moisture, yet most buildings run HVAC on a fixed timetable (for example 07:00–19:00 on weekdays). Empty rooms are conditioned at full rate, and crowded rooms can be under-ventilated.

**Problem.** Monash's Smart Infrastructure programme records occupancy sensors in campus spaces. The question addressed is:

> At the top of each hour, predict the time-weighted mean headcount of each monitored space over the coming hour, using only data available at that moment.

A secondary case study asks whether a physics model of CO₂ build-up can estimate occupancy from a CO₂ sensor, which is cheaper and more common than a people counter.

**Why it matters.** A one-hour lead time is enough for an HVAC controller to pre-condition a space that will be used and to set back one that will not. Over-prediction wastes energy; under-prediction risks stuffy rooms. Both directions are therefore reported separately (bias), not just overall error.

**Intended use.** The forecast is an input to a demand-controlled ventilation schedule for each space, not a safety system. It should be combined with a margin or a real-time override during peaks (Section 4).

**Scope and its limits.** Five spaces are covered, but only two (G.20 and G.25, both in Building 60) could be matched to named rooms because the supplied location register is labelled "Incorrect Data". Room sizes, uses and timetables are unknown. The CO₂ study uses one sensor (G.38) for 15 weeks, and the room is treated as well mixed (a lumped ODE rather than a spatial PDE). These limits are deliberate: they keep the problem tractable within the unit's methods and are revisited in Section 6.

## 2. Inputs, outputs and data

The supplied Dataset 2 contains far less usable data than its sensor register suggests: 5 of 77 registered environmental sensors report, and 3 of those 5 are frozen. Auditing before modelling was therefore the first technical task.

### 2.1 Source data

| File | Contents | Used for |
| --- | --- | --- |
| `5occupancySensor_...9MRows.csv` (1.6 GB) | 9,091,732 state-change events, 5 spaces, 1 device, 24 Nov 2023 – 22 Apr 2026 | Main forecast |
| `5EnvSensor_...180kRows.csv` (353 MB) | 179,447 messages from 5 sensors; readings packed in a JSON field | CO₂ case study |
| `Sensor ID and Locations.xlsx` | Register of 77 env sensors and 89 occupancy rows; OCC tab labelled "Incorrect Data" | Locating sensors |

An occupancy row is written only when a space's state changes, so the data is an irregular event stream (median 4 s between events, silences of hours overnight). An environmental message carries two times: `createdate` (when the server received it) and, inside the JSON, `time` (when the reading was taken).

### 2.2 Data audit

Every check returns evidence rather than silently fixing data (`occupancy/audit.py`).

- **Frozen sensors.** A live sensor measuring a noisy quantity rarely repeats a message exactly, so the ratio of distinct to total messages is about 0.5. Sensors 777, 227 and 241 score 1/10,926, 2/21,363 and 3/58,908. Sensor 777 resends a reading measured on 6 May 2023 throughout 2024; only the measurement time reveals this.
- **Usable CO₂.** Sensor 869 (G.38) is live from 29 Feb to 13 Jun 2024 at 10-minute intervals, then stuck. Sensor 326 is live but absent from the register and shows no occupancy signal (weekday minus weekend daytime CO₂ is −1.5 ppm, against +67 ppm for sensor 869), so it is not used. De-duplicating re-sent readings reduces 1.82 M readings to 435,461.
- **Occupancy integrity.** 4 exact duplicates (removed); no negative headcounts, unknown statuses, or status/headcount contradictions.
- **Outages.** A room cannot plausibly hold exactly the same non-zero count for hours. A silence longer than 1 h after a non-zero count, or 24 h after zero, is treated as missing beyond that limit. This finds 217 gaps totalling about 1,190 hours (sensitivity tested in Section 5).

### 2.3 Inputs and outputs of the solution

| Item | Meaning | Format, units, domain |
| --- | --- | --- |
| Input: occupancy events | State changes per space | `space` (UUID string), `t` (timestamp with UTC offset), `headcount` (integer people ≥ 0), `status` (3 labels) |
| Input: CO₂ readings | Indoor concentration | sensor id, measurement time (Unix s), value (ppm; observed 403–646) |
| Output: hourly occupancy | Time-weighted mean headcount per space-hour | people (real ≥ 0), plus person-hours and coverage (0–1) |
| Output: forecast | Mean headcount over the next hour, issued on the hour | people (real, clipped ≥ 0) |
| Output: CO₂ model | Air-change rate λ, CO₂ per person g, simulated CO₂ | 1/h, ppm/h per person, ppm |

**Invalid, missing and unsupported inputs.** Missing columns, non-DataFrame input, timezone-naive timestamps, negative headcounts, unsorted events, non-positive time steps and malformed JSON raise `ValueError` or `TypeError` with a message; a missing file raises `FileNotFoundError`. A single failed channel inside a JSON message (sensor 326 sends `{"time": t, "errorCode": 4}` for formaldehyde) is skipped without discarding the valid channels. Outage time becomes missing rather than "unchanged"; hours with under 50% trusted coverage are dropped; an unseen space falls back to pooled behaviour. Function-level inputs and outputs are documented in Google-style docstrings rendered as HTML with pdoc (`docs/api`).

## 3. Computational approach

The workflow is a Python package (`occupancy/`) driven by one script per stage, reproducible with `python -m scripts.run_all`. A pipeline diagram is in Appendix A.1. Each stage writes evidence tables to `results/`.

### 3.1 Time-weighted resampling (numerical integration)

Between events the headcount is constant, so N(t) is a step function. The correct hourly value is its integral over the hour divided by the time covered:

$$\bar{N} = \frac{\int N(t)\,dt}{\int dt} = \frac{\sum_i N_i\,\Delta t_i}{\sum_i \Delta t_i}$$

For a step function this sum is exact, not an approximation. Intervals are cut at the outage limit and split at hour boundaries; the numerator is person-hours. A plain mean of events is biased upward because busy periods generate many events: measured over the data, it overstates headcount by a median 1.4–2.1× per space and by 12–20× in one hour in ten (on 10 Apr 2024 at 9 am in G.20: 3.60 plain vs 0.30 true). The result is 84,422 hourly values; 95 hours with under 50% trusted coverage are dropped.

### 3.2 Prediction contract and features

Following the Week 5.6 prediction contract: the forecast is issued at the top of hour τ and predicts the mean over [τ, τ+1 h). Inputs are calendar features in Melbourne time (hour and weekday encoded on the unit circle so 23:00 is next to 00:00; weekend flag) and past headcounts only: lags of 1, 2, 3, 24 and 168 h and the mean of the previous 24 h, plus the space (one-hot). Lags are computed on a complete hourly grid so an outage stays missing instead of silently pulling in a value from a different hour; 82,215 complete rows remain.

### 3.3 Baselines and models

| Model | How it predicts | Why included |
| --- | --- | --- |
| Global mean | One constant (DummyRegressor) | Floor any model must beat |
| Persistence | Next hour = last hour (lag 1) | Strong naive forecast for autocorrelated data |
| Seasonal profile | Mean for this space, weekday and hour | Captures the weekly routine |
| Ridge regression | Linear in features, L2 penalty (Week 5.1) | Simple, interpretable |
| Decision tree | Piecewise-constant splits (5.2) | Non-linear, interpretable |
| k-nearest neighbours | Distance-weighted average of similar hours (3.6) | Interpolation in feature space |
| RBF SVR | Kernel regression with ε-tube (5.3); 5,000-row subsample | Smooth non-linear fit |
| MLP | One hidden layer, ReLU, Adam, early stopping (5.4) | Combines "now" and "usual" non-linearly |

All learned preprocessing (standardisation, one-hot encoding) sits inside a scikit-learn `Pipeline`, so it is fitted on training rows only. Hyperparameters (for example tree depth 6/10/14, MLP width and L2 penalty) were chosen by forward-chaining `TimeSeriesSplit` (4 folds, one-day gap) on the training period. Predictions are clipped at zero.

### 3.4 CO₂ mass-balance model (ODE)

Treating G.38 as a well-mixed room (Week 7.2):

$$V\frac{dC}{dt} = G\,N(t) - Q\,(C - C_b) \quad\Longrightarrow\quad \frac{dy}{dt} = g\,N(t) - \lambda\,y,\qquad y = C - C_b,\; g = \frac{G}{V},\; \lambda = \frac{Q}{V}$$

Assumptions: uniform concentration, constant background C~b~ (5th percentile of readings, 429 ppm), constant CO₂ output per person, and occupancy constant over each 10-minute step. Only g and λ are identifiable; V, Q and G cannot be separated from CO₂ data alone.

The ODE is integrated with forward Euler, y~k+1~ = (1 − hλ)y~k~ + hgN~k~, and backward Euler, y~k+1~ = (y~k~ + hgN~k~)/(1 + hλ). Because the ODE is linear, the implicit step is solved exactly with one division. λ is first estimated from 75 overnight decays (N = 0) by linear least squares on ln(C − C~b~); the daytime λ and g are then fitted on March–April weekdays by a grid over λ with closed-form least squares for g, using linearity of y in g. Occupancy for G.38 is not measured, so a proxy space is used, an explicit assumption of the case study. After removing each series' usual hour-of-week pattern, hourly CO₂ correlates with space 3e5a (r = 0.61), 941f (0.50), 0f3f (0.41), G.25 (0.29) and G.20 (0.20). Space 3e5a has no data in the March–April calibration window, so 941f was chosen.

## 4. Validation and verification

On six months of withheld data the selected MLP beats persistence by 0.015 people MAE (95% CI 0.006–0.023), and the CO₂ solver reproduces the analytic solution with the first-order convergence theory predicts. Verification (are the equations solved correctly?) and validation (is the model right for the data?) were treated separately.

### 4.1 What was validated, and how

**Forecast.** Data were split chronologically: train 30 Nov 2023 – 30 Jun 2025 (54,445 space-hours), validation 1 Jul – 31 Oct 2025 (11,676), test 1 Nov 2025 – 22 Apr 2026 (16,094). A random split was rejected because neighbouring hours are nearly identical (persistence alone gives R² 0.42), so random test hours would have their neighbours in training, and because occupancy drifts (2024 was busier than 2026). The test period was used once, after the model had been chosen.

**Metrics and criteria.** MAE in people is the primary metric because it is directly interpretable; RMSE penalises large misses, which matter most for ventilation; bias shows the direction of error; R² is reported for completeness. The acceptance criterion was that a model must beat persistence on validation with a day-block bootstrap 95% confidence interval excluding zero (2,000 resamples of whole days, because errors within a day are correlated). Error is also broken down by occupancy band, by space and for weekday working hours.

**Did validation influence model selection?** Yes, by design. Cross-validation chose hyperparameters within each family; the validation period chose the family. The top three (SVR 0.334, tree 0.336, MLP 0.337 MAE) were statistically tied (pairwise CIs all include zero), so the pre-stated tie-break, lower RMSE, selected the MLP (0.836 vs 0.877 tree, 0.958 SVR). This decision was committed to git (`results/model_selection.md`) before the test set was touched.

### 4.2 Final test results

| Model | MAE (people) | RMSE | Bias | R² | MAE, weekdays 08–18 |
| --- | --- | --- | --- | --- | --- |
| **MLP (selected)** | **0.283** | **0.828** | +0.016 | **0.510** | **0.552** |
| Decision tree | 0.282 | 0.846 | +0.016 | 0.488 | 0.566 |
| Persistence | 0.298 | 0.977 | 0.000 | 0.317 | 0.591 |
| Seasonal profile | 0.506 | 1.050 | +0.177 | 0.212 | 0.944 |
| Global mean | 0.747 | 1.202 | +0.217 | −0.034 | 0.917 |

The validation ranking held on test, which supports the selection. The gain over persistence is real but modest: 5% in MAE, 15% in RMSE.

![Test week in G.20 (9–13 March 2026): measured headcount, persistence and MLP forecasts.](../results/figures/test_week_g20.png)

Both models lag arrivals by about an hour, and neither anticipates the 29-person spike on 12 March: nothing in the inputs knows about a scheduled class.

![Test MAE by measured headcount band for persistence, seasonal profile and MLP.](../results/figures/test_mae_by_band.png){width=78%}

Error grows with occupancy, and the MLP under-predicts busy hours (bias −1.15 people at 3–6 present, −5.4 above 6, over 77 hours). This regression towards the mean is the main failure mode: it would under-ventilate peaks. By space, the MLP beats persistence in three of the four spaces present in the test period (space 3e5a stops in December 2024) but is slightly worse in G.25 (0.242 vs 0.221). Only 0.006% of test hours exceed the training maximum, so the model is rarely extrapolating; the test period is quieter (mean 0.53 vs 0.74 people), which explains the seasonal profile's positive bias.

**What this means for HVAC.** Of the 5,792 timetabled space-hours in the test period (weekdays 07:00–19:00), 2,285 (39%) were empty, yet a fixed schedule conditions them all. If every hour forecast below 0.5 people is set back, persistence saves 1,869 empty hours but wrongly sets back 582 of 3,507 occupied hours (17%). The MLP saves 1,503 and wrongly sets back only 222 (6%): it gives up some savings in exchange for far fewer under-ventilated hours.

### 4.3 Verification of the numerical methods

- **Resampling:** total person-hours before and after splitting at hour boundaries agree to a relative 1.2 × 10⁻¹⁶, i.e. machine precision (float64 ε = 2.2 × 10⁻¹⁶). A hand-worked example (empty 50 min, three people for 10 min → 0.5) is reproduced exactly.
- **Leakage:** on a synthetic ramp every lag equals the target minus k exactly, and no feature equals the target.
- **ODE solver:** both Euler schemes match the analytic solution for constant occupancy, y(t) = y~ss~ + (y~0~ − y~ss~)e^−λt^; halving h halves the error (log-log slope 1.0, first order); at hλ = 2.5 forward Euler oscillates and grows while backward Euler decays, exactly as the amplification factors predict.
- **Parameter fitting:** manufactured data generated with known g and λ are recovered exactly.

![Convergence of forward and backward Euler against the exact solution (left) and stability beyond hλ = 2 (right).](../results/figures/co2_verification.png)

### 4.4 Validation of the CO₂ model

Fitted values are physically sensible: λ~night~ = 0.041 h⁻¹ (ventilation effectively off overnight) and λ~day~ = 0.95 h⁻¹ (about one air change per hour, typical of a mechanically ventilated room); g = 41 ppm h⁻¹ per person implies an effective volume of about 450 m³ for a sedentary adult emitting 0.0052 L/s of CO₂. At the data step, hλ = 0.16, far below the explicit stability limit of 2. On 31 withheld May–June weekdays, driving the model with the proxy headcount gives CO₂ RMSE 38 ppm against 82 ppm if occupancy is ignored (R² 0.39).

![Held-out days: measured CO₂ and the mass-balance model driven by proxy headcount.](../results/figures/co2_validation.png)

The model dips too low before 08:00 because a single daytime λ is applied before the HVAC actually starts: a model-form error, not a solver error. Inverting the model (headcount from CO₂ by central differences) correlates with the proxy at r = 0.63 hourly but has MAE 0.67 people, and its 10-minute estimates scatter by ±1.3 people because differentiation amplifies the 8 ppm sensor noise (Week 7.1).

### 4.5 What validation cannot establish

It cannot show performance in other buildings or room types (five spaces only), under timetable changes, or in seasons absent from the data. It cannot test whether the occupancy sensors themselves count correctly, because there is no independent ground truth. The CO₂ results rest on an unverified pairing of G.38 with space 941f, so they test the method, not G.38's true occupancy.

## 5. Comparison, sensitivity and optimisation

The MLP was chosen on evidence: it ties for the lowest MAE, has the lowest RMSE, is robust to the main data-cleaning parameter, and predicts quickly. The CO₂ solver was made 38× faster by replacing a Python loop with a compiled recurrence, and profiling shows why.

### 5.1 Comparison of approaches

| Approach | Val MAE | CV MAE (mean ± std, 4 folds) | Fit time | Predict time (11.7k rows) | Interpretability |
| --- | --- | --- | --- | --- | --- |
| MLP (32 units, α = 0.01) | 0.337 | 0.371 ± 0.074 | 4.0 s | 0.02 s | Low |
| Decision tree (depth 14, leaf 100) | 0.336 | 0.374 ± 0.084 | 0.28 s | 0.01 s | High |
| RBF SVR (5k subsample) | 0.334 | 0.391 ± 0.114 | 0.42 s | 1.1 s | Low |
| k-NN (k = 20) | 0.343 | 0.398 ± 0.090 | 0.04 s | 1.1 s | Medium |
| Ridge (α = 0.1) | 0.350 | 0.381 ± 0.077 | 0.05 s | 0.01 s | High |
| Persistence | 0.364 | – | – | instant | High |
| CO₂ inverse (physics) | 0.67\* | – | – | – | High |

\*Against proxy headcount in a different room, so not directly comparable.

The fold-to-fold spread (±0.07–0.11) is larger than the gaps between families, so the families differ little. SVR and k-NN are slow to predict because they compare each query against stored training points, and SVR's training cost (roughly O(n²)–O(n³)) forced subsampling. A Gaussian process was ruled out for the same reason: a 54k × 54k kernel matrix needs 23 GB and about 5 × 10¹³ FLOPs to factorise. The decision tree is a credible alternative where interpretability matters more than RMSE.

### 5.2 Sensitivity analysis

The two judgment-call parameters were varied with the test set untouched (full table in Appendix A.2). Moving the occupied-space hold limit from 0.5 h to 4 h changes total person-hours from 58,830 to 59,174 and MLP validation MAE from 0.341 to 0.335.

The occupied hold limit barely matters (person-hours within 0.6%, MLP MAE within ±1%). A 12 h empty limit discards about 4,000 genuinely silent weekend hours and changes which hours are evaluated, so 24 h was kept. The MLP beats persistence under every setting.

For the CO₂ model, moving C~b~ from the 1st to the 25th percentile (422 to 445 ppm) changes g by ±4% (38.7–41.4 ppm h⁻¹ per person) but λ~day~ by ±13% (0.85–1.10 h⁻¹), making λ~day~ the most sensitive physics parameter. The 25th percentile improves R² to 0.46 but lies above many overnight readings, which implies negative excess CO₂, so the 5th percentile chosen beforehand was kept.

### 5.3 Performance analysis

The machine is a shared cloud VM: 2 vCPU Intel Xeon at 2.1 GHz with AVX-512 and FMA. The theoretical peak is F~T~ = 2 cores × 2.1 GHz × 32 FLOP/cycle = 134 GFLOP/s, assuming two FMA units. likwid could not be installed, so limits were measured: a 3000 × 3000 matrix multiply reached 65 GFLOP/s, and a STREAM-style triad on 320 MB arrays (beyond the 260 MB L3) reached 15.5 GB/s, giving a machine balance of 4.2 FLOP/byte. A first run gave 61 GFLOP/s and 13.4 GB/s, a reminder that shared-VM timings vary; every timing below is the mean ± std of seven `timeit` repeats.

| Kernel | FLOPs / bytes per element | Arithmetic intensity | Time | Achieved | Share of roofline |
| --- | --- | --- | --- | --- | --- |
| Backward Euler, Python loop (N = 10⁶) | 4 / 16 | 0.25 | 244 ± 0.6 ms | 0.016 GFLOP/s | 0.4% |
| Backward Euler, `lfilter` (N = 10⁶) | 3 / 16 | 0.19 | 7.0 ± 1.0 ms | 0.43 GFLOP/s | 15% |
| MLP predict (117k rows) | 1,120 / 136 | 8.2 | 70 ± 0.9 ms | 1.86 GFLOP/s | 2.9% |

![Roofline with measured machine limits and the project kernels.](../results/figures/roofline.png){width=80%}

The solver's intensity (0.19–0.25) places it in the memory-bound region, yet the Python loop achieves only 0.4% of even the memory roof. line_profiler shows 68% of the time on the update line and 32% on the `for` statement itself, about 0.7 µs per step: the bottleneck is interpreter overhead, not hardware. Backward Euler for a linear ODE is a first-order linear recurrence, which `scipy.signal.lfilter` evaluates in C with identical results (tests agree to a relative 10⁻¹²). This is the improved solution: 35–38× faster for N ≥ 10⁴ and still O(N) (scaling study in Appendix A.3). It stays below the roof because each step depends on the last, which prevents vectorisation. The MLP lies in the compute-bound region but reaches only 2.9% of peak, because a 32-unit layer gives thin matrices and scikit-learn adds per-call overhead.

**Precision.** float32 changes the solver output by at most a relative 4.0 × 10⁻⁷ (about 3 machine epsilons, 0.0003 ppm), far below the 8 ppm sensor noise, but gives no speed-up (6.6 vs 6.8 ms) because the sequential recurrence is latency-bound, so float64 was kept.

**Pipeline.** Scalene on the resampling stage (9.1 M events, 73.5 s) shows 67% of the time in native pandas/NumPy code, 28% in system time (memory allocation) and 4% in Python, with a peak of 2.5 GB from boolean masks over 9.1 M-row tables. Building the event intervals once and reusing them for the three resampling passes is the clear next optimisation; it was not done because the stage runs once offline.

## 6. Limitations and lessons learned

The forecast is useful but not sufficient on its own for ventilation control, because it under-predicts exactly the busy hours that matter most.

### 6.1 Limitations

- **Peaks are under-predicted.** Bias of −1.15 people at 3–6 occupants and −5.4 above 6. In deployment, use the forecast with a safety margin, or take the larger of the forecast and the current count during class hours.
- **No knowledge of schedules.** Unscheduled-looking spikes (29 people on 12 March 2026) are invisible to a model that only sees past counts and the clock.
- **Small, partly unidentified sample.** Five spaces, two named; room size and use unknown; one occupancy device. Results may not transfer to other buildings.
- **Sensor accuracy is assumed.** Occupancy counts are treated as truth; there is no independent count to validate them.
- **Judgment calls in cleaning.** The outage limits and coverage threshold are choices; sensitivity testing shows the conclusions do not depend on them within sensible ranges.
- **CO₂ model form.** Well-mixed room, one daytime ventilation rate (the HVAC schedule actually switches λ), constant background, and a proxy occupancy from another space. Only g and λ are identifiable.

### 6.2 Lessons learned

1. **Audit before modelling.** The register promised 77 sensors; three of the five that reported were frozen, detectable only from the measurement timestamp inside the payload.
2. **The averaging method is a modelling decision.** A plain event mean overstated occupancy by 1.4–2.1× typically and over 12× in one hour in ten. Treating the data as a step function and integrating it exactly fixed this, and conservation of person-hours verified it.
3. **A strong naive baseline is hard to beat.** Persistence captured most of the predictable signal; ML added 5–15%. Reporting that honestly, with confidence intervals, is more useful than an inflated headline.
4. **Order of operations protects the test set.** Choosing the model on validation and recording the choice in git before testing meant the test result could be trusted.
5. **Verification and validation answer different questions.** The ODE solver was verified exactly (analytic solution, convergence order, stability limit), yet the model still dips before 08:00, which is a model-form error that no solver refinement can fix.
6. **Profile before optimising.** The solver looked memory-bound on paper but was interpreter-bound in practice; the roofline and line_profiler together pointed to the 38× fix.

### 6.3 Proposed improvements

- Add the university timetable and public holidays as inputs; this targets the largest errors directly.
- Train a quantile or asymmetric-loss model so the forecast deliberately errs on the side of over-ventilation.
- Co-locate a CO₂ sensor with an occupancy sensor to validate the mass-balance model properly, and fit separate λ values for HVAC-on and HVAC-off periods.
- Reuse event intervals across resampling passes to cut the 2.5 GB memory peak.

## 7. AI use and reflection

AI wrote most of the code and the first draft of this report under my direction; I set the scope and key decisions, questioned its claims against the data, and checked every result before it was committed.
**Tool.** Claude (Anthropic), model Claude Opus 5.5, used through claude.ai on 24–25 October 2026.

**What it was used for, and approximate contribution.**

| Task | AI contribution | My role |
| --- | --- | --- |
| Reading the brief, datasets and unit notes; recommending a dataset | High | Chose Dataset 2 after asking how the AI knew the other sensors were frozen and inspecting raw rows myself |
| Data audit and pipeline code | High (drafted all code) | Set requirements; reviewed outputs; ran the pipeline |
| Method choices | Medium (proposed options) | Chose the 1-hour horizon and the hybrid scope; asked to keep scope within the unit's methods |
| Tests, documentation, performance study | High | Checked that tests encode known answers; reviewed figures |
| Report drafting | High (first draft) | Revised wording and checked every number against `results/` |

**Why AI was used.** The project had a two-day window. AI let me spend that time understanding and checking methods rather than typing boilerplate, and its explanations at each step were how I prepared to defend the work without AI in the interview.

**How AI output was checked.** Known-answer tests (a hand-worked 0.5-person example, analytic ODE solutions, manufactured data), conservation checks, inspection of raw data, viewing every figure, and comparing claims with the source files.

**Errors and unhelpful suggestions caught.**

- It described sensor 326 as an "outdoor" sensor; the register does not list 326 at all, so its location is unknown.
- Its JSON parser discarded whole messages when one channel reported an error code, losing 29,043 valid CO₂ readings; caught because the warning count matched sensor 326's message count.
- Its resampling code assumed nanosecond timestamps (pandas 3 stored seconds), putting all data in January 1970, and it extended each sensor's last event 24 h into the future; both caught by unit tests.
- A leave-one-space-out comparison was partly in-sample; it scored the "seen" case on its own training rows.
- It installed a profiler release published the previous day whose interface differed from the unit notes; an older version was pinned.
- When I asked for a longer git history, it declined to backdate commits and instead split the real work into fine-grained commits with true timestamps, which I agreed was the honest approach. In doing so it accidentally committed 64 MB of processed data; this was found because the zipped repository was 110 MB, and the files were purged before anything was pushed. At the end I asked for a full review to keep the project within the unit's scope; this replaced a database library the AI had used for loading (DuckDB, not taught in the unit) with pandas, giving identical results, and made two numbers the AI had quoted from early exploration reproducible from the repository, which corrected one of them (0.53 to 0.50).

**Decisions that remained mine.** The dataset, the forecast horizon, the scope limit, accepting the 1 h / 24 h outage rule after the sensitivity test, accepting the MLP on the pre-stated tie-break, requiring the final review to remove anything outside the unit's methods, and what to claim and not claim in this report.

**Effect on my understanding and workflow.** Asking the AI to explain each step before building it meant I could follow the reasoning (for example why a random split leaks information, or why hλ < 2 governs explicit Euler). The risk is over-trust: several of the errors above were silent and were caught only because tests or a suspicious number forced a check. Disclosure does not replace verification, and I remain responsible for all code, results and claims.

## 8. References

1. Monash University, *MMA3001 Numerical Methods and Machine Learning* unit notes, Weeks 1.1–11.5 (2026): testing and documentation (1.5), performance and arithmetic intensity (2.1), numerical error (2.2), linear systems and conditioning (2.3), interpolation and k-NN (3.4, 3.6), regression models and evaluation (5.1–5.6), numerical integration (6.1), finite differences and Euler methods (7.1–7.3), stability of finite-difference methods (11.4).
2. Monash Smart Infrastructure, *Dataset 2: Occupancy and Environmental Sensor Data* and *Sensor ID and Locations* register, supplied for MMA3001 (2026).
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825–2830.
4. Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261–272 (`scipy.signal.lfilter`, PCHIP interpolation).
5. Williams, S., Waterman, A. and Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65–76.
6. McCalpin, J. D. (1995). Memory bandwidth and machine balance in current high performance computers. *IEEE TCCA Newsletter* (STREAM benchmark).
7. ASHRAE (2022). *Standard 62.1: Ventilation and Acceptable Indoor Air Quality*; per-person CO₂ generation of about 0.0052 L/s for sedentary adults is taken from its supporting literature (Persily, A. and de Jonge, L. (2017). Carbon dioxide generation rates for building occupants. *Indoor Air*, 27(5), 868–879).
8. Berger, E. D., Stern, S. and Pizzorno, J. A. (2023). Triangulating Python performance issues with Scalene. *USENIX OSDI* (Scalene profiler).
9. Anthropic (2026). Claude Opus 5.5, accessed via claude.
Code, results and documentation: the project GitHub repository (**[add link before submission]**), with HTML API documentation in `docs/api` and the test report in `docs/test_report.html`.

<div class="pagebreak"></div>

## Appendix A. Supplementary figures and tables

Supporting material referred to in the main text; appendices do not count towards the page limit.

**A.1 Pipeline.** Stages and their data flow (also in the repository README).

![Pipeline stages and data flow.](pipeline.png){width=62%}

**A.2 Sensitivity to the outage hold limits** (validation period; test set not used).

| Outage hold (occupied / empty) | Person-hours | Model rows | Val MAE persistence | Val MAE MLP |
| --- | --- | --- | --- | --- |
| 0.5 h / 24 h | 58,830 | 81,800 | 0.364 | 0.341 |
| **1 h / 24 h (used)** | 59,040 | 82,215 | 0.364 | 0.337 |
| 2 h / 24 h | 59,134 | 82,536 | 0.364 | 0.337 |
| 4 h / 24 h | 59,174 | 82,661 | 0.364 | 0.335 |
| 1 h / 12 h | 59,027 | 78,167 | 0.376 | 0.350 |
| 1 h / 48 h | 59,041 | 83,055 | 0.363 | 0.343 |

**A.3 Scaling study.** Mean ± std of 7 repeats; both solvers are O(N), and the compiled recurrence is 17× faster at N = 10³ (call overhead dominates) and 35–38× faster for N ≥ 10⁴.

![Run time against problem size for the Python loop and the compiled recurrence.](../results/figures/perf_scaling.png){width=70%}
