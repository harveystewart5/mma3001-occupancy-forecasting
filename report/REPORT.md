---
title: "Forecasting Room Occupancy for HVAC Scheduling"
subtitle: "MMA3001 Numerical Methods and Machine Learning - Project Report (Dataset 2: Monash Smart Infrastructure)"
author: "Harvey"
date: "October 2026"
---

**Summary.** I used Monash occupancy sensor data to predict how many people will be in a room over the next hour, so heating and cooling (HVAC) only needs to run when a room will actually be used. A small neural network predicted next-hour headcount with an average error of 0.28 people (0.55 during weekday working hours), a small but statistically real improvement on the simple baseline "next hour = last hour" (0.30, or 0.59 in working hours). I also tested whether a CO₂ mass-balance ODE could estimate occupancy from a CO₂ sensor. It gave sensible ventilation rates but was too noisy to replace the occupancy sensors.

## 1. Engineering problem

**Context.** Ventilation is sized for the number of people in a room, because people breathe out CO₂. Most buildings still run HVAC on a fixed timetable (e.g. 7am-7pm weekdays), so empty rooms are conditioned anyway and busy rooms can get stuffy.

**Problem.** *At the start of each hour, predict the average number of people in each room for the coming hour, using only data available at that time.* As a smaller second part, I tested whether a CO₂ sensor could be used to estimate occupancy with a physics model.

**Why it matters.** One hour of warning lets the HVAC pre-condition a room that will be used and turn down one that won't. Predicting too high wastes energy and predicting too low leaves people in a poorly ventilated room, so I report which way the model is wrong (bias) as well as its overall error.

**Intended use.** One input to a room-by-room HVAC schedule. It is not a safety system, so it should be used with a margin at busy times.

**Scope limits.** Five spaces, only two of which (G.20 and G.25, Building 60) could be matched to real rooms by finding their space IDs in the location spreadsheet, which is itself labelled "Incorrect Data". Room sizes and timetables are unknown. The CO₂ part uses one sensor (room G.38) for about 15 weeks and treats the room as perfectly mixed.

## 2. Inputs, outputs and data

**Data.** The Monash Smart Infrastructure data [2]: the occupancy file has 9,091,732 rows from 5 spaces (Nov 2023 to Apr 2026). A row is only written when a room's count changes, so it's an irregular event stream. The environmental file has 179,447 messages from 5 CO₂ sensors, with readings stored as JSON text. The sensor list names 77 sensors, but only 5 are in the data.

**Data audit** (`occupancy/audit.py`). Each check reports what it finds instead of silently fixing it.

- **Frozen sensors:** a working sensor almost never repeats a message exactly. Sensors 777, 227 and 241 had only 1, 2 and 3 different messages out of thousands. Sensor 777 kept resending a reading from 6 May 2023, which only shows up in the measurement time inside the message, not the arrival time.
- **Usable CO₂:** sensor 869 (G.38) worked from 29 Feb to 13 Jun 2024. Sensor 326 works but isn't on the sensor list and its CO₂ is no higher on weekdays than weekends (-1.5 ppm vs +67 ppm for sensor 869), so I didn't use it.
- **Occupancy rows:** 4 duplicates removed; no negative headcounts or contradictory statuses.
- **Outages:** a long silence could mean "nothing changed" or "sensor offline". If an occupied room went silent for over 1 hour, or an empty room for over 24 hours, I treated the rest of the gap as missing. This found 217 outages (about 1,190 hours). I test this choice in Section 5.

**Inputs and outputs.**

| | What it is | Format and units |
| --- | --- | --- |
| Input | Occupancy events | room ID, timestamp with time zone, headcount (whole people ≥ 0) |
| Input | CO₂ readings | sensor ID, measurement time, CO₂ in ppm (must be positive; this data ranged 403-646) |
| Output | Hourly occupancy | time-weighted average headcount per room-hour (people ≥ 0) |
| Output | Forecast | predicted average headcount for the next hour, for the HVAC schedule (people; negative predictions are set to 0) |
| Output | CO₂ model | air change rate λ (per hour), CO₂ per person g (ppm/h/person), simulated CO₂ (ppm) |

**Invalid or missing inputs.** Missing columns, wrong types, timestamps without a time zone, negative headcounts, unsorted events, non-positive time steps and broken JSON raise a `ValueError`/`TypeError` with a message, and a missing file raises `FileNotFoundError`. A single failed reading inside a JSON message (sensor 326 sometimes sends an error code instead of a value) is skipped without losing the rest. Outage time is treated as missing, hours less than half covered are dropped, and an unseen room still gets a forecast: its room columns are all zero, so the model relies only on the time and past-headcount inputs. Every function's inputs and outputs are in its docstring and the pdoc HTML docs.

## 3. Computational approach

Figure 1 shows the pipeline.

![Figure 1. Project pipeline. Each stage is one script and the whole thing re-runs with `python -m scripts.run_all`.](pipeline.png)

### 3.1 Events to hourly values (numerical integration)

Between events the headcount is constant, so headcount over time is a step function. The hourly average is the area under it divided by the time covered:

$$\bar{N} = \frac{\int N(t)\,dt}{\int dt} = \frac{\sum_i N_i\,\Delta t_i}{\sum_i \Delta t_i}$$

For a step function this is exact. I chose it because simply averaging the rows is wrong: busy periods create many rows and empty periods very few. Compared with the time-weighted value, the row average was 1.4-2.1 times too high in a typical hour (the median for each room) and more than 12 times too high in one hour in ten (e.g. 3.6 people instead of 0.3 at 9am on 10 April 2024 in G.20). The result was 84,422 hourly values.

### 3.2 What the model can use (prediction contract, Week 5.6)

A forecast made at the start of an hour can only use: the hour and day of week (converted with sine and cosine so 11pm sits next to midnight, like a clock face), whether it's a weekend, which room it is, and **past** headcounts (1, 2, 3, 24 and 168 hours ago, and the average of the last 24 hours). Lags come from a complete hourly timeline, so a missing hour stays missing instead of borrowing the wrong hour. 82,215 rows were usable.

### 3.3 Models

I compared three baselines (always predict the average; persistence, "next hour = last hour"; and the average for that room, weekday and hour) with five models from the unit: ridge regression (5.1), a decision tree (5.2), k-nearest neighbours (3.6), an SVR (5.3, trained on 5,000 rows because it is slow on large data) and a neural network with one hidden layer of 32 neurons (5.4). All models use scikit-learn [3]. Scaling and encoding are done inside a `Pipeline` so they are only learned from training data. I chose each model's settings with `TimeSeriesSplit` (4 folds, one-day gap) on the training period only.

### 3.4 CO₂ mass-balance model (ODE, Week 7)

Treating the room as one well-mixed volume, CO₂ rises because people breathe it out and falls because ventilation removes it:

$$V\frac{dC}{dt} = G\,N(t) - Q\,(C - C_b) \quad\Longrightarrow\quad \frac{dy}{dt} = g\,N(t) - \lambda\,y$$

where y = C - C~b~ is CO₂ above background, g = G/V and λ = Q/V. **Assumptions:** perfect mixing, constant background (429 ppm), the same CO₂ output per person, and constant occupancy within each 10-minute step. Only g and λ can be found from CO₂ data, since V, Q and G only appear as ratios.

I solved it with **forward Euler**, y~k+1~ = (1 - hλ)y~k~ + hgN~k~, and **backward Euler**, y~k+1~ = (y~k~ + hgN~k~)/(1 + hλ). I fitted λ from 75 nights of CO₂ decay in the empty room (a least-squares straight line through ln(C - C~b~), using only readings at least 15 ppm above background, which also skips any readings below it), then fitted the daytime λ and g on March-April weekdays. No CO₂ sensor shares a room with an occupancy sensor, so I used another room's headcount as a stand-in. Space 3e5a matched the CO₂ best (correlation 0.61) but has no March-April data, so I used the next best, 941f (0.50). This is a big assumption.

## 4. Validation method and results

**What was validated.** (1) The forecast against baselines; (2) the numerical methods (verification); (3) the CO₂ model against measured CO₂ (validation).

**Why this method.** I split the data by time: train Nov 2023-Jun 2025, validate Jul-Oct 2025, test Nov 2025-Apr 2026. A random split would let the model see the hours either side of each test hour, and neighbouring hours are very similar (persistence alone gets R² = 0.42 on the validation data), so results would look better than they really are. A time split tests the model the way it would be used.

**Metrics and acceptance criteria.** MAE (average error in people) is my main metric, RMSE weights big misses more, and bias shows over- or under-prediction. To be accepted, a model had to beat persistence on validation with a 95% bootstrap confidence interval (Week 5.5) that excludes zero. I resampled whole days rather than hours, because errors on the same day are related. All five ML models met this rule on validation; the neural network beat persistence by 0.027 people (95% CI 0.016-0.039).

**Did validation influence model selection?** Yes. Cross-validation chose each model's settings and the validation period chose the model. The SVR (MAE 0.334), tree (0.336) and neural network (0.337) were tied within the confidence intervals, so I used validation RMSE to break the tie, which picked the neural network (0.836, against 0.877 for the tree and 0.958 for the SVR; see Table 2). I saved this choice in git **before** using the test set once.

Table 1. Test results (Nov 2025-Apr 2026). The last column is the difference in MAE from persistence with its 95% bootstrap CI; negative means better than persistence.

| Model | MAE (people) | RMSE | Bias | R² | MAE, weekdays 8am-6pm | MAE vs persistence (95% CI) |
| --- | --- | --- | --- | --- | --- | --- |
| **Neural network (selected)** | 0.283 | **0.828** | +0.016 | **0.510** | 0.552 | -0.015 (-0.023 to -0.006) |
| SVR | **0.275** | 0.861 | -0.040 | 0.470 | **0.537** | -0.023 (-0.034 to -0.014) |
| Decision tree | 0.282 | 0.846 | +0.016 | 0.488 | 0.566 | -0.016 (-0.025 to -0.008) |
| k-nearest neighbours | 0.295 | 0.831 | -0.002 | 0.506 | 0.573 | -0.003 (-0.014 to +0.008) |
| Persistence | 0.298 | 0.977 | 0.000 | 0.317 | 0.591 | - |
| Ridge regression | 0.305 | 0.857 | +0.034 | 0.474 | 0.567 | +0.007 (-0.003 to +0.017) |
| Seasonal profile | 0.506 | 1.050 | +0.177 | 0.212 | 0.944 | +0.208 |
| Global mean | 0.747 | 1.202 | +0.217 | -0.034 | 0.917 | +0.449 |

The neural network beat persistence on test (95% CI excludes zero) by 5% on MAE and 15% on RMSE: real, but modest. The test results did not exactly match validation. The SVR had the lowest test MAE, but a higher RMSE and a negative bias (it under-predicts, the riskier direction for ventilation), and k-NN and ridge no longer beat persistence. Because the model had already been chosen and recorded, these results could not change the choice, which is exactly why the test set is only used once.

![Figure 2. A test week in G.20: measured headcount, persistence and neural network forecasts.](../results/figures/test_week_g20.png){width=88%}

Both models are about an hour late when people arrive (Figure 2). Neither predicted the 29-person spike on 12 March: the same hour the week before had only about 4 people, so it was a one-off event rather than a weekly class, and nothing in the inputs could have signalled it.

![Figure 3. Test error grouped by how many people were really in the room.](../results/figures/test_mae_by_band.png){width=58%}

Errors grow with occupancy (Figure 3), and the neural network **under-predicts busy hours** (by 1.15 people when 3-6 were present and 5.4 when more than 6 were). That is its main weakness, because it would under-ventilate at peak times.

**What this means for HVAC.** On weekdays 7am-7pm, 2,285 of 5,792 test room-hours (39%) were empty. Turning HVAC down whenever the forecast is below 0.5 people, persistence would save 1,869 empty hours but wrongly turn down 17% of occupied hours. The neural network would save 1,503 and wrongly turn down only 6%: slightly less saving, but far fewer stuffy rooms. Because the forecast is late on arrivals, most of this value comes from turning HVAC down in hours that stay empty (unused rooms, lunch, after classes) rather than from pre-conditioning a room before the first person arrives. I couldn't convert these hours into kWh because room sizes and HVAC details aren't in the data.

**Verification of the numerical methods.**

- *Resampling:* total person-hours before and after splitting into hours agree to 1.2 × 10⁻¹⁶ (machine precision), and a hand-worked example (empty 50 min, 3 people for 10 min = 0.5) comes out exactly.
- *ODE solver (Figure 4):* both Euler methods match the exact solution y(t) = y~ss~ + (y~0~ - y~ss~)e^-λt^. Halving the time step halves the error (first order). At hλ = 2.5, forward Euler blows up and backward Euler stays stable, matching theory.
- *Fitting:* fake data made with known g and λ is fitted back exactly (manufactured solution).

![Figure 4. Left: error against time step for both Euler methods. Right: forward Euler is unstable when hλ > 2; backward Euler is not.](../results/figures/co2_verification.png){width=86%}

**Validation of the CO₂ model.** The daytime λ = 0.95 per hour (about one air change per hour) is normal for a mechanically ventilated room. The night value, 0.04 per hour, is lower than typical air leakage, which suggests the room is well sealed overnight or the background level drifts, so I treat it with caution. The fitted g = 41 ppm per hour per person implies a room of about 450 m³ using 0.0052 L/s of CO₂ per seated adult [6]. That is plausible, but g also absorbs any difference in size between G.38 and the stand-in room, so it is only a rough check. With 10-minute steps hλ = 0.16, well under the stability limit of 2. On 31 held-out days the CO₂ error was 38 ppm, against 82 ppm if people are ignored. That is a weak comparison, so it mainly shows that occupancy explains a large part of the CO₂ changes (Figure 5).

![Figure 5. Held-out days: measured CO₂ against the model driven by the stand-in headcount.](../results/figures/co2_validation.png){width=82%}

The model drops too low before 8am because it assumes the daytime ventilation rate all day, which is an error in the model's assumptions, not the solver. Working backwards from CO₂ to headcount with central differences (Week 7.1) followed the stand-in room's headcount (correlation 0.63) but had an error of 0.67 people, because differentiating noisy data amplifies the 8 ppm sensor noise.

**What validation cannot establish.** Performance in other buildings or seasons, whether the occupancy sensors count correctly (there is nothing independent to compare against), and G.38's true occupancy, since the CO₂ test uses another room's headcount.

## 5. Comparison and optimisation

### 5.1 Alternative solutions

Table 2. Comparison on the validation period (used for model selection).

| Approach | Val MAE | Val RMSE | CV MAE (mean ± std) | Train time | Predict time | Interpretable? |
| --- | --- | --- | --- | --- | --- | --- |
| Neural network | 0.337 | **0.836** | 0.371 ± 0.074 | 4.0 s | 0.02 s | Low |
| Decision tree | 0.336 | 0.877 | 0.374 ± 0.084 | 0.28 s | 0.01 s | High |
| SVR (5,000 rows) | **0.334** | 0.958 | 0.391 ± 0.114 | 0.42 s | 1.1 s | Low |
| k-nearest neighbours | 0.343 | 0.847 | 0.398 ± 0.090 | 0.04 s | 1.1 s | Medium |
| Ridge regression | 0.350 | 0.878 | 0.381 ± 0.077 | 0.05 s | 0.01 s | High |
| Persistence | 0.364 | 0.992 | - | - | instant | High |

The spread between CV folds (about ±0.07-0.11) is bigger than the gaps between models, so they perform similarly. SVR and k-NN are slow to predict because they compare each new point with stored training data, and a Gaussian process would need about 23 GB of memory for 54,000 rows, so I didn't use one. I chose the neural network for its lowest RMSE and fast predictions; the decision tree would be a good choice if explaining the model mattered more.

### 5.2 Sensitivity analysis

Table 3. Sensitivity to the outage limits (validation data).

| Outage limit (occupied / empty) | Person-hours | Val MAE persistence | Val MAE neural net |
| --- | --- | --- | --- |
| 0.5 h / 24 h | 58,830 | 0.364 | 0.341 |
| **1 h / 24 h (used)** | 59,040 | 0.364 | 0.337 |
| 4 h / 24 h | 59,174 | 0.364 | 0.335 |
| 1 h / 12 h | 59,027 | 0.376 | 0.350 |

Person-hours is the total of people × hours over all rooms. Each setting slightly changes which hours exist, so the MAEs are on slightly different samples, but the occupied-room limit barely changes anything, so the results don't depend on that choice. A 12-hour empty limit throws away about 4,000 genuinely quiet weekend hours, so I kept 24 hours. For the CO₂ model, changing the background level from 422 to 445 ppm changed g by about ±4% but the daytime λ by about ±13%, so λ is the most sensitive physics parameter.

### 5.3 Performance analysis (Week 2.1)

**Machine limits.** Following Week 2.1 [1], the theoretical peak is F~T~ = 2 cores × 2.1 GHz × 32 FLOPs/cycle = 134 GFLOP/s. I couldn't install likwid, so I measured the limits: a large matrix multiply reached 65 GFLOP/s, and a triad (A = B × C + D) on arrays bigger than the cache reached 15.5 GB/s. That gives a machine balance of 4.2 FLOPs/byte. All timings are the mean ± std of 7 `timeit` repeats, because times on a shared machine vary.

| Code | FLOPs / bytes per step | Arithmetic intensity | Time | % of roofline |
| --- | --- | --- | --- | --- |
| Backward Euler, Python loop (10⁶ steps) | 4 / 16 | 0.25 | 244 ± 0.6 ms | 0.4% |
| Backward Euler, `lfilter` (10⁶ steps) | 3 / 16 | 0.19 | 7.0 ± 1.0 ms | 15% |
| Neural network prediction (117k rows\*) | 1,120 / 136 | 8.2 | 70 ± 0.9 ms | 2.9% |

\*The validation set repeated 10 times, to get a longer and more stable timing.

![Figure 6. Roofline plot (Williams et al. [5]) with the measured machine limits and my code.](../results/figures/roofline.png){width=60%}

**Baseline vs improved.** On the roofline (Figure 6), the solver's low arithmetic intensity puts it under the memory-bandwidth limit, but the Python loop only reached 0.4% of that limit. line_profiler showed all the time was in the loop itself, about 0.7 µs per step, so the bottleneck was Python's overhead, not the hardware. Backward Euler here is just "new value = a × old value + b × input", and SciPy's `lfilter` [4] runs exactly that loop in C. It gives identical answers (tests agree to 10⁻¹²) and is **35-38 times faster**, still scaling linearly with the number of steps. It still only reaches 15% of the bandwidth limit because each step has to wait for the previous one, so the real limit is how fast one step can follow another, not memory. On this 15-week dataset the loop only takes milliseconds, so the speed-up matters when the model is fitted many times or scaled to many sensors over several years. The neural network has a higher intensity (8.2), so it is compute-bound, but its small layers mean it only reaches 2.9% of peak.

**Precision.** float32 changed the solver's output by only 4 × 10⁻⁷ (far below the 8 ppm sensor noise) but wasn't any faster, because each step has to wait for the last one, so I kept float64.

## 6. Limitations and lessons learned

**Limitations.** The model under-predicts busy hours, so it would need a safety margin in practice. It is about an hour late on arrivals, so it helps more with turning HVAC down in empty hours than with pre-conditioning before people arrive. It has no timetable information, so it can't see one-off events coming. Only five rooms were available, two identified, all from one sensor device. The occupancy sensors are assumed to be correct. The CO₂ model assumes perfect mixing and one daytime ventilation rate, and uses another room's headcount.

**Lessons learned.**

1. **Check the data first.** Most of the listed sensors were missing or broken, and the frozen ones only showed up in the measurement times.
2. **How you average matters.** Averaging rows instead of time overestimated occupancy by 1.4-2.1 times in a typical hour, and over 12 times in one hour in ten.
3. **Simple baselines are hard to beat.** Persistence already captured most of the pattern; ML added 5-15%. It's better to report that honestly than oversell it.
4. **Protect the test set.** The SVR happened to do slightly better on test MAE, but because the model was chosen before testing, the test result is an honest check rather than a second chance to pick.
5. **Verification and validation are different.** The solver was verified exactly, but the model was still wrong before 8am because of an assumption.
6. **Profile before optimising.** The solver looked memory-limited on paper, but profiling showed Python overhead was the real problem.

**Improvements.** Add the university timetable as an input; smooth the CO₂ data before working backwards to headcount; train the model to lean towards over-predicting, since under-ventilating is worse; and put a CO₂ sensor in the same room as an occupancy sensor to test the physics properly.

## 7. AI use and reflection

**Tools used.** Claude (Anthropic, Claude Opus 5.5) through claude.ai, 24-25 October 2026.
**What it was used for and level of contribution.**

| Task | AI contribution | What I did |
| --- | --- | --- |
| Reading the brief and data; suggesting a dataset | High | Chose Dataset 2 after asking how it knew the sensors were broken, and checked the raw data myself |
| Writing the code | High | Set the requirements, reviewed outputs, ran the pipeline |
| Choosing methods | Medium | Chose the 1-hour forecast and scope, and kept it within the unit's content |
| Tests, docs, performance | High | Checked tests use known answers and reviewed every figure |
| Report | High (first draft) | Rewrote it and checked the numbers against `results/` |

**Why I used it.** I had about two days. Using AI for the code let me spend the time understanding and checking the methods. I had it explain each step before building it, which is also how I prepared for the interview.

**How I checked it.** Tests with known answers (the 0.5-person example, the exact ODE solution, manufactured data), the conservation check, looking at raw data and every figure, and comparing claims with the source files.

**Errors it made.**

- It called sensor 326 an "outdoor" sensor, but the sensor isn't on the list, so its location is unknown.
- Its first JSON reader threw away whole messages when one reading had an error, losing 29,043 readings.
- Its resampling code used the wrong time units (putting the data in 1970) and stretched each sensor's last reading 24 hours into the future. Unit tests caught both.
- While splitting the git history into real, smaller commits (it refused to fake dates), it accidentally committed 64 MB of data, which we removed before uploading.
- It used a database tool (DuckDB) that isn't taught in the unit. In my scope review it was replaced with pandas, with identical results, and a number it had quoted from early exploration was corrected (0.53 to 0.50).

**Decisions that stayed mine.** The dataset, the forecast horizon, the scope, accepting the outage rule after testing it, the model choice by tie-break, asking for the scope reviews, and what to claim in this report.

**Effect on my understanding.** Having each step explained helped me follow the reasoning, for example why a random split leaks information or why forward Euler needs hλ < 2. The main risk was trusting it too much: several of its mistakes gave no error message and were only caught by tests or a number that looked wrong.

## 8. References

1. Monash University (2026). *MMA3001 Numerical Methods and Machine Learning*, unit notes Weeks 1.1-11.5.
2. Monash Smart Infrastructure (2026). *Dataset 2: Occupancy and Environmental Sensor Data*, supplied for MMA3001.
3. Pedregosa, F. et al. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research*, 12, 2825-2830.
4. Virtanen, P. et al. (2020). SciPy 1.0: fundamental algorithms for scientific computing in Python. *Nature Methods*, 17, 261-272.
5. Williams, S., Waterman, A. and Patterson, D. (2009). Roofline: an insightful visual performance model for multicore architectures. *Communications of the ACM*, 52(4), 65-76.
6. Persily, A. and de Jonge, L. (2017). Carbon dioxide generation rates for building occupants. *Indoor Air*, 27(5), 868-879.
7. Anthropic (2026). Claude Opus 5.5, accessed through claude.ai, 24-25 October 2026.

GitHub repository: **[add link before submission]** (includes the HTML code documentation in `docs/api` and the test report in `docs/test_report.html`).
