"""Step 5: tune each model family with time-series CV, compare on validation.

Usage:
    python -m scripts.run_models

Selection protocol (Week 5.6 "safe order of operations"):
1. Test period reserved (not touched here).
2. Hyperparameters per family chosen by forward-chaining CV on train only.
3. Each tuned family refitted on all of train, scored on validation.
4. Validation decides the final model; test is used once in run_final_test.
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from occupancy.evaluate import bootstrap_mae_diff, summarise
from occupancy.features import TARGET, time_split
from occupancy.ml import Forecaster, candidates, time_series_cv
from occupancy.models import GlobalMean, Persistence, SeasonalProfile

PROC = Path("data/processed")
RES = Path("results")
VAL_START, TEST_START = "2025-07-01", "2025-11-01"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_models")


def main() -> None:
    """Tune, fit and validate all models; save tables and predictions."""
    data = pd.read_parquet(PROC / "features.parquet")
    train, val, _ = time_split(data, VAL_START, TEST_START)

    cv_tables, chosen = [], {}
    for cand in candidates():
        t0 = time.perf_counter()
        cv = time_series_cv(cand, train)
        cv["cv_seconds"] = time.perf_counter() - t0
        cv_tables.append(cv)
        # cv rows are in the same order as cand.grid
        chosen[cand.name] = (cand, cand.grid[int(cv["cv_mae_mean"].idxmin())])
    cv_all = pd.concat(cv_tables, ignore_index=True)
    cv_all.to_csv(RES / "cv_results.csv", index=False)

    preds = {
        "Global mean": GlobalMean().fit(train).predict(val),
        "Persistence": Persistence().fit(train).predict(val),
        "Seasonal profile": SeasonalProfile().fit(train).predict(val),
    }
    timing = {}
    for name, (cand, params) in chosen.items():
        f = Forecaster(cand.build(params), cand.max_train_rows)
        t0 = time.perf_counter()
        f.fit(train)
        t1 = time.perf_counter()
        preds[name] = f.predict(val)
        timing[name] = {"fit_s": t1 - t0, "predict_s": time.perf_counter() - t1, "params": params}

    y = val[TARGET].to_numpy()
    busy = ((val["dow"] < 5) & val["hour"].between(8, 17)).to_numpy()
    day = val["bin"].dt.tz_convert("Australia/Melbourne").dt.date.to_numpy()
    rows = []
    for name, p in preds.items():
        d, lo, hi = bootstrap_mae_diff(y, p, preds["Persistence"], day, n_boot=2000)
        rows.append({"model": name, **summarise(y, p),
                     "MAE_work_hours": float(np.mean(np.abs(p[busy] - y[busy]))),
                     "dMAE_vs_persistence": d, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows).sort_values("MAE")
    res.to_csv(RES / "model_comparison_val.csv", index=False)
    log.info("Validation comparison:\n%s", res.round(4).to_string(index=False))
    log.info("Fit/predict times: %s", json.dumps(timing, indent=1, default=str))
    with open(RES / "model_timing.json", "w") as f:
        json.dump(timing, f, indent=2, default=str)
    pd.DataFrame({"space": val["space"].values, "bin": val["bin"].values, "y": y, **preds}).to_parquet(
        PROC / "val_predictions.parquet")


if __name__ == "__main__":
    main()
