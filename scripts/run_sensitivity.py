"""Sensitivity analysis of the two main judgment-call parameters.

Usage:
    python -m scripts.run_sensitivity

1. Outage hold limits (Step 2/3): how long a silent occupancy sensor's last
   value is trusted. Varied around the defaults (1 h occupied, 24 h empty);
   effect on the resampled data and on validation MAE of persistence and
   the selected MLP. The test set is not used.
2. CO2 background C_b (Step 7): percentile of readings used as the
   unoccupied level; effect on fitted lam, g and validation R2.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from occupancy.evaluate import mae
from occupancy.features import TARGET, build_dataset, time_split
from occupancy.ml import Forecaster, candidates
from occupancy.models import Persistence
from occupancy.resample import resample_occupancy
from scripts.run_co2_case_study import load, run

PROC, RES = Path("data/processed"), Path("results")
logging.basicConfig(level=logging.WARNING)
log = logging.getLogger("sensitivity")
log.setLevel(logging.INFO)


def main() -> None:
    """Run both sensitivity studies and save tables."""
    occ = pd.read_parquet(PROC / "occ_events.parquet")
    mlp_cand = {c.name: c for c in candidates()}["Neural network (MLP)"]
    rows = []
    settings = [(0.5, 24), (1, 24), (2, 24), (4, 24), (1, 12), (1, 48)]
    for occ_h, empty_h in settings:
        hourly = resample_occupancy(occ, max_hold_occupied=pd.Timedelta(hours=occ_h),
                                    max_hold_empty=pd.Timedelta(hours=empty_h))
        data = build_dataset(hourly)
        train, val, _ = time_split(data, "2025-07-01", "2025-11-01")
        mlp = Forecaster(mlp_cand.build({"hidden_layer_sizes": (32,), "alpha": 0.01})).fit(train)
        rows.append({"hold_occupied_h": occ_h, "hold_empty_h": empty_h, "hourly_bins": len(hourly),
                     "person_hours": hourly["person_hours"].sum(), "model_rows": len(data),
                     "val_rows": len(val), "val_mean_headcount": val[TARGET].mean(),
                     "val_MAE_persistence": mae(val[TARGET], Persistence().predict(val)),
                     "val_MAE_MLP": mae(val[TARGET], mlp.predict(val))})
        log.info("%s", rows[-1])
    hold = pd.DataFrame(rows)
    hold.to_csv(RES / "sensitivity_hold_limits.csv", index=False)
    log.info("Hold-limit sensitivity:\n%s", hold.round(4).to_string(index=False))

    df = load()
    co2 = []
    for pct in [1, 5, 10, 25]:
        o = run(c_base_pct=pct, save=False, df=df)
        co2.append({"c_base_percentile": pct, "c_base_ppm": o["c_base_ppm"],
                    "lam_night_per_h": o["lam_night_per_h"], "lam_day_per_h": o["lam_day_per_h"],
                    "g_ppm_per_h_per_person": o["g_ppm_per_h_per_person"],
                    "implied_volume_m3": o["implied_volume_m3"], "test_r2": o["test_r2"],
                    "test_rmse_ppm": o["test_rmse_ppm"], "inverse_corr_hourly": o["inverse_corr_hourly"]})
    co2 = pd.DataFrame(co2)
    co2.to_csv(RES / "sensitivity_co2_background.csv", index=False)
    log.info("CO2 background sensitivity:\n%s", co2.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
