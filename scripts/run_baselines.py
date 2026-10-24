"""Step 4: build the feature table, split chronologically, score baselines.

Usage:
    python -m scripts.run_baselines

The **test** period is deliberately NOT scored here. It is used exactly
once, after model selection on the validation period (Week 5.6).
"""

import json
import logging
from pathlib import Path

import pandas as pd

from occupancy.evaluate import summarise
from occupancy.features import TARGET, build_dataset, time_split
from occupancy.models import GlobalMean, Persistence, SeasonalProfile

PROC = Path("data/processed")
RES = Path("results")
VAL_START, TEST_START = "2025-07-01", "2025-11-01"

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_baselines")


def main() -> None:
    """Score baselines on the validation period and leave-one-space-out."""
    hourly = pd.read_parquet(PROC / "occ_hourly.parquet")
    data = build_dataset(hourly)
    data.to_parquet(PROC / "features.parquet")
    train, val, test = time_split(data, VAL_START, TEST_START)
    split_info = {
        name: {"rows": len(d), "from": str(d["bin"].min()), "to": str(d["bin"].max())}
        for name, d in [("train", train), ("val", val), ("test", test)]
    }
    log.info("Split sizes: %s", json.dumps(split_info, indent=1))

    models = {"GlobalMean": GlobalMean(), "Persistence": Persistence(), "SeasonalProfile": SeasonalProfile()}
    rows = []
    busy = (val["dow"] < 5) & val["hour"].between(8, 17)  # weekday working hours
    for name, m in models.items():
        m.fit(train)
        p = m.predict(val)
        rows.append({"model": name, "subset": "val_all", **summarise(val[TARGET], p)})
        rows.append({"model": name, "subset": "val_weekday_8_18", **summarise(val[TARGET][busy], p[busy.to_numpy()])})
    res = pd.DataFrame(rows)
    res.to_csv(RES / "baselines_val.csv", index=False)
    log.info("Validation results:\n%s", res.round(3).to_string(index=False))

    # Leave-one-space-out: can the profile generalise to a room it never saw?
    # Both variants are scored on the validation period only (out of sample).
    loso = []
    for space in sorted(val["space"].unique()):
        te = val[val["space"] == space]
        own = SeasonalProfile().fit(train)  # trained with this space
        unseen = SeasonalProfile().fit(train[train["space"] != space])  # never saw it
        loso.append({
            "held_out": space[:8],
            "MAE_trained_on_space": summarise(te[TARGET], own.predict(te))["MAE"],
            "MAE_space_unseen": summarise(te[TARGET], unseen.predict(te))["MAE"],
        })
    loso = pd.DataFrame(loso)
    loso.to_csv(RES / "baselines_leave_one_space_out.csv", index=False)
    log.info("Leave-one-space-out (validation period):\n%s", loso.round(3).to_string(index=False))
    with open(RES / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)


if __name__ == "__main__":
    main()
