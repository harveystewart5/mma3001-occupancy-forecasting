"""Step 1 of the pipeline: load the raw exports and audit them.

Usage:
    python scripts/run_audit.py

Writes evidence tables to ``results/`` and cleaned intermediate files to
``data/processed/``.
"""

import json
import logging
from pathlib import Path

import pandas as pd

from occupancy import audit, loaders

RAW = Path("data/raw")
PROC = Path("data/processed")
RES = Path("results")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_audit")


def main() -> None:
    """Run all audit checks and save their outputs."""
    PROC.mkdir(parents=True, exist_ok=True)
    RES.mkdir(parents=True, exist_ok=True)

    # ---- Environmental sensors ----
    env_raw = loaders.load_env_raw(RAW / "5EnvSensor_MayToDec2024_180kRows.csv")
    frozen = audit.frozen_sensor_report(env_raw)
    env_long = loaders.explode_env(env_raw)
    lag = audit.measurement_lag(env_long)
    env_report = frozen.join(lag)
    env_report.to_csv(RES / "audit_env_sensors.csv")
    log.info("Env sensor report:\n%s", env_report.to_string())

    env = audit.deduplicate_env(env_long)
    env.to_parquet(PROC / "env_readings.parquet")

    # ---- Occupancy ----
    occ = loaders.load_occupancy(RAW / "5occupancySensor_MayToDec2024_9MRows.csv")
    occ_report = audit.occupancy_audit(occ)
    occ = occ.drop_duplicates().reset_index(drop=True)
    gaps = audit.find_gaps(occ)
    gaps.to_csv(RES / "audit_occupancy_gaps.csv", index=False)

    per_space = occ.groupby("space").agg(
        n_events=("t", "size"), first=("t", "min"), last=("t", "max"), max_headcount=("headcount", "max")
    )
    per_space["n_gaps"] = gaps.groupby("space").size()
    per_space["hours_missing"] = (
        (gaps["end"] - gaps["trusted_until"]).dt.total_seconds().groupby(gaps["space"]).sum() / 3600
    )
    per_space.to_csv(RES / "audit_occupancy_spaces.csv")
    log.info("Per-space summary:\n%s", per_space.to_string())

    with open(RES / "audit_occupancy.json", "w") as f:
        json.dump(occ_report, f, indent=2)
    occ.to_parquet(PROC / "occ_events.parquet")


if __name__ == "__main__":
    main()
