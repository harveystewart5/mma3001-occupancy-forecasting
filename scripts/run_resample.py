"""Step 2 of the pipeline: resample occupancy events to regular bins.

Usage:
    python -m scripts.run_resample

Reads ``data/processed/occ_events.parquet`` (from ``run_audit``) and writes
hourly and 10-minute time-weighted series, plus a summary table and a
plain-mean vs time-weighted comparison for the report.
"""

import logging
from pathlib import Path

import pandas as pd

from occupancy.resample import event_intervals, resample_occupancy, split_by_bin

PROC = Path("data/processed")
RES = Path("results")

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_resample")


def main() -> None:
    """Resample events and save outputs."""
    occ = pd.read_parquet(PROC / "occ_events.parquet")

    hourly = resample_occupancy(occ, freq=pd.Timedelta(hours=1))
    hourly.to_parquet(PROC / "occ_hourly.parquet")
    ten_min = resample_occupancy(occ, freq=pd.Timedelta(minutes=10))
    ten_min.to_parquet(PROC / "occ_10min.parquet")

    # Conservation check on the full dataset (verification).
    iv = event_intervals(occ)
    direct = (iv["headcount"] * (iv["end"] - iv["start"]).dt.total_seconds()).sum() / 3600
    binned = resample_occupancy(occ, min_coverage=0.0)["person_hours"].sum()
    log.info("Person-hours direct %.3f vs binned %.3f (rel diff %.2e)", direct, binned, abs(direct - binned) / direct)

    # Bias of the naive (plain event mean) approach, per hour.
    occ = occ.assign(bin=occ["t"].dt.floor("h"))
    plain = occ.groupby(["space", "bin"])["headcount"].mean().rename("plain_mean")
    cmp = hourly.set_index(["space", "bin"]).join(plain, how="inner")
    cmp = cmp[cmp["headcount"] > 0]
    cmp["ratio"] = cmp["plain_mean"] / cmp["headcount"]
    bias = cmp.groupby(level="space")["ratio"].describe(percentiles=[0.5, 0.9])
    bias.to_csv(RES / "plain_vs_time_weighted.csv")
    log.info("Plain mean / time-weighted mean, per space:\n%s", bias.round(2).to_string())

    summary = hourly.groupby("space").agg(
        hours=("bin", "size"),
        first=("bin", "min"),
        last=("bin", "max"),
        mean_headcount=("headcount", "mean"),
        p95_headcount=("headcount", lambda x: x.quantile(0.95)),
        max_headcount=("headcount", "max"),
        mean_coverage=("coverage", "mean"),
    )
    summary.to_csv(RES / "hourly_summary.csv")
    log.info("Hourly summary:\n%s", summary.round({"mean_headcount": 2, "p95_headcount": 2, "max_headcount": 2, "mean_coverage": 3}).to_string())


if __name__ == "__main__":
    main()
