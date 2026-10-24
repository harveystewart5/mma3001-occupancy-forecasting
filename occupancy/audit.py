"""Audit the raw data for unreliable observations before modelling.

Every check here returns evidence (a table or a count) rather than silently
fixing data, so the report can state *what* was wrong and *how much*.

Checks implemented:

* **Frozen sensors** (`frozen_sensor_report`): a live CO2/temperature sensor
  produces a slightly different payload almost every message. A sensor whose
  messages are nearly all identical is resending a stale reading.
* **Stale measurements** (`measurement_lag`): the gap between when a message
  *arrived* and when the reading inside it was *taken*.
* **Occupancy integrity** (`occupancy_audit`): duplicates, impossible
  headcounts, status/headcount contradictions, unsorted timestamps.
* **Outages** (`find_gaps`): silences in the event stream long enough that
  the last reported state can no longer be trusted.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

VALID_STATUSES = ("CurrentlyOccupied", "RecentlyOccupied", "NotOccupied")
"""Status labels the occupancy system is documented to emit."""


def require_columns(df: pd.DataFrame, required: tuple[str, ...] | list[str]) -> None:
    """Raise if ``df`` lacks any of the ``required`` columns.

    Args:
        df: Table to check.
        required: Column names that must be present.

    Raises:
        TypeError: If ``df`` is not a DataFrame.
        ValueError: If any column is missing.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, got {type(df).__name__}")
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def frozen_sensor_report(
    env_raw: pd.DataFrame, min_distinct_fraction: float = 0.01
) -> pd.DataFrame:
    """Flag sensors whose messages are almost all identical.

    The test statistic is ``n_distinct / n_messages``. A working sensor
    sampling a noisy quantity every 10 minutes scores close to 0.5 (each
    reading is typically delivered about twice). A stuck sensor scores close
    to 0.

    Args:
        env_raw: Table with ``sensorid`` and ``jsondata`` columns.
        min_distinct_fraction: Threshold below which a sensor is flagged as
            frozen. Must lie in (0, 1).

    Returns:
        One row per sensor: ``n_messages``, ``n_distinct``,
        ``distinct_fraction`` and boolean ``frozen``.

    Raises:
        ValueError: If the threshold is outside (0, 1) or columns are missing.
    """
    require_columns(env_raw, ["sensorid", "jsondata"])
    if not 0.0 < min_distinct_fraction < 1.0:
        raise ValueError("min_distinct_fraction must be in (0, 1)")
    rep = env_raw.groupby("sensorid")["jsondata"].agg(n_messages="size", n_distinct="nunique")
    rep["distinct_fraction"] = rep["n_distinct"] / rep["n_messages"]
    rep["frozen"] = rep["distinct_fraction"] < min_distinct_fraction
    for sid, row in rep[rep["frozen"]].iterrows():
        logger.info("Sensor %s frozen: %d distinct of %d messages", sid, row.n_distinct, row.n_messages)
    return rep


def measurement_lag(env_long: pd.DataFrame) -> pd.DataFrame:
    """Summarise how old each reading was when it arrived.

    Args:
        env_long: Output of `occupancy.loaders.explode_env`, with ``sensorid``,
            ``arrived`` and ``t`` (measurement time).

    Returns:
        One row per sensor with the median and maximum lag in hours, and the
        first and last *measurement* times seen.
    """
    require_columns(env_long, ["sensorid", "arrived", "t"])
    lag_h = (env_long["arrived"] - env_long["t"]).dt.total_seconds() / 3600.0
    tmp = env_long.assign(lag_h=lag_h)
    return tmp.groupby("sensorid").agg(
        median_lag_h=("lag_h", "median"),
        max_lag_h=("lag_h", "max"),
        first_measured=("t", "min"),
        last_measured=("t", "max"),
    )


def deduplicate_env(env_long: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per genuine measurement.

    A reading is identified by ``(sensorid, t, name)``. Re-sent copies of the
    same reading are dropped; the earliest arrival is kept. This one rule
    removes both routine double-delivery and the endless re-sends from
    frozen sensors.

    Args:
        env_long: Output of `occupancy.loaders.explode_env`.

    Returns:
        De-duplicated table sorted by sensor, variable and time.
    """
    require_columns(env_long, ["sensorid", "t", "name", "arrived", "value"])
    out = (
        env_long.sort_values("arrived")
        .drop_duplicates(subset=["sensorid", "t", "name"], keep="first")
        .sort_values(["sensorid", "name", "t"])
        .reset_index(drop=True)
    )
    logger.info("De-duplicated env readings: %d -> %d rows", len(env_long), len(out))
    return out


def occupancy_audit(occ: pd.DataFrame) -> dict[str, int]:
    """Count integrity problems in the occupancy event stream.

    Args:
        occ: Output of `occupancy.loaders.load_occupancy` (``space``,
            ``status``, ``headcount``, ``t``).

    Returns:
        Dict of counts:

        * ``n_rows``: total events.
        * ``n_exact_duplicates``: rows identical in every column.
        * ``n_negative_headcount``: headcount < 0 (physically impossible).
        * ``n_unknown_status``: status not in `VALID_STATUSES`.
        * ``n_status_contradiction``: occupied with 0 people, or
          not/recently occupied with > 0 people.
        * ``n_unsorted``: events earlier than the previous event in the same
          space (should be 0 after loading).
    """
    require_columns(occ, ["space", "status", "headcount", "t"])
    hc = occ["headcount"]
    occupied = occ["status"] == "CurrentlyOccupied"
    contradiction = (occupied & (hc == 0)) | (~occupied & (hc > 0))
    dt = occ.groupby("space")["t"].diff()
    result = {
        "n_rows": int(len(occ)),
        "n_exact_duplicates": int(occ.duplicated().sum()),
        "n_negative_headcount": int((hc < 0).sum()),
        "n_unknown_status": int((~occ["status"].isin(VALID_STATUSES)).sum()),
        "n_status_contradiction": int(contradiction.sum()),
        "n_unsorted": int((dt < pd.Timedelta(0)).sum()),
    }
    logger.info("Occupancy audit: %s", result)
    return result


def find_gaps(
    occ: pd.DataFrame,
    max_hold_occupied: pd.Timedelta = pd.Timedelta(hours=1),
    max_hold_empty: pd.Timedelta = pd.Timedelta(hours=24),
) -> pd.DataFrame:
    """Find silences in the event stream that indicate sensor outages.

    Events are only sent when something changes, so a silence is normal: a
    room that is empty overnight sends nothing. But a room cannot plausibly
    hold *exactly* the same non-zero headcount for hours, and a sensor silent
    for days is offline. Two limits are therefore used:

    * after an occupied state (headcount > 0): ``max_hold_occupied``
    * after an empty state (headcount == 0): ``max_hold_empty``

    Args:
        occ: Occupancy events sorted by ``space`` and ``t``.
        max_hold_occupied: Longest believable silence after a non-zero count.
        max_hold_empty: Longest believable silence after a zero count.

    Returns:
        One row per gap with ``space``, ``start`` (last event), ``end`` (next
        event), ``headcount`` (last reported), ``duration`` and
        ``trusted_until`` (``start`` plus the hold limit). Time between
        ``trusted_until`` and ``end`` should be treated as missing.

    Raises:
        ValueError: If a hold limit is not positive.
    """
    require_columns(occ, ["space", "headcount", "t"])
    if max_hold_occupied <= pd.Timedelta(0) or max_hold_empty <= pd.Timedelta(0):
        raise ValueError("Hold limits must be positive")
    nxt = occ.groupby("space")["t"].shift(-1)
    duration = nxt - occ["t"]
    limit = pd.Series(
        np.where(occ["headcount"] > 0, max_hold_occupied, max_hold_empty),
        index=occ.index,
    ).astype("timedelta64[ns]")
    mask = duration > limit
    # Keep everything as pandas Series so timezone information survives.
    gaps = pd.DataFrame(
        {
            "space": occ["space"],
            "start": occ["t"],
            "end": nxt,
            "headcount": occ["headcount"],
            "duration": duration,
            "trusted_until": occ["t"] + limit,
        }
    )[mask]
    logger.info("Found %d outage gaps", len(gaps))
    return gaps.reset_index(drop=True)
