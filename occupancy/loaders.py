"""Read the raw Dataset 2 exports into tidy pandas tables.

The occupancy CSV is ~1.6 GB (9.1 M rows), too large to load comfortably with
``pandas.read_csv``. DuckDB, an in-process SQL engine, streams the file from
disk and only materialises the columns we ask for.

Input format (occupancy CSV, one row per *change* of state):

| column            | type      | units / domain                                  |
|-------------------|-----------|-------------------------------------------------|
| floorspaceid      | str (UUID)| identifies one monitored space                  |
| occupancystatus   | str       | CurrentlyOccupied / RecentlyOccupied / NotOccupied |
| headcount         | int       | people, >= 0                                    |
| collecteddate     | timestamp | time the state began, with UTC offset           |

Input format (environmental CSV, one row per *message received*):

| column     | type      | meaning                                              |
|------------|-----------|------------------------------------------------------|
| sensorid   | int       | sensor identifier                                    |
| createdate | timestamp | time the server received the message ("arrival")     |
| jsondata   | str (JSON)| list of variables, each with ``[{time, value}]``     |

The ``time`` inside ``jsondata`` is the *measurement* time (Unix seconds) and
can differ greatly from ``createdate`` when a sensor is stuck (see
`occupancy.audit`).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import duckdb
import pandas as pd

logger = logging.getLogger(__name__)

OCC_COLUMNS = ("floorspaceid", "occupancystatus", "headcount", "collecteddate")
"""Columns of the occupancy CSV used by this project."""

ENV_COLUMNS = ("sensorid", "createdate", "jsondata")
"""Columns of the environmental CSV used by this project."""


def _check_file(path: str | Path) -> Path:
    """Return ``path`` as a `Path`, raising if it does not exist.

    Args:
        path: File path supplied by the caller.

    Returns:
        The path as a `pathlib.Path`.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Data file not found: {p}")
    return p


def load_occupancy(csv_path: str | Path) -> pd.DataFrame:
    """Load the occupancy event stream.

    Args:
        csv_path: Path to ``5occupancySensor_*.csv``.

    Returns:
        DataFrame with columns ``space`` (str), ``status`` (str),
        ``headcount`` (int64) and ``t`` (datetime64, UTC), sorted by space
        then time.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
        ValueError: If a required column is missing from the file.
    """
    p = _check_file(csv_path)
    con = duckdb.connect()
    header = con.sql(f"select * from read_csv_auto('{p}') limit 0").columns
    missing = set(OCC_COLUMNS) - set(header)
    if missing:
        raise ValueError(f"Occupancy file missing columns: {sorted(missing)}")
    df = con.sql(
        f"""
        select floorspaceid as space,
               occupancystatus as status,
               cast(headcount as bigint) as headcount,
               cast(collecteddate as timestamptz) as t
        from read_csv_auto('{p}')
        order by space, t
        """
    ).df()
    df["t"] = pd.to_datetime(df["t"], utc=True)
    logger.info("Loaded %d occupancy events for %d spaces", len(df), df["space"].nunique())
    return df


def parse_env_payload(jsondata: str) -> list[dict]:
    """Flatten one environmental ``jsondata`` string into readings.

    Args:
        jsondata: JSON text, a list of
            ``{"values": [{"time": int, "value": float}], "variable": {"name": str, "unit": str, ...}}``.

    Returns:
        List of dicts with keys ``name``, ``unit``, ``time`` (Unix seconds,
        int) and ``value`` (float).

    Raises:
        ValueError: If the text is not valid JSON or lacks the expected keys.

    Example:
        >>> parse_env_payload('[{"values":[{"time":1,"value":425}],'
        ...                   '"variable":{"name":"Carbon dioxide","unit":"ppm"}}]')
        [{'name': 'Carbon dioxide', 'unit': 'ppm', 'time': 1, 'value': 425.0}]
    """
    try:
        variables = json.loads(jsondata)
        out = []
        for var in variables:
            meta = var["variable"]
            for v in var["values"]:
                out.append(
                    {
                        "name": str(meta["name"]),
                        "unit": str(meta.get("unit", "")),
                        "time": int(v["time"]),
                        "value": float(v["value"]),
                    }
                )
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise ValueError(f"Malformed jsondata payload: {exc}") from exc
    return out


def load_env_raw(csv_path: str | Path) -> pd.DataFrame:
    """Load the environmental messages without parsing the JSON payload.

    Args:
        csv_path: Path to ``5EnvSensor_*.csv``.

    Returns:
        DataFrame with ``sensorid`` (int64), ``arrived`` (datetime64, UTC)
        and ``jsondata`` (str), one row per received message.

    Raises:
        FileNotFoundError: If ``csv_path`` does not exist.
    """
    p = _check_file(csv_path)
    df = duckdb.sql(
        f"""
        select cast(sensorid as bigint) as sensorid,
               cast(createdate as timestamptz) as arrived,
               jsondata
        from read_csv_auto('{p}')
        """
    ).df()
    df["arrived"] = pd.to_datetime(df["arrived"], utc=True)
    logger.info("Loaded %d environmental messages from %d sensors", len(df), df["sensorid"].nunique())
    return df


def explode_env(env_raw: pd.DataFrame) -> pd.DataFrame:
    """Expand each message into one row per (variable, measurement time).

    Args:
        env_raw: Output of `load_env_raw`.

    Returns:
        Long-format DataFrame with ``sensorid``, ``arrived``, ``t``
        (measurement time, UTC), ``name``, ``unit`` and ``value``. Messages
        with malformed payloads are skipped and counted in the log.
    """
    rows = []
    bad = 0
    # Parse each distinct payload once: frozen sensors resend the same text.
    cache: dict[str, list[dict]] = {}
    for sid, arrived, payload in env_raw[["sensorid", "arrived", "jsondata"]].itertuples(index=False):
        if payload not in cache:
            try:
                cache[payload] = parse_env_payload(payload)
            except ValueError:
                cache[payload] = []
                bad += 1
        for r in cache[payload]:
            rows.append((sid, arrived, r["time"], r["name"], r["unit"], r["value"]))
    if bad:
        logger.warning("Skipped %d malformed payloads", bad)
    out = pd.DataFrame(rows, columns=["sensorid", "arrived", "t", "name", "unit", "value"])
    out["t"] = pd.to_datetime(out["t"], unit="s", utc=True)
    return out
