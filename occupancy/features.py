"""Build model inputs that respect the prediction contract.

Prediction contract
-------------------
* **Issued at** the top of hour ``tau`` (e.g. 10:00).
* **Target**: time-weighted mean headcount of one space over [tau, tau + 1 h).
* **Allowed inputs**: the calendar (known in advance) and headcounts of bins
  that ended at or before ``tau``. Nothing from the target hour or later.
* **Units / domain**: headcount in people, real-valued, >= 0.

Every lag feature is built by shifting a *complete* hourly grid, so ``lag_1``
is always exactly one hour earlier. Hours that were missing (outages) stay
missing instead of silently pulling in a value from a different hour.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from occupancy.audit import require_columns

logger = logging.getLogger(__name__)

LOCAL_TZ = "Australia/Melbourne"
"""Clock time that drives human behaviour in the building."""

LAGS = (1, 2, 3, 24, 168)
"""Lags in hours: last three hours, same hour yesterday, same hour last week."""

CALENDAR_FEATURES = ["hour", "dow", "is_weekend", "month"]
LAG_FEATURES = [f"lag_{k}" for k in LAGS] + ["mean_prev_24"]
FEATURES = CALENDAR_FEATURES + LAG_FEATURES
TARGET = "headcount"


def complete_grid(hourly: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """Reindex each space onto an unbroken time grid, inserting NaN for gaps.

    Args:
        hourly: Output of `occupancy.resample.resample_occupancy`.
        freq: Grid spacing.

    Returns:
        DataFrame with one row per (space, bin) between each space's first
        and last bin; missing bins have NaN headcount.
    """
    require_columns(hourly, ["space", "bin", TARGET])
    parts = []
    for space, g in hourly.groupby("space"):
        idx = pd.date_range(g["bin"].min(), g["bin"].max(), freq=freq, name="bin")
        parts.append(g.set_index("bin")[[TARGET]].reindex(idx).assign(space=space).reset_index())
    return pd.concat(parts, ignore_index=True)[["space", "bin", TARGET]]


def add_calendar_features(df: pd.DataFrame, tz: str = LOCAL_TZ) -> pd.DataFrame:
    """Add local hour, day of week, weekend flag and month.

    Args:
        df: Table with a timezone-aware ``bin`` column.
        tz: Local timezone used to derive clock features.

    Returns:
        Copy of ``df`` with ``hour`` (0-23), ``dow`` (0=Mon), ``is_weekend``
        (0/1) and ``month`` (1-12).

    Raises:
        ValueError: If ``bin`` is not timezone-aware.
    """
    require_columns(df, ["bin"])
    if df["bin"].dt.tz is None:
        raise ValueError("bin must be timezone-aware (UTC)")
    local = df["bin"].dt.tz_convert(tz)
    out = df.copy()
    out["hour"] = local.dt.hour
    out["dow"] = local.dt.dayofweek
    out["is_weekend"] = (out["dow"] >= 5).astype(int)
    out["month"] = local.dt.month
    return out


def add_lag_features(df: pd.DataFrame, lags: tuple[int, ...] = LAGS) -> pd.DataFrame:
    """Add past-only headcount features on a complete hourly grid.

    ``lag_k`` is the headcount ``k`` hours before the target hour.
    ``mean_prev_24`` is the mean of the 24 hours before the target hour
    (ignoring missing hours, requiring at least 12 present).

    Args:
        df: Output of `complete_grid` (sorted by space, bin).
        lags: Positive integer lags in hours.

    Returns:
        Copy of ``df`` with lag columns added.

    Raises:
        ValueError: If any lag is < 1 (lag 0 would leak the target).
    """
    require_columns(df, ["space", "bin", TARGET])
    if any(k < 1 for k in lags):
        raise ValueError("Lags must be >= 1 hour; lag 0 is the target itself")
    out = df.sort_values(["space", "bin"]).copy()
    g = out.groupby("space")[TARGET]
    for k in lags:
        out[f"lag_{k}"] = g.shift(k)
    out["mean_prev_24"] = g.transform(lambda s: s.shift(1).rolling(24, min_periods=12).mean())
    return out


def build_dataset(hourly: pd.DataFrame) -> pd.DataFrame:
    """Full feature table: complete grid + calendar + lags, rows with NaN dropped.

    Args:
        hourly: Output of `occupancy.resample.resample_occupancy`.

    Returns:
        DataFrame with ``space``, ``bin``, `FEATURES` and `TARGET`; only rows
        where the target and every feature are present.
    """
    df = add_lag_features(add_calendar_features(complete_grid(hourly)))
    n0 = len(df)
    df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
    logger.info("Feature table: %d rows (%d dropped for missing target/lags)", len(df), n0 - len(df))
    return df


def time_split(
    df: pd.DataFrame, val_start: str, test_start: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological train / validation / test split.

    Args:
        df: Feature table with a ``bin`` column.
        val_start: First timestamp (inclusive) of the validation period.
        test_start: First timestamp (inclusive) of the test period.

    Returns:
        ``(train, val, test)``, each a DataFrame.

    Raises:
        ValueError: If ``val_start`` is not before ``test_start``.
    """
    v, t = pd.Timestamp(val_start, tz="UTC"), pd.Timestamp(test_start, tz="UTC")
    if not v < t:
        raise ValueError("val_start must be earlier than test_start")
    train = df[df["bin"] < v]
    val = df[(df["bin"] >= v) & (df["bin"] < t)]
    test = df[df["bin"] >= t]
    return train, val, test


def one_hot_space(df: pd.DataFrame, spaces: list[str]) -> np.ndarray:
    """One-hot encode the space ID against a fixed list of known spaces.

    An unseen space encodes as all zeros, so models can still be applied to
    it (used for the leave-one-space-out test).

    Args:
        df: Table with a ``space`` column.
        spaces: Ordered list of known space IDs.

    Returns:
        Array of shape (len(df), len(spaces)).
    """
    return (df["space"].to_numpy()[:, None] == np.asarray(spaces)[None, :]).astype(float)
