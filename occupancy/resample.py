"""Convert the irregular occupancy event stream into a regular time series.

Each event says "from time t_i the headcount is N_i", and that stays true
until the next event at t_{i+1}. Headcount is therefore a *step function* of
time, and the correct average over a bin is the time-weighted mean

    N_bar = sum_i N_i * dt_i / sum_i dt_i ,

which is the exact integral of the step function divided by the covered
time. The numerator, in hours, is *person-hours*.

A plain mean over events is biased upwards because busy periods generate many
events (the count flickers) while empty periods generate almost none.

Pipeline:

1. `event_intervals` - turn events into [start, end) intervals, cutting each
   one at the outage hold limit (see `occupancy.audit.find_gaps`).
2. `split_by_bin` - split intervals at bin boundaries (default 1 hour).
3. `resample_occupancy` - aggregate pieces per (space, bin) into the
   time-weighted headcount, person-hours and coverage.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from occupancy.audit import require_columns

logger = logging.getLogger(__name__)


def time_weighted_mean(values: np.ndarray, durations: np.ndarray) -> float:
    """Average a piecewise-constant signal, weighting by how long each value held.

    Args:
        values: Value of the signal on each interval.
        durations: Length of each interval (any consistent unit), all >= 0.

    Returns:
        ``sum(values * durations) / sum(durations)``.

    Raises:
        ValueError: If shapes differ, any duration is negative, or the total
            duration is zero.

    Example:
        >>> time_weighted_mean(np.array([0, 3]), np.array([50, 10]))
        0.5
    """
    v = np.asarray(values, dtype=float)
    d = np.asarray(durations, dtype=float)
    if v.shape != d.shape:
        raise ValueError("values and durations must have the same shape")
    if np.any(d < 0):
        raise ValueError("durations must be non-negative")
    total = d.sum()
    if total == 0:
        raise ValueError("total duration is zero; mean undefined")
    return float((v * d).sum() / total)


def event_intervals(
    occ: pd.DataFrame,
    max_hold_occupied: pd.Timedelta = pd.Timedelta(hours=1),
    max_hold_empty: pd.Timedelta = pd.Timedelta(hours=24),
) -> pd.DataFrame:
    """Turn events into the intervals over which each headcount is trusted.

    Interval ``i`` runs from ``t_i`` to ``min(t_{i+1}, t_i + hold)``, where
    ``hold`` depends on whether the headcount is zero. The last event of each
    space has no known end and is dropped.

    Args:
        occ: Events with ``space``, ``headcount`` and ``t``, sorted by space
            then time.
        max_hold_occupied: Hold limit after a non-zero headcount.
        max_hold_empty: Hold limit after a zero headcount.

    Returns:
        DataFrame with ``space``, ``start``, ``end``, ``headcount``; only
        intervals of positive length.

    Raises:
        ValueError: If a hold limit is not positive or events are unsorted.
    """
    require_columns(occ, ["space", "headcount", "t"])
    if max_hold_occupied <= pd.Timedelta(0) or max_hold_empty <= pd.Timedelta(0):
        raise ValueError("Hold limits must be positive")
    if (occ.groupby("space")["t"].diff() < pd.Timedelta(0)).any():
        raise ValueError("Events must be sorted by space and time")
    nxt = occ.groupby("space")["t"].shift(-1)
    hold = pd.Series(
        np.where(occ["headcount"].to_numpy() > 0, max_hold_occupied, max_hold_empty),
        index=occ.index,
    ).astype("timedelta64[ns]")
    cap = occ["t"] + hold
    # Explicit choice rather than np.minimum: np.minimum(NaT, x) returns x,
    # which would extend the final event of each space into the future.
    end = nxt.where(nxt <= cap, cap).where(nxt.notna())
    out = pd.DataFrame(
        {"space": occ["space"], "start": occ["t"], "end": end, "headcount": occ["headcount"]}
    )
    out = out[out["end"].notna() & (out["end"] > out["start"])].reset_index(drop=True)
    return out


def split_by_bin(intervals: pd.DataFrame, freq: pd.Timedelta = pd.Timedelta(hours=1)) -> pd.DataFrame:
    """Split intervals at bin boundaries so each piece lies in one bin.

    Example: an interval 08:50-09:10 with ``freq`` = 1 h becomes two pieces,
    08:50-09:00 in bin 08:00 and 09:00-09:10 in bin 09:00.

    Bins are aligned to UTC multiples of ``freq``; for 1 h bins this matches
    Melbourne clock hours because the UTC offset is a whole number of hours.

    Args:
        intervals: Output of `event_intervals`.
        freq: Bin width. Must be positive.

    Returns:
        DataFrame with ``space``, ``bin`` (bin start), ``headcount`` and
        ``seconds`` (length of the piece).

    Raises:
        ValueError: If ``freq`` is not positive.
    """
    require_columns(intervals, ["space", "start", "end", "headcount"])
    if freq <= pd.Timedelta(0):
        raise ValueError("freq must be positive")
    f = pd.Timedelta(freq).as_unit("ns").value  # nanoseconds
    # pandas may store timestamps in s, ms, us or ns; force ns before taking
    # the integer representation, otherwise units silently mismatch.
    s = intervals["start"].dt.as_unit("ns").astype("int64").to_numpy()
    e = intervals["end"].dt.as_unit("ns").astype("int64").to_numpy()
    first_bin = (s // f) * f
    last_bin = ((e - 1) // f) * f  # bin containing the last instant before e
    n_pieces = (last_bin - first_bin) // f + 1

    # Repeat each interval once per bin it touches, then clip to that bin.
    # Worked example: intervals A (2 bins) and B (1 bin) give n_pieces = [2, 1]
    #   idx    = [0, 0, 1]  -> which interval each piece came from
    #   offset = [0, 1, 0]  -> 0 for an interval's first bin, 1 for its second...
    # This is the vectorised equivalent of a double for-loop over intervals and
    # their bins, which would be far too slow in Python for 9 million events.
    idx = np.repeat(np.arange(len(intervals)), n_pieces)
    offset = np.arange(len(idx)) - np.repeat(np.cumsum(n_pieces) - n_pieces, n_pieces)
    bin_start = first_bin[idx] + offset * f
    piece_start = np.maximum(s[idx], bin_start)
    piece_end = np.minimum(e[idx], bin_start + f)

    return pd.DataFrame(
        {
            "space": intervals["space"].to_numpy()[idx],
            "bin": pd.to_datetime(bin_start, utc=True),
            "headcount": intervals["headcount"].to_numpy()[idx],
            "seconds": (piece_end - piece_start) / 1e9,
        }
    )


def resample_occupancy(
    occ: pd.DataFrame,
    freq: pd.Timedelta = pd.Timedelta(hours=1),
    min_coverage: float = 0.5,
    max_hold_occupied: pd.Timedelta = pd.Timedelta(hours=1),
    max_hold_empty: pd.Timedelta = pd.Timedelta(hours=24),
) -> pd.DataFrame:
    """Time-weighted headcount per space per bin.

    Args:
        occ: Cleaned occupancy events (``space``, ``headcount``, ``t``),
            sorted by space then time.
        freq: Bin width (default 1 hour).
        min_coverage: Minimum fraction of a bin that must be covered by
            trusted data; bins below this are dropped. In [0, 1].
        max_hold_occupied: See `event_intervals`.
        max_hold_empty: See `event_intervals`.

    Returns:
        DataFrame with ``space``, ``bin``, ``headcount`` (time-weighted mean,
        people), ``person_hours`` and ``coverage`` (0-1).

    Raises:
        ValueError: If ``min_coverage`` is outside [0, 1].
    """
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError("min_coverage must be in [0, 1]")
    pieces = split_by_bin(event_intervals(occ, max_hold_occupied, max_hold_empty), freq)
    pieces["person_seconds"] = pieces["headcount"] * pieces["seconds"]
    g = pieces.groupby(["space", "bin"], sort=True)[["person_seconds", "seconds"]].sum()
    out = pd.DataFrame(
        {
            "headcount": g["person_seconds"] / g["seconds"],
            "person_hours": g["person_seconds"] / 3600.0,
            "coverage": g["seconds"] / freq.total_seconds(),
        }
    ).reset_index()
    kept = out[out["coverage"] >= min_coverage].reset_index(drop=True)
    logger.info(
        "Resampled to %d bins; dropped %d bins with coverage < %.2f",
        len(kept), len(out) - len(kept), min_coverage,
    )
    return kept
