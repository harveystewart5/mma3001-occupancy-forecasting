"""Tests for occupancy.resample.

The key verification is the worked example from the report: a room empty for
50 minutes then holding ~3 people for 10 minutes has a true hourly mean of
0.5, while the plain mean of its events is far higher.
"""

import numpy as np
import pandas as pd
import pytest

from occupancy.resample import (
    event_intervals,
    resample_occupancy,
    split_by_bin,
    time_weighted_mean,
)


def _occ(rows, space="A"):
    df = pd.DataFrame(rows, columns=["t", "headcount"])
    df["t"] = pd.to_datetime(df["t"], utc=True)
    df.insert(0, "space", space)
    return df


# ---------- time_weighted_mean ----------

def test_time_weighted_mean_worked_example():
    assert time_weighted_mean(np.array([0, 3]), np.array([50, 10])) == pytest.approx(0.5)


def test_time_weighted_mean_rejects_negative_duration():
    with pytest.raises(ValueError):
        time_weighted_mean(np.array([1, 2]), np.array([10, -1]))


def test_time_weighted_mean_rejects_zero_total():
    with pytest.raises(ValueError):
        time_weighted_mean(np.array([1]), np.array([0]))


# ---------- event_intervals ----------

def test_event_intervals_cuts_at_hold_limit():
    occ = _occ([["2024-01-01 09:00", 4], ["2024-01-01 12:00", 0]])
    iv = event_intervals(occ)  # 4 people held 3 h > 1 h limit
    assert len(iv) == 1  # last event has no end and is dropped
    assert iv.loc[0, "end"] == pd.Timestamp("2024-01-01 10:00", tz="UTC")


def test_event_intervals_rejects_unsorted():
    occ = _occ([["2024-01-01 10:00", 1], ["2024-01-01 09:00", 0]])
    with pytest.raises(ValueError):
        event_intervals(occ)


# ---------- split_by_bin ----------

def test_split_by_bin_interval_crossing_boundary():
    iv = pd.DataFrame(
        {
            "space": ["A"],
            "start": pd.to_datetime(["2024-01-01 08:50"], utc=True),
            "end": pd.to_datetime(["2024-01-01 09:10"], utc=True),
            "headcount": [2],
        }
    )
    pieces = split_by_bin(iv)
    assert list(pieces["bin"]) == list(pd.to_datetime(["2024-01-01 08:00", "2024-01-01 09:00"], utc=True))
    assert list(pieces["seconds"]) == [600.0, 600.0]


def test_split_by_bin_interval_spanning_many_bins():
    iv = pd.DataFrame(
        {
            "space": ["A"],
            "start": pd.to_datetime(["2024-01-01 00:30"], utc=True),
            "end": pd.to_datetime(["2024-01-01 03:15"], utc=True),
            "headcount": [0],
        }
    )
    pieces = split_by_bin(iv)
    assert list(pieces["seconds"]) == [1800.0, 3600.0, 3600.0, 900.0]


def test_split_by_bin_rejects_zero_freq():
    iv = pd.DataFrame(columns=["space", "start", "end", "headcount"])
    with pytest.raises(ValueError):
        split_by_bin(iv, freq=pd.Timedelta(0))


# ---------- resample_occupancy ----------

def test_resample_matches_worked_example_and_beats_plain_mean():
    # Empty 9:00-9:50, then 10 one-minute counts of 3,2,3,4,3,2,3,4,3,3 (mean 3).
    counts = [3, 2, 3, 4, 3, 2, 3, 4, 3, 3]
    rows = [["2024-01-01 09:00", 0]]
    rows += [[f"2024-01-01 09:{50 + i}", c] for i, c in enumerate(counts)]
    rows += [["2024-01-01 10:00", 0]]
    occ = _occ(rows)
    out = resample_occupancy(occ)
    hour9 = out[out["bin"] == pd.Timestamp("2024-01-01 09:00", tz="UTC")].iloc[0]
    assert hour9["headcount"] == pytest.approx(0.5)
    assert hour9["person_hours"] == pytest.approx(0.5)
    assert hour9["coverage"] == pytest.approx(1.0)
    plain_mean = occ["headcount"].iloc[:-1].mean()  # events within the hour
    assert plain_mean > 5 * hour9["headcount"]


def test_resample_conserves_person_hours():
    # Splitting intervals at bin edges must not create or destroy person-time.
    rng = np.random.default_rng(0)
    t = pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(np.sort(rng.uniform(0, 72 * 3600, 500)), unit="s")
    occ = pd.DataFrame({"space": "A", "t": t, "headcount": rng.integers(0, 10, 500)})
    iv = event_intervals(occ)
    direct = (iv["headcount"] * (iv["end"] - iv["start"]).dt.total_seconds()).sum() / 3600
    out = resample_occupancy(occ, min_coverage=0.0)
    assert out["person_hours"].sum() == pytest.approx(direct)


def test_resample_drops_low_coverage_bins():
    # 3 people at 09:00, sensor goes silent: trusted only until 10:00.
    occ = _occ([["2024-01-01 09:00", 3], ["2024-01-01 15:00", 0], ["2024-01-01 16:00", 0]])
    out = resample_occupancy(occ, min_coverage=0.5)
    assert pd.Timestamp("2024-01-01 12:00", tz="UTC") not in set(out["bin"])


def test_resample_rejects_bad_coverage():
    with pytest.raises(ValueError):
        resample_occupancy(_occ([["2024-01-01", 0]]), min_coverage=2.0)
