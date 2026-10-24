"""Tests for occupancy.audit and occupancy.loaders.parse_env_payload.

Each test builds a tiny hand-made table whose correct answer is known, so the
expected values can be checked by eye.
"""

import json

import pandas as pd
import pytest

from occupancy.audit import (
    deduplicate_env,
    find_gaps,
    frozen_sensor_report,
    measurement_lag,
    occupancy_audit,
    require_columns,
)
from occupancy.loaders import load_occupancy, parse_env_payload


def _payload(time: int, co2: float) -> str:
    return json.dumps(
        [{"values": [{"time": time, "value": co2}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}}]
    )


def _occ(rows):
    df = pd.DataFrame(rows, columns=["space", "status", "headcount", "t"])
    df["t"] = pd.to_datetime(df["t"], utc=True)
    return df


# ---------- parse_env_payload ----------

def test_parse_env_payload_extracts_reading():
    out = parse_env_payload(_payload(1712703315, 458))
    assert out == [{"name": "Carbon dioxide", "unit": "ppm", "time": 1712703315, "value": 458.0}]


def test_parse_env_payload_skips_failed_channel_keeps_others():
    # Real pattern from sensor 326: formaldehyde channel reports an error code.
    payload = json.dumps(
        [
            {"values": [{"time": 5, "value": 447}], "variable": {"name": "Carbon dioxide", "unit": "ppm"}},
            {"values": [{"time": 5, "errorCode": 4}], "variable": {"name": "Formaldehyde", "unit": "ug/m3"}},
        ]
    )
    out = parse_env_payload(payload)
    assert [r["name"] for r in out] == ["Carbon dioxide"]


def test_parse_env_payload_rejects_bad_json():
    with pytest.raises(ValueError):
        parse_env_payload("{not json")


def test_parse_env_payload_rejects_missing_keys():
    with pytest.raises(ValueError):
        parse_env_payload('[{"values": [{"time": 1}]}]')


# ---------- frozen_sensor_report ----------

def test_frozen_sensor_detected():
    # Sensor 1 resends one payload 200 times; sensor 2 sends 200 different ones.
    env = pd.DataFrame(
        {
            "sensorid": [1] * 200 + [2] * 200,
            "jsondata": [_payload(0, 425)] * 200 + [_payload(i, 400 + i) for i in range(200)],
        }
    )
    rep = frozen_sensor_report(env)
    assert bool(rep.loc[1, "frozen"]) is True
    assert bool(rep.loc[2, "frozen"]) is False
    assert rep.loc[1, "distinct_fraction"] == pytest.approx(1 / 200)


def test_frozen_sensor_rejects_bad_threshold():
    env = pd.DataFrame({"sensorid": [1], "jsondata": ["x"]})
    with pytest.raises(ValueError):
        frozen_sensor_report(env, min_distinct_fraction=1.5)


# ---------- measurement_lag / deduplicate_env ----------

def _env_long():
    return pd.DataFrame(
        {
            "sensorid": [7, 7, 7],
            "arrived": pd.to_datetime(
                ["2024-04-10 09:01", "2024-04-10 09:12", "2024-04-10 09:13"], utc=True
            ),
            "t": pd.to_datetime(["2024-04-10 09:00", "2024-04-10 09:10", "2024-04-10 09:10"], utc=True),
            "name": ["Carbon dioxide"] * 3,
            "value": [458.0, 450.0, 450.0],
        }
    )


def test_measurement_lag_hours():
    lag = measurement_lag(_env_long())
    # Lags are 1, 2 and 3 minutes; median 2 min, max 3 min.
    assert lag.loc[7, "median_lag_h"] == pytest.approx(2 / 60)
    assert lag.loc[7, "max_lag_h"] == pytest.approx(3 / 60)


def test_deduplicate_env_keeps_first_arrival():
    out = deduplicate_env(_env_long())
    assert len(out) == 2
    assert out.loc[1, "arrived"] == pd.Timestamp("2024-04-10 09:12", tz="UTC")


# ---------- occupancy_audit ----------

def _problem_occ():
    return _occ(
        [
            ["A", "CurrentlyOccupied", 2, "2024-01-01 09:00"],
            ["A", "CurrentlyOccupied", 2, "2024-01-01 09:00"],  # exact duplicate
            ["A", "CurrentlyOccupied", 0, "2024-01-01 09:05"],  # contradiction
            ["A", "NotOccupied", -1, "2024-01-01 09:06"],  # negative
            ["A", "Bogus", 0, "2024-01-01 09:04"],  # unknown status, unsorted
        ]
    )


@pytest.mark.parametrize(
    "key, expected",
    [("n_rows", 5), ("n_exact_duplicates", 1), ("n_negative_headcount", 1),
     ("n_unknown_status", 1), ("n_status_contradiction", 1), ("n_unsorted", 1)],
)
def test_occupancy_audit_counts_each_problem(key, expected):
    # One planted problem of each kind; each count is checked separately.
    assert occupancy_audit(_problem_occ())[key] == expected


def test_occupancy_audit_clean_data_has_no_problems():
    occ = _occ(
        [
            ["A", "CurrentlyOccupied", 3, "2024-01-01 09:00"],
            ["A", "NotOccupied", 0, "2024-01-01 10:00"],
        ]
    )
    rep = occupancy_audit(occ)
    assert sum(v for k, v in rep.items() if k != "n_rows") == 0


def test_occupancy_audit_rejects_missing_column():
    with pytest.raises(ValueError):
        occupancy_audit(pd.DataFrame({"space": ["A"]}))


# ---------- find_gaps ----------

def test_find_gaps_uses_different_limits_for_occupied_and_empty():
    occ = _occ(
        [
            ["A", "CurrentlyOccupied", 4, "2024-01-01 09:00"],  # silent 3 h while occupied -> gap
            ["A", "NotOccupied", 0, "2024-01-01 12:00"],  # silent 12 h while empty -> fine
            ["A", "CurrentlyOccupied", 1, "2024-01-02 00:00"],  # last event, no gap
        ]
    )
    gaps = find_gaps(occ)
    assert len(gaps) == 1
    assert gaps.loc[0, "headcount"] == 4
    assert gaps.loc[0, "trusted_until"] == pd.Timestamp("2024-01-01 10:00", tz="UTC")


def test_find_gaps_rejects_non_positive_limit():
    occ = _occ([["A", "NotOccupied", 0, "2024-01-01"]])
    with pytest.raises(ValueError):
        find_gaps(occ, max_hold_occupied=pd.Timedelta(0))


# ---------- input validation ----------

def test_require_columns_rejects_non_dataframe():
    with pytest.raises(TypeError):
        require_columns([1, 2, 3], ["a"])


def test_load_occupancy_missing_file():
    with pytest.raises(FileNotFoundError):
        load_occupancy("does/not/exist.csv")
