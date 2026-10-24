"""Tests for features (prediction contract), metrics and baseline models."""

import numpy as np
import pandas as pd
import pytest

from occupancy.evaluate import bias, bootstrap_mae_diff, mae, r2, rmse
from occupancy.features import (
    FEATURES,
    add_calendar_features,
    add_lag_features,
    build_dataset,
    complete_grid,
    time_split,
)
from occupancy.models import GlobalMean, Persistence, SeasonalProfile


def _hourly(n=400, space="A", start="2024-01-01"):
    bins = pd.date_range(start, periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({"space": space, "bin": bins, "headcount": np.arange(n, dtype=float)})


# ---------- features: prediction contract ----------

def test_lag_features_are_strictly_past():
    # headcount = hour index, so lag_k must equal target - k exactly.
    df = add_lag_features(complete_grid(_hourly()))
    row = df.iloc[200]
    for k in (1, 2, 3, 24, 168):
        assert row[f"lag_{k}"] == row["headcount"] - k


def test_no_feature_equals_target_leakage():
    df = build_dataset(_hourly())
    for col in FEATURES:
        assert not np.allclose(df[col], df["headcount"]), f"{col} leaks the target"


def test_lag_zero_rejected():
    with pytest.raises(ValueError):
        add_lag_features(complete_grid(_hourly()), lags=(0, 1))


def test_complete_grid_inserts_missing_hours_as_nan():
    h = _hourly(10).drop(index=[4, 5])
    g = complete_grid(h)
    assert len(g) == 10
    assert g["headcount"].isna().sum() == 2


def test_lag_does_not_jump_over_gap():
    # If hour 5 is missing, lag_1 at hour 6 must be NaN, not hour 4's value.
    h = _hourly(10).drop(index=[5])
    df = add_lag_features(complete_grid(h))
    assert np.isnan(df.iloc[6]["lag_1"])


def test_calendar_features_use_melbourne_time():
    # 2024-01-01 00:00 UTC is 11:00 AEDT on Monday 1 Jan.
    df = add_calendar_features(_hourly(1))
    assert df.loc[0, "hour"] == 11
    assert df.loc[0, "dow"] == 0


def test_calendar_features_reject_naive_time():
    df = pd.DataFrame({"bin": pd.date_range("2024-01-01", periods=2, freq="1h")})
    with pytest.raises(ValueError):
        add_calendar_features(df)


def test_time_split_is_chronological_and_disjoint():
    df = _hourly(1000)
    tr, va, te = time_split(df, "2024-01-20", "2024-02-01")
    assert tr["bin"].max() < va["bin"].min()
    assert va["bin"].max() < te["bin"].min()
    assert len(tr) + len(va) + len(te) == len(df)


def test_time_split_rejects_reversed_dates():
    with pytest.raises(ValueError):
        time_split(_hourly(), "2024-02-01", "2024-01-01")


# ---------- metrics ----------

def test_metrics_known_values():
    y, p = np.array([0.0, 2.0, 4.0]), np.array([1.0, 2.0, 2.0])
    assert mae(y, p) == pytest.approx(1.0)
    assert rmse(y, p) == pytest.approx(np.sqrt(5 / 3))
    assert bias(y, p) == pytest.approx(-1 / 3)
    assert r2(y, y) == pytest.approx(1.0)


def test_metrics_reject_nan():
    with pytest.raises(ValueError):
        mae([1.0, np.nan], [1.0, 1.0])


def test_metrics_reject_shape_mismatch():
    with pytest.raises(ValueError):
        rmse([1.0, 2.0], [1.0])


def test_bootstrap_detects_clearly_better_model():
    rng = np.random.default_rng(1)
    y = rng.normal(size=2000)
    good, bad = y + 0.1 * rng.normal(size=2000), y + 1.0 * rng.normal(size=2000)
    est, lo, hi = bootstrap_mae_diff(y, good, bad, groups=np.arange(2000) // 24, n_boot=300)
    assert hi < 0  # good model's MAE is lower, interval excludes 0
    assert lo <= est <= hi


def test_bootstrap_rejects_zero_resamples():
    with pytest.raises(ValueError):
        bootstrap_mae_diff([1.0], [1.0], [1.0], groups=[0], n_boot=0)


# ---------- baselines ----------

def test_seasonal_profile_recovers_exact_weekly_pattern():
    # Headcount depends only on hour of week -> profile must be perfect.
    bins = pd.date_range("2024-01-01", periods=24 * 7 * 6, freq="1h", tz="UTC")
    df = add_calendar_features(pd.DataFrame({"space": "A", "bin": bins}))
    df["headcount"] = (df["dow"] * 24 + df["hour"]) % 7.0
    model = SeasonalProfile().fit(df.iloc[: 24 * 7 * 4])
    pred = model.predict(df.iloc[24 * 7 * 4 :])
    assert mae(df["headcount"].iloc[24 * 7 * 4 :], pred) == pytest.approx(0.0)


def test_seasonal_profile_falls_back_for_unseen_space():
    train = add_calendar_features(_hourly(24 * 7, space="A"))
    test = add_calendar_features(_hourly(24, space="B"))
    pred = SeasonalProfile().fit(train).predict(test)
    assert np.isfinite(pred).all()


def test_persistence_returns_lag_1():
    df = pd.DataFrame({"lag_1": [1.0, 2.0]})
    assert list(Persistence().fit(df).predict(df)) == [1.0, 2.0]


def test_global_mean_constant():
    df = pd.DataFrame({"headcount": [1.0, 3.0]})
    assert list(GlobalMean().fit(df).predict(df)) == [2.0, 2.0]
