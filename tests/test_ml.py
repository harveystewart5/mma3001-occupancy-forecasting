"""Tests for occupancy.ml."""

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from occupancy.features import build_dataset
from occupancy.ml import Candidate, Forecaster, add_cyclic_features, make_pipeline, time_series_cv


def _data(n=24 * 7 * 6, spaces=("A", "B")):
    parts = []
    for i, s in enumerate(spaces):
        bins = pd.date_range("2024-01-01", periods=n, freq="1h", tz="UTC")
        hc = 2.0 + np.sin(2 * np.pi * np.arange(n) / 24) + i  # daily cycle
        parts.append(pd.DataFrame({"space": s, "bin": bins, "headcount": np.clip(hc, 0, None)}))
    return build_dataset(pd.concat(parts, ignore_index=True))


def test_cyclic_features_wrap_midnight():
    df = add_cyclic_features(pd.DataFrame({"hour": [0, 23], "dow": [0, 6]}))
    # 23:00 and 00:00 are close on the circle (distance ~0.26), not 23 apart.
    d = np.hypot(df["hour_sin"].diff().iloc[1], df["hour_cos"].diff().iloc[1])
    assert d == pytest.approx(2 * np.sin(np.pi / 24))


def test_cyclic_features_reject_bad_hour():
    with pytest.raises(ValueError):
        add_cyclic_features(pd.DataFrame({"hour": [24], "dow": [0]}))


def test_linear_model_learns_lag_relationship():
    # Daily sine: lag_24 equals the target exactly, so a linear model is ~perfect.
    df = _data()
    f = Forecaster(make_pipeline(LinearRegression())).fit(df)
    assert np.mean(np.abs(f.predict(df) - df["headcount"])) < 1e-6


def test_predictions_non_negative():
    df = _data()
    df["headcount"] = 0.0
    df.loc[df.index[:10], "headcount"] = -5.0  # force a model that could go negative
    f = Forecaster(make_pipeline(LinearRegression())).fit(df)
    assert (f.predict(df) >= 0).all()


def test_unseen_space_does_not_crash():
    df = _data()
    f = Forecaster(make_pipeline(LinearRegression())).fit(df[df["space"] == "A"])
    assert np.isfinite(f.predict(df[df["space"] == "B"])).all()


def test_subsampling_limits_training_rows():
    df = _data()
    f = Forecaster(make_pipeline(LinearRegression()), max_train_rows=50).fit(df)
    assert f.pipeline.named_steps["pre"].n_features_in_ > 0
    assert f.pipeline.named_steps["model"].n_features_in_ > 0


def test_time_series_cv_returns_one_row_per_grid_point():
    cand = Candidate("lin", lambda p: make_pipeline(LinearRegression(**p)), [{"fit_intercept": True}, {"fit_intercept": False}])
    res = time_series_cv(cand, _data(), n_splits=3)
    assert len(res) == 2
    assert (res["cv_mae_mean"] >= 0).all()
