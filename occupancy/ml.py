"""Machine-learning forecasters from the unit (Weeks 3.6 and 5.1-5.4).

Each model is a scikit-learn ``Pipeline`` so every learned preprocessing step
(standardisation, one-hot encoding of the space) is fitted on training rows
only (Week 5.6 section 5.6.8 - no preprocessing leakage).

Inputs (see `occupancy.features`):

* cyclic hour and weekday, ``sin(2 pi h/24)``, ``cos(2 pi h/24)`` etc. - a
  fixed engineering conversion (no learned statistics), so 23:00 and 00:00
  are numerically close;
* weekend flag;
* lagged headcounts (1, 2, 3, 24, 168 h) and mean of the previous 24 h;
* the space ID, one-hot encoded (unknown spaces -> all zeros).

Predictions are clipped at 0 because headcount cannot be negative.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

from occupancy.audit import require_columns
from occupancy.features import LAG_FEATURES, TARGET

logger = logging.getLogger(__name__)

NUMERIC = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"] + LAG_FEATURES
CATEGORICAL = ["space"]


def add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode hour (period 24) and weekday (period 7) on the unit circle.

    Args:
        df: Table with integer ``hour`` and ``dow`` columns.

    Returns:
        Copy with ``hour_sin``, ``hour_cos``, ``dow_sin``, ``dow_cos``.

    Raises:
        ValueError: If hour is outside 0-23 or dow outside 0-6.
    """
    require_columns(df, ["hour", "dow"])
    if not df["hour"].between(0, 23).all() or not df["dow"].between(0, 6).all():
        raise ValueError("hour must be in 0-23 and dow in 0-6")
    out = df.copy()
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["dow_sin"] = np.sin(2 * np.pi * out["dow"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["dow"] / 7)
    return out


def make_pipeline(estimator, scale: bool = True) -> Pipeline:
    """Wrap an estimator with leakage-safe preprocessing.

    Args:
        estimator: Any scikit-learn regressor.
        scale: Standardise numeric features (needed for SVR, kNN, MLP, Ridge;
            harmless but unnecessary for trees).

    Returns:
        Unfitted ``Pipeline``.
    """
    numeric = StandardScaler() if scale else "passthrough"
    pre = ColumnTransformer(
        [
            ("num", numeric, NUMERIC),
            ("space", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", estimator)])


@dataclass
class Candidate:
    """A model family with a small hyperparameter grid.

    Attributes:
        name: Display name.
        build: Function mapping a params dict to an unfitted Pipeline.
        grid: List of params dicts to try.
        max_train_rows: If set, fit on a random subsample of this size
            (SVR training cost grows ~n^2 to n^3 with rows).
    """

    name: str
    build: callable
    grid: list[dict]
    max_train_rows: int | None = None
    cv_results: list[dict] = field(default_factory=list)


def candidates(seed: int = 0) -> list[Candidate]:
    """The model families compared in this project.

    Args:
        seed: Random seed for reproducible models.

    Returns:
        List of `Candidate` objects.
    """
    return [
        Candidate("Ridge regression", lambda p: make_pipeline(Ridge(**p)),
                  [{"alpha": 0.1}, {"alpha": 10.0}, {"alpha": 1000.0}]),
        Candidate("Decision tree", lambda p: make_pipeline(DecisionTreeRegressor(random_state=seed, **p), scale=False),
                  [{"max_depth": d, "min_samples_leaf": m} for d in (6, 10, 14) for m in (20, 100)]),
        Candidate("k-nearest neighbours", lambda p: make_pipeline(KNeighborsRegressor(weights="distance", **p)),
                  [{"n_neighbors": k} for k in (5, 20, 50)]),
        Candidate("RBF SVR", lambda p: make_pipeline(SVR(kernel="rbf", **p)),
                  [{"C": c, "epsilon": 0.1, "gamma": "scale"} for c in (1.0, 10.0)], max_train_rows=5000),
        Candidate("Neural network (MLP)", lambda p: make_pipeline(MLPRegressor(
                      activation="relu", solver="adam", learning_rate_init=0.001, max_iter=300,
                      early_stopping=True, validation_fraction=0.15, n_iter_no_change=10,
                      random_state=seed, **p)),
                  [{"hidden_layer_sizes": h, "alpha": a} for h in ((32,), (64, 32)) for a in (1e-4, 1e-2)]),
    ]


class Forecaster:
    """Adapter giving a Pipeline the project's ``fit(df)`` / ``predict(df)`` API.

    Args:
        pipeline: Unfitted scikit-learn Pipeline from `make_pipeline`.
        max_train_rows: Optional random subsample size for training.
        seed: Random seed for subsampling.
    """

    def __init__(self, pipeline: Pipeline, max_train_rows: int | None = None, seed: int = 0):
        self.pipeline = pipeline
        self.max_train_rows = max_train_rows
        self.seed = seed

    def fit(self, df: pd.DataFrame) -> "Forecaster":
        """Fit on a feature table.

        Args:
            df: Table with ``hour``, ``dow``, lag features, ``space`` and target.

        Returns:
            self
        """
        d = add_cyclic_features(df)
        if self.max_train_rows is not None and len(d) > self.max_train_rows:
            d = d.sample(self.max_train_rows, random_state=self.seed)
        self.pipeline.fit(d[NUMERIC + CATEGORICAL], d[TARGET].to_numpy())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Predict non-negative headcounts.

        Args:
            df: Feature table.

        Returns:
            Array of predictions, clipped at 0.
        """
        d = add_cyclic_features(df)
        return np.clip(self.pipeline.predict(d[NUMERIC + CATEGORICAL]), 0.0, None)


def time_series_cv(
    cand: Candidate, train: pd.DataFrame, n_splits: int = 4, gap_hours: int = 24, seed: int = 0
) -> pd.DataFrame:
    """Score every grid point with forward-chaining time-series CV (Week 5.6.7).

    Rows are sorted by time first because ``TimeSeriesSplit`` uses row order.
    The gap (in rows) is ``gap_hours`` x number of spaces, so roughly one day
    separates each training block from its validation block.

    Args:
        cand: Model family and grid.
        train: Development data (never the test set).
        n_splits: Number of forward folds.
        gap_hours: Gap between train and validation blocks, in hours.
        seed: Random seed.

    Returns:
        One row per params dict: mean and std of fold MAE.
    """
    d = train.sort_values("bin").reset_index(drop=True)
    gap = gap_hours * d["space"].nunique()
    folds = list(TimeSeriesSplit(n_splits=n_splits, gap=gap).split(d))
    rows = []
    for params in cand.grid:
        maes = []
        for tr_idx, va_idx in folds:
            f = Forecaster(cand.build(params), cand.max_train_rows, seed).fit(d.iloc[tr_idx])
            y = d.iloc[va_idx][TARGET].to_numpy()
            maes.append(float(np.mean(np.abs(f.predict(d.iloc[va_idx]) - y))))
        rows.append({"model": cand.name, "params": str(params),
                     "cv_mae_mean": float(np.mean(maes)), "cv_mae_std": float(np.std(maes)),
                     "fold_maes": [round(m, 4) for m in maes]})
        logger.info("%s %s CV MAE %.4f +/- %.4f", cand.name, params, rows[-1]["cv_mae_mean"], rows[-1]["cv_mae_std"])
    return pd.DataFrame(rows)
