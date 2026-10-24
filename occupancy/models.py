"""Forecasters: simple baselines (this module) and ML models (added later).

All models follow the scikit-learn pattern ``fit(df)`` / ``predict(df)`` on
the feature table from `occupancy.features.build_dataset`.

Baselines, from weakest to strongest:

* `GlobalMean`     - one number for every hour (like ``DummyRegressor``).
* `Persistence`    - "next hour looks like the last hour" (``lag_1``).
* `SeasonalProfile`- average headcount for this space, weekday and hour in
  the training data ("what usually happens at 10 am on a Tuesday here").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from occupancy.audit import require_columns
from occupancy.features import TARGET


class GlobalMean:
    """Predict the training-set mean headcount everywhere."""

    def fit(self, df: pd.DataFrame) -> "GlobalMean":
        """Store the mean target.

        Args:
            df: Training table with the target column.

        Returns:
            self
        """
        require_columns(df, [TARGET])
        self.mean_ = float(df[TARGET].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return the stored mean for every row.

        Args:
            df: Any table.

        Returns:
            Array of length ``len(df)``.
        """
        return np.full(len(df), self.mean_)


class Persistence:
    """Predict that the coming hour equals the last observed hour."""

    def fit(self, df: pd.DataFrame) -> "Persistence":
        """No parameters to learn.

        Args:
            df: Ignored.

        Returns:
            self
        """
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return ``lag_1``.

        Args:
            df: Table with a ``lag_1`` column.

        Returns:
            Array of predictions.
        """
        require_columns(df, ["lag_1"])
        return df["lag_1"].to_numpy(dtype=float)


class SeasonalProfile:
    """Mean headcount by (space, day of week, hour) from the training data.

    Falls back to the (day of week, hour) profile pooled over all spaces when
    a space was not in the training data, and to the global mean if even that
    combination is missing.
    """

    def fit(self, df: pd.DataFrame) -> "SeasonalProfile":
        """Compute the per-space and pooled weekly profiles.

        Args:
            df: Training table with ``space``, ``dow``, ``hour`` and target.

        Returns:
            self
        """
        require_columns(df, ["space", "dow", "hour", TARGET])
        self.space_profile_ = df.groupby(["space", "dow", "hour"])[TARGET].mean()
        self.pooled_profile_ = df.groupby(["dow", "hour"])[TARGET].mean()
        self.global_ = float(df[TARGET].mean())
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Look up the profile value for each row.

        Args:
            df: Table with ``space``, ``dow`` and ``hour``.

        Returns:
            Array of predictions.
        """
        require_columns(df, ["space", "dow", "hour"])
        key = pd.MultiIndex.from_frame(df[["space", "dow", "hour"]])
        p = self.space_profile_.reindex(key).to_numpy()
        pooled = self.pooled_profile_.reindex(pd.MultiIndex.from_frame(df[["dow", "hour"]])).to_numpy()
        p = np.where(np.isnan(p), pooled, p)
        return np.where(np.isnan(p), self.global_, p)
