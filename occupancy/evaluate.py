"""Error metrics and uncertainty for comparing forecasters.

* MAE  - mean absolute error, in people. Primary metric: directly
  interpretable ("off by 0.4 people on average") and not dominated by the
  rare very busy hours.
* RMSE - root mean square error, in people. Penalises large misses, which
  matter for ventilation sizing.
* Bias - mean (prediction - truth). Positive = over-predicts (wastes energy),
  negative = under-predicts (stuffy rooms).
* R^2  - fraction of variance explained relative to predicting the mean.

`bootstrap_mae_diff` puts a confidence interval on the difference in MAE
between two models, resampling whole *days* (block bootstrap) because errors
within a day are correlated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _check(y_true, y_pred) -> tuple[np.ndarray, np.ndarray]:
    """Validate and convert metric inputs.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        Both as float arrays.

    Raises:
        ValueError: On shape mismatch, empty input or non-finite values.
    """
    t = np.asarray(y_true, dtype=float)
    p = np.asarray(y_pred, dtype=float)
    if t.shape != p.shape:
        raise ValueError(f"Shape mismatch: {t.shape} vs {p.shape}")
    if t.size == 0:
        raise ValueError("Empty input")
    if not (np.isfinite(t).all() and np.isfinite(p).all()):
        raise ValueError("Inputs contain NaN or inf")
    return t, p


def mae(y_true, y_pred) -> float:
    """Mean absolute error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        mean(|y_pred - y_true|).
    """
    t, p = _check(y_true, y_pred)
    return float(np.mean(np.abs(p - t)))


def rmse(y_true, y_pred) -> float:
    """Root mean square error.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        sqrt(mean((y_pred - y_true)^2)).
    """
    t, p = _check(y_true, y_pred)
    return float(np.sqrt(np.mean((p - t) ** 2)))


def bias(y_true, y_pred) -> float:
    """Mean signed error (prediction minus truth).

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        mean(y_pred - y_true).
    """
    t, p = _check(y_true, y_pred)
    return float(np.mean(p - t))


def r2(y_true, y_pred) -> float:
    """Coefficient of determination.

    Args:
        y_true: Observed values (must not all be equal).
        y_pred: Predicted values.

    Returns:
        1 - SS_res / SS_tot.

    Raises:
        ValueError: If ``y_true`` has zero variance.
    """
    t, p = _check(y_true, y_pred)
    ss_tot = np.sum((t - t.mean()) ** 2)
    if ss_tot == 0:
        raise ValueError("R^2 undefined: y_true has zero variance")
    return float(1.0 - np.sum((t - p) ** 2) / ss_tot)


def summarise(y_true, y_pred) -> dict[str, float]:
    """All four metrics in one dict.

    Args:
        y_true: Observed values.
        y_pred: Predicted values.

    Returns:
        ``{"MAE", "RMSE", "Bias", "R2"}``.
    """
    return {"MAE": mae(y_true, y_pred), "RMSE": rmse(y_true, y_pred),
            "Bias": bias(y_true, y_pred), "R2": r2(y_true, y_pred)}


def bootstrap_mae_diff(
    y_true, pred_a, pred_b, groups, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float, float]:
    """Block-bootstrap confidence interval for MAE(a) - MAE(b).

    Whole groups (e.g. calendar days) are resampled with replacement so the
    within-day correlation of errors is preserved.

    Args:
        y_true: Observed values.
        pred_a: Predictions of model A.
        pred_b: Predictions of model B.
        groups: Group label per row (same length).
        n_boot: Number of bootstrap resamples (> 0).
        seed: Random seed for reproducibility.

    Returns:
        ``(estimate, lower_2.5%, upper_97.5%)``. If the interval excludes 0,
        the difference is unlikely to be chance.

    Raises:
        ValueError: If ``n_boot`` < 1 or lengths differ.
    """
    if n_boot < 1:
        raise ValueError("n_boot must be >= 1")
    t, a = _check(y_true, pred_a)
    _, b = _check(y_true, pred_b)
    g = np.asarray(groups)
    if g.shape != t.shape:
        raise ValueError("groups must have the same length as y_true")
    diff = np.abs(a - t) - np.abs(b - t)
    codes, uniq = pd.factorize(g)
    sums = np.bincount(codes, weights=diff)
    counts = np.bincount(codes)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(n_boot, len(uniq)))
    stats = sums[pick].sum(axis=1) / counts[pick].sum(axis=1)
    return float(diff.mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))
