"""CO2 mass-balance case study: can CO2 tell us how many people are in a room?

Well-mixed (lumped) room model (Week 7.2):

    V dC/dt = G N(t) - Q (C - C_b)

* C   - indoor CO2 concentration [ppm]
* C_b - background (unoccupied, ventilated) concentration [ppm]
* N   - number of people
* G   - CO2 generated per person [ppm m^3 / s]
* Q   - ventilation airflow [m^3 / s];  V - room volume [m^3]

Dividing by V gives the form we solve, with excess concentration y = C - C_b:

    dy/dt = g N(t) - lam * y,   g = G/V [ppm/s per person],  lam = Q/V [1/s]

Only g and lam are identifiable from CO2 data; V, Q and G cannot be separated
without extra information (identifiability).

Time stepping with step h and forcing N_k held constant over the step:

* Forward (explicit) Euler: y_{k+1} = y_k + h (g N_k - lam y_k)
  -> amplification factor (1 - h lam); stable only if h lam < 2.
* Backward (implicit) Euler: y_{k+1} = y_k + h (g N_k - lam y_{k+1})
  -> y_{k+1} = (y_k + h g N_k) / (1 + h lam); amplification 1/(1 + h lam)
  is < 1 for every h > 0 (unconditionally stable). Because the ODE is linear,
  the implicit equation is solved exactly with one division - no Newton
  iteration needed.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter


def _check_params(g: float, lam: float, h: float) -> None:
    """Validate physical parameters.

    Args:
        g: Generation per person (>= 0).
        lam: Air-exchange rate (>= 0).
        h: Time step (> 0).

    Raises:
        ValueError: If any parameter is outside its physical domain.
    """
    if h <= 0:
        raise ValueError("Time step h must be positive")
    if lam < 0 or g < 0:
        raise ValueError("g and lam must be non-negative")


def simulate(
    n: np.ndarray, g: float, lam: float, h: float, y0: float = 0.0, method: str = "backward"
) -> np.ndarray:
    """Integrate dy/dt = g N - lam y with Euler's method.

    Args:
        n: Occupancy N_k for each step (people), length K.
        g: CO2 generation per person per unit volume [ppm/s/person].
        lam: Air-exchange rate Q/V [1/s].
        h: Time step [s].
        y0: Initial excess concentration [ppm].
        method: ``"forward"`` (explicit) or ``"backward"`` (implicit).

    Returns:
        Array of length K + 1: y_0 ... y_K [ppm].

    Raises:
        ValueError: On invalid parameters, unknown method, or negative N.
    """
    _check_params(g, lam, h)
    n = np.asarray(n, dtype=float)
    if np.any(n < 0):
        raise ValueError("Occupancy must be non-negative")
    y = np.empty(n.size + 1)
    y[0] = y0
    if method == "forward":
        a = 1.0 - h * lam
        for k in range(n.size):
            y[k + 1] = a * y[k] + h * g * n[k]
    elif method == "backward":
        inv = 1.0 / (1.0 + h * lam)
        for k in range(n.size):
            y[k + 1] = (y[k] + h * g * n[k]) * inv
    else:
        raise ValueError("method must be 'forward' or 'backward'")
    return y


def analytic_constant(t: np.ndarray, n: float, g: float, lam: float, y0: float) -> np.ndarray:
    """Exact solution for constant occupancy (verification reference).

    y(t) = y_ss + (y0 - y_ss) exp(-lam t),  y_ss = g n / lam.

    Args:
        t: Times [s].
        n: Constant occupancy.
        g: Generation per person [ppm/s/person].
        lam: Air-exchange rate [1/s], > 0.
        y0: Initial excess [ppm].

    Returns:
        y(t) [ppm].

    Raises:
        ValueError: If lam <= 0.
    """
    if lam <= 0:
        raise ValueError("lam must be positive for the analytic solution")
    y_ss = g * n / lam
    return y_ss + (y0 - y_ss) * np.exp(-lam * np.asarray(t, dtype=float))


def fit_decay_rate(t: np.ndarray, c: np.ndarray, c_base: float, min_excess: float = 15.0) -> tuple[float, float]:
    """Fit lam from an unoccupied decay by linear least squares on log excess.

    With N = 0: ln(C - C_b) = ln(y0) - lam t, a straight line in t.

    Args:
        t: Times since start of decay [s].
        c: Measured CO2 [ppm].
        c_base: Background concentration C_b [ppm].
        min_excess: Points with C - C_b below this are discarded, because the
            log of a small, noisy excess is dominated by sensor noise.

    Returns:
        ``(lam, cond)``: fitted rate [1/s] and the 2-norm condition number of
        the least-squares design matrix [1, t] (Week 2.3).

    Raises:
        ValueError: If fewer than 3 usable points remain.
    """
    t = np.asarray(t, dtype=float)
    excess = np.asarray(c, dtype=float) - c_base
    keep = excess > min_excess
    if keep.sum() < 3:
        raise ValueError("Need at least 3 points above min_excess to fit a decay")
    A = np.column_stack([np.ones(keep.sum()), t[keep]])
    coef, *_ = np.linalg.lstsq(A, np.log(excess[keep]), rcond=None)
    return float(-coef[1]), float(np.linalg.cond(A))


def fit_generation(responses: list[np.ndarray], targets: list[np.ndarray]) -> float:
    """Least-squares fit of g, using linearity of the model in g.

    For fixed lam, y = (free decay of y0) + g * R(t), where R is the response
    to the occupancy record with g = 1 and y0 = 0. So g minimises
    sum ||target - g R||^2  ->  g = <R, target> / <R, R>.

    Args:
        responses: Unit-g responses R for each segment.
        targets: Measured excess minus free decay for each segment.

    Returns:
        Fitted g (>= 0).

    Raises:
        ValueError: If the responses are all zero (no occupancy to learn from).
    """
    r = np.concatenate(responses)
    y = np.concatenate(targets)
    denom = float(r @ r)
    if denom == 0:
        raise ValueError("Responses are all zero; g is not identifiable")
    return max(0.0, float(r @ y) / denom)


def estimate_occupancy(c: np.ndarray, h: float, g: float, lam: float, c_base: float) -> np.ndarray:
    """Invert the model: N = (dC/dt + lam (C - C_b)) / g.

    The derivative uses central differences (``np.gradient``, Week 7.1).
    Differentiation amplifies sensor noise: noise sigma in C gives roughly
    sigma / (sqrt(2) h) in dC/dt, so this estimate is noisy.

    Args:
        c: Measured CO2 at uniform spacing h [ppm].
        h: Sample spacing [s].
        g: Fitted generation [ppm/s/person], > 0.
        lam: Fitted air-exchange rate [1/s].
        c_base: Background CO2 [ppm].

    Returns:
        Estimated occupancy per sample (may be negative due to noise).

    Raises:
        ValueError: If g <= 0 or h <= 0.
    """
    if g <= 0:
        raise ValueError("g must be positive to invert the model")
    _check_params(g, lam, h)
    c = np.asarray(c, dtype=float)
    dcdt = np.gradient(c, h)
    return (dcdt + lam * (c - c_base)) / g


def simulate_fast(
    n: np.ndarray, g: float, lam: float, h: float, y0: float = 0.0, dtype=np.float64
) -> np.ndarray:
    """Backward Euler via a compiled linear recurrence (performance version).

    The backward Euler update y_{k+1} = a y_k + b N_k with a = 1/(1 + h lam)
    and b = h g / (1 + h lam) is a first-order linear recurrence (an IIR
    filter). ``scipy.signal.lfilter`` evaluates it in compiled C, removing the
    Python interpreter overhead of `simulate`. Same arithmetic, same answer.

    Args:
        n: Occupancy per step, length K.
        g: Generation per person [ppm/s/person].
        lam: Air-exchange rate [1/s].
        h: Time step [s].
        y0: Initial excess [ppm].
        dtype: ``np.float64`` (default) or ``np.float32`` for the precision study.

    Returns:
        Array of length K + 1 in the requested dtype.

    Raises:
        ValueError: On invalid parameters or negative occupancy.
    """
    _check_params(g, lam, h)
    n = np.asarray(n, dtype=dtype)
    if np.any(n < 0):
        raise ValueError("Occupancy must be non-negative")
    a = dtype(1.0 / (1.0 + h * lam))
    b = dtype(h * g) * a
    # y_{k+1} - a y_k = b N_k ; initial condition carries y0 into the filter state.
    body, _ = lfilter([b], [dtype(1.0), -a], n, zi=np.array([a * dtype(y0)], dtype=dtype))
    return np.concatenate([np.array([y0], dtype=dtype), body.astype(dtype)])
