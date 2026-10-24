"""Verification tests for the CO2 mass-balance solver.

Verification = "are we solving the equations right?": compare against the
exact solution, check the order of convergence, and check the stability
limit predicted by theory.
"""

import numpy as np
import pytest

from occupancy.physics import (
    analytic_constant,
    estimate_occupancy,
    fit_decay_rate,
    fit_generation,
    simulate,
    simulate_fast,
)

G, LAM = 0.01, 1.0 / 3600  # 0.01 ppm/s per person; 1 air change per hour


def _err(method, h, T=4 * 3600, n=10, y0=50.0):
    k = int(T / h)
    y = simulate(np.full(k, n), G, LAM, h, y0, method)
    return abs(y[-1] - analytic_constant(T, n, G, LAM, y0))


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_matches_analytic_solution(method):
    assert _err(method, 1.0) < 0.01  # ppm, with 1 s steps


@pytest.mark.parametrize("method", ["forward", "backward"])
def test_first_order_convergence(method):
    # Halving h should halve the error (order 1): slope of log(err) vs log(h) ~ 1.
    hs = np.array([600.0, 300.0, 150.0, 75.0])
    errs = np.array([_err(method, h) for h in hs])
    slope = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert slope == pytest.approx(1.0, abs=0.1)


def test_steady_state_reached():
    y = simulate(np.full(200, 10), G, LAM, 600.0, 0.0, "backward")
    assert y[-1] == pytest.approx(G * 10 / LAM, rel=1e-6)


def test_forward_euler_unstable_beyond_limit():
    # h*lam = 2.5 > 2: |1 - h lam| = 1.5 -> errors grow without bound.
    h = 2.5 / LAM
    y = simulate(np.zeros(40), G, LAM, h, 1.0, "forward")
    assert abs(y[-1]) > 1e6


def test_backward_euler_stable_beyond_limit():
    h = 2.5 / LAM
    y = simulate(np.zeros(40), G, LAM, h, 1.0, "backward")
    assert np.all(np.diff(np.abs(y)) <= 0) and abs(y[-1]) < 1e-6


def test_simulate_rejects_bad_inputs():
    with pytest.raises(ValueError):
        simulate(np.ones(3), G, LAM, h=0.0)
    with pytest.raises(ValueError):
        simulate(np.array([-1.0]), G, LAM, h=1.0)
    with pytest.raises(ValueError):
        simulate(np.ones(3), G, LAM, h=1.0, method="rk4")


def test_fit_decay_rate_recovers_lambda():
    t = np.arange(0, 6 * 3600, 600.0)
    c = 440 + 80 * np.exp(-LAM * t)
    lam, cond = fit_decay_rate(t, c, c_base=440)
    assert lam == pytest.approx(LAM, rel=1e-9)
    assert cond > 1  # design matrix [1, t] is poorly scaled in seconds


def test_fit_decay_rate_rejects_too_few_points():
    with pytest.raises(ValueError):
        fit_decay_rate(np.arange(5.0), np.full(5, 441.0), c_base=440)


def test_fit_generation_manufactured_solution():
    # Manufactured data: simulate with known g, then recover it.
    rng = np.random.default_rng(0)
    n = rng.integers(0, 8, 300).astype(float)
    r = simulate(n, 1.0, LAM, 600.0)[1:]
    assert fit_generation([r], [0.0042 * r]) == pytest.approx(0.0042)


def test_fit_generation_rejects_no_occupancy():
    with pytest.raises(ValueError):
        fit_generation([np.zeros(5)], [np.zeros(5)])


def test_inverse_recovers_occupancy_without_noise():
    t = np.arange(0, 12 * 3600, 60.0)
    n_true = 5.0 * (np.sin(2 * np.pi * t / (6 * 3600)) + 1)  # smooth occupancy
    y = simulate(n_true[:-1], G, LAM, 60.0)
    c = 440 + y
    n_est = estimate_occupancy(c, 60.0, G, LAM, 440)
    assert np.median(np.abs(n_est[5:-5] - n_true[5:-5])) < 0.3


def test_simulate_fast_matches_loop():
    rng = np.random.default_rng(3)
    n = rng.integers(0, 20, 5000).astype(float)
    np.testing.assert_allclose(simulate_fast(n, G, LAM, 600.0, 37.0), simulate(n, G, LAM, 600.0, 37.0), rtol=1e-12)


def test_simulate_fast_float32_close_to_float64():
    n = np.full(10000, 5.0)
    y64 = simulate_fast(n, G, LAM, 600.0)
    y32 = simulate_fast(n, G, LAM, 600.0, dtype=np.float32)
    assert y32.dtype == np.float32
    assert np.max(np.abs(y32 - y64) / np.max(y64)) < 1e-5


def test_simulate_fast_rejects_negative_occupancy():
    with pytest.raises(ValueError):
        simulate_fast(np.array([1.0, -2.0]), G, LAM, 600.0)
