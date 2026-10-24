"""Step 7: CO2 mass-balance case study (sensor 869, G.38, Mar-Jun 2024).

Usage:
    python -m scripts.run_co2_case_study

Assumption (stated in the report): no CO2 sensor shares a room with an
occupancy sensor, so occupancy space 941f is used as a *proxy* for G.38
occupants (best statistical match in the audit). Results test whether the
method is workable, not the exact occupancy of G.38.

1. Background C_b and night-time decay rate lam_night (unoccupied, N = 0).
2. Daytime fit of (g, lam_day): grid over lam, closed-form least squares g.
3. Forward validation: simulate CO2 from headcount on held-out May-Jun days.
4. Inverse: estimate headcount from CO2 and compare with the proxy.
5. Verification figure: convergence and stability of forward/backward Euler.
"""

import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occupancy.physics import (analytic_constant, estimate_occupancy, fit_decay_rate,
                               fit_generation, simulate)
from occupancy.plotting import INK, NEUTRAL, SERIES, style

PROC, RES, FIG = Path("data/processed"), Path("results"), Path("results/figures")
H = 600.0  # s, sensor sampling interval
TZ = "Australia/Melbourne"
SPLIT = pd.Timestamp("2024-05-01", tz=TZ)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("co2")


def load() -> pd.DataFrame:
    """Align CO2 (sensor 869) and proxy headcount (941f) on a 10-min grid."""
    env = pd.read_parquet(PROC / "env_readings.parquet")
    co2 = env[(env.sensorid == 6012002000869) & (env.name == "Carbon dioxide")]
    co2 = co2.assign(bin=co2["t"].dt.floor("10min")).groupby("bin")["value"].mean().rename("c")
    occ = pd.read_parquet(PROC / "occ_10min.parquet")
    occ = occ[occ.space.str.startswith("941f")].set_index("bin")["headcount"].rename("n")
    df = pd.concat([co2, occ], axis=1, sort=True)
    df = df[(df.index >= co2.index.min()) & (df.index <= co2.index.max())]
    idx = pd.date_range(df.index.min(), df.index.max(), freq="10min")
    df = df.reindex(idx)
    # Sensor period jitters (9.97-10.02 min), so ~9% of 10-min slots are empty.
    # Fill gaps of <= 2 slots (20 min) with shape-preserving PCHIP (Week 3.4);
    # longer gaps stay missing.
    df["c_raw"] = df["c"]
    df["c"] = df["c"].interpolate(method="pchip", limit=2, limit_area="inside")
    df.index = df.index.tz_convert(TZ)
    return df


def proxy_evidence() -> dict:
    """Evidence for choosing the proxy space and for discarding sensor 326.

    1. For each occupancy space, correlate hourly headcount with hourly CO2
       from sensor 869 (Mar-Jun 2024) after subtracting each series' mean for
       that hour of the week. Removing the shared weekly routine means a high
       correlation reflects the same day-to-day fluctuations, not just the
       fact that both are busier on weekday afternoons.
    2. For each live CO2 sensor, the median weekday 10:00-16:00 CO2 minus the
       weekend median: a room with people in it should be clearly higher on
       weekdays.

    Returns:
        Dict with ``residual_corr_with_869`` and ``overlap`` (hours in common,
        and how many fall in the Mar-Apr calibration window) per space, and
        ``weekday_minus_weekend_ppm`` per sensor.
    """
    env = pd.read_parquet(PROC / "env_readings.parquet")
    co2 = env[env.name == "Carbon dioxide"]
    occ = pd.read_parquet(PROC / "occ_hourly.parquet")

    def hourly(sensor):
        s = co2[co2.sensorid == sensor].set_index("t")["value"].sort_index()
        return s.resample("1h").mean().tz_convert(TZ)

    c869 = hourly(6012002000869)
    c869 = c869[c869.index < pd.Timestamp("2024-06-14", tz=TZ)].dropna()

    def deseasonalise(s):
        how = s.index.dayofweek * 24 + s.index.hour
        return s - s.groupby(how).transform("mean")

    corr, overlap = {}, {}
    for space, g in occ.groupby("space"):
        n = g.set_index("bin")["headcount"].tz_convert(TZ)
        both = pd.concat([c869.rename("c"), n.rename("n")], axis=1, join="inner").dropna()
        if len(both) > 24 * 14:
            corr[space[:8]] = float(deseasonalise(both.c).corr(deseasonalise(both.n)))
            # The proxy must cover the Mar-Apr calibration window, not just May-Jun.
            overlap[space[:8]] = {"hours": len(both), "hours_before_split": int((both.index < SPLIT).sum())}

    contrast = {}
    for sensor in (6012002000869, 6012002000326):
        s = hourly(sensor).dropna()
        if sensor == 6012002000869:
            s = s[s.index < pd.Timestamp("2024-06-14", tz=TZ)]
        day = s[s.index.hour.isin(range(10, 16))]
        contrast[str(sensor)[-3:]] = float(day[day.index.dayofweek < 5].median() - day[day.index.dayofweek >= 5].median())
    return {"residual_corr_with_869": corr, "overlap": overlap, "weekday_minus_weekend_ppm": contrast}


def day_segments(df: pd.DataFrame, start_h: int, end_h: int, weekdays_only: bool = True):
    """Yield complete daily windows [start_h, end_h) with no missing data."""
    for d, g in df.groupby(df.index.date):
        if weekdays_only and pd.Timestamp(d).dayofweek >= 5:
            continue
        w = g[(g.index.hour >= start_h) & (g.index.hour < end_h)]
        if len(w) == (end_h - start_h) * 6 and w[['c', 'n']].notna().all().all():
            yield pd.Timestamp(d, tz=TZ), w


def run(c_base_pct: float = 5.0, save: bool = True, df: pd.DataFrame | None = None) -> dict:
    """Run the case study.

    Args:
        c_base_pct: Percentile of all CO2 readings used as background C_b.
        save: Write JSON results and figures (False for sensitivity runs).
        df: Pre-loaded data (avoids reloading in sensitivity loops).

    Returns:
        Dict of fitted parameters and validation metrics.
    """
    FIG.mkdir(parents=True, exist_ok=True)
    df = load() if df is None else df
    out = {"proxy_selection": proxy_evidence() if save else None,
           "n_10min_bins": int(len(df)), "co2_coverage_raw": float(df.c_raw.notna().mean()),
           "co2_coverage_after_pchip": float(df.c.notna().mean())}

    # ---- 1. Background and night decay (people assumed absent 20:00-06:00) ----
    wk = df[df.index.dayofweek >= 5].c_raw.dropna()
    c_base = float(np.percentile(df.c.dropna(), c_base_pct))
    out.update(c_base_ppm=c_base, weekend_median_ppm=float(wk.median()))
    lams, conds, conds_h = [], [], []
    for d, g in df.groupby(df.index.date):
        if pd.Timestamp(d).dayofweek >= 5:
            continue
        start = pd.Timestamp(d, tz=TZ) + pd.Timedelta(hours=20)
        w = df.loc[start: start + pd.Timedelta(hours=10)].c.dropna()
        if len(w) < 30:
            continue
        try:
            lam, cond = fit_decay_rate((w.index - start).total_seconds().to_numpy(), w.values, c_base)
        except ValueError:
            continue
        _, cond_h = fit_decay_rate((w.index - start).total_seconds().to_numpy() / 3600, w.values, c_base)
        lams.append(lam); conds.append(cond); conds_h.append(cond_h)
    lams = np.array(lams)
    lam_night = float(np.median(lams))
    out.update(n_decay_nights=len(lams), lam_night_per_h=lam_night * 3600,
               lam_night_iqr_per_h=[float(np.percentile(lams, 25) * 3600), float(np.percentile(lams, 75) * 3600)],
               decay_lstsq_cond_median_t_in_s=float(np.median(conds)),
               decay_lstsq_cond_median_t_in_h=float(np.median(conds_h)))

    # ---- 2. Daytime fit (06:00-20:00) on Mar-Apr ----
    segs = list(day_segments(df, 6, 20))
    train = [w for d, w in segs if d < SPLIT]
    test = [w for d, w in segs if d >= SPLIT]

    def responses(ws, lam):
        """Unit-g responses R and decay-corrected targets T for each day (see fit_generation)."""
        R, T = [], []
        for w in ws:
            y = w.c.values - c_base
            k = np.arange(len(w))
            R.append(simulate(w.n.values[:-1], 1.0, lam, H, 0.0)[1:])
            T.append(y[1:] - y[0] * (1 + H * lam) ** (-k[1:]))  # remove free decay of y0 (backward Euler)
        return R, T

    grid = np.linspace(0.05, 3.0, 60) / 3600
    sse = []
    for lam in grid:
        R, T = responses(train, lam)
        g = fit_generation(R, T)
        sse.append(sum(float(np.sum((t - g * r) ** 2)) for r, t in zip(R, T)))
    lam_day = float(grid[int(np.argmin(sse))])
    R, T = responses(train, lam_day)
    g_fit = fit_generation(R, T)
    out.update(n_train_days=len(train), n_test_days=len(test), lam_day_per_h=lam_day * 3600,
               g_ppm_per_h_per_person=g_fit * 3600, h_lam_day=H * lam_day,
               forward_euler_stable_limit_h_s=2 / lam_day)
    # Effective volume, assuming a sedentary adult emits ~0.0052 L/s CO2 (ASHRAE 62.1 basis)
    out["implied_volume_m3"] = 0.0052e-3 * 1e6 / g_fit

    # ---- 3. Forward validation on May-Jun ----
    Rt, Tt = responses(test, lam_day)
    obs = np.concatenate(Tt); pred = g_fit * np.concatenate(Rt)
    ss_res = float(np.sum((obs - pred) ** 2)); ss_tot = float(np.sum((obs - obs.mean()) ** 2))
    out.update(test_rmse_ppm=float(np.sqrt(np.mean((obs - pred) ** 2))),
               test_rmse_ppm_no_occupancy=float(np.sqrt(np.mean(obs ** 2))),
               test_r2=1 - ss_res / ss_tot)

    # ---- 4. Inverse: headcount from CO2 ----
    n_est = pd.Series(estimate_occupancy(df.c.values, H, g_fit, lam_day, c_base), index=df.index)
    hourly = pd.DataFrame({"est": n_est, "proxy": df.n}).resample("1h").mean().dropna()
    work = hourly[(hourly.index.dayofweek < 5) & (hourly.index.hour.isin(range(8, 18)))]
    out.update(inverse_corr_hourly=float(hourly.corr().iloc[0, 1]),
               inverse_corr_work_hours=float(work.corr().iloc[0, 1]),
               inverse_mae_hourly=float(np.mean(np.abs(hourly.est - hourly.proxy))),
               inverse_est_std_10min=float(n_est.std()),
               noise_std_ppm_est=float(np.std(np.diff(wk.values)) / np.sqrt(2)))
    if not save:
        return out
    with open(RES / "co2_case_study.json", "w") as f:
        json.dump(out, f, indent=2)
    log.info("CO2 case study:\n%s", json.dumps(out, indent=1))

    # ---- Figures ----
    style()
    # (a) verification: convergence + stability
    g0, l0, T_end, y0 = 0.01, 1 / 3600, 4 * 3600.0, 50.0
    hs = np.array([1200, 600, 300, 150, 75, 37.5])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for meth, c in [("forward", SERIES[1]), ("backward", SERIES[0])]:
        err = [abs(simulate(np.full(int(T_end / h), 10), g0, l0, h, y0, meth)[-1]
                   - analytic_constant(T_end, 10, g0, l0, y0)) for h in hs]
        axes[0].loglog(hs, err, "o-", color=c, ms=5, label=f"{meth.capitalize()} Euler")
    axes[0].loglog(hs, 0.02 * hs / hs[0] * 10, "--", color=NEUTRAL, lw=1, label="slope 1 (first order)")
    axes[0].set(xlabel="Time step h (s)", ylabel="|error| at t = 4 h (ppm)", title="Convergence vs exact solution")
    axes[0].legend()
    h_big = 2.5 / l0; k = np.arange(12)
    axes[1].plot(k * h_big / 3600, simulate(np.zeros(11), g0, l0, h_big, 1.0, "forward"), "o-", color=SERIES[1], ms=5, label="Forward Euler (h·λ = 2.5)")
    axes[1].plot(k * h_big / 3600, simulate(np.zeros(11), g0, l0, h_big, 1.0, "backward"), "o-", color=SERIES[0], ms=5, label="Backward Euler (h·λ = 2.5)")
    axes[1].plot(np.linspace(0, 11 * h_big / 3600, 200), np.exp(-np.linspace(0, 11 * h_big, 200) * l0), color=INK, lw=1.2, label="Exact")
    axes[1].set(xlabel="Time (h)", ylabel="Excess CO$_2$ (normalised)", title="Stability beyond h·λ = 2", ylim=(-60, 60))
    axes[1].legend(loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "co2_verification.png"); plt.close(fig)

    # (b) validation week: measured vs simulated CO2
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for i, w in enumerate(test[:5]):
        y = w.c.values - c_base; kk = np.arange(len(w))
        sim = c_base + y[0] * (1 + H * lam_day) ** (-kk) + np.r_[0, g_fit * simulate(w.n.values[:-1], 1.0, lam_day, H)[1:]]
        ax.plot(w.index, w.c.values, color=INK, lw=1.3, label="Measured CO$_2$" if i == 0 else None)
        ax.plot(w.index, sim, color=SERIES[0], lw=2, label="Mass-balance model (proxy headcount)" if i == 0 else None)
    ax.set(ylabel="CO$_2$ (ppm)", title="Held-out days, 06:00-20:00: model driven by proxy headcount")
    ax.set_ylim(top=650)
    ax.legend(loc="upper left", ncol=2); fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIG / "co2_validation.png"); plt.close(fig)
    return out


def main() -> None:
    """Run with default settings and save outputs."""
    run()


if __name__ == "__main__":
    main()
