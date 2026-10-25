"""Step 6: one-time evaluation on the held-out test period.

Usage:
    python -m scripts.run_final_test

The model was selected BEFORE this script was run (results/model_selection.md).
All models are refitted on train + validation and scored once on test.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occupancy.evaluate import bootstrap_mae_diff, summarise
from occupancy.features import LAG_FEATURES, TARGET, time_split
from occupancy.ml import Forecaster, candidates
from occupancy.models import GlobalMean, Persistence, SeasonalProfile
from occupancy.plotting import INK, NEUTRAL, SERIES, style

PROC, RES, FIG = Path("data/processed"), Path("results"), Path("results/figures")
VAL_START, TEST_START = "2025-07-01", "2025-11-01"
# Settings chosen by cross-validation in run_models. The MLP was selected on
# validation (results/model_selection.md); the other families are scored on
# test for reporting only and cannot change that choice.
SELECTED = {"Neural network (MLP)": {"hidden_layer_sizes": (32,), "alpha": 0.01},
            "Decision tree": {"max_depth": 14, "min_samples_leaf": 100},
            "RBF SVR": {"C": 1.0, "epsilon": 0.1, "gamma": "scale"},
            "k-nearest neighbours": {"n_neighbors": 20},
            "Ridge regression": {"alpha": 0.1}}
NAMES = {"cca1ac3e": "G.20", "227f1068": "G.25"}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_final_test")


def main() -> None:
    """Refit on development data, score test once, analyse errors, draw figures."""
    FIG.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(PROC / "features.parquet")
    train, val, test = time_split(data, VAL_START, TEST_START)
    dev = pd.concat([train, val])

    # ---- Range coverage (Week 5.6.9): is test inside the training ranges? ----
    cov = []
    for c in LAG_FEATURES + [TARGET]:
        cov.append({"feature": c, "dev_max": dev[c].max(), "test_max": test[c].max(),
                    "test_frac_above_dev_max": float((test[c] > dev[c].max()).mean()),
                    "dev_mean": dev[c].mean(), "test_mean": test[c].mean()})
    pd.DataFrame(cov).to_csv(RES / "test_range_coverage.csv", index=False)

    # ---- Fit and predict ----
    cands = {c.name: c for c in candidates()}
    preds = {"Global mean": GlobalMean().fit(dev).predict(test),
             "Persistence": Persistence().fit(dev).predict(test),
             "Seasonal profile": SeasonalProfile().fit(dev).predict(test)}
    for name, params in SELECTED.items():
        preds[name] = Forecaster(cands[name].build(params), cands[name].max_train_rows).fit(dev).predict(test)

    y = test[TARGET].to_numpy()
    day = test["bin"].dt.tz_convert("Australia/Melbourne").dt.date.to_numpy()
    busy = ((test["dow"] < 5) & test["hour"].between(8, 17)).to_numpy()
    rows = []
    for name, p in preds.items():
        d, lo, hi = bootstrap_mae_diff(y, p, preds["Persistence"], day, n_boot=2000)
        rows.append({"model": name, **summarise(y, p), "MAE_work_hours": float(np.abs(p - y)[busy].mean()),
                     "dMAE_vs_persistence": d, "ci_lo": lo, "ci_hi": hi})
    res = pd.DataFrame(rows).sort_values("MAE")
    res.to_csv(RES / "final_test.csv", index=False)
    log.info("FINAL TEST:\n%s", res.round(4).to_string(index=False))

    # ---- Error by occupancy band and by space (Week 5.5) ----
    bands = pd.cut(y, [-0.01, 0.05, 1, 3, 6, np.inf], labels=["empty (<0.05)", "0.05-1", "1-3", "3-6", ">6"])
    by_band = pd.DataFrame({"band": bands, **{k: preds[k] - y for k in ["Persistence", "Seasonal profile", "Neural network (MLP)"]}})
    band_tab = by_band.groupby("band", observed=True).agg(["size", lambda e: np.abs(e).mean(), "mean"])
    band_tab.to_csv(RES / "final_test_error_by_band.csv")
    log.info("Error by band:\n%s", band_tab.round(3).to_string())
    sp = pd.DataFrame({"space": test["space"].str[:8].values, "y": y, "mlp": preds["Neural network (MLP)"], "pers": preds["Persistence"]})
    sp_tab = sp.groupby("space").apply(lambda g: pd.Series({
        "hours": len(g), "mean_headcount": g.y.mean(),
        "MAE_MLP": np.abs(g.mlp - g.y).mean(), "MAE_persistence": np.abs(g.pers - g.y).mean()}), include_groups=False)
    sp_tab.to_csv(RES / "final_test_by_space.csv")
    log.info("By space:\n%s", sp_tab.round(3).to_string())

    # ---- Engineering interpretation: HVAC set-back decisions ----
    # A fixed timetable runs HVAC 07:00-19:00 on weekdays. A forecast-driven
    # controller would set back any timetabled hour forecast to have < 0.5
    # people on average. How many truly empty hours would it save, and how
    # often would it wrongly set back an occupied hour?
    timetabled = ((test["dow"] < 5) & test["hour"].between(7, 18)).to_numpy()
    empty = y < 0.5
    hvac = []
    for name in ["Persistence", "Neural network (MLP)"]:
        setback = preds[name] < 0.5
        hvac.append({
            "model": name,
            "timetabled_hours": int(timetabled.sum()),
            "truly_empty_hours": int((timetabled & empty).sum()),
            "empty_hours_correctly_set_back": int((timetabled & empty & setback).sum()),
            "occupied_hours_wrongly_set_back": int((timetabled & ~empty & setback).sum()),
            "occupied_hours": int((timetabled & ~empty).sum()),
        })
    hvac = pd.DataFrame(hvac)
    hvac.to_csv(RES / "final_test_hvac_setback.csv", index=False)
    log.info("HVAC set-back interpretation:\n%s", hvac.to_string(index=False))

    # ---- Figures ----
    style()
    # 1. An example teaching week in G.20
    g20 = test["space"].str.startswith("cca1ac3e").to_numpy()
    local = test["bin"].dt.tz_convert("Australia/Melbourne")
    wk = g20 & (local >= "2026-03-09").to_numpy() & (local < "2026-03-14").to_numpy()
    fig, ax = plt.subplots(figsize=(9, 3.4))
    t = local[wk]
    ax.plot(t, y[wk], color=INK, lw=1.5, label="Measured")
    ax.plot(t, preds["Persistence"][wk], color=SERIES[1], lw=1.5, label="Persistence (baseline)")
    ax.plot(t, preds["Neural network (MLP)"][wk], color=SERIES[0], lw=2, label="MLP (selected)")
    ax.set(ylabel="Headcount (people)", title="G.20, Mon 9 - Fri 13 Mar 2026 (test period): 1-hour-ahead forecasts")
    ax.legend(loc="upper left")
    fig.autofmt_xdate()
    fig.tight_layout(); fig.savefig(FIG / "test_week_g20.png"); plt.close(fig)

    # 2. MAE by occupancy band
    mae_band = by_band.groupby("band", observed=True)[["Persistence", "Seasonal profile", "Neural network (MLP)"]].agg(lambda e: np.abs(e).mean())
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(mae_band)); w = 0.26
    for i, (col, c) in enumerate(zip(mae_band.columns, [SERIES[1], SERIES[2], SERIES[0]])):
        ax.bar(x + (i - 1) * w, mae_band[col], w * 0.92, color=c, label=col)
    ax.set_xticks(x, mae_band.index)
    ax.set(xlabel="Measured headcount band (people)", ylabel="MAE (people)", title="Test error grows with occupancy")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "test_mae_by_band.png"); plt.close(fig)

    # 3. Residuals vs prediction (MLP)
    p = preds["Neural network (MLP)"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(p, p - y, s=8, alpha=0.25, color=SERIES[0], edgecolors="none")
    ax.axhline(0, color=NEUTRAL, lw=1)
    ax.set(xlabel="Predicted headcount (people)", ylabel="Prediction - measured (people)",
           title="MLP residuals on the test period")
    fig.tight_layout(); fig.savefig(FIG / "test_residuals_mlp.png"); plt.close(fig)


if __name__ == "__main__":
    main()
