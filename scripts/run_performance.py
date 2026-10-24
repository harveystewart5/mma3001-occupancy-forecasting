"""Step 8: performance analysis (Week 2.1 methods).

Usage:
    python -m scripts.run_performance

1. Machine limits: theoretical peak F_T from lscpu; measured peak (large
   matrix multiply) and measured memory bandwidth (STREAM-style triad), since
   likwid cannot be installed in this environment.
2. Arithmetic intensity (FLOPs / bytes moved) for each kernel, from counting
   operations per element.
3. Timing with ``timeit.repeat`` (mean +/- std over repeats, like %timeit -r).
4. Baseline vs improved: Python-loop backward Euler vs compiled recurrence.
5. Scaling study over problem size.
6. float32 vs float64: speed and accuracy.
7. line_profiler on the baseline solver loop.
8. Roofline plot.
"""

import json
import logging
import subprocess
import timeit
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from occupancy.features import time_split
from occupancy.ml import Forecaster, candidates
from occupancy.physics import simulate, simulate_fast
from occupancy.plotting import INK, INK_2, NEUTRAL, SERIES, style

PROC, RES, FIG = Path("data/processed"), Path("results"), Path("results/figures")
G, LAM, H = 41.4 / 3600, 0.95 / 3600, 600.0  # fitted CO2 parameters (Step 7)
REPEAT = 7

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("perf")


def bench(fn, repeat: int = REPEAT) -> tuple[float, float]:
    """Time ``fn`` like ``%timeit -r repeat``: mean and std of per-call time.

    Args:
        fn: Zero-argument callable.
        repeat: Number of repeats.

    Returns:
        (mean_seconds, std_seconds) per call.
    """
    t1 = timeit.timeit(fn, number=1)
    number = max(1, int(0.2 / max(t1, 1e-9)))  # ~0.2 s per repeat
    times = np.array(timeit.repeat(fn, number=number, repeat=repeat)) / number
    return float(times.mean()), float(times.std())


def main() -> None:
    """Run all measurements and save tables and the roofline figure."""
    FIG.mkdir(parents=True, exist_ok=True)
    out = {}

    # ---- 1. Machine limits ----
    out["lscpu"] = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
    cores, ghz = 2, 2.1
    # AVX-512 with FMA: 8 doubles x 2 (FMA = mul + add) x 2 FMA units per cycle = 32 FLOP/cycle.
    out["F_T_GFLOPs"] = cores * ghz * 32
    out["F_T_assumption"] = "2 vCPU x 2.1 GHz x 32 DP FLOP/cycle (AVX-512, 2 FMA units); 1 FMA unit would halve this"

    n = 3000
    A, B = np.random.rand(n, n), np.random.rand(n, n)
    t, s = bench(lambda: A @ B, repeat=5)
    out["peak_matmul"] = {"n": n, "time_s": t, "std_s": s, "GFLOPs": 2 * n**3 / t / 1e9}

    m = 40_000_000  # 320 MB per array: well past the 260 MB L3 cache
    b, c, d, a = (np.random.rand(m) for _ in range(4))

    def triad():
        """a = b * c + d using in-place ufuncs (no hidden temporaries)."""
        np.multiply(b, c, out=a)  # read b, c; write a  -> 24 B/element
        np.add(a, d, out=a)  # read a, d; write a  -> 24 B/element

    t, s = bench(triad, repeat=5)
    out["triad"] = {"elements": m, "time_s": t, "std_s": s, "bytes_per_elem": 48, "flops_per_elem": 2,
                    "GBs": 48 * m / t / 1e9, "GFLOPs": 2 * m / t / 1e9, "AI": 2 / 48}
    del b, c, d, a
    B_meas = out["triad"]["GBs"]
    out["machine_balance_meas_FLOP_per_B"] = out["peak_matmul"]["GFLOPs"] / B_meas

    # ---- 2-4. Kernels: baseline vs improved CO2 solver ----
    N = 1_000_000
    occ = np.random.default_rng(0).integers(0, 20, N).astype(float)
    kernels = {}
    t, s = bench(lambda: simulate(occ, G, LAM, H, 0.0, "backward"), repeat=REPEAT)
    # per step: h*g, *n, +y, *inv = 4 FLOPs; load n (8 B) + store y (8 B) = 16 B
    kernels["BE loop (Python)"] = {"time_s": t, "std_s": s, "flops": 4 * N, "bytes": 16 * N}
    t, s = bench(lambda: simulate_fast(occ, G, LAM, H), repeat=REPEAT)
    # b*n + a*y = 3 FLOPs; same 16 B
    kernels["BE lfilter (compiled)"] = {"time_s": t, "std_s": s, "flops": 3 * N, "bytes": 16 * N}
    kernels["BE lfilter (compiled)"]["speedup_vs_loop"] = kernels["BE loop (Python)"]["time_s"] / t

    # MLP prediction on the real feature table (selected model)
    data = pd.read_parquet(PROC / "features.parquet")
    train, val, _ = time_split(data, "2025-07-01", "2025-11-01")
    cand = {c.name: c for c in candidates()}["Neural network (MLP)"]
    mlp = Forecaster(cand.build({"hidden_layer_sizes": (32,), "alpha": 0.01})).fit(train)
    Xv = pd.concat([val] * 10, ignore_index=True)  # ~117k rows
    t, s = bench(lambda: mlp.predict(Xv), repeat=REPEAT)
    d_in = 16  # 11 numeric + 5 one-hot space columns
    # per row: 2*d_in*32 (layer 1) + 32 (ReLU) + 2*32 (output) FLOPs; read d_in doubles, write 1
    kernels["MLP predict"] = {"time_s": t, "std_s": s, "flops": len(Xv) * (2 * d_in * 32 + 32 + 64),
                              "bytes": len(Xv) * (d_in + 1) * 8}
    for k in kernels.values():
        k["AI"] = k["flops"] / k["bytes"]
        k["GFLOPs"] = k["flops"] / k["time_s"] / 1e9
        k["bound_by_roofline"] = "memory" if k["AI"] < out["machine_balance_meas_FLOP_per_B"] else "compute"
        k["roofline_limit_GFLOPs"] = min(out["peak_matmul"]["GFLOPs"], B_meas * k["AI"])
        k["fraction_of_roofline"] = k["GFLOPs"] / k["roofline_limit_GFLOPs"]
    out["kernels"] = kernels

    # ---- 5. Scaling study ----
    scaling = []
    for n_steps in [1_000, 10_000, 100_000, 1_000_000]:
        x = occ[:n_steps]
        tl, sl = bench(lambda: simulate(x, G, LAM, H), repeat=REPEAT)
        tf, sf = bench(lambda: simulate_fast(x, G, LAM, H), repeat=REPEAT)
        scaling.append({"N": n_steps, "loop_s": tl, "loop_std": sl, "fast_s": tf, "fast_std": sf, "speedup": tl / tf})
    pd.DataFrame(scaling).to_csv(RES / "perf_scaling.csv", index=False)
    out["scaling"] = scaling

    # ---- 6. float32 vs float64 ----
    y64 = simulate_fast(occ, G, LAM, H, dtype=np.float64)
    y32 = simulate_fast(occ, G, LAM, H, dtype=np.float32)
    t64, s64 = bench(lambda: simulate_fast(occ, G, LAM, H, dtype=np.float64))
    t32, s32 = bench(lambda: simulate_fast(occ, G, LAM, H, dtype=np.float32))
    out["precision"] = {"N": N, "time64_s": t64, "std64": s64, "time32_s": t32, "std32": s32,
                        "max_abs_diff_ppm": float(np.max(np.abs(y32 - y64))),
                        "max_rel_diff": float(np.max(np.abs(y32 - y64)) / np.max(np.abs(y64))),
                        "eps32": float(np.finfo(np.float32).eps), "eps64": float(np.finfo(np.float64).eps),
                        "sensor_noise_ppm": 8.3}

    # ---- 7. line_profiler on the baseline loop ----
    from line_profiler import LineProfiler
    lp = LineProfiler(simulate)
    lp.runcall(simulate, occ[:200_000], G, LAM, H)
    with open(RES / "profile_line_simulate.txt", "w") as f:
        lp.print_stats(stream=f)

    with open(RES / "performance.json", "w") as f:
        json.dump({k: v for k, v in out.items() if k != "lscpu"}, f, indent=2)
    (RES / "lscpu.txt").write_text(out["lscpu"])
    log.info("Performance:\n%s", json.dumps({k: v for k, v in out.items() if k != "lscpu"}, indent=1))

    # ---- 8. Roofline figure ----
    style()
    ai = np.logspace(-2.5, 2.5, 200)
    peak, ft = out["peak_matmul"]["GFLOPs"], out["F_T_GFLOPs"]
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    ax.loglog(ai, np.minimum(ft, B_meas * ai), color=NEUTRAL, lw=1.2, ls="--", label=f"Theoretical F_T = {ft:.0f} GFLOP/s")
    ax.loglog(ai, np.minimum(peak, B_meas * ai), color=INK, lw=2, label=f"Measured roof: {peak:.0f} GFLOP/s, {B_meas:.1f} GB/s")
    pts = [("Triad (bandwidth test)", out["triad"]["AI"], out["triad"]["GFLOPs"], NEUTRAL)]
    pts += [(name, k["AI"], k["GFLOPs"], col) for (name, k), col in
            zip(kernels.items(), [SERIES[1], SERIES[0], SERIES[2]])]
    markers = ["s", "o", "o", "D"]
    offsets = [((-10, 6), "right"), ((8, -12), "left"), ((8, -4), "left"), ((8, -4), "left")]
    for (name, x, y, col), mk, (off, ha) in zip(pts, markers, offsets):
        ax.loglog(x, y, mk, color=col, ms=9, mec="white", mew=1.5)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=off, ha=ha, fontsize=9, color=INK_2)
    ax.set(xlabel="Arithmetic intensity (FLOP / byte)", ylabel="Performance (GFLOP/s)",
           title="Roofline: measured machine limits and project kernels", ylim=(1e-3, 3e2))
    ax.legend(loc="upper left")
    fig.tight_layout(); fig.savefig(FIG / "roofline.png"); plt.close(fig)

    # Scaling figure
    sc = pd.DataFrame(scaling)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.errorbar(sc.N, sc.loop_s, yerr=sc.loop_std, fmt="o-", color=SERIES[1], ms=6, label="Python loop (baseline)")
    ax.errorbar(sc.N, sc.fast_s, yerr=sc.fast_std, fmt="o-", color=SERIES[0], ms=6, label="Compiled recurrence (improved)")
    ax.set(xscale="log", yscale="log", xlabel="Number of time steps N", ylabel="Run time (s), mean ± std of 7",
           title="Backward Euler: both O(N), compiled version faster")
    ax.legend(); fig.tight_layout(); fig.savefig(FIG / "perf_scaling.png"); plt.close(fig)


if __name__ == "__main__":
    main()
