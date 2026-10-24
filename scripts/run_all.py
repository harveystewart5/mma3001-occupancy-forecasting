"""Run the full pipeline in order.

Usage:
    python -m scripts.run_all            # everything (~10 min on 2 vCPU)
    python -m scripts.run_all --skip-perf

Each stage reads the previous stage's outputs from ``data/processed``.
"""

import argparse
import logging
import time

from scripts import (run_audit, run_baselines, run_co2_case_study, run_final_test, run_models,
                     run_performance, run_resample, run_sensitivity)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
log = logging.getLogger("run_all")


def main() -> None:
    """Run every stage, logging the time taken by each."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--skip-perf", action="store_true", help="skip performance and sensitivity studies")
    args = ap.parse_args()
    stages = [run_audit, run_resample, run_baselines, run_models, run_final_test, run_co2_case_study]
    if not args.skip_perf:
        stages += [run_sensitivity, run_performance]
    for stage in stages:
        t0 = time.perf_counter()
        stage.main()
        log.info("%s finished in %.1f s", stage.__name__, time.perf_counter() - t0)


if __name__ == "__main__":
    main()
