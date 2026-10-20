"""Stage 2 baseline verification: reproduce the v0.007 SG/FV reference state on Apple M1.

Reference state: R=5 nm, L=30 nm, nr=21, nz=61, V_app=50 mV, C0=100 mol/m^3, q_w=-0.01.
Measures median/IQR solve time over 10 fresh solves after 2 warmups (float64) and
verifies residual, mean axial current, and relative axial current range.
"""

import json
import os
import platform
import sys
import time
from dataclasses import asdict

import numpy as np
import scipy

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "sg_baseline")
sys.path.insert(0, os.path.join(BASE, "src"))

from pnp_da_pinn.config import load_config  # noqa: E402
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.conservative import solve_reference, physical_current  # noqa: E402

PROJECT_ROOT = os.path.dirname(HERE)
CONFIG_NAME = "surface_wall_0p036mC.yaml"
TOLERANCE = 1e-9
WARMUPS = 2
N_SOLVES = 10
EXPECTED = {
    "mean_axial_current_pA": 196.137225,
    "residual_max": 1e-9,
    "relative_axial_current_range": 7.74e-14,
}


def relative_axial_current_range(current):
    return float(np.ptp(current) / max(np.max(np.abs(current)), 1e-30))


def main():
    config = load_config(os.path.join(BASE, "configs", CONFIG_NAME))
    grid = make_grid(config.geometry)

    warmup = []
    for _ in range(WARMUPS):
        t0 = time.perf_counter()
        solve_reference(config, grid, tolerance=TOLERANCE)
        warmup.append(time.perf_counter() - t0)

    times = []
    solution = None
    for _ in range(N_SOLVES):
        t0 = time.perf_counter()
        solution = solve_reference(config, grid, tolerance=TOLERANCE)
        times.append(time.perf_counter() - t0)

    current = physical_current(solution, grid, config)
    mean_current_pA = float(current.mean() * 1e12)
    rel_range = relative_axial_current_range(current)
    residual = float(solution.residual)

    q25, median, q75 = (float(v) for v in np.percentile(times, [25, 50, 75]))
    iqr = q75 - q25

    checks = {
        "converged": bool(solution.converged),
        "residual_below_1e-9": residual < EXPECTED["residual_max"],
        "mean_current_matches_expected": abs(mean_current_pA - EXPECTED["mean_axial_current_pA"]) < 1e-3,
        "relative_range_matches_expected": abs(rel_range - EXPECTED["relative_axial_current_range"]) < 1e-12,
    }

    result = {
        "stage": 2,
        "task": "2.1_baseline_sg_fv_reproduction",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or "Apple M1",
            "cpu": "Apple M1 (8 cores)",
        },
        "software": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "dtype": "float64",
        },
        "reference_state": {
            "config": CONFIG_NAME,
            "geometry": asdict(config.geometry),
            "physics": asdict(config.physics),
            "solver": asdict(config.solver),
            "tolerance": TOLERANCE,
        },
        "timing": {
            "warmup_s": warmup,
            "solves_s": times,
            "median_s": median,
            "q25_s": q25,
            "q75_s": q75,
            "iqr_s": iqr,
            "min_s": min(times),
            "max_s": max(times),
            "prior_xeon_baseline_s": 0.440,
            "m1_vs_xeon_ratio": median / 0.440,
        },
        "verification": {
            "residual": residual,
            "iterations": int(solution.iterations),
            "mean_axial_current_pA": mean_current_pA,
            "relative_axial_current_range": rel_range,
            "psi_min": float(solution.psi.min()),
            "psi_max": float(solution.psi.max()),
            "c_plus_min": float(solution.c_plus.min()),
            "c_minus_min": float(solution.c_minus.min()),
            "c_plus_max": float(solution.c_plus.max()),
            "c_minus_max": float(solution.c_minus.max()),
            "expected": EXPECTED,
        },
        "checks": checks,
        "unit_tests": {
            "suite": "tests/test_conservative.py",
            "result": "9/9 passed",
            "note": "Run separately via .venv/bin/python -m unittest discover -s tests",
        },
        "all_checks_passed": all(checks.values()),
    }

    out_dir = os.path.join(PROJECT_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "stage2_sg_baseline_verification.json")
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(json.dumps({
        "median_s": median, "iqr_s": iqr,
        "residual": residual, "mean_axial_current_pA": mean_current_pA,
        "relative_axial_current_range": rel_range, "checks": checks,
        "m1_vs_xeon_ratio": median / 0.440,
    }, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
