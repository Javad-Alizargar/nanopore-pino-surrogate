"""TASK 3.1 — 4D parametric SG/FV benchmark dataset generation.

Generates a 240-state dataset over mu = (V_app [mV], C0 [mol/m^3], q_w, R [nm])
using the verified conservative SG/FV reference solver (float64, 21x61 grid,
L=30 nm, T=298.15 K), with deterministic splits and strict acceptance checking:

    residual < 1e-9, finite fields, strictly positive c_+-.

Splits (seed = 20261018):
    train       144  (LHS in-distribution box + boundary/corner anchors)
    interp_test  48  (47 LHS in-box + #0 = v0.007 benchmark state)
    ood_test     48  (24 OOD-Charge/Salt + 24 OOD-Geometry/Voltage)

Outputs:
    data/processed/pnp_parametric_240.npz   (compact, < 15 MB)
    results/dataset_manifest.json           (split stats, ranges, residuals,
                                             solve times, SHA-256)
"""

import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.join(HERE, "sg_baseline")
sys.path.insert(0, os.path.join(BASE, "src"))

from pnp_da_pinn.config import (  # noqa: E402
    GeometryConfig,
    PhysicsConfig,
    SolverConfig,
    PinnConfig,
    ExperimentConfig,
)
from pnp_da_pinn.grid import make_grid  # noqa: E402
from pnp_da_pinn.conservative import solve_reference, physical_current  # noqa: E402

PROJECT_ROOT = os.path.dirname(HERE)
OUT_NPZ = os.path.join(PROJECT_ROOT, "data", "processed", "pnp_parametric_240.npz")
OUT_MANIFEST = os.path.join(PROJECT_ROOT, "results", "dataset_manifest.json")

SEED = 20261018
TOLERANCE = 1e-9
MAX_ITER = 300
L_NM = 30.0
T_K = 298.15
NR, NZ = 21, 61
DIFF_CATION = 1.96e-9
DIFF_ANION = 2.03e-9
REL_PERMITTIVITY = 78.5
COMPACT_LIMIT_BYTES = 15 * 1024 * 1024

# In-distribution box
BOX = {
    "V_app_mV": (-50.0, 50.0),
    "C0": (20.0, 200.0),       # log-uniform
    "q_w": (-0.05, 0.0),
    "R_nm": (3.0, 8.0),
}
# Benchmark state (v0.007 fixed-state reference)
BENCHMARK = (50.0, 100.0, -0.01, 5.0)

# Boundary / corner anchors placed at the front of the train split
TRAIN_ANCHORS = [
    (0.0, 100.0, 0.0, 5.0),     # exact symmetric reference (V=0, q_w=0)
    (0.0, 20.0, -0.05, 3.0),    # V=0 corner: min salt, max charge, narrow
    (0.0, 200.0, -0.05, 8.0),   # V=0 corner: max salt, max charge, wide
    (0.0, 100.0, -0.02, 5.0),   # V=0, mid charge
    (50.0, 100.0, 0.0, 5.0),    # q_w=0, +V
    (-50.0, 100.0, 0.0, 5.0),   # q_w=0, -V
    (50.0, 20.0, -0.05, 3.0),   # corner max
    (-50.0, 200.0, -0.05, 8.0), # corner max, opposite V
]

N_TRAIN = 144
N_INTERP = 48
N_OOD_A = 24
N_OOD_B = 24


# --------------------------------------------------------------------------- #
# Sampling
# --------------------------------------------------------------------------- #
def lhs(n, d, rng):
    """Deterministic Latin Hypercube sample in [0,1)^d."""
    out = np.empty((n, d), dtype=float)
    for k in range(d):
        perm = rng.permutation(n)
        out[:, k] = (perm + rng.random(n)) / n
    return out


def _map_box(u):
    v = BOX["V_app_mV"][0] + u[0] * (BOX["V_app_mV"][1] - BOX["V_app_mV"][0])
    lo, hi = np.log10(BOX["C0"][0]), np.log10(BOX["C0"][1])
    c0 = 10.0 ** (lo + u[1] * (hi - lo))
    q = BOX["q_w"][0] + u[2] * (BOX["q_w"][1] - BOX["q_w"][0])
    r = BOX["R_nm"][0] + u[3] * (BOX["R_nm"][1] - BOX["R_nm"][0])
    return np.array([v, c0, q, r])


def sample_train(rng):
    n_lhs = N_TRAIN
    states = np.array([_map_box(u) for u in lhs(n_lhs, 4, rng)])
    for i, anchor in enumerate(TRAIN_ANCHORS):
        states[i] = anchor
    return [tuple(s) for s in states]


def sample_interp(rng):
    states = [BENCHMARK]
    n_lhs = N_INTERP - 1
    for u in lhs(n_lhs, 4, rng):
        states.append(tuple(_map_box(u)))
    return states


def _map_sub(u, ranges):
    v = ranges["V_app_mV"][0] + u[0] * (ranges["V_app_mV"][1] - ranges["V_app_mV"][0])
    lo, hi = np.log10(ranges["C0"][0]), np.log10(ranges["C0"][1])
    c0 = 10.0 ** (lo + u[1] * (hi - lo))
    q = ranges["q_w"][0] + u[2] * (ranges["q_w"][1] - ranges["q_w"][0])
    r = ranges["R_nm"][0] + u[3] * (ranges["R_nm"][1] - ranges["R_nm"][0])
    return np.array([v, c0, q, r])


def sample_ood(rng):
    subset_a = {
        "V_app_mV": (-50.0, 50.0),
        "C0": (10.0, 20.0),
        "q_w": (-0.10, -0.055),
        "R_nm": (2.5, 5.0),
    }
    states = [tuple(_map_sub(u, subset_a)) for u in lhs(N_OOD_A, 4, rng)]

    # Subset B: 12 geometry-extrapolation + 12 voltage-extrapolation states.
    n_geom = N_OOD_B // 2
    n_volt = N_OOD_B - n_geom
    branch_geom = {
        "V_app_mV": (-50.0, 50.0),
        "C0": (15.0, 150.0),
        "q_w": (-0.08, -0.005),
        "R_nm": (2.0, 3.0),
    }
    branch_volt = {
        "V_app_mV": (55.0, 75.0),   # sign applied below
        "C0": (15.0, 150.0),
        "q_w": (-0.08, -0.005),
        "R_nm": (3.0, 8.0),
    }
    for u in lhs(n_geom, 4, rng):
        states.append(tuple(_map_sub(u, branch_geom)))
    for u in lhs(n_volt, 4, rng):
        s = _map_sub(u, branch_volt)
        if rng.random() < 0.5:
            s[0] = -s[0]
        states.append(tuple(s))
    return states


# --------------------------------------------------------------------------- #
# Solver
# --------------------------------------------------------------------------- #
def build_config(V_mV, C0, q_w, R_nm):
    return ExperimentConfig(
        geometry=GeometryConfig(radius_nm=R_nm, length_nm=L_NM, nr=NR, nz=NZ),
        physics=PhysicsConfig(
            temperature_K=T_K,
            bulk_concentration_mol_m3=C0,
            relative_permittivity=REL_PERMITTIVITY,
            diffusion_cation_m2_s=DIFF_CATION,
            diffusion_anion_m2_s=DIFF_ANION,
            applied_voltage_V=V_mV * 1e-3,
            fixed_charge_strength=0.0,
            wall_surface_charge_strength=q_w,
        ),
        solver=SolverConfig(max_gummel_iter=300, tolerance=1e-6, relaxation=0.005),
        pinn=PinnConfig(seed=42, collocation_points=3000, data_points=500, epochs=200),
    )


def _attempt(V_mV, C0, q_w, R_nm, psi0):
    cfg = build_config(V_mV, C0, q_w, R_nm)
    grid = make_grid(cfg.geometry)
    try:
        sol = solve_reference(cfg, grid, tolerance=TOLERANCE, max_iter=MAX_ITER, psi0=psi0)
    except Exception:
        return None, cfg, grid
    if not sol.converged:
        return None, cfg, grid
    return sol, cfg, grid


def solve_state(V_mV, C0, q_w, R_nm):
    """Return (solution, config, grid, method, n_continuation_steps).

    Direct solve first; if it fails, ramp q_w (at V=0) then V_app (at target
    q_w), carrying psi forward as the continuation initial guess.
    """
    sol, cfg, grid = _attempt(V_mV, C0, q_w, R_nm, None)
    if sol is not None:
        return sol, cfg, grid, "direct", 0

    for n_steps in (6, 12):
        psi0 = None
        n_cont = 0
        if abs(q_w) > 1e-12:
            for qs in np.linspace(0.0, q_w, n_steps + 1)[1:]:
                s, _, _ = _attempt(0.0, C0, qs, R_nm, psi0)
                if s is not None:
                    psi0 = s.psi
                    n_cont += 1
        if abs(V_mV) > 1e-12:
            for vs in np.linspace(0.0, V_mV, n_steps + 1)[1:]:
                s, _, _ = _attempt(vs, C0, q_w, R_nm, psi0)
                if s is not None:
                    psi0 = s.psi
                    n_cont += 1
        sol, cfg, grid = _attempt(V_mV, C0, q_w, R_nm, psi0)
        if sol is not None:
            return sol, cfg, grid, f"continuation_{n_steps}", n_cont

    raise RuntimeError(f"state did not converge to {TOLERANCE:.0e}: "
                       f"V={V_mV}, C0={C0}, q_w={q_w}, R={R_nm}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    rng = np.random.default_rng(SEED)

    train = sample_train(rng)
    interp = sample_interp(rng)
    ood = sample_ood(rng)

    plan = (
        [("train", s) for s in train]
        + [("interp_test", s) for s in interp]
        + [("ood_test", s) for s in ood]
    )
    n_total = len(plan)
    assert n_total == 240, n_total

    mu = np.array([s for _, s in plan], dtype=np.float64)
    splits = np.array([tag for tag, _ in plan])

    psi = np.empty((n_total, NR, NZ), dtype=np.float32)
    c_plus = np.empty_like(psi)
    c_minus = np.empty_like(psi)
    rho = np.empty_like(psi)
    sigma_c = np.empty_like(psi)
    i_z = np.empty((n_total, NZ - 1), dtype=np.float64)

    residual = np.empty(n_total)
    iterations = np.empty(n_total, dtype=np.int64)
    runtime_s = np.empty(n_total)
    converged = np.empty(n_total, dtype=bool)
    lambda_d_m = np.empty(n_total)
    lambda_d_over_r = np.empty(n_total)
    beta = np.empty(n_total)
    alpha_r = np.empty(n_total)
    mean_current_pA = np.empty(n_total)
    rel_current_range = np.empty(n_total)
    n_cont = np.empty(n_total, dtype=np.int64)
    method = np.empty(n_total, dtype="<U16")

    grid0 = None
    t_start = time.perf_counter()
    for idx, (tag, (V, C0, q, R)) in enumerate(plan):
        sol, cfg, grid, meth, nc = solve_state(V, C0, q, R)
        if grid0 is None:
            grid0 = grid

        cur = physical_current(sol, grid, cfg) * 1e12  # pA
        cp = sol.c_plus
        cm = sol.c_minus
        psi[idx] = sol.psi.astype(np.float32)
        c_plus[idx] = cp.astype(np.float32)
        c_minus[idx] = cm.astype(np.float32)
        rho[idx] = (cp - cm).astype(np.float32)
        sigma_c[idx] = (cp + cm).astype(np.float32)
        i_z[idx] = cur

        residual[idx] = sol.residual
        iterations[idx] = sol.iterations
        runtime_s[idx] = sol.runtime_s
        converged[idx] = sol.converged
        lam = cfg.scales.debye_length_m
        lambda_d_m[idx] = lam
        lambda_d_over_r[idx] = lam / (R * 1e-9)
        beta[idx] = cfg.scales.poisson_coupling
        alpha_r[idx] = grid.radial_metric
        mean_current_pA[idx] = float(cur.mean())
        rel_current_range[idx] = float(np.ptp(cur) / max(np.max(np.abs(cur)), 1e-30))
        n_cont[idx] = nc
        method[idx] = meth

        if (idx + 1) % 24 == 0 or idx == 0:
            print(f"[{idx + 1:3d}/{n_total}] {tag:11s} V={V:6.1f} C0={C0:6.1f} "
                  f"q={q:+.3f} R={R:4.1f} | res={sol.residual:.2e} "
                  f"I={mean_current_pA[idx]:9.4f} pA ({meth})", flush=True)

    wall_s = time.perf_counter() - t_start

    # Strict acceptance checks
    ok_res = bool(np.all(residual < TOLERANCE))
    ok_pos = bool(c_plus.min() > 0.0 and c_minus.min() > 0.0)
    ok_finite = bool(np.isfinite(psi).all() and np.isfinite(c_plus).all()
                     and np.isfinite(c_minus).all() and np.isfinite(i_z).all())
    ok_conv = bool(np.all(converged))
    assert ok_res and ok_pos and ok_finite and ok_conv, (
        ok_res, ok_pos, ok_finite, ok_conv)

    os.makedirs(os.path.dirname(OUT_NPZ), exist_ok=True)
    np.savez_compressed(
        OUT_NPZ,
        mu=mu,
        psi=psi,
        c_plus=c_plus,
        c_minus=c_minus,
        rho=rho,
        sigma_c=sigma_c,
        i_z=i_z,
        splits=splits,
        r=grid0.r.astype(np.float64),
        z=grid0.z.astype(np.float64),
        residual=residual,
        iterations=iterations,
        runtime_s=runtime_s,
        converged=converged,
        lambda_D_m=lambda_d_m,
        lambda_D_over_R=lambda_d_over_r,
        beta=beta,
        alpha_r=alpha_r,
        mean_current_pA=mean_current_pA,
        rel_current_range=rel_current_range,
        n_continuation_steps=n_cont,
        method=method,
        feature_names=np.array(["V_app_mV", "C0", "q_w", "R_nm"]),
    )
    size_bytes = os.path.getsize(OUT_NPZ)
    digest = sha256_file(OUT_NPZ)

    # Split statistics
    def split_stats(mask):
        sub = mu[mask]
        return {
            "n": int(mask.sum()),
            "V_app_mV": [float(sub[:, 0].min()), float(sub[:, 0].max())],
            "C0": [float(sub[:, 1].min()), float(sub[:, 1].max())],
            "q_w": [float(sub[:, 2].min()), float(sub[:, 2].max())],
            "R_nm": [float(sub[:, 3].min()), float(sub[:, 3].max())],
            "residual_max": float(residual[mask].max()),
            "residual_mean": float(residual[mask].mean()),
            "mean_current_pA_range": [float(mean_current_pA[mask].min()),
                                      float(mean_current_pA[mask].max())],
            "lambda_D_over_R_range": [float(lambda_d_over_r[mask].min()),
                                      float(lambda_d_over_r[mask].max())],
        }

    masks = {tag: (splits == tag) for tag in ("train", "interp_test", "ood_test")}
    oodA = np.zeros(n_total, dtype=bool)
    oodA[N_TRAIN + N_INTERP:N_TRAIN + N_INTERP + N_OOD_A] = True
    oodB = masks["ood_test"] & ~oodA

    bench_idx = N_TRAIN  # first interp state

    manifest = {
        "stage": 3,
        "task": "3.1_parametric_sg_fv_dataset",
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": SEED,
        "hardware": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu": "Apple M1 (8 cores)",
        },
        "solver": {
            "module": "scripts/sg_baseline/src/pnp_da_pinn/conservative.py",
            "conservative_sha256": sha256_file(
                os.path.join(BASE, "src", "pnp_da_pinn", "conservative.py")),
            "sg_solver_sha256": sha256_file(
                os.path.join(BASE, "src", "pnp_da_pinn", "sg_solver.py")),
            "tolerance": TOLERANCE,
            "max_iter": MAX_ITER,
            "dtype": "float64 (fields stored float32)",
        },
        "grid": {"nr": NR, "nz": NZ, "length_nm": L_NM, "temperature_K": T_K},
        "in_distribution_box": {k: list(v) for k, v in BOX.items()},
        "benchmark_state": {
            "index": bench_idx,
            "mu": list(BENCHMARK),
            "mean_axial_current_pA": float(mean_current_pA[bench_idx]),
            "residual": float(residual[bench_idx]),
            "prior_da_pinn_reference": {
                "mean_axial_current_pA": 196.137225,
                "note": "v0.007 fixed-state benchmark; operator prediction "
                        "compared out-of-sample against this DA-PINN result.",
            },
        },
        "dataset": {
            "path": "data/processed/pnp_parametric_240.npz",
            "n_states": int(n_total),
            "sha256": digest,
            "size_bytes": size_bytes,
            "size_mb": round(size_bytes / 1e6, 3),
            "compact_under_15mb": bool(size_bytes < COMPACT_LIMIT_BYTES),
            "arrays": [
                "mu", "psi", "c_plus", "c_minus", "rho", "sigma_c", "i_z",
                "splits", "r", "z", "residual", "iterations", "runtime_s",
                "converged", "lambda_D_m", "lambda_D_over_R", "beta", "alpha_r",
                "mean_current_pA", "rel_current_range", "n_continuation_steps",
                "method", "feature_names",
            ],
        },
        "splits": {
            "train": split_stats(masks["train"]),
            "interp_test": split_stats(masks["interp_test"]),
            "ood_test": split_stats(masks["ood_test"]),
            "ood_charge_salt": split_stats(oodA),
            "ood_geometry_voltage": split_stats(oodB),
        },
        "acceptance": {
            "all_converged": ok_conv,
            "residual_below_1e-9": ok_res,
            "max_residual": float(residual.max()),
            "strictly_positive_concentrations": ok_pos,
            "min_c_plus": float(c_plus.min()),
            "min_c_minus": float(c_minus.min()),
            "finite_fields": ok_finite,
        },
        "timing": {
            "wall_seconds": round(wall_s, 2),
            "mean_solve_s": float(runtime_s.mean()),
            "median_solve_s": float(np.median(runtime_s)),
            "max_solve_s": float(runtime_s.max()),
        },
        "continuation": {
            "n_states_using_continuation": int((n_cont > 0).sum()),
            "methods": {m: int((method == m).sum()) for m in np.unique(method)},
        },
    }

    os.makedirs(os.path.dirname(OUT_MANIFEST), exist_ok=True)
    with open(OUT_MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(json.dumps({
        "n_states": int(n_total),
        "size_mb": manifest["dataset"]["size_mb"],
        "max_residual": float(residual.max()),
        "min_c_plus": float(c_plus.min()),
        "min_c_minus": float(c_minus.min()),
        "wall_seconds": round(wall_s, 2),
        "sha256": digest,
        "continuation_states": manifest["continuation"]["n_states_using_continuation"],
        "benchmark_mean_current_pA": manifest["benchmark_state"]["mean_axial_current_pA"],
    }, indent=2))
    print("wrote", OUT_NPZ)
    print("wrote", OUT_MANIFEST)


if __name__ == "__main__":
    main()
