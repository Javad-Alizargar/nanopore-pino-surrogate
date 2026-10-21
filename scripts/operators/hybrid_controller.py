"""Sensitivity-controlled neural-assisted PNP correction utilities.

The controller combines the frozen charge-balanced operator with discrete
Scharfetter--Gummel finite-volume evaluations.  It certifies a fast state with
a Weyl singular-value gate, flags unidentifiable charge directions, and falls
back to a damped Newton--Gummel correction when the surrogate bias exceeds the
observable tolerance.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import brentq

from .constants import (
    DIFF_ANION, DIFF_CATION, EPSILON_0, FARADAY, GAS_CONSTANT, L_NM,
    REL_PERMITTIVITY, T_K,
)


V_T = GAS_CONSTANT * T_K / FARADAY
EPS = REL_PERMITTIVITY * EPSILON_0


def q_from_sigma(sigma_mC_m2: np.ndarray | float, R_nm: np.ndarray | float):
    """Dimensionless wall charge from physical surface charge density."""
    sigma = np.asarray(sigma_mC_m2, dtype=float) * 1.0e-3
    radius = np.asarray(R_nm, dtype=float) * 1.0e-9
    return sigma * radius / EPS / V_T


def sigma_from_q(q_w: np.ndarray | float, R_nm: np.ndarray | float):
    """Physical surface charge density in mC m^-2 from dimensionless q_w."""
    q = np.asarray(q_w, dtype=float)
    radius = np.asarray(R_nm, dtype=float) * 1.0e-9
    return q * EPS * V_T / radius * 1.0e3


def weyl_gate(K: np.ndarray, eps_K: float, observable_error: float,
              noise_floor: float = 1.0) -> dict:
    """Certify a fast state using the smallest singular value of K.

    K is the noise-whitened Jacobian of observables with respect to
    (ln R, ln |q_w|).  The observable_error is expressed as a fraction of the
    noise floor.  A state is accepted when the estimated observable error is
    below 0.1 noise units and the sensitivity perturbation is below 20% of
    sigma_min(K).  If eps_K reaches sigma_min, the weak direction is declared
    unidentifiable rather than fitted.
    """
    K = np.atleast_2d(np.asarray(K, dtype=float))
    sv = np.linalg.svd(K, compute_uv=False)
    smin = float(sv[-1]) if sv.size else 0.0
    smax = float(sv[0]) if sv.size else 0.0
    if smin <= max(eps_K, 1e-14):
        status = "UNIDENTIFIABLE_AT_NOISE_FLOOR"
    elif observable_error < 0.1 * noise_floor and eps_K < 0.2 * smin:
        status = "ACCEPT_FAST"
    else:
        status = "POLISH_SGFV"
    return {
        "status": status,
        "sigma_min": smin,
        "sigma_max": smax,
        "kappa": float((smax / max(smin, 1e-300)) ** 2),
        "eps_K": float(eps_K),
        "weyl_margin": float(smin - eps_K),
        "observable_error_noise_units": float(observable_error),
    }


def exact_sensitivity(fun, theta: np.ndarray, delta: float = 1.0e-3) -> tuple[np.ndarray, np.ndarray]:
    """Central-difference Jacobian of ``fun`` in coordinates (ln R, ln q)."""
    theta = np.asarray(theta, dtype=float)
    base = np.asarray(fun(theta), dtype=float)
    cols = []
    for k in range(theta.size):
        step = np.zeros_like(theta)
        step[k] = delta
        plus = np.asarray(fun(theta * np.exp(step)), dtype=float)
        minus = np.asarray(fun(theta * np.exp(-step)), dtype=float)
        cols.append((plus - minus) / (2.0 * delta))
    return base, np.column_stack(cols)


def newton_step(K: np.ndarray, residual: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Damped least-squares parameter update in (ln R, ln q)."""
    K = np.atleast_2d(np.asarray(K, dtype=float))
    residual = np.asarray(residual, dtype=float)
    w = 1.0 / np.maximum(np.asarray(sigma, dtype=float), 1e-14)
    Kw = K * w[:, None]
    rw = residual * w
    try:
        delta = np.linalg.solve(Kw.T @ Kw + 1e-12 * np.eye(K.shape[1]), Kw.T @ rw)
    except np.linalg.LinAlgError:
        delta = np.linalg.lstsq(Kw, rw, rcond=None)[0]
    return delta


class ReversalSurrogate:
    """Ridge calibration of V_rev from operator observables and (q, R)."""

    def __init__(self, ridge: float = 1e-8):
        self.ridge = ridge
        self.coef: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.scale: np.ndarray | None = None

    @staticmethod
    def features(q: float, R: float, C0: float, selectivity: float,
                 log_ratio: float) -> np.ndarray:
        lr = np.log(max(R, 1e-9))
        lc = np.log(max(C0, 1e-9))
        return np.array([
            1.0, log_ratio, selectivity, selectivity ** 2,
            q, q ** 2, lr, q * lr, lc, selectivity * log_ratio,
        ])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ReversalSurrogate":
        X = np.atleast_2d(np.asarray(X, dtype=float))
        y = np.asarray(y, dtype=float)
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0) + 1e-12
        Z = (X - self.mean) / self.scale
        A = Z.T @ Z + self.ridge * np.eye(Z.shape[1])
        self.coef = np.linalg.solve(A, Z.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef is None:
            raise RuntimeError("ReversalSurrogate is not fitted")
        X = np.atleast_2d(np.asarray(X, dtype=float))
        return ((X - self.mean) / self.scale) @ self.coef


def tms_reversal(C0: float, ratio: float = 5.0, q_w: float = 0.0,
                 R_nm: float = 5.0) -> float:
    """Henderson/TMS concentration-cell reversal potential in mV.

    The voltage convention is psi(r,0)=V_app/V_T and psi(r,1)=0.  For
    D_+ < D_- and c_L > c_R the diffusion term is positive.  The Donnan term
    uses the pore-averaged fixed charge implied by the wall flux,
    X = -2 q_w / beta times the squared pore aspect ratio, evaluated at both
    reservoir concentrations.
    """
    d = (DIFF_ANION - DIFF_CATION) / (DIFF_ANION + DIFF_CATION)
    v_diff = d * V_T * math.log(ratio)
    lam = math.sqrt(EPS * GAS_CONSTANT * T_K / (2.0 * FARADAY ** 2 * C0))
    beta = 0.5 * (L_NM * 1e-9 / lam) ** 2
    aspect = (L_NM / max(R_nm, 1e-12)) ** 2
    x_fixed = -2.0 * q_w / max(beta, 1e-30) * aspect
    c_hi = max(ratio, 1e-12)
    c_lo = 1.0
    v_don = V_T * (math.asinh(x_fixed / (2.0 * c_hi)) -
                   math.asinh(x_fixed / (2.0 * c_lo)))
    return float((v_diff + v_don) * 1e3)


def fixed_one_step_gummel(psi, c_plus, c_minus, beta, q_w, left_psi, grid,
                          c_left: float = 1.0, c_right: float = 1.0):
    """One conservative SG/FV Gummel step using the reference discretization.

    The concentration Dirichlet values are imposed inside the banded ion
    systems, never by solving with unity and overwriting endpoints.
    """
    from pnp_da_pinn.conservative import ConservativePNP

    psi = np.asarray(psi, dtype=float).copy()
    problem = ConservativePNP(grid, beta, left_psi, q_w,
                              right_psi=0.0, c_left=c_left, c_right=c_right)
    return problem.gummel_step(psi)


def conservative_gummel_solve(psi, beta, q_w, left_psi, grid,
                              c_left: float = 1.0, c_right: float = 1.0,
                              tolerance: float = 1e-9, max_iter: int = 100,
                              relaxation: float = 0.5):
    """Damped Gummel/Krylov correction using the conservative operator."""
    from pnp_da_pinn.conservative import ConservativePNP

    problem = ConservativePNP(grid, beta, left_psi, q_w,
                              right_psi=0.0, c_left=c_left, c_right=c_right)
    psi = np.asarray(psi, dtype=float).copy()
    psi, cp, cm = problem.gummel_step(psi)
    history = []
    for it in range(1, max_iter + 1):
        target, cp_t, cm_t = problem.gummel_step(psi)
        step = float(np.max(np.abs(target - psi)))
        history.append(step)
        if step < tolerance:
            cp, cm = cp_t, cm_t
            break
        damp = min(relaxation, 1.0)
        psi = (1.0 - damp) * psi + damp * target
        cp, cm = cp_t, cm_t
    return psi, cp, cm, len(history), history


def axial_current_nonuniformity(cp, cm, psi, grid) -> float:
    from pnp_da_pinn.baseline import PnpSolution
    from pnp_da_pinn.sg_solver import sg_current_profile
    sol = PnpSolution(psi, cp, cm, 0, 0.0, 0.0, True)
    cur = sg_current_profile(sol, grid)
    # The two reservoir half-cells are Dirichlet supply boundaries; evaluate
    # conservation on the pore-interior faces, where continuity is enforced.
    interior = cur[1:-1] if cur.size > 2 else cur
    denom = max(float(np.max(np.abs(interior))), 1e-30)
    return float((np.max(interior) - np.min(interior)) / denom)


def run_corrector_unit_test() -> dict:
    """Reference-preservation test for the conservative corrector."""
    from pnp_da_pinn.config import (
        ExperimentConfig, GeometryConfig, PhysicsConfig, PinnConfig, SolverConfig,
    )
    from pnp_da_pinn.conservative import solve_reference
    from pnp_da_pinn.grid import make_grid

    cases = [
        dict(q=-0.01, R=5.0, C0=100.0, c_left=1.0, c_right=1.0, V=25.0),
        dict(q=-0.50, R=5.0, C0=100.0, c_left=1.0, c_right=1.0, V=25.0),
        dict(q=-0.10, R=5.0, C0=20.0, c_left=2.5, c_right=0.5, V=25.0),
    ]
    rows = []
    for case in cases:
        cfg = ExperimentConfig(
            geometry=GeometryConfig(radius_nm=case["R"], length_nm=L_NM, nr=21, nz=61),
            physics=PhysicsConfig(
                temperature_K=T_K, bulk_concentration_mol_m3=case["C0"],
                relative_permittivity=REL_PERMITTIVITY,
                diffusion_cation_m2_s=DIFF_CATION, diffusion_anion_m2_s=DIFF_ANION,
                applied_voltage_V=float(case["V"]) * 1e-3, fixed_charge_strength=0.0,
                wall_surface_charge_strength=case["q"],
            ),
            solver=SolverConfig(max_gummel_iter=800, tolerance=1e-12, relaxation=0.005),
            pinn=PinnConfig(seed=42, collocation_points=100, data_points=50, epochs=1),
        )
        grid = make_grid(cfg.geometry)
        ref = solve_reference(cfg, grid, tolerance=1e-12, max_iter=800,
                              c_left=case["c_left"], c_right=case["c_right"])
        psi1, cp1, cm1 = fixed_one_step_gummel(
            ref.psi, ref.c_plus, ref.c_minus, cfg.scales.poisson_coupling,
            case["q"], case["V"] * 1e-3 / V_T, grid, c_left=case["c_left"], c_right=case["c_right"],
        )
        dp = float(np.max(np.abs(psi1 - ref.psi)))
        dc = float(max(np.max(np.abs(cp1 - ref.c_plus)),
                       np.max(np.abs(cm1 - ref.c_minus))))
        nonuni = axial_current_nonuniformity(cp1, cm1, psi1, grid)
        rows.append({**case, "dpsi": dp, "dc": dc,
                     "current_nonuniformity": nonuni,
                     "pass": bool(dp < 1e-9 and dc < 1e-9 and nonuni < 1e-11)})
    return {"cases": rows, "pass": all(r["pass"] for r in rows)}


def whitened_jacobian(fun, m: np.ndarray, sigmas: np.ndarray,
                      delta: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Noise-whitened Jacobian in m=(ln R, ln|sigma|).

    Every observable row is divided by its measurement standard deviation, so
    K is dimensionless and mixes no pA/dimensionless units.
    """
    m = np.asarray(m, dtype=float)
    sigmas = np.maximum(np.asarray(sigmas, dtype=float), 1e-14)
    base = np.asarray(fun(m), dtype=float)
    cols = []
    for k in range(m.size):
        step = np.zeros_like(m)
        step[k] = delta
        plus = np.asarray(fun(m * np.exp(step)), dtype=float)
        minus = np.asarray(fun(m * np.exp(-step)), dtype=float)
        cols.append((plus - minus) / (2.0 * delta))
    K = np.column_stack(cols) / sigmas[:, None]
    return base, K


class ReferenceFreeController:
    """Neural warm start plus conservative SG/FV Gummel correction.

    ``controller_solve`` accepts only the physical parameter vector mu; it has
    no access to a reference solution or to the true observation vector.
    """

    def __init__(self, op, tau_psi: float = 1e-4, tau_cons: float = 1e-3):
        self.op = op
        self.tau_psi = float(tau_psi)
        self.tau_cons = float(tau_cons)

    def _setup(self, mu, c_left, c_right):
        from pnp_da_pinn.config import GeometryConfig
        from pnp_da_pinn.grid import make_grid

        V, C0, q, R = (float(x) for x in mu)
        geom = GeometryConfig(radius_nm=R, length_nm=L_NM, nr=21, nz=61)
        grid = make_grid(geom)
        lam = math.sqrt(EPS * GAS_CONSTANT * T_K / (2.0 * FARADAY ** 2 * C0))
        beta = 0.5 * (L_NM * 1e-9 / lam) ** 2
        return V, C0, q, R, grid, beta

    def _config(self, mu, grid):
        from pnp_da_pinn.config import (
            ExperimentConfig, GeometryConfig, PhysicsConfig, PinnConfig, SolverConfig,
        )
        V, C0, q, R = (float(x) for x in mu)
        return ExperimentConfig(
            geometry=GeometryConfig(radius_nm=R, length_nm=L_NM,
                                    nr=grid.shape[0], nz=grid.shape[1]),
            physics=PhysicsConfig(
                temperature_K=T_K, bulk_concentration_mol_m3=C0,
                relative_permittivity=REL_PERMITTIVITY,
                diffusion_cation_m2_s=DIFF_CATION, diffusion_anion_m2_s=DIFF_ANION,
                applied_voltage_V=V * 1e-3, fixed_charge_strength=0.0,
                wall_surface_charge_strength=q,
            ),
            solver=SolverConfig(max_gummel_iter=50, tolerance=1e-6, relaxation=0.005),
            pinn=PinnConfig(seed=42, collocation_points=10, data_points=10, epochs=1),
        )

    def initial_fields(self, mu, c_left=1.0, c_right=1.0):
        V, C0, q, R, grid, beta = self._setup(mu, c_left, c_right)
        fields = self.op.fields(np.array([[V, C0, q, R]]))
        psi = np.asarray(fields["psi"][0], dtype=float).copy()
        cp = np.asarray(fields["c_plus"][0], dtype=float).copy()
        cm = np.asarray(fields["c_minus"][0], dtype=float).copy()
        if abs(c_left - 1.0) > 1e-15 or abs(c_right - 1.0) > 1e-15:
            z = np.linspace(0.0, 1.0, grid.shape[1])[None, :]
            profile = (1.0 - z) * c_left + z * c_right
            cp *= profile
            cm *= profile
            cp[:, 0], cp[:, -1] = c_left, c_right
            cm[:, 0], cm[:, -1] = c_left, c_right
        return psi, cp, cm, grid, beta, V * 1e-3 / V_T

    def controller_solve(self, mu, task: str = "forward",
                         c_left: float = 1.0, c_right: float = 1.0):
        start = time.perf_counter()
        psi0, cp0, cm0, grid, beta, left_psi = self.initial_fields(mu, c_left, c_right)
        q = float(mu[2])
        psi1, cp1, cm1 = fixed_one_step_gummel(
            psi0, cp0, cm0, beta, q, left_psi, grid, c_left, c_right,
        )
        dpsi = float(np.max(np.abs(psi1 - psi0)))
        cons = axial_current_nonuniformity(cp1, cm1, psi1, grid)
        iterations = 1
        converged = False
        if dpsi <= self.tau_psi and cons <= self.tau_cons:
            converged = True
        else:
            from pnp_da_pinn.conservative import solve_reference
            cfg = self._config(mu, grid)
            warm = psi1 if (dpsi < 0.1 and cons < 0.5) else psi0
            sol = solve_reference(cfg, grid, tolerance=1e-6, max_iter=80,
                                  psi0=warm, c_left=c_left, c_right=c_right)
            psi1, cp1, cm1 = sol.psi, sol.c_plus, sol.c_minus
            iterations += int(sol.iterations)
            dpsi = float(np.max(np.abs(psi1 - psi0)))
            cons = axial_current_nonuniformity(cp1, cm1, psi1, grid)
            converged = bool(sol.converged)
        I = axial_current_mean(cp1, cm1, psi1, grid, float(mu[0]), float(mu[1]), float(mu[3]))
        return {
            "task": task, "psi": psi1, "c_plus": cp1, "c_minus": cm1,
            "current_pA": I, "iterations": iterations,
            "delta_psi": dpsi, "conservation_defect": cons,
            "converged": converged,
            "runtime_s": time.perf_counter() - start,
        }


def axial_current_mean(cp, cm, psi, grid, V_mV: float, C0: float, R_nm: float) -> float:
    from pnp_da_pinn.baseline import PnpSolution, current_scale_A
    from pnp_da_pinn.conservative import physical_current
    from pnp_da_pinn.config import (
        ExperimentConfig, GeometryConfig, PhysicsConfig, PinnConfig, SolverConfig,
    )
    cfg = ExperimentConfig(
        geometry=GeometryConfig(radius_nm=R_nm, length_nm=L_NM, nr=grid.shape[0], nz=grid.shape[1]),
        physics=PhysicsConfig(
            temperature_K=T_K, bulk_concentration_mol_m3=C0,
            relative_permittivity=REL_PERMITTIVITY,
            diffusion_cation_m2_s=DIFF_CATION, diffusion_anion_m2_s=DIFF_ANION,
            applied_voltage_V=float(V_mV) * 1e-3, fixed_charge_strength=0.0,
            wall_surface_charge_strength=0.0,
        ),
        solver=SolverConfig(max_gummel_iter=10, tolerance=1e-6, relaxation=0.005),
        pinn=PinnConfig(seed=42, collocation_points=10, data_points=10, epochs=1),
    )
    sol = PnpSolution(psi, cp, cm, 0, 0.0, 0.0, True)
    return float(np.mean(physical_current(sol, grid, cfg)) * 1e12)
