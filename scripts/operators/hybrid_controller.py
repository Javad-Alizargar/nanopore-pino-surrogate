"""Sensitivity-controlled neural-assisted PNP correction utilities.

The controller combines the frozen charge-balanced operator with discrete
Scharfetter--Gummel finite-volume evaluations.  It certifies a fast state with
a Weyl singular-value gate, flags unidentifiable charge directions, and falls
back to a damped Newton--Gummel correction when the surrogate bias exceeds the
observable tolerance.
"""

from __future__ import annotations

import numpy as np

from .constants import EPSILON_0, FARADAY, GAS_CONSTANT, REL_PERMITTIVITY, T_K


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


def fixed_one_step_gummel(psi, c_plus, c_minus, beta, q_w, left_psi, grid,
                          c_left: float = 1.0, c_right: float = 1.0):
    """One discrete SG/FV Gummel correction from an operator state."""
    from pnp_da_pinn.sg_solver import (
        solve_np_scharfetter_gummel, solve_poisson_linear,
        _apply_psi_boundary_values,
    )

    psi = np.asarray(psi, dtype=float).copy()
    cp = solve_np_scharfetter_gummel(psi, grid, ion_sign=1)
    cm = solve_np_scharfetter_gummel(psi, grid, ion_sign=-1)
    cp[:, 0] = c_left; cp[:, -1] = c_right
    cm[:, 0] = c_left; cm[:, -1] = c_right
    psi_new = solve_poisson_linear(cp, cm, np.zeros(grid.shape), grid, beta,
                                    left_psi, 0.0, q_w)
    _apply_psi_boundary_values(psi_new, left_psi, 0.0, q_w, grid.dr)
    return psi_new, cp, cm
