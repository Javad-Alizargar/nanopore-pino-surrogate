from dataclasses import dataclass
import time

import numpy as np

from .config import ExperimentConfig
from .grid import AxisymmetricGrid
from .scales import FARADAY, voltage_to_psi


@dataclass
class PnpSolution:
    psi: np.ndarray
    c_plus: np.ndarray
    c_minus: np.ndarray
    iterations: int
    residual: float
    runtime_s: float
    converged: bool


def _axisymmetric_laplace_update(field: np.ndarray, grid: AxisymmetricGrid, i: int, j: int) -> float:
    r = max(grid.r[i], 1e-8)
    radial = (field[i + 1, j] + field[i - 1, j]) / grid.dr**2
    radial += (field[i + 1, j] - field[i - 1, j]) / (2.0 * r * grid.dr)
    axial = (field[i, j + 1] + field[i, j - 1]) / grid.dz**2
    denom = 2.0 / grid.dr**2 + 2.0 / grid.dz**2
    return (radial + axial) / denom


def _solve_poisson_sor(
    psi: np.ndarray,
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    beta: float,
    grid: AxisymmetricGrid,
    left_psi: float,
    right_psi: float,
    fixed_charge: np.ndarray,
    max_iter: int = 120,
    tol: float = 1e-6,
    omega: float = 1.6,
) -> np.ndarray:
    out = psi.copy()
    for _ in range(max_iter):
        old = out.copy()
        charge = c_plus - c_minus + fixed_charge
        for i in range(1, len(grid.r) - 1):
            for j in range(1, len(grid.z) - 1):
                source = beta * charge[i, j]
                update = _axisymmetric_laplace_update(out, grid, i, j)
                update += source / (2.0 / grid.dr**2 + 2.0 / grid.dz**2)
                out[i, j] = (1.0 - omega) * out[i, j] + omega * update
        _apply_psi_boundaries(out, left_psi, right_psi)
        if np.max(np.abs(out - old)) < tol:
            break
    return out


def _apply_psi_boundaries(psi: np.ndarray, left_psi: float, right_psi: float) -> None:
    psi[:, 0] = left_psi
    psi[:, -1] = right_psi
    psi[0, :] = psi[1, :]
    psi[-1, :] = psi[-2, :]


def _apply_concentration_boundaries(c: np.ndarray) -> None:
    c[:, 0] = 1.0
    c[:, -1] = 1.0
    c[0, :] = c[1, :]
    c[-1, :] = c[-2, :]
    np.maximum(c, 1e-8, out=c)


def _boltzmann_relaxation(psi: np.ndarray, ion_sign: float) -> np.ndarray:
    shifted = psi - 0.5 * (psi[:, :1] + psi[:, -1:])
    clipped = np.clip(-ion_sign * shifted, -20.0, 20.0)
    return np.exp(clipped)


def solve_pnp_gummel(config: ExperimentConfig, grid: AxisymmetricGrid) -> PnpSolution:
    """Small, stable dimensionless Gummel baseline for early experiments.

    This is intentionally conservative. It is adequate for smoke tests and
    first DA-PINN data generation, but production paper results should upgrade
    this to finite-volume/Scharfetter-Gummel flux discretization.
    """
    start = time.perf_counter()
    scales = config.scales
    beta = scales.poisson_coupling
    left_psi = voltage_to_psi(config.physics.applied_voltage_V, scales)
    right_psi = 0.0

    psi = left_psi * (1.0 - grid.zz)
    c_plus = np.ones(grid.shape)
    c_minus = np.ones(grid.shape)
    fixed_charge = np.full(grid.shape, config.physics.fixed_charge_strength, dtype=float)

    residual = np.inf
    converged = False
    for iteration in range(1, config.solver.max_gummel_iter + 1):
        old_psi = psi.copy()
        old_cp = c_plus.copy()
        old_cm = c_minus.copy()

        target_cp = _boltzmann_relaxation(psi, ion_sign=1.0)
        target_cm = _boltzmann_relaxation(psi, ion_sign=-1.0)
        relax = config.solver.relaxation
        c_plus = (1.0 - relax) * c_plus + relax * target_cp
        c_minus = (1.0 - relax) * c_minus + relax * target_cm
        _apply_concentration_boundaries(c_plus)
        _apply_concentration_boundaries(c_minus)

        psi = _solve_poisson_sor(
            psi=psi,
            c_plus=c_plus,
            c_minus=c_minus,
            beta=beta,
            grid=grid,
            left_psi=left_psi,
            right_psi=right_psi,
            fixed_charge=fixed_charge,
            tol=config.solver.tolerance * 0.1,
        )

        residual = max(
            float(np.max(np.abs(psi - old_psi))),
            float(np.max(np.abs(c_plus - old_cp))),
            float(np.max(np.abs(c_minus - old_cm))),
        )
        if residual < config.solver.tolerance:
            converged = True
            break

    return PnpSolution(
        psi=psi,
        c_plus=c_plus,
        c_minus=c_minus,
        iterations=iteration,
        residual=residual,
        runtime_s=time.perf_counter() - start,
        converged=converged,
    )


def compute_np_fluxes(solution: PnpSolution, grid: AxisymmetricGrid) -> dict[str, np.ndarray]:
    dpsi_dr, dpsi_dz = np.gradient(solution.psi, grid.dr, grid.dz, edge_order=1)
    dcp_dr, dcp_dz = np.gradient(solution.c_plus, grid.dr, grid.dz, edge_order=1)
    dcm_dr, dcm_dz = np.gradient(solution.c_minus, grid.dr, grid.dz, edge_order=1)
    jpr = -(dcp_dr + solution.c_plus * dpsi_dr)
    jpz = -(dcp_dz + solution.c_plus * dpsi_dz)
    jmr = -(dcm_dr - solution.c_minus * dpsi_dr)
    jmz = -(dcm_dz - solution.c_minus * dpsi_dz)
    return {"jpr": jpr, "jpz": jpz, "jmr": jmr, "jmz": jmz}


def dimensionless_current_profile(solution: PnpSolution, grid: AxisymmetricGrid) -> np.ndarray:
    flux = compute_np_fluxes(solution, grid)
    integrand = (flux["jpz"] - flux["jmz"]) * grid.r[:, None]
    return 2.0 * np.pi * np.trapz(integrand, grid.r, axis=0)


def current_scale_A(config: ExperimentConfig) -> float:
    scales = config.scales
    radius_m = config.geometry.radius_nm * 1e-9
    length_m = config.geometry.length_nm * 1e-9
    diffusivity = 0.5 * (
        config.physics.diffusion_cation_m2_s + config.physics.diffusion_anion_m2_s
    )
    return FARADAY * diffusivity * config.physics.bulk_concentration_mol_m3 * radius_m**2 / length_m


def physical_current_profile_A(solution: PnpSolution, grid: AxisymmetricGrid, config: ExperimentConfig) -> np.ndarray:
    return dimensionless_current_profile(solution, grid) * current_scale_A(config)


def flux_conservation_error(solution: PnpSolution, grid: AxisymmetricGrid) -> float:
    current = dimensionless_current_profile(solution, grid)
    denom = max(float(np.max(np.abs(current))), 1e-12)
    return float((np.max(current) - np.min(current)) / denom)

