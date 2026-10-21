from dataclasses import dataclass

import numpy as np

from .config import ExperimentConfig
from .edl import edl_initial_state
from .grid import AxisymmetricGrid
from .scales import PhysicalScales, voltage_to_psi
from .sg_solver import (
    sg_current_profile,
    sg_flux_conservation_error,
    solve_pnp_sg_state,
    wall_fixed_charge,
)


@dataclass
class ContinuationResult:
    solution: object
    diagnostics: list[dict[str, float | int | bool]]


def _beta_for_bulk(config: ExperimentConfig, bulk_concentration_mol_m3: float) -> float:
    scales = PhysicalScales(
        temperature_K=config.physics.temperature_K,
        bulk_concentration_mol_m3=bulk_concentration_mol_m3,
        relative_permittivity=config.physics.relative_permittivity,
        length_m=config.scales.length_m,
    )
    return scales.poisson_coupling


def run_bulk_voltage_charge_continuation(
    config: ExperimentConfig,
    grid: AxisymmetricGrid,
    start_bulk_mol_m3: float = 1.0,
    bulk_stages: int = 6,
    voltage_stages: int = 6,
    charge_stages: int = 8,
    max_iter_per_stage: int | None = None,
    tolerance: float | None = None,
    relaxation: float | None = None,
    schedule_mode: str = "bulk_then_charge",
) -> ContinuationResult:
    target_bulk = config.physics.bulk_concentration_mol_m3
    target_voltage = config.physics.applied_voltage_V
    target_charge_strength = config.physics.fixed_charge_strength
    target_surface_charge = config.physics.wall_surface_charge_strength
    max_iter = max_iter_per_stage or config.solver.max_gummel_iter
    tol = tolerance or config.solver.tolerance
    relax = relaxation or config.solver.relaxation

    if target_bulk <= 0 or start_bulk_mol_m3 <= 0:
        raise ValueError("Bulk concentrations must be positive.")

    bulk_values = np.geomspace(start_bulk_mol_m3, target_bulk, bulk_stages)
    voltage_values = np.linspace(0.0, target_voltage, voltage_stages)
    charge_values = np.linspace(0.0, target_charge_strength, charge_stages)
    surface_charge_values = np.linspace(0.0, target_surface_charge, charge_stages)

    schedule: list[tuple[float, float, float, str]] = []
    schedule.append((float(start_bulk_mol_m3), 0.0, 0.0, "initial"))
    for voltage in voltage_values[1:]:
        schedule.append((float(start_bulk_mol_m3), float(voltage), 0.0, "voltage"))

    if schedule_mode == "surface_then_bulk":
        if abs(target_surface_charge) > 1e-15:
            for surface_charge in surface_charge_values[1:]:
                schedule.append((float(start_bulk_mol_m3), float(target_voltage), float(surface_charge), "surface_charge"))
        for bulk in bulk_values[1:]:
            schedule.append((float(bulk), float(target_voltage), float(target_surface_charge), "surface_charge_bulk"))
        if abs(target_charge_strength) > 1e-15:
            for charge in charge_values[1:]:
                schedule.append((float(target_bulk), float(target_voltage), float(charge), "volume_charge"))
    elif schedule_mode == "bulk_then_charge":
        for bulk in bulk_values[1:]:
            schedule.append((float(bulk), float(target_voltage), 0.0, "bulk"))
        if abs(target_charge_strength) > 1e-15:
            for charge in charge_values[1:]:
                schedule.append((float(target_bulk), float(target_voltage), float(charge), "volume_charge"))
        if abs(target_surface_charge) > 1e-15:
            for surface_charge in surface_charge_values[1:]:
                schedule.append((float(target_bulk), float(target_voltage), float(surface_charge), "surface_charge"))
    else:
        raise ValueError(f"Unknown schedule_mode: {schedule_mode}")

    psi = None
    c_plus = None
    c_minus = None
    diagnostics: list[dict[str, float | int | bool]] = []
    solution = None

    for stage_index, (bulk, voltage, charge, stage_type) in enumerate(schedule, start=1):
        beta = _beta_for_bulk(config, bulk)
        # Thermal voltage is independent of bulk concentration.
        left_psi = voltage_to_psi(voltage, config.scales)
        if stage_type in {"surface_charge", "surface_charge_bulk"}:
            fixed_charge = wall_fixed_charge(grid, target_charge_strength)
            wall_surface_charge = charge
        else:
            fixed_charge = wall_fixed_charge(grid, charge)
            wall_surface_charge = 0.0
        psi_initial = psi
        c_plus_initial = c_plus
        c_minus_initial = c_minus
        if stage_type == "voltage" and abs(charge) < 1e-15:
            psi_initial = left_psi * (1.0 - grid.zz)
        if (
            stage_type in {"surface_charge", "surface_charge_bulk"}
            and abs(wall_surface_charge) > 1e-15
        ):
            psi_initial, c_plus_initial, c_minus_initial = edl_initial_state(
                grid=grid,
                beta=beta,
                left_psi=left_psi,
                wall_surface_charge=wall_surface_charge,
            )
        solution = solve_pnp_sg_state(
            grid=grid,
            beta=beta,
            left_psi=left_psi,
            fixed_charge=fixed_charge,
            wall_surface_charge=wall_surface_charge,
            max_iter=max_iter,
            tolerance=tol,
            relaxation=relax,
            psi0=psi_initial,
            c_plus0=c_plus_initial,
            c_minus0=c_minus_initial,
        )
        psi = solution.psi
        c_plus = solution.c_plus
        c_minus = solution.c_minus
        current = sg_current_profile(solution, grid)
        conservation = sg_flux_conservation_error(solution, grid)
        finite = bool(np.isfinite(psi).all() and np.isfinite(c_plus).all() and np.isfinite(c_minus).all())
        accepted_for_continuation = bool(
            finite
            and (
                solution.converged
                or (
                    stage_type in {"surface_charge", "surface_charge_bulk"}
                    and conservation < 0.02
                    and solution.residual < 0.2
                )
            )
        )
        diagnostics.append(
            {
                "stage": stage_index,
                "stage_type": stage_type,
                "bulk_mol_m3": bulk,
                "voltage_V": voltage,
                "charge_strength": charge,
                "wall_surface_charge": wall_surface_charge,
                "beta": beta,
                "converged": bool(solution.converged),
                "accepted_for_continuation": accepted_for_continuation,
                "iterations": int(solution.iterations),
                "residual": float(solution.residual),
                "runtime_s": float(solution.runtime_s),
                "finite": finite,
                "psi_min": float(np.nanmin(psi)),
                "psi_max": float(np.nanmax(psi)),
                "c_plus_min": float(np.nanmin(c_plus)),
                "c_plus_max": float(np.nanmax(c_plus)),
                "c_minus_min": float(np.nanmin(c_minus)),
                "c_minus_max": float(np.nanmax(c_minus)),
                "current_min": float(np.nanmin(current)),
                "current_max": float(np.nanmax(current)),
                "current_conservation_error": float(conservation),
            }
        )
        if not accepted_for_continuation:
            break

    if solution is None:
        raise RuntimeError("Continuation schedule was empty.")
    return ContinuationResult(solution=solution, diagnostics=diagnostics)


