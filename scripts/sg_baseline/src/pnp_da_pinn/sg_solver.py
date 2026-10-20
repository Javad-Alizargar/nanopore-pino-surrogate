from dataclasses import dataclass
import time

import numpy as np
from scipy.optimize import NoConvergence, anderson, root
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import MatrixRankWarning, spsolve
import warnings

from .baseline import PnpSolution
from .config import ExperimentConfig
from .grid import AxisymmetricGrid
from .scales import voltage_to_psi


def bernoulli(x: np.ndarray | float) -> np.ndarray | float:
    x_arr = np.asarray(x, dtype=float)
    x_clip = np.clip(x_arr, -80.0, 80.0)
    out = np.empty_like(x_clip, dtype=float)
    small = np.abs(x_clip) < 1e-8
    out[small] = 1.0 - x_clip[small] / 2.0 + x_clip[small] ** 2 / 12.0
    out[~small] = x_clip[~small] / np.expm1(x_clip[~small])
    if np.isscalar(x):
        return float(out)
    return out


@dataclass(frozen=True)
class SgDiagnostics:
    iterations: int
    residual: float
    runtime_s: float
    converged: bool


def _idx(i: int, j: int, nz: int) -> int:
    return i * nz + j


def _apply_psi_boundary_values(
    psi: np.ndarray,
    left_psi: float,
    right_psi: float,
    wall_surface_charge: float = 0.0,
    dr: float | None = None,
) -> None:
    psi[0, :] = psi[1, :]
    radial_step = (1.0 / (psi.shape[0] - 1)) if dr is None else dr
    psi[-1, :] = psi[-2, :] + wall_surface_charge * radial_step
    psi[:, 0] = left_psi
    psi[:, -1] = right_psi


def solve_poisson_linear(
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    fixed_charge: np.ndarray,
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    right_psi: float,
    wall_surface_charge: float = 0.0,
) -> np.ndarray:
    nr, nz = grid.shape
    n = nr * nz
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n, dtype=float)

    def add(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(value)

    for i in range(nr):
        for j in range(nz):
            row = _idx(i, j, nz)
            if j == 0:
                add(row, row, 1.0)
                b[row] = left_psi
            elif j == nz - 1:
                add(row, row, 1.0)
                b[row] = right_psi
            elif i == 0:
                add(row, row, 1.0)
                add(row, _idx(1, j, nz), -1.0)
            elif i == nr - 1:
                add(row, row, 1.0)
                add(row, _idx(nr - 2, j, nz), -1.0)
                b[row] = wall_surface_charge * grid.dr
            else:
                r = max(grid.r[i], 1e-8)
                ae = grid.radial_metric * (1.0 / grid.dr**2 + 1.0 / (2.0 * r * grid.dr))
                aw = grid.radial_metric * (1.0 / grid.dr**2 - 1.0 / (2.0 * r * grid.dr))
                an = 1.0 / grid.dz**2
                ass = 1.0 / grid.dz**2
                ap = -(ae + aw + an + ass)
                add(row, _idx(i + 1, j, nz), ae)
                add(row, _idx(i - 1, j, nz), aw)
                add(row, _idx(i, j + 1, nz), an)
                add(row, _idx(i, j - 1, nz), ass)
                add(row, row, ap)
                b[row] = -beta * (c_plus[i, j] - c_minus[i, j] + fixed_charge[i, j])

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        return spsolve(csr_matrix((data, (rows, cols)), shape=(n, n)), b).reshape((nr, nz))


def solve_np_scharfetter_gummel(psi: np.ndarray, grid: AxisymmetricGrid, ion_sign: int) -> np.ndarray:
    nr, nz = grid.shape
    n = nr * nz
    rows: list[int] = []
    cols: list[int] = []
    data: list[float] = []
    b = np.zeros(n, dtype=float)

    def add(row: int, col: int, value: float) -> None:
        rows.append(row)
        cols.append(col)
        data.append(value)

    for i in range(nr):
        for j in range(nz):
            row = _idx(i, j, nz)
            if j == 0 or j == nz - 1:
                add(row, row, 1.0)
                b[row] = 1.0
                continue
            if i == 0:
                add(row, row, 1.0)
                add(row, _idx(1, j, nz), -1.0)
                continue
            if i == nr - 1:
                delta = ion_sign * (psi[i, j] - psi[i - 1, j])
                ratio = bernoulli(delta) / max(bernoulli(-delta), 1e-12)
                add(row, row, 1.0)
                add(row, _idx(nr - 2, j, nz), -ratio)
                continue

            coeff_center = 0.0
            neighbors: list[tuple[int, int, float]] = []
            r_center = max(grid.r[i], 1e-8)
            for ni, nj, h in [
                (i + 1, j, grid.dr),
                (i - 1, j, grid.dr),
                (i, j + 1, grid.dz),
                (i, j - 1, grid.dz),
            ]:
                delta = ion_sign * (psi[ni, nj] - psi[i, j])
                b_forward = bernoulli(delta)
                b_backward = bernoulli(-delta)
                is_radial = ni != i
                if is_radial:
                    face_r = grid.r[i] + (0.5 * grid.dr if ni > i else -0.5 * grid.dr)
                    face_weight = max(face_r, 0.0) / r_center
                    weight = grid.radial_metric * face_weight / h**2
                else:
                    weight = 1.0 / h**2
                coeff_center += weight * b_forward
                neighbors.append((ni, nj, -weight * b_backward))

            add(row, row, coeff_center)
            for ni, nj, coeff in neighbors:
                add(row, _idx(ni, nj, nz), coeff)

    with warnings.catch_warnings():
        warnings.simplefilter("error", MatrixRankWarning)
        c = spsolve(csr_matrix((data, (rows, cols)), shape=(n, n)), b).reshape((nr, nz))
    return np.maximum(c, 1e-12)


def wall_fixed_charge(grid: AxisymmetricGrid, strength: float, width: float = 0.08) -> np.ndarray:
    """Dimensionless fixed charge localized near the nanopore wall."""
    if abs(strength) < 1e-15:
        return np.zeros(grid.shape, dtype=float)
    shell = np.exp(-((1.0 - grid.rr) / width) ** 2)
    return strength * shell


def _state_residual(target: tuple[np.ndarray, np.ndarray, np.ndarray], old: tuple[np.ndarray, np.ndarray, np.ndarray]) -> float:
    return float(max(np.max(np.abs(target[k] - old[k])) for k in range(3)))


def _solution_from_state(
    psi: np.ndarray,
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    iterations: int = 0,
    residual: float = np.inf,
    converged: bool = False,
) -> PnpSolution:
    return PnpSolution(
        psi=psi,
        c_plus=c_plus,
        c_minus=c_minus,
        iterations=iterations,
        residual=residual,
        runtime_s=0.0,
        converged=converged,
    )


def _state_quality(
    psi: np.ndarray,
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    grid: AxisymmetricGrid,
    residual: float,
) -> tuple[float, float]:
    conservation = sg_flux_conservation_error(_solution_from_state(psi, c_plus, c_minus), grid)
    return max(float(residual), float(conservation)), float(conservation)


def _fixed_point_state_residual(
    psi: np.ndarray,
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    right_psi: float,
    fixed_charge: np.ndarray,
    wall_surface_charge: float,
) -> float:
    cp_target = solve_np_scharfetter_gummel(psi, grid, ion_sign=1)
    cm_target = solve_np_scharfetter_gummel(psi, grid, ion_sign=-1)
    psi_target = solve_poisson_linear(
        cp_target,
        cm_target,
        fixed_charge,
        grid,
        beta,
        left_psi,
        right_psi,
        wall_surface_charge,
    )
    _apply_psi_boundary_values(psi_target, left_psi, right_psi, wall_surface_charge, grid.dr)
    return _state_residual((psi_target, cp_target, cm_target), (psi, c_plus, c_minus))


def _apply_concentration_boundaries(c_plus: np.ndarray, c_minus: np.ndarray) -> None:
    c_plus[:, 0] = 1.0
    c_plus[:, -1] = 1.0
    c_minus[:, 0] = 1.0
    c_minus[:, -1] = 1.0
    np.maximum(c_plus, 1e-12, out=c_plus)
    np.maximum(c_minus, 1e-12, out=c_minus)


def _solve_stage(
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    right_psi: float,
    fixed_charge: np.ndarray,
    wall_surface_charge: float,
    psi: np.ndarray,
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    max_iter: int,
    tolerance: float,
    base_relaxation: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, bool]:
    _apply_psi_boundary_values(psi, left_psi, right_psi, wall_surface_charge, grid.dr)
    _apply_concentration_boundaries(c_plus, c_minus)
    residual = np.inf
    previous_step = np.inf
    accepted_iterations = 0

    for iteration in range(1, max_iter + 1):
        old = (psi.copy(), c_plus.copy(), c_minus.copy())
        try:
            cp_new = solve_np_scharfetter_gummel(psi, grid, ion_sign=1)
            cm_new = solve_np_scharfetter_gummel(psi, grid, ion_sign=-1)
            psi_new = solve_poisson_linear(cp_new, cm_new, fixed_charge, grid, beta, left_psi, right_psi, wall_surface_charge)
        except Exception:
            return psi, c_plus, c_minus, iteration, np.inf, False

        _apply_psi_boundary_values(psi_new, left_psi, right_psi, wall_surface_charge, grid.dr)
        target = (psi_new, cp_new, cm_new)
        residual = _state_residual(target, old)
        if not np.isfinite(residual):
            return psi, c_plus, c_minus, iteration, np.inf, False
        if residual < tolerance:
            return psi_new, cp_new, cm_new, iteration, residual, True

        relax = min(base_relaxation, 0.5)
        accepted = False
        for _ in range(14):
            cand_psi = (1.0 - relax) * psi + relax * psi_new
            cand_cp = (1.0 - relax) * c_plus + relax * cp_new
            cand_cm = (1.0 - relax) * c_minus + relax * cm_new
            _apply_psi_boundary_values(cand_psi, left_psi, right_psi, wall_surface_charge, grid.dr)
            _apply_concentration_boundaries(cand_cp, cand_cm)
            step = _state_residual((cand_psi, cand_cp, cand_cm), old)
            if np.isfinite(step) and step <= max(previous_step * 1.25, tolerance * 10.0):
                psi, c_plus, c_minus = cand_psi, cand_cp, cand_cm
                previous_step = step
                accepted = True
                accepted_iterations = iteration
                break
            relax *= 0.5
        if not accepted:
            return psi, c_plus, c_minus, iteration, residual, False

    return psi, c_plus, c_minus, accepted_iterations, residual, False


def _solve_stage_pseudo_transient(
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    right_psi: float,
    fixed_charge: np.ndarray,
    wall_surface_charge: float,
    psi: np.ndarray,
    c_plus: np.ndarray,
    c_minus: np.ndarray,
    max_iter: int,
    tolerance: float,
    base_relaxation: float,
    current_tolerance: float = 0.02,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, bool]:
    _apply_psi_boundary_values(psi, left_psi, right_psi, wall_surface_charge, grid.dr)
    _apply_concentration_boundaries(c_plus, c_minus)

    residual = np.inf
    _, conservation = _state_quality(psi, c_plus, c_minus, grid, residual)
    best_state = (psi.copy(), c_plus.copy(), c_minus.copy())
    best_score = np.inf
    best_residual = np.inf
    relax = min(max(base_relaxation, 1e-5), 0.2)

    for iteration in range(1, max_iter + 1):
        old = (psi.copy(), c_plus.copy(), c_minus.copy())
        try:
            cp_target = solve_np_scharfetter_gummel(psi, grid, ion_sign=1)
            cm_target = solve_np_scharfetter_gummel(psi, grid, ion_sign=-1)
            psi_target = solve_poisson_linear(
                cp_target,
                cm_target,
                fixed_charge,
                grid,
                beta,
                left_psi,
                right_psi,
                wall_surface_charge,
            )
        except Exception:
            return best_state[0], best_state[1], best_state[2], iteration, residual, False

        _apply_psi_boundary_values(psi_target, left_psi, right_psi, wall_surface_charge, grid.dr)
        residual = _state_residual((psi_target, cp_target, cm_target), old)
        if not np.isfinite(residual):
            return best_state[0], best_state[1], best_state[2], iteration, np.inf, False

        state_score, conservation = _state_quality(psi, c_plus, c_minus, grid, residual)
        if state_score < best_score:
            best_state = (psi.copy(), c_plus.copy(), c_minus.copy())
            best_score = state_score
            best_residual = residual
        if residual < tolerance and conservation < current_tolerance:
            return psi, c_plus, c_minus, iteration, residual, True

        accepted = False
        trial_relax = relax
        for _ in range(18):
            cand_psi = (1.0 - trial_relax) * psi + trial_relax * psi_target
            cand_cp = (1.0 - trial_relax) * c_plus + trial_relax * cp_target
            cand_cm = (1.0 - trial_relax) * c_minus + trial_relax * cm_target
            _apply_psi_boundary_values(cand_psi, left_psi, right_psi, wall_surface_charge, grid.dr)
            _apply_concentration_boundaries(cand_cp, cand_cm)
            if not (np.isfinite(cand_psi).all() and np.isfinite(cand_cp).all() and np.isfinite(cand_cm).all()):
                trial_relax *= 0.5
                continue
            if cand_cp.min() < 1e-10 or cand_cm.min() < 1e-10:
                trial_relax *= 0.5
                continue

            candidate_step = _state_residual((cand_psi, cand_cp, cand_cm), old)
            candidate_score, candidate_conservation = _state_quality(
                cand_psi,
                cand_cp,
                cand_cm,
                grid,
                max(residual * trial_relax, candidate_step),
            )
            conservation_gate = max(conservation * 1.05, current_tolerance)
            if candidate_score <= max(best_score * 1.02, tolerance * 10.0) or candidate_conservation <= conservation_gate:
                psi, c_plus, c_minus = cand_psi, cand_cp, cand_cm
                relax = min(trial_relax * 1.15, 0.2)
                accepted = True
                if candidate_score < best_score:
                    best_state = (psi.copy(), c_plus.copy(), c_minus.copy())
                    best_score = candidate_score
                    best_residual = max(residual * trial_relax, candidate_step)
                break
            trial_relax *= 0.5

        if not accepted:
            relax *= 0.5
            if relax < 1e-8:
                best_residual = _fixed_point_state_residual(
                    best_state[0],
                    best_state[1],
                    best_state[2],
                    grid,
                    beta,
                    left_psi,
                    right_psi,
                    fixed_charge,
                    wall_surface_charge,
                )
                return best_state[0], best_state[1], best_state[2], iteration, best_residual, False

    best_residual = _fixed_point_state_residual(
        best_state[0],
        best_state[1],
        best_state[2],
        grid,
        beta,
        left_psi,
        right_psi,
        fixed_charge,
        wall_surface_charge,
    )
    return best_state[0], best_state[1], best_state[2], max_iter, best_residual, False


def _solve_stage_anderson(
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    right_psi: float,
    fixed_charge: np.ndarray,
    wall_surface_charge: float,
    psi: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, bool]:
    nr, nz = grid.shape

    def fixed_point_residual(flat_psi: np.ndarray) -> np.ndarray:
        psi_iter = flat_psi.reshape((nr, nz)).copy()
        _apply_psi_boundary_values(psi_iter, left_psi, right_psi, wall_surface_charge, grid.dr)
        cp_new = solve_np_scharfetter_gummel(psi_iter, grid, ion_sign=1)
        cm_new = solve_np_scharfetter_gummel(psi_iter, grid, ion_sign=-1)
        psi_new = solve_poisson_linear(
            cp_new,
            cm_new,
            fixed_charge,
            grid,
            beta,
            left_psi,
            right_psi,
            wall_surface_charge,
        )
        _apply_psi_boundary_values(psi_new, left_psi, right_psi, wall_surface_charge, grid.dr)
        return (psi_new - psi_iter).ravel()

    start = psi.copy()
    _apply_psi_boundary_values(start, left_psi, right_psi, wall_surface_charge, grid.dr)
    iterations = 0
    residual = np.inf

    def callback(_: np.ndarray, f: np.ndarray) -> None:
        nonlocal iterations, residual
        iterations += 1
        residual = float(np.max(np.abs(f)))

    try:
        solved = anderson(
            fixed_point_residual,
            start.ravel(),
            maxiter=max_iter,
            f_tol=tolerance,
            M=8,
            w0=0.01,
            callback=callback,
        )
        psi_solved = solved.reshape((nr, nz)).copy()
        _apply_psi_boundary_values(psi_solved, left_psi, right_psi, wall_surface_charge, grid.dr)
        c_plus = solve_np_scharfetter_gummel(psi_solved, grid, ion_sign=1)
        c_minus = solve_np_scharfetter_gummel(psi_solved, grid, ion_sign=-1)
        final_residual = float(np.max(np.abs(fixed_point_residual(psi_solved.ravel()))))
        return psi_solved, c_plus, c_minus, iterations, final_residual, final_residual < tolerance
    except NoConvergence as err:
        candidate = np.asarray(err.args[0], dtype=float).reshape((nr, nz)).copy()
        _apply_psi_boundary_values(candidate, left_psi, right_psi, wall_surface_charge, grid.dr)
        c_plus = solve_np_scharfetter_gummel(candidate, grid, ion_sign=1)
        c_minus = solve_np_scharfetter_gummel(candidate, grid, ion_sign=-1)
        final_residual = float(np.max(np.abs(fixed_point_residual(candidate.ravel()))))
        if not np.isfinite(final_residual) or final_residual > 1.0:
            return psi, np.ones(grid.shape), np.ones(grid.shape), max(iterations, 1), residual, False
        return candidate, c_plus, c_minus, max(iterations, 1), final_residual, final_residual < tolerance
    except Exception:
        return psi, np.ones(grid.shape), np.ones(grid.shape), max(iterations, 1), residual, False


def _solve_stage_reduced_krylov(
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    right_psi: float,
    fixed_charge: np.ndarray,
    wall_surface_charge: float,
    psi: np.ndarray,
    max_iter: int,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, float, bool]:
    mask = np.zeros(grid.shape, dtype=bool)
    mask[1:-1, 1:-1] = True

    def pack(field: np.ndarray) -> np.ndarray:
        return field[mask]

    def unpack(values: np.ndarray) -> np.ndarray:
        out = psi.copy()
        out[mask] = values
        _apply_psi_boundary_values(out, left_psi, right_psi, wall_surface_charge, grid.dr)
        return out

    def residual_fn(values: np.ndarray) -> np.ndarray:
        psi_iter = unpack(values)
        cp_target = solve_np_scharfetter_gummel(psi_iter, grid, ion_sign=1)
        cm_target = solve_np_scharfetter_gummel(psi_iter, grid, ion_sign=-1)
        psi_target = solve_poisson_linear(
            cp_target,
            cm_target,
            fixed_charge,
            grid,
            beta,
            left_psi,
            right_psi,
            wall_surface_charge,
        )
        _apply_psi_boundary_values(psi_target, left_psi, right_psi, wall_surface_charge, grid.dr)
        return (psi_target - psi_iter)[mask]

    try:
        result = root(
            residual_fn,
            pack(psi),
            method="krylov",
            options={"maxiter": max_iter, "fatol": tolerance},
        )
        psi_solved = unpack(result.x)
        c_plus = solve_np_scharfetter_gummel(psi_solved, grid, ion_sign=1)
        c_minus = solve_np_scharfetter_gummel(psi_solved, grid, ion_sign=-1)
        residual = float(np.max(np.abs(residual_fn(result.x))))
        finite = bool(np.isfinite(psi_solved).all() and np.isfinite(c_plus).all() and np.isfinite(c_minus).all())
        positive = bool(c_plus.min() > 0.0 and c_minus.min() > 0.0)
        return psi_solved, c_plus, c_minus, int(result.nit), residual, bool(residual < tolerance and finite and positive)
    except Exception:
        return psi, np.ones(grid.shape), np.ones(grid.shape), 0, np.inf, False



def solve_pnp_sg_state(
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    fixed_charge: np.ndarray,
    wall_surface_charge: float,
    max_iter: int,
    tolerance: float,
    relaxation: float,
    psi0: np.ndarray | None = None,
    c_plus0: np.ndarray | None = None,
    c_minus0: np.ndarray | None = None,
) -> PnpSolution:
    start = time.perf_counter()
    right_psi = 0.0
    psi = left_psi * (1.0 - grid.zz) if psi0 is None else psi0.copy()
    c_plus = np.ones(grid.shape) if c_plus0 is None else c_plus0.copy()
    c_minus = np.ones(grid.shape) if c_minus0 is None else c_minus0.copy()

    if abs(wall_surface_charge) > 1e-15 and np.max(np.abs(fixed_charge)) < 1e-15:
        psi_a, cp_a, cm_a, iterations, residual, converged = _solve_stage_pseudo_transient(
            grid=grid,
            beta=beta,
            left_psi=left_psi,
            right_psi=right_psi,
            fixed_charge=fixed_charge,
            wall_surface_charge=wall_surface_charge,
            psi=psi,
            c_plus=c_plus,
            c_minus=c_minus,
            max_iter=max_iter,
            tolerance=tolerance,
            base_relaxation=relaxation,
        )
        if converged:
            psi, c_plus, c_minus = psi_a, cp_a, cm_a
        else:
            psi_k, cp_k, cm_k, iter_k, residual_k, converged_k = _solve_stage_reduced_krylov(
                grid=grid,
                beta=beta,
                left_psi=left_psi,
                right_psi=right_psi,
                fixed_charge=fixed_charge,
                wall_surface_charge=wall_surface_charge,
                psi=psi_a,
                max_iter=max(20, min(max_iter, 40)),
                tolerance=tolerance,
            )
            iterations += iter_k
            if converged_k:
                psi, c_plus, c_minus = psi_k, cp_k, cm_k
                residual = residual_k
                converged = True
            elif _state_quality(psi_a, cp_a, cm_a, grid, residual)[1] < 0.02 and residual < 0.2:
                psi, c_plus, c_minus = psi_a, cp_a, cm_a
            else:
                psi_b, cp_b, cm_b, iter_b, residual_b, converged_b = _solve_stage_anderson(
                    grid=grid,
                    beta=beta,
                    left_psi=left_psi,
                    right_psi=right_psi,
                    fixed_charge=fixed_charge,
                    wall_surface_charge=wall_surface_charge,
                    psi=psi_a,
                    max_iter=max_iter,
                    tolerance=tolerance,
                )
                if converged_b:
                    psi, c_plus, c_minus = psi_b, cp_b, cm_b
                    iterations += iter_b
                    residual = residual_b
                    converged = True
                else:
                    psi, c_plus, c_minus = psi_a, cp_a, cm_a
    else:
        psi, c_plus, c_minus, iterations, residual, converged = _solve_stage(
            grid=grid,
            beta=beta,
            left_psi=left_psi,
            right_psi=right_psi,
            fixed_charge=fixed_charge,
            wall_surface_charge=wall_surface_charge,
            psi=psi,
            c_plus=c_plus,
            c_minus=c_minus,
            max_iter=max_iter,
            tolerance=tolerance,
            base_relaxation=relaxation,
        )
    return PnpSolution(
        psi=psi,
        c_plus=c_plus,
        c_minus=c_minus,
        iterations=iterations,
        residual=residual,
        runtime_s=time.perf_counter() - start,
        converged=converged,
    )
def solve_pnp_sg(config: ExperimentConfig, grid: AxisymmetricGrid) -> PnpSolution:
    start = time.perf_counter()
    beta = config.scales.poisson_coupling
    target_left_psi = voltage_to_psi(config.physics.applied_voltage_V, config.scales)
    target_charge = wall_fixed_charge(grid, config.physics.fixed_charge_strength)
    target_surface_charge = config.physics.wall_surface_charge_strength
    right_psi = 0.0

    stage_count = 12 if (np.max(np.abs(target_charge)) > 1e-15 or abs(target_surface_charge) > 1e-15) else 1

    psi = target_left_psi * (1.0 - grid.zz)
    c_plus = np.ones(grid.shape)
    c_minus = np.ones(grid.shape)
    total_iterations = 0
    residual = np.inf
    converged = True

    for stage in range(1, stage_count + 1):
        alpha = stage / stage_count
        left_psi = target_left_psi
        fixed_charge = alpha * target_charge
        wall_surface_charge = alpha * target_surface_charge
        psi, c_plus, c_minus, iters, residual, ok = _solve_stage(
            grid=grid,
            beta=beta,
            left_psi=left_psi,
            right_psi=right_psi,
            fixed_charge=fixed_charge,
            wall_surface_charge=wall_surface_charge,
            psi=psi,
            c_plus=c_plus,
            c_minus=c_minus,
            max_iter=config.solver.max_gummel_iter,
            tolerance=config.solver.tolerance,
            base_relaxation=config.solver.relaxation,
        )
        total_iterations += iters
        if not ok:
            converged = False
            break

    return PnpSolution(
        psi=psi,
        c_plus=c_plus,
        c_minus=c_minus,
        iterations=total_iterations,
        residual=residual,
        runtime_s=time.perf_counter() - start,
        converged=converged,
    )


def sg_face_flux_z(c: np.ndarray, psi: np.ndarray, grid: AxisymmetricGrid, ion_sign: int) -> np.ndarray:
    nr, nz = c.shape
    flux = np.zeros((nr, nz - 1), dtype=float)
    for j in range(nz - 1):
        delta = ion_sign * (psi[:, j + 1] - psi[:, j])
        flux[:, j] = -(c[:, j + 1] * bernoulli(-delta) - c[:, j] * bernoulli(delta)) / grid.dz
    return flux


def sg_current_profile(solution: PnpSolution, grid: AxisymmetricGrid) -> np.ndarray:
    jpz = sg_face_flux_z(solution.c_plus, solution.psi, grid, ion_sign=1)
    jmz = sg_face_flux_z(solution.c_minus, solution.psi, grid, ion_sign=-1)
    weights = annular_control_volume_weights(grid)
    return 2.0 * np.pi * np.sum((jpz - jmz) * weights[:, None], axis=0)


def annular_control_volume_weights(grid: AxisymmetricGrid) -> np.ndarray:
    """Weights for integral_0^1 f(r) r dr on a node-centered radial grid."""
    edges = np.empty(grid.r.size + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (grid.r[:-1] + grid.r[1:])
    return 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)


def sg_flux_conservation_error(solution: PnpSolution, grid: AxisymmetricGrid) -> float:
    current = sg_current_profile(solution, grid)
    denom = max(float(np.max(np.abs(current))), 1e-12)
    return float((np.max(current) - np.min(current)) / denom)





