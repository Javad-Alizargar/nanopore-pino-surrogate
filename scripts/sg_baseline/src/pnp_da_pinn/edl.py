import numpy as np
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

from .grid import AxisymmetricGrid


def radial_debye_huckel_profile(grid: AxisymmetricGrid, beta: float, wall_surface_charge: float) -> np.ndarray:
    """Return a linearized radial EDL perturbation satisfying the wall Neumann condition.

    The profile solves u_rr + r^-1 u_r - 2 beta u = 0 with symmetry at r=0 and
    u_r(1)=wall_surface_charge. It is intended as an initial condition, not as a
    replacement for the nonlinear PNP solve.
    """
    nr = grid.r.size
    if nr < 3 or abs(wall_surface_charge) < 1e-15:
        return np.zeros(nr, dtype=float)

    a = lil_matrix((nr, nr), dtype=float)
    b = np.zeros(nr, dtype=float)

    a[0, 0] = 1.0
    a[0, 1] = -1.0

    for i in range(1, nr - 1):
        r = max(float(grid.r[i]), 1e-8)
        ae = 1.0 / grid.dr**2 + 1.0 / (2.0 * r * grid.dr)
        aw = 1.0 / grid.dr**2 - 1.0 / (2.0 * r * grid.dr)
        ap = -(ae + aw + 2.0 * beta / grid.radial_metric)
        a[i, i + 1] = ae
        a[i, i - 1] = aw
        a[i, i] = ap

    a[-1, -1] = 1.0
    a[-1, -2] = -1.0
    b[-1] = wall_surface_charge * grid.dr

    profile = np.asarray(spsolve(a.tocsr(), b), dtype=float)
    return profile - profile[0]


def edl_initial_state(
    grid: AxisymmetricGrid,
    beta: float,
    left_psi: float,
    wall_surface_charge: float,
    max_abs_psi: float = 12.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reservoir-compatible EDL initial state for charged-wall PNP stages."""
    radial = radial_debye_huckel_profile(grid, beta, wall_surface_charge)
    axial_envelope = grid.zz * (1.0 - grid.zz)
    perturbation = np.clip(radial[:, None] * axial_envelope, -max_abs_psi, max_abs_psi)

    psi = left_psi * (1.0 - grid.zz) + perturbation
    c_plus = np.exp(np.clip(-perturbation, -30.0, 30.0))
    c_minus = np.exp(np.clip(perturbation, -30.0, 30.0))

    psi[0, :] = psi[1, :]
    psi[-1, 1:-1] = psi[-2, 1:-1] + wall_surface_charge * grid.dr
    psi[:, 0] = left_psi
    psi[:, -1] = 0.0
    c_plus[:, 0] = 1.0
    c_plus[:, -1] = 1.0
    c_minus[:, 0] = 1.0
    c_minus[:, -1] = 1.0
    return psi, c_plus, c_minus
