"""Node-centred cylindrical finite volumes with blocking radial boundaries.

The only eliminated unknowns are the axial reservoir values. Radial end nodes
own half control volumes; prescribed Poisson wall flux enters the load vector.
"""
import time

import numpy as np
from scipy.optimize import root
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu, spsolve

from .baseline import PnpSolution, current_scale_A
from .scales import voltage_to_psi
from .sg_solver import bernoulli, annular_control_volume_weights, sg_face_flux_z


class ConservativePNP:
    def __init__(self, grid, beta, left_psi, wall_charge, fixed_charge=None,
                 right_psi=0.0, c_left=1.0, c_right=1.0):
        self.grid = grid
        self.beta = beta
        self.left = left_psi
        self.right = right_psi
        self.c_left = float(c_left)
        self.c_right = float(c_right)
        nr, nz = grid.shape
        self.n = nr * nz
        indices = np.arange(self.n).reshape(grid.shape)
        self.boundary = np.concatenate((indices[:, 0], indices[:, -1]))
        self.left_boundary = indices[:, 0]
        self.right_boundary = indices[:, -1]
        self.free = indices[:, 1:-1].ravel()
        weights = annular_control_volume_weights(grid)
        self.row, self.col, self.weight = [], [], []
        for i in range(nr):
            for j in range(1, nz - 1):
                neighbors = [(i, j - 1, 1 / grid.dz**2), (i, j + 1, 1 / grid.dz**2)]
                if i > 0:
                    neighbors.append((i - 1, j, grid.radial_metric * (grid.r[i] - grid.dr / 2) / (weights[i] * grid.dr)))
                if i < nr - 1:
                    neighbors.append((i + 1, j, grid.radial_metric * (grid.r[i] + grid.dr / 2) / (weights[i] * grid.dr)))
                for ni, nj, weight in neighbors:
                    self.row.append(indices[i, j])
                    self.col.append(indices[ni, nj])
                    self.weight.append(weight)
        self.row = np.asarray(self.row)
        self.col = np.asarray(self.col)
        self.weight = np.asarray(self.weight)
        self.poisson_matrix = self.matrix(np.ones_like(self.weight), np.ones_like(self.weight))
        self.poisson_lu = splu(self.poisson_matrix.tocsc())
        self.load = np.zeros(grid.shape)
        self.load[:, 0] = left_psi
        self.load[:, -1] = right_psi
        self.load[-1, 1:-1] += grid.radial_metric * wall_charge / weights[-1]
        if fixed_charge is not None:
            self.load[:, 1:-1] += beta * fixed_charge[:, 1:-1]

    def matrix(self, center, neighbor):
        diagonal = np.bincount(self.row, weights=self.weight * center, minlength=self.n)
        diagonal[self.boundary] = 1.0
        return coo_matrix((np.concatenate((diagonal, -self.weight * neighbor)),
                          (np.concatenate((np.arange(self.n), self.row)),
                           np.concatenate((np.arange(self.n), self.col)))), shape=(self.n, self.n)).tocsr()

    def ions(self, psi, sign):
        delta = sign * (psi.ravel()[self.col] - psi.ravel()[self.row])
        # J(i->j) = [B(delta)c_i - B(-delta)c_j] / h.
        matrix = self.matrix(bernoulli(delta), bernoulli(-delta))
        rhs = np.zeros(self.n)
        rhs[self.left_boundary] = self.c_left
        rhs[self.right_boundary] = self.c_right
        return spsolve(matrix, rhs).reshape(self.grid.shape)

    def unpack(self, values):
        psi = np.zeros(self.grid.shape)
        psi[:, 0] = self.left
        psi[:, -1] = self.right
        psi.ravel()[self.free] = values
        return psi

    def poisson_from_concentrations(self, cp, cm):
        rhs = self.load.copy()
        rhs[:, 1:-1] += self.beta * (cp - cm)[:, 1:-1]
        return self.poisson_lu.solve(rhs.ravel()).reshape(self.grid.shape)

    def gummel_step(self, psi):
        cp = self.ions(psi, 1)
        cm = self.ions(psi, -1)
        return self.poisson_from_concentrations(cp, cm), cp, cm

    def residual(self, values):
        psi = self.unpack(values)
        cp, cm = self.ions(psi, 1), self.ions(psi, -1)
        rhs = self.load.copy()
        rhs[:, 1:-1] += self.beta * (cp - cm)[:, 1:-1]
        target = self.poisson_lu.solve(rhs.ravel())
        return target[self.free] - values

    def solve(self, tolerance=1e-9, max_iter=100, psi0=None):
        start = time.perf_counter()
        initial = (self.left * (1 - self.grid.zz) + self.right * self.grid.zz) if psi0 is None else psi0
        result = root(self.residual, initial.ravel()[self.free], method="krylov",
                      options={"fatol": tolerance, "maxiter": max_iter})
        psi = self.unpack(result.x)
        cp, cm = self.ions(psi, 1), self.ions(psi, -1)
        residual = float(np.max(np.abs(self.residual(result.x))))
        finite = all(np.isfinite(a).all() for a in (psi, cp, cm))
        ok = finite and cp.min() > 0 and cm.min() > 0 and residual < tolerance
        return PnpSolution(psi, cp, cm, int(result.nit), residual, time.perf_counter() - start, bool(ok))


def solve_reference(config, grid, tolerance=1e-9, max_iter=100, psi0=None,
                    c_left=1.0, c_right=1.0, right_psi=0.0):
    if config.physics.fixed_charge_strength != 0:
        raise ValueError("Revision benchmark uses surface charge only.")
    problem = ConservativePNP(grid, config.scales.poisson_coupling,
                              voltage_to_psi(config.physics.applied_voltage_V, config.scales),
                              config.physics.wall_surface_charge_strength,
                              right_psi=right_psi, c_left=c_left, c_right=c_right)
    solution = problem.solve(tolerance, max_iter, psi0)
    if not solution.converged:
        raise RuntimeError(f"Reference rejected: residual={solution.residual:.3e}")
    return solution


def physical_current(solution, grid, config):
    dp = config.physics.diffusion_cation_m2_s
    dm = config.physics.diffusion_anion_m2_s
    mean_d = (dp + dm) / 2
    jp = sg_face_flux_z(solution.c_plus, solution.psi, grid, 1)
    jm = sg_face_flux_z(solution.c_minus, solution.psi, grid, -1)
    weights = annular_control_volume_weights(grid)
    return 2 * np.pi * np.sum((dp * jp - dm * jm) * weights[:, None], axis=0) * current_scale_A(config) / mean_d
