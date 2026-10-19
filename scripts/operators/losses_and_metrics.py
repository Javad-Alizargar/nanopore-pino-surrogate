"""Vectorized PNP loss terms and the exact SG axial-current evaluator.

All operators are vectorized over a batch of predicted fields on the fixed
nr x nz grid. Residuals replicate the reference finite-volume / Scharfetter-
Gummel discretization of ``scripts/sg_baseline`` so the physics loss is
consistent with the verified solver.
"""

import numpy as np
import torch

from .constants import (
    DIFF_ANION,
    DIFF_CATION,
    FARADAY,
    L_NM,
    NR,
    NZ,
    annular_weights,
    make_grid,
)
from .features import raw_features

_MEAN_D = 0.5 * (DIFF_CATION + DIFF_ANION)


class GridTensors:
    """Fixed-grid tensors shared by all loss terms."""

    def __init__(self, nr=NR, nz=NZ, dtype=torch.float32):
        r, z, rr, zz, dr, dz = make_grid(nr, nz)
        edges = np.empty(nr + 1)
        edges[0] = 0.0
        edges[-1] = 1.0
        edges[1:-1] = 0.5 * (r[:-1] + r[1:])
        self.nr, self.nz = nr, nz
        self.r = torch.tensor(r, dtype=dtype)
        self.z = torch.tensor(z, dtype=dtype)
        self.dr = dr
        self.dz = dz
        self.weights = torch.tensor(annular_weights(r), dtype=dtype)
        self.edges = torch.tensor(edges, dtype=dtype)  # nr+1 edges


def bernoulli_torch(x):
    x_clip = torch.clamp(x, -80.0, 80.0)
    small = torch.abs(x_clip) < 1e-8
    series = 1.0 - x_clip / 2.0 + x_clip**2 / 12.0
    x_safe = torch.where(small, torch.ones_like(x_clip), x_clip)
    exact = x_safe / torch.expm1(x_safe)
    return torch.where(small, series, exact)


def sg_face_flux_z_torch(c, psi, grid, sign):
    dz = grid.dz
    delta = sign * (psi[:, :, 1:] - psi[:, :, :-1])
    return -(c[:, :, 1:] * bernoulli_torch(-delta) - c[:, :, :-1] * bernoulli_torch(delta)) / dz


def sg_face_flux_r_torch(c, psi, grid, sign):
    dr = grid.dr
    delta = sign * (psi[:, 1:, :] - psi[:, :-1, :])
    return -(c[:, 1:, :] * bernoulli_torch(-delta) - c[:, :-1, :] * bernoulli_torch(delta)) / dr


def sg_current_pA(c_plus, c_minus, psi, mu, grid):
    """Exact SG axial current profile I(z) in pA. Returns (B, nz-1)."""
    grid = grid if isinstance(grid, GridTensors) else GridTensors()
    jp = sg_face_flux_z_torch(c_plus, psi, grid, 1)
    jm = sg_face_flux_z_torch(c_minus, psi, grid, -1)
    w = grid.weights.to(psi.dtype)
    dimless = 2.0 * np.pi * torch.sum((DIFF_CATION * jp - DIFF_ANION * jm) * w[None, :, None], dim=1)
    c0 = mu[:, 1]
    r_m = mu[:, 3] * 1e-9
    scale = FARADAY * c0 * r_m**2 / (L_NM * 1e-9)  # current_scale_A / mean_d
    return dimless * scale[:, None] * 1e12


def poisson_residual(psi, c_plus, c_minus, mu, grid=None):
    grid = grid if isinstance(grid, GridTensors) else GridTensors()
    raw = raw_features(mu)
    alpha_r = raw[:, 4].reshape(-1, 1, 1)
    beta = raw[:, 5].reshape(-1, 1, 1)

    r = grid.r.to(psi.dtype)
    w = grid.weights.to(psi.dtype)
    dr, dz = grid.dr, grid.dz

    psi_i = psi[:, 1:-1, 1:-1]
    psi_im1 = psi[:, :-2, 1:-1]
    psi_ip1 = psi[:, 2:, 1:-1]
    coef_im1 = ((r[1:-1] - dr / 2.0) / dr).reshape(1, -1, 1)
    coef_ip1 = ((r[1:-1] + dr / 2.0) / dr).reshape(1, -1, 1)
    w_i = w[1:-1].reshape(1, -1, 1)
    radial = alpha_r * (coef_im1 * (psi_i - psi_im1) + coef_ip1 * (psi_i - psi_ip1)) / w_i

    axial = (2.0 * psi_i - psi[:, 1:-1, :-2] - psi[:, 1:-1, 2:]) / dz**2
    rhs = beta * (c_plus[:, 1:-1, 1:-1] - c_minus[:, 1:-1, 1:-1])
    return axial + radial - rhs


def continuity_residual(c, psi, mu, grid=None, sign=1):
    grid = grid if isinstance(grid, GridTensors) else GridTensors()
    alpha_r = raw_features(mu)[:, 4].reshape(-1, 1, 1)
    jz = sg_face_flux_z_torch(c, psi, grid, sign)   # (B,nr,nz-1)
    jr = sg_face_flux_r_torch(c, psi, grid, sign)   # (B,nr-1,nz)
    edges = grid.edges.to(psi.dtype)
    r = grid.r.to(psi.dtype)
    dr, dz = grid.dr, grid.dz

    d_z = (jz[:, :, 1:] - jz[:, :, :-1]) / dz                       # (B,nr,nz-2)
    ep = edges[2:grid.nr]      # face radius i+1 for i=1..nr-2
    em = edges[1:grid.nr - 1]  # face radius i   for i=1..nr-2
    num = ep.reshape(1, -1, 1) * jr[:, 1:, :] - em.reshape(1, -1, 1) * jr[:, :-1, :]
    d_r = num / (r[1:-1].reshape(1, -1, 1) * dr)                    # (B,nr-2,nz)
    return alpha_r * d_r[:, :, 1:-1] + d_z[:, 1:-1, :]


def wall_bc_residual(psi, mu, grid=None):
    """Radial potential derivative at the wall must equal q_w."""
    grid = grid if isinstance(grid, GridTensors) else GridTensors()
    q_w = raw_features(mu)[:, 2].reshape(-1, 1)
    dpsi_dr = (psi[:, -1, :] - psi[:, -2, :]) / grid.dr
    return dpsi_dr - q_w


def wall_flux_residual(c, psi, grid=None, sign=1):
    """Zero radial SG ion flux at the wall (last radial face)."""
    grid = grid if isinstance(grid, GridTensors) else GridTensors()
    jr = sg_face_flux_r_torch(c, psi, grid, sign)
    return jr[:, -1, :]


def normalized_mse(pred, target, scale):
    return ((pred - target) / scale) ** 2


def operator_data_loss(pred, target, scales, fields=("psi", "c_plus", "c_minus")):
    """Mean normalized field MSE over the requested fields.

    pred/target: dict of (B,nr,nz) tensors; scales: dict of scalar scales.
    """
    total = 0.0
    for name in fields:
        total = total + normalized_mse(pred[name], target[name], scales[name]).mean()
    return total / len(fields)


def net_charge_loss(pred, target, scale=1.0):
    return normalized_mse(pred["rho"], target["rho"], scale).mean()


def pnp_physics_loss(pred, mu, grid=None, w_wall=1.0):
    grid = grid if isinstance(grid, GridTensors) else GridTensors()
    psi, cp, cm = pred["psi"], pred["c_plus"], pred["c_minus"]
    res_poisson = poisson_residual(psi, cp, cm, mu, grid).abs().mean()
    res_cont = 0.5 * (
        continuity_residual(cp, psi, mu, grid, 1).abs().mean()
        + continuity_residual(cm, psi, mu, grid, -1).abs().mean()
    )
    res_wall = (
        wall_bc_residual(psi, mu, grid).abs().mean()
        + wall_flux_residual(cp, psi, grid, 1).abs().mean()
        + wall_flux_residual(cm, psi, grid, -1).abs().mean()
    )
    return {
        "poisson": res_poisson,
        "continuity": res_cont,
        "wall": w_wall * res_wall,
    }
