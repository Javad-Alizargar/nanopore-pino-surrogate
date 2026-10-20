"""Shared physical constants and grid specification for the operator package.

All quantities are dimensionless / SI as noted. The reference configuration is
the v0.007 SG/FV benchmark: L=30 nm, T=298.15 K, relative permittivity 78.5.
"""

import math

import numpy as np
import torch

FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
EPSILON_0 = 8.8541878128e-12
REL_PERMITTIVITY = 78.5
T_K = 298.15

# Dimensionless thermal potential V_T = R T / F  (volts per unit psi)
V_T = GAS_CONSTANT * T_K / FARADAY

L_NM = 30.0
NR = 21
NZ = 61

DIFF_CATION = 1.96e-9
DIFF_ANION = 2.03e-9

# lambda_D = sqrt(A_LAMBDA_M2 / C0) with C0 in mol/m^3 and lambda_D in metres
A_LAMBDA_M2 = EPSILON_0 * REL_PERMITTIVITY * GAS_CONSTANT * T_K / (2.0 * FARADAY**2)

FEATURE_NAMES = [
    "psi_app",
    "log10_C0",
    "q_w",
    "R_nm",
    "alpha_r",
    "beta",
    "lambda_D_over_R",
]


def make_grid(nr: int = NR, nz: int = NZ):
    """Return (r, z, rr, zz, dr, dz) on the unit square (outer radius r=1)."""
    r = np.linspace(0.0, 1.0, nr)
    z = np.linspace(0.0, 1.0, nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    dr = float(r[1] - r[0])
    dz = float(z[1] - z[0])
    return r, z, rr, zz, dr, dz


def annular_weights(r: np.ndarray) -> np.ndarray:
    edges = np.empty(r.size + 1, dtype=float)
    edges[0] = 0.0
    edges[-1] = 1.0
    edges[1:-1] = 0.5 * (r[:-1] + r[1:])
    return 0.5 * (edges[1:] ** 2 - edges[:-1] ** 2)


def grid_coords_normalized(nr: int = NR, nz: int = NZ) -> torch.Tensor:
    """Normalized grid coordinates (nr, nz, 2) mapped to [-1, 1]."""
    r, z, *_ = make_grid(nr, nz)
    rn = 2.0 * r - 1.0
    zn = 2.0 * z - 1.0
    rr, zz = np.meshgrid(rn, zn, indexing="ij")
    return torch.tensor(np.stack([rr, zz], axis=-1), dtype=torch.float32)
