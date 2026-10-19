"""Parameter-conditioned Exact-Boundary PI-DeepONet.

Branch net encodes the normalized parameter features; trunk net encodes
coordinates (r, z) with multi-scale radial features to resolve the EDL near the
wall r=1. Inner products produce the three raw fields, passed through the same
exact-boundary ansatz as the FNO.
"""

import numpy as np
import torch
import torch.nn as nn

from .boundary_ansatz import BoundaryAnsatz
from .constants import NR, NZ, make_grid
from .features import ParamNormalizer

_RADIAL_SCALES = (1.0, 3.0, 10.0, 30.0)


def _mlp(dims):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers.append(nn.GELU())
    return nn.Sequential(*layers)


class DeepONet(nn.Module):
    def __init__(self, mode="reparam", normalizer=None, p=64, branch_width=64,
                 trunk_width=64, depth=3, n_outputs=3, n_params=7, nr=NR, nz=NZ):
        super().__init__()
        if normalizer is None:
            normalizer = ParamNormalizer([0.0] * n_params, [1.0] * n_params)
        self.mode = mode
        self.norm = normalizer
        self.nr, self.nz = nr, nz
        self.n_params = n_params
        self.p = p
        self.n_outputs = n_outputs

        trunk_in = 2 + len(_RADIAL_SCALES)
        self.branch = _mlp([n_params] + [branch_width] * depth + [n_outputs * p])
        self.trunk = _mlp([trunk_in] + [trunk_width] * depth + [p])
        self.bias = nn.Parameter(torch.zeros(n_outputs))

        r, z, *_ = make_grid(nr, nz)
        rn = 2.0 * r - 1.0
        zn = 2.0 * z - 1.0
        rr, zz = np.meshgrid(rn, zn, indexing="ij")
        coords = np.stack([rr, zz], axis=-1)
        self.register_buffer("grid_coords", torch.tensor(coords, dtype=torch.float32))

        self.ansatz = BoundaryAnsatz(mode, nz)

    def _trunk_features(self, coords):
        r = (coords[..., 0] + 1.0) * 0.5   # (nr,nz)
        z = (coords[..., 1] + 1.0) * 0.5
        feats = [r, z]
        for s in _RADIAL_SCALES:
            feats.append(torch.exp(-s * (1.0 - r)))
        return torch.stack(feats, dim=-1)

    def forward(self, mu, coords=None):
        raw, norm = self.norm(mu)
        psi_app = raw[:, 0]
        B = mu.shape[0]

        if coords is None:
            coords = self.grid_coords.to(mu.dtype)

        trunk_feats = self._trunk_features(coords).reshape(-1, 2 + len(_RADIAL_SCALES))
        trunk = self.trunk(trunk_feats)              # (N, p)
        branch = self.branch(norm).reshape(B, self.n_outputs, self.p)

        out = torch.matmul(branch, trunk.t())        # (B, n_outputs, N)
        out = out + self.bias[None, :, None]
        out = out.reshape(B, self.n_outputs, self.nr, self.nz)
        return self.ansatz(out, psi_app)
