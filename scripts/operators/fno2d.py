"""Parameter-conditioned Exact-Boundary 2D FNO.

Spectral convolution is implemented with explicit precomputed real cosine/sine
basis matrices on the fixed nr x nz grid (real matmuls only). This is
mathematically identical to a truncated 2D FFT for the retained modes, and is
100% ONNX / onnxruntime-web compatible (no Complex64 ops, no torch.fft).
"""

import math

import numpy as np
import torch
import torch.nn as nn

from .boundary_ansatz import BoundaryAnsatz
from .constants import NR, NZ, make_grid
from .features import ParamNormalizer, raw_features


class SpectralConv2D(nn.Module):
    def __init__(self, in_ch, out_ch, modes_r, modes_z, nr=NR, nz=NZ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.nr, self.nz = nr, nz
        self.modes_r, self.modes_z = modes_r, modes_z

        kr = np.arange(modes_r)[:, None]
        r = np.arange(nr)[None, :]
        kr2 = np.arange(modes_z)[:, None]
        z = np.arange(nz)[None, :]
        self.register_buffer("Cr", torch.tensor(np.cos(2 * np.pi * kr * r / nr), dtype=torch.float32))
        self.register_buffer("Sr", torch.tensor(np.sin(2 * np.pi * kr * r / nr), dtype=torch.float32))
        self.register_buffer("Cz", torch.tensor(np.cos(2 * np.pi * kr2 * z / nz), dtype=torch.float32))
        self.register_buffer("Sz", torch.tensor(np.sin(2 * np.pi * kr2 * z / nz), dtype=torch.float32))

        scale = 1.0 / (in_ch * modes_r * modes_z)
        self.weight_real = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_r, modes_z))
        self.weight_imag = nn.Parameter(scale * torch.randn(in_ch, out_ch, modes_r, modes_z))

    def forward(self, x):
        B, C, nr, nz = x.shape
        Cr, Sr = self.Cr.to(x.dtype), self.Sr.to(x.dtype)
        Cz, Sz = self.Cz.to(x.dtype), self.Sz.to(x.dtype)

        xm = x.reshape(B * C, nr, nz)
        Ar = torch.matmul(Cr, xm)   # (B*C, K1, nz)
        Sx = torch.matmul(Sr, xm)
        Re = torch.matmul(Ar, Cz.t()) - torch.matmul(Sx, Sz.t())
        Im = -(torch.matmul(Sx, Cz.t()) + torch.matmul(Ar, Sz.t()))

        M = self.modes_r * self.modes_z
        Re = Re.reshape(B, C, M).permute(2, 0, 1)   # (M, B, C)
        Im = Im.reshape(B, C, M).permute(2, 0, 1)
        Wr = self.weight_real.reshape(self.in_ch, self.out_ch, M).permute(2, 0, 1).to(x.dtype)
        Wi = self.weight_imag.reshape(self.in_ch, self.out_ch, M).permute(2, 0, 1).to(x.dtype)

        Yre = torch.matmul(Re, Wr) - torch.matmul(Im, Wi)   # (M, B, out)
        Yim = torch.matmul(Re, Wi) + torch.matmul(Im, Wr)

        Yre = Yre.permute(1, 0, 2).reshape(B, self.modes_r, self.modes_z, self.out_ch)
        Yim = Yim.permute(1, 0, 2).reshape(B, self.modes_r, self.modes_z, self.out_ch)
        Yre = Yre.permute(0, 3, 1, 2).reshape(B * self.out_ch, self.modes_r, self.modes_z)
        Yim = Yim.permute(0, 3, 1, 2).reshape(B * self.out_ch, self.modes_r, self.modes_z)

        CrT, SrT = Cr.t(), Sr.t()
        rr = torch.matmul(torch.matmul(CrT, Yre), Cz)
        ss = torch.matmul(torch.matmul(SrT, Yre), Sz)
        sc = torch.matmul(torch.matmul(SrT, Yim), Cz)
        cs = torch.matmul(torch.matmul(CrT, Yim), Sz)
        out = (rr - ss - sc - cs) / (nr * nz)
        return out.reshape(B, self.out_ch, nr, nz)


class FiLM(nn.Module):
    def __init__(self, n_feat, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_feat, width), nn.GELU(), nn.Linear(width, 2 * width)
        )

    def forward(self, feats):
        scale, shift = self.net(feats).chunk(2, dim=-1)
        return scale, shift


class FNO2D(nn.Module):
    def __init__(self, mode="reparam", normalizer=None, width=32, modes_r=10,
                 modes_z=16, n_layers=4, n_params=7, nr=NR, nz=NZ):
        super().__init__()
        if normalizer is None:
            normalizer = ParamNormalizer([0.0] * n_params, [1.0] * n_params)
        self.mode = mode
        self.norm = normalizer
        self.nr, self.nz = nr, nz
        self.n_params = n_params

        r, z, *_ = make_grid(nr, nz)
        rn = 2.0 * r - 1.0
        zn = 2.0 * z - 1.0
        rr, zz = np.meshgrid(rn, zn, indexing="ij")
        coords = np.stack([rr, zz], axis=-1)  # (nr,nz,2)
        self.register_buffer("grid_coords", torch.tensor(coords, dtype=torch.float32))

        self.lift = nn.Conv2d(2 + n_params, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv2D(width, width, modes_r, modes_z, nr, nz) for _ in range(n_layers)]
        )
        self.pointwise = nn.ModuleList([nn.Conv2d(width, width, 1) for _ in range(n_layers)])
        self.film = nn.ModuleList([FiLM(n_params, width) for _ in range(n_layers)])
        self.act = nn.GELU()
        self.proj = nn.Conv2d(width, 3, 1)
        self.ansatz = BoundaryAnsatz(mode, nz)

    def forward(self, mu, coords=None):
        raw, norm = self.norm(mu)
        psi_app = raw[:, 0]
        B = mu.shape[0]

        if coords is None:
            coords = self.grid_coords.to(mu.dtype)
        coord = coords.permute(2, 0, 1).unsqueeze(0).expand(B, -1, -1, -1)

        feat = norm.reshape(B, self.n_params, 1, 1).expand(-1, -1, self.nr, self.nz)
        h = self.lift(torch.cat([coord, feat], dim=1))

        for spec, point, film in zip(self.spectral, self.pointwise, self.film):
            h = self.act(spec(h) + point(h))
            scale, shift = film(norm)
            h = h * (1.0 + scale[..., None, None]) + shift[..., None, None]

        out = self.proj(h)
        return self.ansatz(out, psi_app)
