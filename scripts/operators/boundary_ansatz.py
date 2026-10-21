"""Exact reservoir boundary + positivity ansatz.

Potential (both modes):

    psi(r,z;mu) = psi_app(mu) (1 - z) + z(1-z) G_psi(r,z;mu)

so psi(r,0)=psi_app and psi(r,1)=0 exactly (reservoir Dirichlet).

Mode ``reparam`` (charge-balanced reparameterization):
    rho   = z(1-z) G_rho
    s     = 2 exp(clip(z(1-z) G_s, -15, 15))
    c_pm  = 0.5 (sqrt(s^2 + rho^2) +/- rho)
  => c_pm = 1 exactly at z in {0,1}; c_+ - c_- == rho identically; c_pm > 0
     for all real G_s, G_rho.

Mode ``direct_c`` (ablation / baseline):
    c_pm  = exp(clip(z(1-z) G_pm, -15, 15))
    rho   = c_+ - c_-   (post-hoc subtraction)

Mode ``linear_ortho_projection`` (Stage-11 constraint-scheme benchmark):
    sigma = 2 + z(1-z) G_sigma
    rho_t = z(1-z) G_rho
    c_pm  = Softplus_beta(0.5 (sigma +/- rho_t)) / beta, beta = 20
  => positivity by projection, but c_+ - c_- != rho_t after the projection
     (the algebraic-consistency mismatch is measured as ``rho_latent`` vs rho).
"""

import torch
import torch.nn as nn

from .constants import NZ

_CLIP = 15.0
_SOFTPLUS_BETA = 20.0


def salt_mean(c_plus: torch.Tensor, c_minus: torch.Tensor) -> torch.Tensor:
    """Exact reparam ``s`` target: s = 2 sqrt(c_+ c_-) (geometric salt mean).

    Identity (enforced in training and unit-checked): for c_pm > 0,
        sqrt(s^2 + rho^2) = sqrt(4 c_+ c_- + (c_+ - c_-)^2) = c_+ + c_-
    so reconstruct_concentrations(s, rho) returns c_pm exactly, at zero loss.
    """
    return 2.0 * torch.sqrt(c_plus * c_minus)


def reconstruct_concentrations(s: torch.Tensor, rho: torch.Tensor):
    """Invert (s, rho) -> (c_+, c_-) with the algebraic identity above."""
    root = torch.sqrt(s * s + rho * rho)
    return 0.5 * (root + rho), 0.5 * (root - rho)


class BoundaryAnsatz(nn.Module):
    def __init__(self, mode: str = "reparam", nz: int = NZ):
        super().__init__()
        if mode not in ("reparam", "direct_c", "linear_ortho_projection"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        z = torch.linspace(0.0, 1.0, nz, dtype=torch.float32)
        self.register_buffer("z", z.reshape(1, 1, 1, nz))

    def forward(self, raw: torch.Tensor, psi_app: torch.Tensor,
                c_left: torch.Tensor | None = None,
                c_right: torch.Tensor | None = None) -> dict:
        """raw: (B, 3, nr, nz); psi_app: (B,) dimensionless.

        Optional unequal reservoir concentrations ``c_left``/``c_right``
        (dimensionless, shape (B,)) preserve the exact boundary values.
        """
        z = self.z.to(raw.dtype)
        env = z * (1.0 - z)
        psi_app = psi_app.reshape(-1, 1, 1, 1)

        psi = psi_app * (1.0 - z) + env * raw[:, 0:1]
        g2 = raw[:, 1:2]
        g3 = raw[:, 2:3]

        if self.mode == "reparam":
            rho = env * g3
            if c_left is None and c_right is None:
                salt_scale = 1.0
            else:
                if c_left is None or c_right is None:
                    raise ValueError("both c_left and c_right are required")
                cl = c_left.reshape(-1, 1, 1, 1)
                cr = c_right.reshape(-1, 1, 1, 1)
                salt_scale = cl * (1.0 - z) + cr * z
            s = 2.0 * salt_scale * torch.exp(torch.clamp(env * g2, -_CLIP, _CLIP))
            root = torch.sqrt(s * s + rho * rho)
            c_plus = 0.5 * (root + rho)
            c_minus = 0.5 * (root - rho)
            rho_latent = rho
        elif self.mode == "linear_ortho_projection":
            sigma = 2.0 + env * g2
            rho_latent = env * g3
            b = _SOFTPLUS_BETA
            c_plus = torch.nn.functional.softplus(b * 0.5 * (sigma + rho_latent)) / b
            c_minus = torch.nn.functional.softplus(b * 0.5 * (sigma - rho_latent)) / b
            rho = c_plus - c_minus
            s = c_plus + c_minus
        else:
            c_plus = torch.exp(torch.clamp(env * g2, -_CLIP, _CLIP))
            c_minus = torch.exp(torch.clamp(env * g3, -_CLIP, _CLIP))
            rho = c_plus - c_minus
            s = c_plus + c_minus
            rho_latent = rho

        return {
            "psi": psi.squeeze(1),
            "c_plus": c_plus.squeeze(1),
            "c_minus": c_minus.squeeze(1),
            "rho": rho.squeeze(1),
            "s": s.squeeze(1),
            "rho_latent": rho_latent.squeeze(1),
        }
