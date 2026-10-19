from dataclasses import dataclass

import torch
import torch.nn as nn


class PnpPINN(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        in_features = 2
        for _ in range(depth):
            layers.append(nn.Linear(in_features, width))
            layers.append(nn.SiLU())
            in_features = width
        layers.append(nn.Linear(width, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, r: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([r, z], dim=1))


@dataclass(frozen=True)
class AnsatzConfig:
    left_psi: float
    right_psi: float = 0.0
    bulk_c: float = 1.0


def predict_fields(model: PnpPINN, r: torch.Tensor, z: torch.Tensor, cfg: AnsatzConfig):
    raw = model(r, z)
    envelope = z * (1.0 - z)
    psi_bg = cfg.left_psi * (1.0 - z) + cfg.right_psi * z
    psi = psi_bg + envelope * raw[:, 0:1]
    c_plus = cfg.bulk_c * torch.exp(torch.clamp(envelope * raw[:, 1:2], -20.0, 20.0))
    c_minus = cfg.bulk_c * torch.exp(torch.clamp(envelope * raw[:, 2:3], -20.0, 20.0))
    return psi, c_plus, c_minus


def grad(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return torch.autograd.grad(y, x, torch.ones_like(y), create_graph=True, retain_graph=True)[0]


def axisymmetric_divergence(
    jr: torch.Tensor,
    jz: torch.Tensor,
    r: torch.Tensor,
    z: torch.Tensor,
    radial_metric: float,
) -> torch.Tensor:
    r_safe = torch.where(r <= 0, torch.full_like(r, 1e-5), r)
    derivative = grad(jr, r)
    radial = torch.where(r == 0, 2.0 * derivative, derivative + jr / r_safe)
    return radial_metric * radial + grad(jz, z)


def pnp_residuals(
    model: PnpPINN,
    r: torch.Tensor,
    z: torch.Tensor,
    ansatz: AnsatzConfig,
    beta: float,
    radial_metric: float,
):
    r = r.clone().detach().requires_grad_(True)
    z = z.clone().detach().requires_grad_(True)
    psi, c_plus, c_minus = predict_fields(model, r, z, ansatz)

    dpsi_dr = grad(psi, r)
    dpsi_dz = grad(psi, z)
    d2psi_dr2 = grad(dpsi_dr, r)
    d2psi_dz2 = grad(dpsi_dz, z)
    r_safe = torch.where(r <= 0, torch.full_like(r, 1e-5), r)
    radial_laplacian = torch.where(r == 0, 2.0 * d2psi_dr2, d2psi_dr2 + dpsi_dr / r_safe)
    poisson = radial_metric * radial_laplacian + d2psi_dz2 + beta * (c_plus - c_minus)

    dcp_dr = grad(c_plus, r)
    dcp_dz = grad(c_plus, z)
    dcm_dr = grad(c_minus, r)
    dcm_dz = grad(c_minus, z)

    jp_r = -(dcp_dr + c_plus * dpsi_dr)
    jp_z = -(dcp_dz + c_plus * dpsi_dz)
    jm_r = -(dcm_dr - c_minus * dpsi_dr)
    jm_z = -(dcm_dz - c_minus * dpsi_dz)

    np_plus = axisymmetric_divergence(jp_r, jp_z, r, z, radial_metric)
    np_minus = axisymmetric_divergence(jm_r, jm_z, r, z, radial_metric)
    return {
        "poisson": poisson,
        "np_plus": np_plus,
        "np_minus": np_minus,
        "psi": psi,
        "c_plus": c_plus,
        "c_minus": c_minus,
        "jp_z": jp_z,
        "jm_z": jm_z,
    }


def sample_collocation(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    r = torch.rand((n, 1), generator=generator)
    z = torch.rand((n, 1), generator=generator)
    return r, z
