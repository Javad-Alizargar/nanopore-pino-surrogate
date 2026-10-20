from dataclasses import dataclass

import numpy as np

from .config import GeometryConfig


@dataclass(frozen=True)
class AxisymmetricGrid:
    r: np.ndarray
    z: np.ndarray
    rr: np.ndarray
    zz: np.ndarray
    dr: float
    dz: float
    radial_metric: float

    @property
    def shape(self) -> tuple[int, int]:
        return self.rr.shape


def make_grid(geometry: GeometryConfig) -> AxisymmetricGrid:
    r = np.linspace(0.0, 1.0, geometry.nr)
    z = np.linspace(0.0, 1.0, geometry.nz)
    rr, zz = np.meshgrid(r, z, indexing="ij")
    return AxisymmetricGrid(
        r=r,
        z=z,
        rr=rr,
        zz=zz,
        dr=float(r[1] - r[0]),
        dz=float(z[1] - z[0]),
        radial_metric=float((geometry.length_nm / geometry.radius_nm) ** 2),
    )
