from dataclasses import dataclass
from pathlib import Path

import yaml

from .scales import PhysicalScales, nm_to_m


@dataclass(frozen=True)
class GeometryConfig:
    radius_nm: float
    length_nm: float
    nr: int
    nz: int


@dataclass(frozen=True)
class PhysicsConfig:
    temperature_K: float
    bulk_concentration_mol_m3: float
    relative_permittivity: float
    diffusion_cation_m2_s: float
    diffusion_anion_m2_s: float
    applied_voltage_V: float
    fixed_charge_strength: float
    wall_surface_charge_strength: float = 0.0


@dataclass(frozen=True)
class SolverConfig:
    max_gummel_iter: int
    tolerance: float
    relaxation: float


@dataclass(frozen=True)
class PinnConfig:
    seed: int
    collocation_points: int
    data_points: int
    epochs: int


@dataclass(frozen=True)
class ExperimentConfig:
    geometry: GeometryConfig
    physics: PhysicsConfig
    solver: SolverConfig
    pinn: PinnConfig

    @property
    def scales(self) -> PhysicalScales:
        return PhysicalScales(
            temperature_K=self.physics.temperature_K,
            bulk_concentration_mol_m3=self.physics.bulk_concentration_mol_m3,
            relative_permittivity=self.physics.relative_permittivity,
            length_m=nm_to_m(self.geometry.length_nm),
        )


def load_config(path: str | Path) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    raw["physics"].setdefault("wall_surface_charge_strength", 0.0)
    return ExperimentConfig(
        geometry=GeometryConfig(**raw["geometry"]),
        physics=PhysicsConfig(**raw["physics"]),
        solver=SolverConfig(**raw["solver"]),
        pinn=PinnConfig(**raw["pinn"]),
    )

