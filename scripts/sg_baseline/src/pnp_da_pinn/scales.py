from dataclasses import dataclass

import numpy as np


FARADAY = 96485.33212
GAS_CONSTANT = 8.314462618
EPSILON_0 = 8.8541878128e-12


@dataclass(frozen=True)
class PhysicalScales:
    temperature_K: float
    bulk_concentration_mol_m3: float
    relative_permittivity: float
    length_m: float

    @property
    def thermal_voltage_V(self) -> float:
        return GAS_CONSTANT * self.temperature_K / FARADAY

    @property
    def permittivity_F_m(self) -> float:
        return EPSILON_0 * self.relative_permittivity

    @property
    def debye_length_m(self) -> float:
        numerator = self.permittivity_F_m * GAS_CONSTANT * self.temperature_K
        denominator = 2.0 * FARADAY**2 * self.bulk_concentration_mol_m3
        return float(np.sqrt(numerator / denominator))

    @property
    def poisson_coupling(self) -> float:
        """Dimensionless beta in laplacian(psi) = -beta * (c_plus - c_minus)."""
        return float((self.length_m / self.debye_length_m) ** 2 / 2.0)


def nm_to_m(value_nm: float) -> float:
    return value_nm * 1e-9


def voltage_to_psi(voltage_V: float, scales: PhysicalScales) -> float:
    return voltage_V / scales.thermal_voltage_V


def psi_to_voltage(psi: np.ndarray, scales: PhysicalScales) -> np.ndarray:
    return psi * scales.thermal_voltage_V
