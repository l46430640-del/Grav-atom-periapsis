"""Normalization of the companion tide and scalar flux."""
from __future__ import annotations
import math
from scipy.special import sph_harm_y

def directed_companion_coefficient(ell: int, azimuthal: int, theta: float = math.pi/2) -> complex:
    """Companion harmonic only; the source already contains its Gaunt overlap."""
    if ell < 0 or abs(azimuthal) > ell or not 0 <= theta <= math.pi:
        raise ValueError('Invalid companion harmonic')
    return complex(4*math.pi/(2*ell+1)*sph_harm_y(ell, azimuthal, theta, 0).conjugate())


def leading_tidal_flux_scale(*, alpha: float, parent_frequency: float, parent_charge: float,
                            mass_ratio: float, cloud_energy: float, ell: int = 1, azimuthal: int = -1) -> float:
    """Convert the solver's Im(R* Delta Rprime) to physical number flux.

    The omitted KG forcing coefficient is 2 mu^2 q times the companion
    harmonic. The supplied parent has Hermitian KG charge Q, and the cloud
    energy convention is M_c=omega_parent*N. The physical current is twice
    the solver current. Spatial source deformations retain this coefficient;
    this function does not establish their independent metric derivation.
    """
    values = (alpha, parent_frequency, parent_charge, mass_ratio, cloud_energy)
    if not all(math.isfinite(value) for value in values):
        raise ValueError('Normalization inputs must be finite')
    if min(alpha, parent_frequency, parent_charge) <= 0 or min(mass_ratio, cloud_energy) < 0:
        raise ValueError('Invalid frequency, charge or mass normalization')
    coupling = 2*alpha**2*mass_ratio*directed_companion_coefficient(ell, azimuthal)
    occupation = cloud_energy/parent_frequency
    return float(2*occupation/parent_charge*abs(coupling)**2)
