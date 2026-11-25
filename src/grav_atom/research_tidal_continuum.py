"""Open higher harmonics of a hydrogenic 211 cloud on a retrograde orbit.

These are weak-field Born rates at fixed orbit. They supply an external
drift for studying a different, still-bound tidal harmonic.
"""
from types import SimpleNamespace
import math

import numpy as np

from .research_cloud_scattering import coulomb_waves
from .research_collective_mode import angular
from .research_rydberg_storage import radial
from .source_normalization import directed_companion_coefficient


def source_grid(radius, order=32, cutoff=80., breakpoints=()):
    if radius <= 0 or cutoff <= max(16., radius) or order < 12:
        raise ValueError('Require a positive orbit inside the integration range')
    if any(point <= 0 or point >= cutoff for point in breakpoints):
        raise ValueError('Additional radial breakpoints must be inside the domain')
    edges = np.unique(np.r_[0., 2.**np.arange(-1, math.ceil(math.log2(cutoff))+1),
                            cutoff, radius, breakpoints])
    edges = edges[edges <= cutoff]
    nodes, weights = np.polynomial.legendre.leggauss(order)
    half = np.diff(edges)/2
    r = ((edges[1:]+edges[:-1])[:, None]/2+half[:, None]*nodes).ravel()
    w = (half[:, None]*weights).ravel()
    return SimpleNamespace(r=r, w=w, half=half)


def open_harmonic(radius, harmonic, *, extra_ell=6, order=32, cutoff=80., tolerance=2e-11):
    """Return C_h with I_h/N_parent=C_h q^2 alpha^3/M, h=-g>=2.

    Energy in Bohr binding units is e=-1/8+h*R^(-3/2). Dipole recoil is
    absent for these harmonics; inner and outer companion multipoles remain.
    """
    if not isinstance(harmonic, int) or harmonic < 2 or extra_ell < 0 or extra_ell % 2:
        raise ValueError('Require h>=2 and an even nonnegative angular extension')
    energy = -1/8+harmonic*radius**(-1.5)
    if energy <= 0:
        raise ValueError('This harmonic must be above its ionization threshold')
    grid = source_grid(radius, order, cutoff)
    r, w = grid.r, grid.w
    magnetic = 1-harmonic
    ls = list(range(harmonic-1, harmonic+extra_ell, 2))
    waves = coulomb_waves(grid, ls, energy, tolerance=tolerance)
    parent = radial(2, 1, r)
    rows = []
    for ell in ls:
        potential = np.zeros(len(r), complex)
        for rank in (ell-1, ell+1):
            if rank < harmonic:
                continue
            coefficient = (directed_companion_coefficient(rank, -harmonic)
                           *angular(ell, magnetic, 1, 1, rank, -harmonic))
            potential -= coefficient*np.minimum(r, radius)**rank/np.maximum(r, radius)**(rank+1)
        amplitude = np.dot(w*r*r*parent*waves[ell]['normalized'], potential)
        rows.append(dict(ell=ell, amplitude=[float(amplitude.real), float(amplitude.imag)],
                         coefficient=float(2*np.pi*abs(amplitude)**2)))
    return dict(harmonic=harmonic, radius=radius, energy=energy, order=order,
                cutoff=cutoff, extra_ell=extra_ell, tolerance=tolerance,
                coefficient=sum(row['coefficient'] for row in rows), channels=rows)


def open_sum(radius, *, maximum_harmonic=10, **options):
    rows = [open_harmonic(radius, h, **options) for h in range(3, maximum_harmonic+1)]
    return dict(radius=radius, harmonics=rows,
                population_coefficient=sum(row['coefficient'] for row in rows),
                torque_coefficient=sum(row['harmonic']*row['coefficient'] for row in rows))


def external_drift(alpha, q, cloud, radius, torque_coefficient):
    """d[2 M|Omega|]/d(t/M), excluding the target h=2 bound resonance."""
    if min(alpha, q, radius) <= 0 or min(cloud, torque_coefficient) < 0:
        raise ValueError('Invalid drift scales')
    u = alpha**3*radius**(-1.5)
    gw = 2*(96/5)*q*u**(11/3)
    ion = 6*cloud*q*alpha**2*u**(4/3)*torque_coefficient
    return dict(omega=u, gw=gw, ionization=ion, total=gw+ion)
