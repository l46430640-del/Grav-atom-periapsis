"""Leading Coulomb continuum loss for inclined circular and eccentric orbits.

Rotate the bare 211 parent into the orbital frame. Circular harmonics have
one initial m for a given final m, so their rotated populations add
incoherently. Eccentric projections retain all three parent amplitudes.
"""

import math
import time
from functools import lru_cache

import numpy as np

from .research_cloud_scattering import coulomb_waves
from .research_collective_mode import angular
from .research_rydberg_storage import radial
from .research_tidal_continuum import source_grid
from .source_normalization import directed_companion_coefficient


def rotation_weights(inclination):
    z = np.cos(inclination)
    return {-1:(1-z)**2/4, 0:(1-z*z)/2, 1:(1+z)**2/4}


def circular_harmonic(radius, harmonic, *, extra_ell=6, order=32,
                      cutoff=80., tolerance=2e-11, binding_energy=-.125,
                      orbital_frequency=None):
    """Coefficients Gamma/N = q**2 alpha**3/M C for initial m=-1,0,1.

    Optional energies change the on-shell Coulomb energy. The source remains
    hydrogenic and no finite-cloud potential is added to propagation.
    """
    if (not isinstance(harmonic, int) or harmonic < 1 or extra_ell < 0
            or extra_ell % 2):
        raise ValueError("Require positive harmonic and even angular extension")
    frequency = radius**(-1.5) if orbital_frequency is None else orbital_frequency
    energy = binding_energy+harmonic*frequency
    if energy <= 0:
        raise ValueError("The selected harmonic is closed")
    grid = source_grid(radius, order, cutoff)
    r, w = grid.r, grid.w
    ls = list(range(harmonic-1, harmonic+extra_ell+2, 2))
    waves = coulomb_waves(grid, ls, energy, tolerance=tolerance)
    parent = radial(2, 1, r)
    channels = []
    for initial_m in (-1, 0, 1):
        for ell in ls:
            final_m = initial_m+harmonic
            if abs(final_m) > ell:
                continue
            source = np.zeros_like(r, dtype=complex)
            for rank in (ell-1, ell+1):
                if rank < harmonic:
                    continue
                coefficient = directed_companion_coefficient(rank, harmonic)*angular(
                    ell, final_m, 1, initial_m, rank, harmonic)
                profile = np.minimum(r, radius)**rank/np.maximum(r, radius)**(rank+1)
                if rank == 1:
                    # Acceleration of the central black hole cancels the inner dipole.
                    profile -= r/radius**2
                source -= coefficient*profile
            amplitude = np.dot(w*r*r*parent*waves[ell]['normalized'], source)
            channels.append(dict(initial_m=initial_m, ell=ell, final_m=final_m,
                amplitude=[float(amplitude.real), float(amplitude.imag)],
                coefficient=float(2*math.pi*abs(amplitude)**2)))
    return dict(radius=radius, harmonic=harmonic, energy=energy, extra_ell=extra_ell,
        order=order, cutoff=cutoff, tolerance=tolerance, binding_energy=binding_energy,
        orbital_frequency=frequency, channels=channels,
        coefficients={str(m):sum(c['coefficient'] for c in channels if c['initial_m'] == m)
                      for m in (-1, 0, 1)})


def inclined_sum(harmonics, inclination):
    weights = rotation_weights(inclination)
    rates = [sum(weights[m]*row['coefficients'][str(m)] for m in weights)
             for row in harmonics]
    return dict(inclination=float(inclination), population_coefficient=float(sum(rates)),
        energy_coefficient=float(sum(row['harmonic']*rate for row, rate in zip(harmonics, rates))),
        harmonic_coefficients=[dict(harmonic=row['harmonic'], coefficient=float(rate))
                               for row, rate in zip(harmonics, rates)])


def rotated_amplitudes(inclination, omega):
    """Y_11 in the periapsis frame, using R_x(I) R_z(omega)."""
    z = np.cos(inclination)
    return np.array([-(1-z)*np.exp(-1j*omega)/2,
                     1j*np.sin(inclination)/np.sqrt(2),
                     (1+z)*np.exp(1j*omega)/2])


def eccentric_projection(a, eccentricity, *, maximum_harmonic=20, ell_max=10,
                          order=32, anomaly_order=48, cutoff=80., tolerance=2e-11,
                          budget=240, binding_energy=-.125, parent_radial=None,
                          continuum_waves=None):
    """Real Born amplitudes resolved in the three rotated parent components.

    Orbital Fourier integrals are split where the companion radius equals
    each field radius. The reported kernels include the central recoil.
    """
    if not 0 <= eccentricity < .9 or maximum_harmonic < 1 or ell_max < 2:
        raise ValueError("Require moderate eccentricity and positive cutoffs")
    start = time.perf_counter()
    e = eccentricity
    grid = source_grid(a, order, cutoff, breakpoints=(a*(1-e), a*(1+e)))
    r = grid.r
    harmonics = np.array([h for h in range(1, maximum_harmonic+1)
                         if binding_energy+h*a**(-1.5) > 0])
    if len(harmonics) == 0:
        raise ValueError("No open harmonics in the selected range")
    kernels = {}
    if e > 0:
        node, weights = np.polynomial.legendre.leggauss(anomaly_order)
        split = np.arccos(np.clip((1-r/a)/e, -1, 1))
        widths = np.column_stack([split, np.pi-split])/2
        centers = np.column_stack([split, np.pi+split])/2
        anomaly = (centers[:, :, None]+widths[:, :, None]*node).reshape(len(r), -1)
        quadrature = (widths[:, :, None]*weights/np.pi).reshape(len(r), -1)
        measure = 1-e*np.cos(anomaly)
        orbital_r = a*measure
        mean = anomaly-e*np.sin(anomaly)
        true = np.arctan2(np.sqrt(1-e*e)*np.sin(anomaly), np.cos(anomaly)-e)
    for rank in range(ell_max+2):
        if time.perf_counter()-start > budget:
            raise TimeoutError("Eccentric projection exceeded its time budget")
        if e > 0:
            profile = (np.minimum(r[:, None], orbital_r)**rank
                       /np.maximum(r[:, None], orbital_r)**(rank+1))
            if rank == 1:
                profile -= r[:, None]/orbital_r**2
        for magnetic in range(-rank, rank+1, 2):
            if e == 0:
                profile = np.minimum(r, a)**rank/np.maximum(r, a)**(rank+1)
                if rank == 1:
                    profile -= r/a**2
                kernels[rank, magnetic] = (harmonics[:, None] == magnetic)*profile
            else:
                phase = harmonics[:, None, None]*mean[None, :, :]-magnetic*true[None, :, :]
                kernels[rank, magnetic] = np.einsum('hrk,rk->hr',
                    np.cos(phase), quadrature*measure*profile)
    rows = []
    parent = radial(2, 1, r) if parent_radial is None else parent_radial(r)
    wave_provider = coulomb_waves if continuum_waves is None else continuum_waves
    for index, h in enumerate(harmonics):
        if time.perf_counter()-start > budget:
            raise TimeoutError("Eccentric projection exceeded its time budget")
        energy = binding_energy+h*a**(-1.5)
        waves = wave_provider(grid, list(range(ell_max+1)), energy, tolerance=tolerance)
        channels = []
        for ell in range(ell_max+1):
            radial_weight = grid.w*r*r*parent*waves[ell]['normalized']
            for final_m in range(-ell, ell+1):
                amplitudes = []
                for initial_m in (-1, 0, 1):
                    amplitude = 0j
                    magnetic = final_m-initial_m
                    for rank in (ell-1, ell+1):
                        if (rank, magnetic) not in kernels:
                            continue
                        coefficient = directed_companion_coefficient(rank, magnetic)*angular(
                            ell, final_m, 1, initial_m, rank, magnetic)
                        amplitude -= coefficient*np.dot(radial_weight, kernels[rank, magnetic][index])
                    if abs(amplitude.imag) > 1e-12:
                        raise ArithmeticError("The periapsis-frame amplitude should be real")
                    amplitudes.append(float(amplitude.real))
                channels.append(dict(ell=ell, final_m=final_m, amplitudes=amplitudes))
        rows.append(dict(harmonic=int(h), energy=float(energy), channels=channels))
    return dict(a=a, eccentricity=e, maximum_harmonic=maximum_harmonic,
        ell_max=ell_max, order=order, anomaly_order=anomaly_order, cutoff=cutoff,
        tolerance=tolerance, binding_energy=binding_energy, harmonics=rows, seconds=time.perf_counter()-start)


def coherent_flux(projection, inclination, omega):
    """C_N and C_E, plus the same flux with uniform periapsis averaging."""
    rotation = rotated_amplitudes(inclination, omega)
    rows = []
    for row in projection['harmonics']:
        amplitudes = np.array([channel['amplitudes'] for channel in row['channels']])
        coherent = 2*np.pi*np.sum(abs(amplitudes@rotation)**2)
        averaged = 2*np.pi*np.sum(abs(amplitudes)**2*abs(rotation)**2)
        rows.append(dict(harmonic=row['harmonic'], coherent=float(coherent), averaged=float(averaged)))
    return dict(inclination=float(inclination), omega=float(omega),
        population_coefficient=sum(row['coherent'] for row in rows),
        energy_coefficient=sum(row['harmonic']*row['coherent'] for row in rows),
        averaged_population_coefficient=sum(row['averaged'] for row in rows),
        averaged_energy_coefficient=sum(row['harmonic']*row['averaged'] for row in rows),
        harmonics=rows)


@lru_cache(maxsize=32)
def angular_matrices(ell):
    m = np.arange(-ell, ell+1)
    plus = np.diag(np.sqrt(ell*(ell+1)-m[:-1]*(m[:-1]+1)), -1)
    return ((plus+plus.T)/2, (plus-plus.T)/(2j), np.diag(m))


def torque_coefficients(projection, inclination, omega):
    """Scalar torque in the periapsis frame, including bound-state loss.

    Multiply by epsilon*q**2*alpha**2*M for dJ_scalar/dt. Orbital reaction
    has the opposite sign. The incoming term is the anticommutator with
    the loss matrix; subtracting one fixed initial m is insufficient.
    """
    c = rotated_amplitudes(inclination, omega)
    loss = np.zeros((3, 3))
    outgoing = np.zeros(3)
    for row in projection['harmonics']:
        for ell in range(projection['ell_max']+1):
            block = np.array([channel['amplitudes'] for channel in row['channels']
                              if channel['ell'] == ell])
            loss += 2*np.pi*block.T@block
            amplitude = block@c
            outgoing += 2*np.pi*np.array([np.vdot(amplitude, matrix@amplitude).real
                                         for matrix in angular_matrices(ell)])
    incoming = -np.array([np.vdot(c, matrix@loss@c).real for matrix in angular_matrices(1)])
    total = outgoing+incoming
    cloud_axis = np.array([np.sin(inclination)*np.sin(omega),
                          np.sin(inclination)*np.cos(omega), np.cos(inclination)])
    return dict(outgoing=list(outgoing), bound_change=list(incoming), total=list(total),
        cloud_axis_projection=float(np.dot(cloud_axis, total)),
        loss_matrix=list(map(list, loss)),
        number_loss_coefficient=float(np.vdot(c, loss@c).real))
