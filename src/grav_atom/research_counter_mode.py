"""Resolved counterrotating collective mode of a finite 211 cloud.

The Newtonian conservative Hessian includes the conjugate m=3 partner.
This diagnoses the tidal gap and overlap; it is not a dressed dissipative
three-cloud history or a relativistic response calculation.
"""
import math

import numpy as np
from scipy.linalg import eig, eigh

from .research_collective_mode import (
    RadialGrid, make_sector, stationary_parent, transition_density,
    momentum_projection, angular)
from .research_radial_response import normal_from_background
from .research_moving_resonance import EXCHANGE


def paired_operator(grid, parent, upper, lower, fraction):
    background = stationary_parent(grid, parent, fraction)
    return paired_from_background(grid, parent, upper, lower, fraction, background)


def paired_from_background(grid, parent, upper, lower, fraction, background):
    """Build the paired Hessian on a supplied real parent profile."""
    if parent.m != 1 or upper.m+lower.m != 2 or upper.m == 1:
        raise ValueError('Require distinct conjugate sectors around the m=1 parent')
    ku = normal_from_background(grid, upper, parent, background)
    kv = normal_from_background(grid, lower, parent, background)
    coupling = np.zeros((len(ku), len(kv)))
    weights = grid.w*grid.r**2
    order = upper.m-parent.m
    parity = (-1.)**order
    reciprocity_error = 0.
    for rank in range(abs(order), max(max(upper.ls), max(lower.ls))+max(parent.ls)+1):
        fu = transition_density(upper, parent, background['fields'], rank)
        fv = transition_density(lower, parent, background['fields'], rank)
        coefficient = -parity*4*math.pi/(2*rank+1)
        uv = coefficient*fu.T@(weights[:, None]*grid.potential(fv, rank))
        vu = coefficient*fv.T@(weights[:, None]*grid.potential(fu, rank))
        reciprocity_error = max(reciprocity_error, float(np.max(abs(uv-vu.T))))
        coupling += (uv+vu.T)/2
    tu = momentum_projection(grid, upper, parent, background)
    tv = momentum_projection(grid, lower, parent, background)
    coupling += np.real(tu@tv.T)
    au = upper.h0-background['chemical']*np.eye(len(ku))+fraction*ku
    av = lower.h0-background['chemical']*np.eye(len(kv))+fraction*kv
    matrix = np.block([[au, fraction*coupling], [-fraction*coupling.T, -av]])
    return dict(matrix=matrix, upper=au, lower=av, coupling=coupling,
        background=background, reciprocity_error=reciprocity_error)


def configuration(size=20, order=40, maximum_parent_l=3):
    if maximum_parent_l not in (1, 3, 5):
        raise ValueError('Use an odd parent angular cutoff no larger than 5')
    grid = RadialGrid(order)
    parent = make_sector(grid, 1, tuple(range(1, maximum_parent_l+1, 2)), size)
    upper = make_sector(grid, -1, tuple(range(1, maximum_parent_l+1, 2)), size)
    lower = make_sector(grid, 3, tuple(range(3, maximum_parent_l+3, 2)), size)
    return grid, parent, upper, lower


def quadrupole_projection(grid, sector, parent, fields):
    order = sector.m-parent.m
    return np.concatenate([sector.functions[l].T@(grid.w*grid.r**4*sum(
        angular(l, sector.m, a, parent.m, 2, order)*fields[a] for a in parent.ls))
        for l in sector.ls])


def resolved_counter_mode(config, fraction):
    grid, parent, upper, lower = config
    operator = paired_operator(grid, parent, upper, lower, fraction)
    reference = eigh(upper.h0)[1][:, 0]
    values, vectors = eig(operator['matrix'])
    split = len(reference)
    overlaps = abs(reference@vectors[:split])**2
    index = int(np.argmax(overlaps))
    value, vector = values[index], vectors[:, index]
    norm = float(np.vdot(vector[:split], vector[:split]).real
                 -np.vdot(vector[split:], vector[split:]).real)
    if norm <= 0 or abs(value.imag) > 1e-10:
        raise ArithmeticError('Counterrotating branch lost real positive-norm character')
    vector = vector/math.sqrt(norm)
    if np.dot(reference, vector[:split]).real < 0:
        vector *= -1
    return operator, float(value.real), vector.real, reference


def inspect_counter_mode(config, fraction):
    grid, parent, upper, lower = config
    operator, value, vector, reference = resolved_counter_mode(config, fraction)
    split = len(reference)
    fields = operator['background']['fields']
    pu = quadrupole_projection(grid, upper, parent, fields)
    pv = quadrupole_projection(grid, lower, parent, fields)
    element = float((np.dot(vector[:split], pu)+np.dot(vector[split:], pv)).real)
    radial = grid.r*np.exp(-grid.r/2)/(2*math.sqrt(6))
    bare = angular(1, -1, 1, 1, 2, -2)*np.dot(grid.w*grid.r**4, radial**2)
    fixed_values, fixed_vectors = eigh(operator['upper'])
    fixed_index = int(np.argmax(abs(reference@fixed_vectors)**2))
    return dict(fraction=fraction, frequency=value,
        gap=-value, leading_gap=EXCHANGE*fraction,
        gap_ratio=(None if fraction == 0 else -value/(EXCHANGE*fraction)),
        normal_frequency=float(fixed_values[fixed_index]),
        conjugate_frequency_shift=float(value-fixed_values[fixed_index]),
        conjugate_norm=float(np.vdot(vector[split:], vector[split:]).real),
        krein_norm=float(np.vdot(vector[:split], vector[:split]).real
                         -np.vdot(vector[split:], vector[split:]).real),
        quadrupole_element=element, bare_quadrupole_element=float(bare),
        quadrupole_ratio=float(abs(element/bare)),
        background_residual=operator['background']['residual'],
        mode_residual=float(np.linalg.norm(operator['matrix']@vector-value*vector)),
        reciprocity_error=operator['reciprocity_error'],
        overlap_error=max(s.overlap_error for s in [parent, upper, lower]))
