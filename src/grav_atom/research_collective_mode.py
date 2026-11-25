"""Projected Schrodinger-Poisson modes with central recoil.

The conservative background and Bogoliubov operator follow the same
Newtonian energy functional. Horizon absorption is a first-order flux
functional, not a resolved Kerr inner region. Length/time units are a_B
and 1/E_B. The cloud mass fraction is the nonlinear coupling.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.linalg import eig, eigh
from scipy.special import eval_genlaguerre, gammaln
from sympy.physics.wigner import gaunt


@lru_cache(maxsize=None)
def angular(left_l, left_m, right_l, right_m, rank, order):
    """Integral Y_left* Y_right Y_rank, in Condon-Shortley convention."""
    if left_m != right_m+order:
        return 0.
    return float((-1)**left_m*gaunt(
        left_l, right_l, rank, -left_m, right_m, order))


class RadialGrid:
    def __init__(self, order=40):
        if not isinstance(order, int) or not 12 <= order <= 128:
            raise ValueError('Require 12 to 128 Gauss points per radial panel')
        nodes, weights = np.polynomial.legendre.leggauss(order)
        vandermonde = np.polynomial.legendre.legvander(nodes, order)
        integral = np.empty((order, order))
        integral[:, 0] = nodes+1
        for k in range(1, order):
            integral[:, k] = (vandermonde[:, k+1]-vandermonde[:, k-1])/(2*k+1)
        cardinal = vandermonde[:, :order].T*((2*np.arange(order)+1)/2)[:, None]
        self.local_lower = integral@(cardinal*weights[None, :])
        self.local_upper = weights[None, :]-self.local_lower
        edges = np.array([0., .5, 1., 2., 4., 8., 16., 32., 64., 128., 256., 512.])
        self.half = np.diff(edges)/2
        self.r = ((edges[:-1]+edges[1:])[:, None]/2
                  +self.half[:, None]*nodes).ravel()
        self.w = (self.half[:, None]*weights).ravel()
        self.order = order
        self.panels = len(self.half)
        self.weights = weights

    def cumulative(self, values):
        """Spectral partial integrals within each Gauss panel."""
        values = np.asarray(values)
        scalar = values.ndim == 1
        values = values[:, None] if scalar else values
        panels = values.reshape(self.panels, self.order, -1)
        totals = np.einsum('i,pij,p->pj', self.weights, panels, self.half)
        below = np.vstack([np.zeros_like(totals[:1]), np.cumsum(totals, axis=0)[:-1]])
        above = np.vstack([np.cumsum(totals[::-1], axis=0)[::-1][1:],
                           np.zeros_like(totals[:1])])
        lower = (np.einsum('ik,pkj,p->pij', self.local_lower, panels, self.half)
                 +below[:, None, :]).reshape(len(self.r), -1)
        upper = (np.einsum('ik,pkj,p->pij', self.local_upper, panels, self.half)
                 +above[:, None, :]).reshape(len(self.r), -1)
        return (lower[:, 0], upper[:, 0]) if scalar else (lower, upper)

    def potential(self, density, rank):
        scalar = density.ndim == 1
        radius = self.r if scalar else self.r[:, None]
        lower = self.cumulative(radius**(rank+2)*density)[0]
        upper = self.cumulative(radius**(1-rank)*density)[1]
        return lower/radius**(rank+1)+radius**rank*upper


@dataclass
class Sector:
    m: int
    ls: tuple
    size: int
    functions: dict
    derivatives: dict
    h0: np.ndarray
    overlap_error: float
    origin: np.ndarray

    def block(self, index):
        return slice(index*self.size, (index+1)*self.size)


def make_sector(grid, m, ls, size=20, decay=.5):
    if (not isinstance(size, int) or not 4 <= size <= 48
            or not np.isfinite(decay) or decay <= 0
            or not ls or len(set(ls)) != len(ls)
            or any(not isinstance(l, int) or l < abs(m) for l in ls)):
        raise ValueError('Invalid radial basis or angular sector')
    radius, weights = grid.r, grid.w
    functions, derivatives = {}, {}
    h0 = np.zeros((len(ls)*size, len(ls)*size))
    origin = np.zeros(len(ls)*size)
    error = 0.
    x = 2*decay*radius
    for index, l in enumerate(ls):
        norm = (2*decay)**1.5*np.exp(
            .5*(gammaln(np.arange(size)+1)-gammaln(np.arange(size)+2*l+3)))
        lag = np.column_stack([eval_genlaguerre(n, 2*l+2, x) for n in range(size)])
        dlag = np.column_stack([np.zeros_like(x) if n == 0 else
            -eval_genlaguerre(n-1, 2*l+3, x) for n in range(size)])
        prefactor = x[:, None]**l*np.exp(-x[:, None]/2)*norm[None, :]
        functions[l] = f = prefactor*lag
        derivatives[l] = df = prefactor*(2*decay*dlag
                                       +(l/radius[:, None]-decay)*lag)
        overlap = f.T@((weights*radius**2)[:, None]*f)
        error = max(error, float(np.max(np.abs(overlap-np.eye(size)))))
        block = slice(index*size, (index+1)*size)
        h0[block, block] = (.5*df.T@((weights*radius**2)[:, None]*df)
            +f.T@((weights*(l*(l+1)/2-radius))[:, None]*f))
        if l == 0:
            origin[block] = norm*np.exp(gammaln(np.arange(size)+3)
                                      -gammaln(np.arange(size)+1)-gammaln(3))
    return Sector(m, tuple(ls), size, functions, derivatives, h0, error, origin)


def background_fields(sector, coefficients):
    return {l: sector.functions[l]@coefficients[sector.block(i)]
            for i, l in enumerate(sector.ls)}


def density_potential(grid, parent, fields, contact=0.):
    """Gravity plus contact*density in units of the cloud coupling times E_B."""
    result = {}
    for rank in range(0, 2*max(parent.ls)+1, 2):
        density = np.zeros_like(grid.r)
        for left in parent.ls:
            for right in parent.ls:
                coefficient = angular(left, parent.m, right, parent.m, rank, 0)
                if coefficient:
                    density += coefficient*fields[left]*fields[right]
        result[rank] = (-4*math.pi/(2*rank+1)*grid.potential(density, rank)
                        +contact*density)
    return result


def hartree(grid, sector, potentials):
    result = np.zeros_like(sector.h0)
    weights = grid.w*grid.r**2
    for i, left in enumerate(sector.ls):
        for j, right in enumerate(sector.ls):
            potential = sum(angular(left, sector.m, right, sector.m, rank, 0)*value
                            for rank, value in potentials.items())
            result[sector.block(i), sector.block(j)] = (
                sector.functions[left].T@((weights*potential)[:, None]
                                         *sector.functions[right]))
    return (result+result.T)/2


def stationary_parent(grid, sector, fraction, tolerance=2e-13, max_iterations=80,
                      *, contact=0.):
    if not np.isfinite(fraction) or not 0 <= fraction <= .2:
        raise ValueError('Require cloud mass fraction between zero and 0.2')
    if not np.isfinite(contact):
        raise ValueError('Require a finite contact coupling')
    _, vectors = eigh(sector.h0)
    coefficients = vectors[:, 0]
    for iteration in range(max_iterations):
        fields = background_fields(sector, coefficients)
        potential = density_potential(grid, sector, fields, contact)
        operator = sector.h0+fraction*hartree(grid, sector, potential)
        values, vectors = eigh(operator)
        new = vectors[:, 0]
        if np.dot(new, coefficients) < 0:
            new *= -1
        difference = np.linalg.norm(new-coefficients)
        coefficients = new
        if difference < tolerance:
            break
    else:
        raise ArithmeticError('The projected parent did not converge')
    fields = background_fields(sector, coefficients)
    potential = density_potential(grid, sector, fields, contact)
    interaction = hartree(grid, sector, potential)
    chemical = float(coefficients@(sector.h0+fraction*interaction)@coefficients)
    return dict(coefficients=coefficients, fields=fields, potential=potential,
                chemical=chemical, iterations=iteration+1,
                residual=float(np.linalg.norm(
                    (sector.h0+fraction*interaction)@coefficients-chemical*coefficients)),
                energy=float(coefficients@(sector.h0+.5*fraction*interaction)@coefficients))


def transition_density(sector, parent, fields, rank):
    order = sector.m-parent.m
    columns = []
    for l in sector.ls:
        radial = sum(angular(l, sector.m, a, parent.m, rank, order)*fields[a]
                     for a in parent.ls)
        columns.append(sector.functions[l]*radial[:, None])
    return np.hstack(columns)


def momentum_projection(grid, sector, parent, background):
    """Project -i grad(phi); no finite-basis commutator approximation."""
    result = np.zeros((sector.h0.shape[0], 3), dtype=complex)
    weights = grid.w*grid.r**2
    root = math.sqrt(2*math.pi/3)
    for i, l in enumerate(sector.ls):
        for j, a in enumerate(parent.ls):
            minus = angular(l, sector.m, a, parent.m, 1, -1)
            plus = angular(l, sector.m, a, parent.m, 1, 1)
            zero = angular(l, sector.m, a, parent.m, 1, 0)
            angles = np.array([root*(minus-plus), 1j*root*(minus+plus),
                               math.sqrt(4*math.pi/3)*zero])
            if not np.any(angles):
                continue
            radial = background['fields'][a]
            derivative = parent.derivatives[a]@background['coefficients'][parent.block(j)]
            gradient = derivative+(a*(a+1)+2-l*(l+1))/2*radial/grid.r
            projected = sector.functions[l].T@(weights*gradient)
            result[sector.block(i)] += -1j*projected[:, None]*angles[None, :]
    return result


def collective_operator(grid, parent, upper, lower, fraction, *, contact=0.):
    """Real Bogoliubov operator; contact<0 denotes a local attraction."""
    if upper.m != parent.m+1 or lower.m != parent.m-1:
        raise ValueError('This calculation requires azimuthal offsets +1 and -1')
    background = stationary_parent(grid, parent, fraction, contact=contact)
    ku = np.zeros_like(upper.h0)
    kv = np.zeros_like(lower.h0)
    coupling = np.zeros((len(ku), len(kv)))
    weights = grid.w*grid.r**2
    reciprocity_error = 0.
    for rank in range(1, max(max(upper.ls), max(lower.ls))+max(parent.ls)+1):
        fu = transition_density(upper, parent, background['fields'], rank)
        fv = transition_density(lower, parent, background['fields'], rank)
        pu, pv = grid.potential(fu, rank), grid.potential(fv, rank)
        coefficient = 4*math.pi/(2*rank+1)
        ku -= coefficient*fu.T@(weights[:, None]*pu)
        kv -= coefficient*fv.T@(weights[:, None]*pv)
        uv = coefficient*fu.T@(weights[:, None]*pv)
        vu = coefficient*fv.T@(weights[:, None]*pu)
        reciprocity_error = max(reciprocity_error, float(np.max(np.abs(uv-vu.T))))
        coupling += (uv+vu.T)/2
        if contact:
            ku += contact*fu.T@(weights[:, None]*fu)
            kv += contact*fv.T@(weights[:, None]*fv)
            coupling -= contact*fu.T@(weights[:, None]*fv)
    tu = momentum_projection(grid, upper, parent, background)
    tv = momentum_projection(grid, lower, parent, background)
    ku += np.real(tu@tu.conj().T)
    kv += np.real(tv@tv.conj().T)
    coupling += np.real(tu@tv.T)
    ku, kv = (ku+ku.T)/2, (kv+kv.T)/2
    au = (upper.h0-background['chemical']*np.eye(len(ku))
          +fraction*(hartree(grid, upper, background['potential'])+ku))
    av = (lower.h0-background['chemical']*np.eye(len(kv))
          +fraction*(hartree(grid, lower, background['potential'])+kv))
    matrix = np.block([[au, fraction*coupling], [-fraction*coupling.T, -av]])
    return dict(matrix=matrix, upper=au, lower=av, coupling=coupling,
                background=background, reciprocity_error=reciprocity_error)


def growing_branch(operator, upper, lower):
    """Track the mode with largest overlap with the bare 322 radial ground state."""
    matrix = operator['matrix']
    values, vectors = eig(matrix)
    bare_values, bare_vectors = eigh(upper.h0)
    reference = bare_vectors[:, 0]
    overlaps = np.abs(reference@vectors[:len(reference), :])**2
    index = int(np.argmax(overlaps))
    value, vector = values[index], vectors[:, index]
    nu = len(reference)
    norm = float(np.vdot(vector[:nu], vector[:nu]).real
                 -np.vdot(vector[nu:], vector[nu:]).real)
    if norm <= 0 or abs(value.imag) > 1e-10:
        raise ArithmeticError('Tracked branch is not a real positive-norm mode')
    origin = lower.origin@vector[nu:]
    return dict(value=float(value.real), vector=vector, norm=norm,
                bare_energy=float(bare_values[0]),
                upper_overlap=float(overlaps[index]),
                origin_real=float(origin.real), origin_imag=float(origin.imag),
                absorption_coefficient=float(abs(origin)**2/norm),
                residual=float(np.linalg.norm(matrix@vector-value*vector)))


def energy_functional(grid, states, fraction, *, contact=0.):
    """Evaluate the unexpanded energy for independent Hessian checks.

    states is a list of (Sector, coefficient vector) pairs. Do not normalize
    the perturbed state: its norm is needed for the grand-potential Hessian.
    """
    if not np.isfinite(contact):
        raise ValueError('Require a finite contact coupling')
    components = []
    kinetic = 0.
    norm = 0.
    for sector, coefficients in states:
        kinetic += float(np.vdot(coefficients, sector.h0@coefficients).real)
        norm += float(np.vdot(coefficients, coefficients).real)
        for index, l in enumerate(sector.ls):
            block = coefficients[sector.block(index)]
            components.append((l, sector.m, sector.functions[l]@block,
                               sector.derivatives[l]@block))
    weights = grid.w*grid.r**2
    gravitational = 0.
    local = 0.
    for rank in range(2*max(c[0] for c in components)+1):
        for order in range(-rank, rank+1):
            density = np.zeros_like(grid.r, dtype=complex)
            for left_l, left_m, left, _ in components:
                for right_l, right_m, right, _ in components:
                    coefficient = (-1)**order*angular(
                        right_l, right_m, left_l, left_m, rank, -order)
                    if coefficient:
                        density += coefficient*left*right.conj()
            if np.any(density):
                potential = -4*math.pi/(2*rank+1)*grid.potential(density, rank)
                gravitational += .5*float(np.vdot(density, weights*potential).real)
                local += .5*contact*float(np.vdot(density, weights*density).real)
    momentum = np.zeros(3, dtype=complex)
    root = math.sqrt(2*math.pi/3)
    for left_l, left_m, left, _ in components:
        for right_l, right_m, right, derivative in components:
            minus = angular(left_l, left_m, right_l, right_m, 1, -1)
            plus = angular(left_l, left_m, right_l, right_m, 1, 1)
            zero = angular(left_l, left_m, right_l, right_m, 1, 0)
            angles = np.array([root*(minus-plus), 1j*root*(minus+plus),
                               math.sqrt(4*math.pi/3)*zero])
            radial = derivative+(right_l*(right_l+1)+2-left_l*(left_l+1))/2*right/grid.r
            momentum += -1j*np.vdot(left, weights*radial)*angles
    if np.max(np.abs(momentum.imag)) > 1e-10:
        raise ArithmeticError('Hermitian momentum acquired an imaginary expectation')
    recoil = .5*float(np.dot(momentum.real, momentum.real))
    return dict(norm=norm, kinetic=kinetic, gravitational=gravitational,
                recoil=recoil, contact=local,
                energy=kinetic+fraction*(gravitational+recoil+local))


def absorbed_eigenvalue(operator, upper, lower, strength):
    """Finite-basis pole check for -i*strength*|origin><origin| absorption."""
    if not np.isfinite(strength) or strength <= 0:
        raise ValueError('Require positive dimensionless absorption strength')
    matrix = operator['matrix'].astype(complex)
    nu = len(upper.h0)
    matrix[nu:, nu:] -= 1j*strength*np.outer(lower.origin, lower.origin)
    values, vectors = eig(matrix)
    reference = eigh(upper.h0)[1][:, 0]
    index = int(np.argmax(np.abs(reference@vectors[:nu, :])**2))
    value = values[index]
    return dict(strength=strength, real=float(value.real), imag=float(value.imag),
                residual=float(np.linalg.norm(matrix@vectors[:, index]
                                              -value*vectors[:, index])))
