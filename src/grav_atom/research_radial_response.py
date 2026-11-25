"""Radial completion of the cloud's internal m=0 absorption response.

Square-integrable Laguerre functions represent both Coulomb bound states and
positive-energy pseudostates. Fine splittings are retained as a finite-rank
bound-state operator; this is not a relativistic continuum calculation.
The contact absorber is checked against its first-order flux limit.
"""
import math

import numpy as np
from scipy.linalg import eigh

from .research_collective_mode import (
    RadialGrid, Sector, make_sector, density_potential, hartree,
    transition_density, momentum_projection)
from .research_competing_levels import hydrogen_polynomial, radial_source
from .research_counter_absorption import CounterAbsorptionModel, A, B, C
from .research_moving_resonance import EXCHANGE


def hydrogen_fields(grid, state):
    n, l, m = state
    polynomial = hydrogen_polynomial(state)
    wave = np.exp(-grid.r/n)*sum(float(v)*grid.r**p for p, v in polynomial.items())
    derivative = np.exp(-grid.r/n)*sum(float(v)*(
        p*grid.r**(p-1)-grid.r**p/n) for p, v in polynomial.items())
    sector = Sector(m, (l,), 1, {l: wave[:, None]}, {l: derivative[:, None]},
                    np.zeros((1, 1)), 0., np.zeros(1))
    return sector, dict(fields={l: wave}, coefficients=np.ones(1))


def normal_operator(grid, target, state):
    parent, background = hydrogen_fields(grid, state)
    return normal_from_background(grid, target, parent, background)


def normal_from_background(grid, target, parent, background):
    """Normal gravitational Hessian for a real resolved parent profile."""
    result = hartree(grid, target, density_potential(grid, parent, background['fields']))
    weights = grid.w*grid.r**2
    for rank in range(abs(target.m-parent.m), max(target.ls)+max(parent.ls)+1):
        density = transition_density(target, parent, background['fields'], rank)
        if np.any(density):
            result -= 4*math.pi/(2*rank+1)*density.T@(
                weights[:, None]*grid.potential(density, rank))
    momentum = momentum_projection(grid, target, parent, background)
    result += np.real(momentum@momentum.conj().T)
    return (result+result.T)/2


class RadialCounterResponse(CounterAbsorptionModel):
    def __init__(self, parent, *, size=28, order=48, decay=1/3, maximum_l=4,
                 fine_n=6, bound_only=False, dispersive_scale=1., first_order=False):
        if maximum_l not in (2, 4, 6) or fine_n not in (3, 4, 6, 8):
            raise ValueError('Use angular cutoffs 2, 4, 6 and fine cutoffs 3, 4, 6, 8')
        super().__init__(parent, maximum_n=3, dispersive_scale=dispersive_scale)
        self.states = None  # This basis has no finite list of hydrogenic state labels.
        self.first_order = first_order
        self.grid = grid = RadialGrid(order)
        self.sector = sector = make_sector(grid, 0, tuple(range(0, maximum_l+1, 2)),
                                          size=size, decay=decay)
        weights = grid.w*grid.r**2
        dimension = len(sector.h0)
        identity = np.eye(dimension)
        normal = np.array([normal_operator(grid, sector, s) for s in (A, B, C)])
        vertex = np.concatenate([sector.functions[l].T@(
            weights*radial_source(grid.r, (B, C), A, l)['total']) for l in sector.ls])
        fine_value = lambda n, l: -1/(8*n**4)+(2*l-3*n+1)/(n**4*(l+.5))
        fine = -fine_value(3, 2)*identity
        for n in range(1, fine_n+1):
            for l in sector.ls:
                if l >= n:
                    continue
                _, background = hydrogen_fields(grid, (n, l, 0))
                projection = np.zeros(dimension)
                projection[sector.block(sector.ls.index(l))] = (
                    sector.functions[l].T@(weights*background['fields'][l]))
                fine += fine_value(n, l)*np.outer(projection, projection)
        values, vectors = eigh(sector.h0)
        self.bare_spectrum = values
        # A negative-energy projection isolates the omission made by bound-only models.
        transform = vectors[:, values < 0] if bound_only else identity
        self.identity = np.eye(transform.shape[1])
        self.bohr = transform.T@(sector.h0+identity/18)@transform
        self.normal = np.einsum('ij,sik,kl->sjl', transform, normal, transform)
        self.fine = transform.T@fine@transform
        self.vertex = transform.T@vertex
        self.origin = transform.T@sector.origin*math.sqrt(27)/2
        self.absorbing = np.outer(self.origin, self.origin)
        self.matrices = self.normal-self.chemical[:, None, None]*self.identity
        self.matrices += np.array([EXCHANGE, self.cross_difference, -EXCHANGE])[:, None, None]*self.identity
        self.transform = transform
        self.configuration = dict(size=size, order=order, decay=decay, maximum_l=maximum_l,
            fine_n=fine_n, bound_only=bound_only, first_order=first_order,
            dimension=len(self.identity),
            negative_bare_states=int(sum(values < 0)), positive_pseudostates=int(sum(values > 0)),
            overlap_error=sector.overlap_error)

    def response(self, alpha0, mass, j, a, b, c):
        response = super().response(alpha0, mass, j, a, b, c)
        if not self.first_order:
            return response
        # Keep the matched horizon loss only to first order. Resumming an
        # unrenormalized point absorber has no infinite-basis physical limit.
        delta, width = response['delta'], response['width']
        u = np.linalg.solve(delta, self.vertex)
        contact = np.linalg.solve(delta, self.origin)
        amplitude = np.dot(self.origin, u)
        prefactor = self.rate_scale*alpha0**6*mass**2
        matrices = [d.real for d in response['matrix_derivatives']]
        spin = j/mass**2
        root = math.sqrt(1-spin*spin)
        log_width_gradient = np.array([
            5/mass+2*spin**2/(mass*root*(1+root)),
            -spin/(mass**2*root*(1+root)), 0., 0., 0.])
        imaginary = .5*prefactor*width*amplitude**2
        kernel = prefactor*np.dot(self.vertex, u)+1j*imaginary
        gradient = np.array([-prefactor*np.dot(u, d@u) for d in matrices], dtype=complex)
        gradient[0] += 2*kernel.real/mass
        imag_gradient = imaginary*log_width_gradient
        imag_gradient[0] += 2*imaginary/mass
        imag_gradient -= prefactor*width*amplitude*np.array([np.dot(contact, d@u) for d in matrices])
        gradient += 1j*imag_gradient
        response.update(kernel=kernel, gradient=gradient, nonhermitian=delta,
            matrix_derivatives=matrices, inverse_vertex=u.astype(complex),
            occupation=a*b*c*prefactor*u**2,
            fractions=prefactor*np.dot(u, u)*np.array([b*c, a*c, a*b]),
            horizon_occupation=a*b*c*prefactor*amplitude**2)
        return response

    def diagnostics(self, alpha0, mass, j, a, b, c):
        response = self.response(alpha0, mass, j, a, b, c)
        real_solution = np.linalg.solve(response['delta'], self.vertex)
        prefactor = self.rate_scale*alpha0**6*mass**2
        amplitude = np.dot(self.origin, real_solution)
        weak_imag = .5*prefactor*response['width']*amplitude**2
        coincident = np.dot(self.origin, np.linalg.solve(response['delta'], self.origin))
        return dict(kernel_real=float(response['kernel'].real),
            kernel_imag=float(response['kernel'].imag),
            conservative_zero_width=float(prefactor*np.dot(self.vertex, real_solution)),
            first_order_absorption=float(weak_imag),
            contact_resummation_parameter=float(.5*response['width']*coincident),
            kernel_gradient_real=response['gradient'].real.tolist(),
            kernel_gradient_imag=response['gradient'].imag.tolist(),
            gap_gradient=self.gap_gradient(alpha0, mass, j, a, b, c).tolist(),
            smallest_detuning=float(min(abs(np.linalg.eigvalsh(response['delta'])))),
            maximum_response_fraction=float(max(response['fractions'])),
            optical_theorem_relative_error=float(abs(
                2*a*b*c*response['kernel'].imag
                -response['width']*response['horizon_occupation'])
                /max(abs(2*a*b*c*response['kernel'].imag), 1e-300)))
