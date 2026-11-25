"""Absorptive bound-state response to the 322 + 21-1 -> 211 source.

The nearly degenerate n=3 pair, or an extended m=0 basis, includes self-gravity mixing.
Elimination is a local weak-coupling approximation. Its conservative term
is differentiated from one energy function; the dissipative flux obeys the
same response optical theorem. This is not a complete multimode cloud model.
"""
from functools import lru_cache
import math

import mpmath as mp
import numpy as np

from .research_competing_levels import A, B, source_terms, hydrogen_polynomial, number
from .research_moving_resonance import EXCHANGE
from .research_tidal_history import ThreeModeModel


C = (2, 1, -1)
STATES = ((3, 0, 0), (3, 2, 0))


def locked_flux_fractions(lever):
    """Necessary flux partition for a stationary dilute pair on a moving resonance."""
    if not 0 <= lever <= 1:
        raise ValueError('Nonnegative stationary fluxes require 0 <= lever <= 1')
    return dict(escape=2*(1-lever)/(4-lever), induced_absorption=3*lever/(4-lever))


@lru_cache(maxsize=256)
def bound_vertex(pumps, final, output, digits=60):
    if sum(s[2] for s in pumps)-final[2] != output[2]:
        return 0.
    with mp.workdps(digits):
        terms = source_terms(pumps, final, output[1])
        polynomial = hydrogen_polynomial(output)
        result = mp.mpf(0)
        for name in ['direct', 'recoil']:
            for (p, decay), value in terms[name].items():
                for k, coefficient in polynomial.items():
                    power = p+k+2
                    result += (coefficient*value*mp.factorial(power)
                        /(number(decay)+mp.mpf(1)/output[0])**(power+1))
        return float(result)


def extended_response(alpha0, mass, j, a, b, c, *, maximum_n=4):
    """Finite-bound-basis check with coherent monopole absorption at the origin."""
    states = tuple((n, l, 0) for n in range(1, maximum_n+1) for l in (0, 2) if l < n)
    identity = np.eye(len(states))
    normal = np.array([[[bound_vertex((s, right), s, left)
        for right in states] for left in states] for s in (A, B, C)])
    chemical = np.array([bound_vertex((s, B), s, B) for s in (A, B, C)])
    cross_difference = chemical[0]-chemical[2]
    matrices = normal-chemical[:, None, None]*identity
    matrices += np.array([EXCHANGE, cross_difference, -EXCHANGE])[:, None, None]*identity
    fine = lambda s: -1/(8*s[0]**4)+(2*s[1]-3*s[0]+1)/(s[0]**4*(s[1]+.5))
    delta = np.diag([alpha0**3*mass**2*(-1/(2*s[0]**2)+1/18)
        +alpha0**5*mass**4*(fine(s)-fine(B)) for s in states])
    delta += ((2-16/135)*alpha0**6*j*mass**3/12*identity
        +alpha0**3*mass*np.einsum('i,ijk->jk', [a, b, c], matrices))
    spin = j/mass**2
    origin = np.array([1/s[0]**1.5 if s[1] == 0 else 0. for s in states])
    width = 8*(1+math.sqrt(1-spin*spin))*alpha0**6*mass**5*np.outer(origin, origin)
    vertex = np.array([bound_vertex((B, C), A, s) for s in states])
    response = np.linalg.solve(delta-.5j*width, vertex)
    kernel = alpha0**6*mass**2*np.dot(vertex, response)
    return dict(states=states, kernel_real=float(kernel.real), kernel_imag=float(kernel.imag),
        induced_flux=float(2*a*b*c*kernel.imag),
        direct_horizon_flux=float(a*b*c*alpha0**6*mass**2*np.real(response.conj()@width@response)),
        response_population=float(a*b*c*alpha0**6*mass**2*np.sum(abs(response)**2)))


class CounterAbsorptionModel(ThreeModeModel):
    def __init__(self, parent, *, rate_scale=1., dispersive_scale=1., normal_scale=1., maximum_n=None):
        super().__init__(parent)
        if rate_scale < 0 or normal_scale <= 0 or not 0 <= dispersive_scale <= 1:
            raise ValueError('Require nonnegative rate and physical diagnostic scales')
        if maximum_n is not None and (not isinstance(maximum_n, int) or not 3 <= maximum_n <= 10):
            raise ValueError('Bound-basis checks require 3 <= maximum_n <= 10')
        self.rate_scale, self.dispersive_scale = rate_scale, dispersive_scale
        self.normal_scale = normal_scale
        self.states = (STATES if maximum_n is None else
            tuple((n, l, 0) for n in range(1, maximum_n+1) for l in (0, 2) if l < n))
        self.identity = np.eye(len(self.states))
        self.normal = np.array([[[bound_vertex((s, right), s, left)
            for right in self.states] for left in self.states] for s in [A, B, C]])
        self.chemical = np.array([bound_vertex((s, B), s, B) for s in [A, B, C]])
        self.vertex = np.array([bound_vertex((B, C), A, d) for d in self.states])
        self.matrices = self.normal-self.chemical[:, None, None]*self.identity
        self.matrices += np.array([EXCHANGE, self.cross_difference, -EXCHANGE])[:, None, None]*self.identity
        fine = lambda s: -1/(8*s[0]**4)+(2*s[1]-3*s[0]+1)/(s[0]**4*(s[1]+.5))
        self.fine = np.diag([fine(s)-fine(B) for s in self.states])
        self.bohr = np.diag([-1/(2*s[0]**2)+1/18 for s in self.states])
        self.hf_ratio = 2-16/135
        self.origin = np.array([(3/s[0])**1.5 if s[1] == 0 else 0. for s in self.states])
        self.absorbing = np.outer(self.origin, self.origin)

    def response(self, alpha0, mass, j, a, b, c):
        spin = j/mass**2
        root = math.sqrt(1-spin*spin)
        rplus = 1+root
        h = alpha0**6*j*mass**3/12
        density = np.einsum('i,ijk->jk', [a, b, c], self.matrices)*self.normal_scale
        delta = (alpha0**3*mass**2*self.bohr+alpha0**5*mass**4*self.fine+self.hf_ratio*h*self.identity
                 +alpha0**3*mass*density)
        width = 8*rplus*alpha0**6*mass**5/27
        nonhermitian = delta-.5j*width*self.absorbing
        inverse_vertex = np.linalg.solve(nonhermitian, self.vertex)
        prefactor = self.rate_scale*alpha0**6*mass**2
        kernel = prefactor*np.dot(self.vertex, inverse_vertex)
        width_m = width*(5/mass+2*spin**2/(mass*root*rplus))
        width_j = -width*spin/(mass**2*root*rplus)
        derivatives = [2*alpha0**3*mass*self.bohr+4*alpha0**5*mass**3*self.fine
            +3*self.hf_ratio*h/mass*self.identity+alpha0**3*density
            -.5j*width_m*self.absorbing,
            self.hf_ratio*alpha0**6*mass**3/12*self.identity-.5j*width_j*self.absorbing]
        derivatives += list(alpha0**3*mass*self.normal_scale*self.matrices)
        gradient = np.array([-prefactor*np.dot(inverse_vertex, d@inverse_vertex) for d in derivatives])
        gradient[0] += 2*kernel/mass
        occupation = a*b*c*prefactor*abs(inverse_vertex)**2
        response_norm = prefactor*float(np.sum(abs(inverse_vertex)**2))
        return dict(kernel=kernel, gradient=gradient, delta=delta, width=width,
                    nonhermitian=nonhermitian, matrix_derivatives=derivatives,
                    occupation=occupation, inverse_vertex=inverse_vertex,
                    fractions=response_norm*np.array([b*c, a*c, a*b]),
                    horizon_occupation=a*b*c*prefactor*abs(np.dot(self.origin, inverse_vertex))**2)

    def gap_gradient(self, alpha0, mass, j, a, b, c):
        """Analytic derivatives of the real tidal gap, including response dispersion."""
        h = alpha0**6*j*mass**3/12
        gradient = np.array([6*h/mass+alpha0**3*(EXCHANGE*(a-c)+self.cross_difference*b),
            alpha0**6*mass**3/6, alpha0**3*mass*EXCHANGE,
            alpha0**3*mass*self.cross_difference, -alpha0**3*mass*EXCHANGE])
        response = self.response(alpha0,mass,j,a,b,c)
        k, dk = response['kernel'].real, response['gradient'].real
        u = response['inverse_vertex']
        prefactor = self.rate_scale*alpha0**6*mass**2
        qmatrix = alpha0**3*mass*self.normal_scale*(self.matrices[0]-self.matrices[2])
        partner = np.linalg.solve(response['nonhermitian'],qmatrix@u)
        q = response['gradient'][2]-response['gradient'][4]
        dq = np.array([2*prefactor*np.dot(partner,d@u) for d in response['matrix_derivatives']])
        dq[0] += 3*q/mass
        extra = b*(a-c)*dk-a*b*c*dq.real
        extra[2] += b*k-b*c*q.real
        extra[3] += (a-c)*k-a*c*q.real
        extra[4] += -b*k-a*b*q.real
        return gradient+self.dispersive_scale*extra

    def energy(self, alpha0, mass, j, a, b, c):
        base = super().energy(alpha0, mass, j, a, b, c)
        response = self.response(alpha0, mass, j, a, b, c)
        k, dk = response['kernel'], response['gradient']
        factor = self.dispersive_scale/alpha0
        shift = -factor*a*b*c*k.real
        gradient = -factor*(a*b*c*dk.real+np.array([0., 0., b*c, a*c, a*b])*k.real)
        for name, correction in zip(['clock', 'spin_derivative', 'wa', 'wb', 'wc'], gradient):
            base[name] += correction
        base['energy'] += shift
        base['gap'] += alpha0*(gradient[2]-gradient[4])
        base.update(wh=2*base['wa']-base['wb'], we=2*base['wb']-base['wa'],
            induced_coefficient=2*k.imag, induced_energy=shift,
            induced_occupation=float(sum(response['occupation'])),
            induced_response_fraction=float(max(response['fractions'])),
            induced_monopole=float(response['horizon_occupation']))
        return base

    def rates(self, alpha0, mass, j, a, b, c):
        v = super().rates(alpha0, mass, j, a, b, c)
        flux = v['induced_coefficient']*a*b*c
        dm = (v['wb']+v['wc']-v['wa'])*flux/v['clock']
        v.update(da=v['da']+flux, db=v['db']-flux, dc=v['dc']-flux,
                 dm=v['dm']+dm, horizon_area_driver=v['horizon_area_driver']+dm,
                 induced_flux=flux,
                 counter_fractional_rate=v['gamma_c']-v['induced_coefficient']*a*b)
        return v
