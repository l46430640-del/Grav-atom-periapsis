"""Leading gravitational quartic channels beyond the 211/322 truncation.

Exact hydrogenic polynomials and Coulomb Laplace moments are used. Central
recoil is canonical. Finite-cloud scattering and horizon multipoles above
the included dipole diagnostic are outside this model.
"""
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
import math

import mpmath as mp
import numpy as np
from scipy.linalg import eigh
from scipy.special import gammainc, gammaincc

from boson_ep.models import State
from boson_ep.spectrum import detweiler_coefficient
from .research_collective_mode import (
    angular, RadialGrid, Sector, density_potential, hartree,
    transition_density, momentum_projection)


A = (2, 1, 1)
B = (3, 2, 2)


def number(value):
    return mp.mpf(value.numerator)/value.denominator if isinstance(value, Fraction) else mp.mpf(value)


def energy(state):
    return Fraction(-1, 2*state[0]**2)


def hydrogen_polynomial(state):
    n, l, m = state
    if not all(isinstance(v, int) for v in state) or not 0 <= abs(m) <= l < n:
        raise ValueError('Require hydrogenic integer quantum numbers')
    norm = mp.mpf(2)/n**2*mp.sqrt(mp.factorial(n-l-1)/mp.factorial(n+l))
    return {l+t: norm*(-1)**t*mp.binomial(n+l, n-l-1-t)*(mp.mpf(2)/n)**(l+t)
            /mp.factorial(t) for t in range(n-l)}


def product(left, right):
    result = defaultdict(mp.mpf)
    for p, a in left.items():
        for q, b in right.items():
            result[p+q] += a*b
    return dict(result)


def angle_product(i, j, final, output_l, rank):
    output_m = i[2]+j[2]-final[2]
    if output_l < abs(output_m):
        return 0.
    order = i[2]-final[2]
    return (angular(i[1], i[2], final[1], final[2], rank, order)
            *angular(output_l, output_m, j[1], j[2], rank, order))


def source_terms(pumps, final, output_l):
    """Sum ordered contractions once for identical pumps and twice otherwise."""
    i, j = pumps
    pairs = [(i, j)] if i == j else [(i, j), (j, i)]
    direct, recoil, contact = [defaultdict(mp.mpf) for _ in range(3)]
    for i, j in pairs:
        pi, pj, pk = map(hydrogen_polynomial, (i, j, final))
        density = product(pi, pk)
        decay = Fraction(1, i[0])+Fraction(1, final[0])
        beta = Fraction(1, j[0])
        s = number(decay)
        contact_angle = 0.
        for rank in range(i[1]+final[1]+1):
            angles = angle_product(i, j, final, output_l, rank)
            contact_angle += angles
            if not angles:
                continue
            coefficient = number(4*math.pi/(2*rank+1)*angles)
            for p, value in density.items():
                lo, hi = p+rank+2, p+1-rank
                if hi < 0:
                    raise ArithmeticError('Selected density has a nonintegrable multipole')
                low = value*mp.factorial(lo)/s**(lo+1)
                high = value*mp.factorial(hi)/s**(hi+1)
                for q, outer in pj.items():
                    direct[q-rank-1, beta] -= coefficient*outer*low
                    for k in range(lo+1):
                        direct[q-rank-1+k, beta+decay] += (
                            coefficient*outer*low*s**k/mp.factorial(k))
                    for k in range(hi+1):
                        direct[q+rank+k, beta+decay] -= (
                            coefficient*outer*high*s**k/mp.factorial(k))
        for p, value in product(density, pj).items():
            contact[p, beta+decay] += number(contact_angle)*value
        dipole = sum(value*mp.factorial(p+3)/s**(p+4) for p, value in density.items())
        coefficient = number(4*math.pi/3*angle_product(i, j, final, output_l, 1))
        coefficient *= number(energy(i)-energy(final))*dipole
        # p=i[H,r] makes this recoil source valid before the emitted leg is on shell.
        for p, value in pj.items():
            q = p+1
            recoil[q-2, beta] += coefficient*value*(output_l*(output_l+1)-q*(q+1))/2
            recoil[q-1, beta] += coefficient*value*(number(beta)*(q+1)-1)
    return dict(direct=dict(direct), recoil=dict(recoil), contact=dict(contact))


def coulomb_laplace(l, power, decay, emitted):
    k, s = mp.sqrt(2*emitted), number(decay)
    exponent = power+l+1
    if exponent <= 0:
        raise ValueError('Individual Coulomb moment does not converge')
    a, c, z = l+1+1j/k, 2*l+2, 2j*k/(s+1j*k)
    if exponent >= c:
        # Euler's terminating form resolves exact bound/continuum orthogonality zeros.
        degree = exponent-c
        hyper = ((1-z)**(c-a-exponent)*sum(
            mp.rf(c-a, j)*(-1)**j*mp.binomial(degree, j)/mp.rf(c, j)*z**j
            for j in range(degree+1)))
    else:
        # For positive integer b<c, integrate Euler's polynomial weight exactly.
        def elementary(n):
            return sum((-1)**j*mp.binomial(n, j)
                *(1-(1-z)**(j-a+1))/(j-a+1) for j in range(n+1))/z**(n+1)
        hyper = (mp.factorial(c-1)/mp.factorial(exponent-1)/mp.factorial(c-exponent-1)
                 *sum((-1)**j*mp.binomial(c-exponent-1, j)*elementary(exponent-1+j)
                      for j in range(c-exponent)))
    value = (mp.sqrt(2/(mp.pi*k))*mp.coulombc(l, -1/k)*k**(l+1)
        *mp.gamma(exponent)/(s+1j*k)**exponent
        *hyper)
    if abs(value.imag) > mp.mpf('1e-40')*max(1, abs(value.real)):
        raise ArithmeticError('A real continuum moment acquired an imaginary part')
    return value.real


def monopole_laplace(power, decay, emitted):
    kappa = mp.sqrt(-2*emitted)
    nu, s = 1/kappa, number(decay)
    if power < 0:
        raise ValueError('Individual Whittaker moment does not converge')
    return (2*kappa*mp.gamma(power+2)*mp.gamma(power+1)
        /mp.gamma(power+2-nu)/(s+kappa)**(power+2)
        *mp.hyp2f1(power+2, 1-nu, power+2-nu, (s-kappa)/(s+kappa)))


@lru_cache(maxsize=256)
def channel(pumps, final, *, digits=60):
    """Origin amplitude for a monopole, or all allowed continuum amplitudes.

    The output frequency is fixed by energy conservation. Absorption is
    reported as 2*|R0|^2, before multiplying rplus*alpha^6 and populations.
    Continuum coefficients are 2*pi*|V|^2 before alpha^3 and populations.
    """
    with mp.workdps(digits):
        emitted = number(energy(pumps[0])+energy(pumps[1])-energy(final))
        m = pumps[0][2]+pumps[1][2]-final[2]
        if emitted == 0:
            raise ValueError('This calculation does not include exact thresholds')
        if emitted < 0 and m != 0:
            raise ValueError('Only monopole horizon absorption is matched here')
        partial = []
        maximum_l = pumps[0][1]+pumps[1][1]+final[1]
        ls = [0] if emitted < 0 else range(abs(m), maximum_l+1)
        for l in ls:
            terms = source_terms(pumps, final, l)
            if not any(terms['direct'].values()) and not any(terms['recoil'].values()):
                continue
            if emitted < 0:
                nu = 1/mp.sqrt(-2*emitted)
                if abs(nu-mp.nint(nu)) < mp.mpf('1e-30'):
                    raise ValueError('On-shell bound pole needs its frequency shift and width')
                factor = -2*mp.gamma(1-nu)
                integral = lambda p, s: factor*monopole_laplace(p+1, s, emitted)
            else:
                integral = lambda p, s: coulomb_laplace(l, p+2, s, emitted)
            moments = {}
            parts = {}
            for name, expansion in terms.items():
                value = mp.mpf(0)
                for key, coefficient in expansion.items():
                    if not coefficient:
                        continue
                    if key not in moments:
                        moments[key] = integral(*key)
                    value += coefficient*moments[key]
                parts[name] = float(value)
            total = parts['direct']+parts['recoil']
            factor = 2 if emitted < 0 else 2*math.pi
            partial.append(dict(l=l, **parts, total=total, coefficient=factor*total**2))
        return dict(pumps=[list(s) for s in pumps], final=list(final),
            output_m=m, output_energy=float(emitted), digits=digits, partial=partial,
            coefficient=sum(c['coefficient'] for c in partial),
            contact_literature_coefficient=sum(c['contact']**2 for c in partial)
                *(1/32 if emitted < 0 else math.pi/32))


def radial_source(radius, pumps, final, output_l):
    """Independent real-axis source using incomplete-gamma multipoles."""
    radius = np.asarray(radius, dtype=float)
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError('Require positive finite radii')
    pairs = [pumps] if pumps[0] == pumps[1] else [pumps, pumps[::-1]]
    direct, recoil = np.zeros_like(radius), np.zeros_like(radius)
    with mp.workdps(50):
        for i, j in pairs:
            pi, pj, pk = map(hydrogen_polynomial, (i, j, final))
            density = product(pi, pk)
            decay, beta = 1/i[0]+1/final[0], 1/j[0]
            outer = sum(float(c)*radius**p for p, c in pj.items())*np.exp(-beta*radius)
            for rank in range(i[1]+final[1]+1):
                angle = 4*math.pi/(2*rank+1)*angle_product(i, j, final, output_l, rank)
                if not angle:
                    continue
                potential = np.zeros_like(radius)
                for p, value in density.items():
                    lo, hi = p+rank+2, p+1-rank
                    potential += float(value)*(
                        math.factorial(lo)/decay**(lo+1)*gammainc(lo+1, decay*radius)
                        /radius**(rank+1)
                        +math.factorial(hi)/decay**(hi+1)*gammaincc(hi+1, decay*radius)
                        *radius**rank)
                direct -= angle*outer*potential
            dipole = sum(float(c)*math.factorial(p+3)/decay**(p+4)
                         for p, c in density.items())
            coefficient = (4*math.pi/3*angle_product(i, j, final, output_l, 1)
                           *float(energy(i)-energy(final))*dipole)
            operator = np.zeros_like(radius)
            for p, value in pj.items():
                q = p+1
                operator += float(value)*np.exp(-beta*radius)*(
                    (output_l*(output_l+1)-q*(q+1))/2*radius**(q-2)
                    +(beta*(q+1)-1)*radius**(q-1))
            recoil += coefficient*operator
    return dict(direct=direct, recoil=recoil, total=direct+recoil)


def quadrature(pumps, final, *, order=240, cutoff=180.):
    if not isinstance(order, int) or order < 64 or not np.isfinite(cutoff) or cutoff < 40:
        raise ValueError('Require at least 64 points and cutoff >=40')
    emitted = float(energy(pumps[0])+energy(pumps[1])-energy(final))
    exact = channel(pumps, final)
    edges = np.array([0.]+[x for x in [.5, 1., 2., 4., 8., 16., 32., 64., 128., 256.]
                          if x < cutoff]+[cutoff])
    nodes, weights = np.polynomial.legendre.leggauss(max(16, order//(len(edges)-1)))
    half = np.diff(edges)/2
    radius = (((edges[:-1]+edges[1:])/2)[:, None]+half[:, None]*nodes).ravel()
    weights = (half[:, None]*weights).ravel()
    rows = []
    with mp.workdps(30):
        for c in exact['partial']:
            l = c['l']
            if emitted < 0:
                kappa = mp.sqrt(-2*emitted)
                nu = 1/kappa
                wave = np.array([float(-2*mp.gamma(1-nu)
                    *mp.whitw(nu, .5, 2*kappa*r)*r) for r in radius])
            else:
                k = mp.sqrt(2*emitted)
                wave = np.array([float(mp.sqrt(2/(mp.pi*k))
                    *mp.coulombf(l, -1/k, k*r)*r) for r in radius])
            source = radial_source(radius, pumps, final, l)
            rows.append(dict(l=l, **{name: float(np.dot(weights, wave*value))
                                     for name, value in source.items()}))
    return dict(pumps=pumps, final=final, order=order, actual_nodes=len(radius),
                cutoff=cutoff, partial=rows)


@lru_cache(maxsize=128)
def multipole_origin(pumps, final, l=1, order=144, cutoff=100.):
    """Independent real-axis regular coefficient R_l/r^l, below threshold."""
    emitted = float(energy(pumps[0])+energy(pumps[1])-energy(final))
    m = pumps[0][2]+pumps[1][2]-final[2]
    if emitted >= 0 or abs(m) > l or l < 0:
        raise ValueError('Require a subthreshold allowed angular channel')
    nodes, weights = np.polynomial.legendre.leggauss(order)
    r, weights = cutoff*(nodes+1)/2, cutoff*weights/2
    with mp.workdps(30):
        kappa, nu = mp.sqrt(-2*emitted), 1/mp.sqrt(-2*emitted)
        if nu >= l+1 and abs(nu-mp.nint(nu)) < mp.mpf('1e-12'):
            raise ValueError('Resonant forced state needs detuning and width')
        factor = -2*(2*kappa)**l*mp.gamma(l+1-nu)/mp.gamma(2*l+2)
        wave = np.array([float(factor*mp.whitw(nu, l+.5, 2*kappa*x)*x) for x in r])
    source = radial_source(r, pumps, final, l)
    return dict(pumps=pumps, final=final, l=l, m=m, output_energy=emitted,
                order=order, cutoff=cutoff,
                **{name: float(np.dot(weights, wave*value)) for name, value in source.items()})


def horizon_coefficient(l, m, alpha, spin, frequency_over_mu, clock=1.):
    """Matched occupation flux per |R_l/r^l|^2, in units of 1/M_H."""
    rplus = 1+math.sqrt(1-spin*spin)
    delta = 2*rplus*alpha*frequency_over_mu/clock-spin*m
    product = math.prod(j*j*(1-spin*spin)+delta*delta for j in range(1, l+1))
    coefficient = 4**l*math.factorial(l)**2/math.factorial(2*l)**2
    return coefficient*clock*delta*product*alpha**(4*l+5)


@lru_cache(maxsize=128)
def cross_energy(left, right, order=48):
    """Direct/exchange/recoil coefficient between two orthogonal bound states."""
    if left == right:
        raise ValueError('Cross-energy formula assumes distinct states')
    grid = RadialGrid(order)
    r, weights = grid.r, grid.w*grid.r**2
    with mp.workdps(50):
        fields = []
        for state in [left, right]:
            p = hydrogen_polynomial(state)
            fields.append(sum(float(c)*r**k for k, c in p.items())*np.exp(-r/state[0]))
    u, v = fields
    direct, exchange = 0., 0.
    for rank in range(2*max(left[1], right[1])+1):
        factor = 4*math.pi/(2*rank+1)
        angle = (angular(left[1], left[2], left[1], left[2], rank, 0)
                 *angular(right[1], right[2], right[1], right[2], rank, 0))
        if angle:
            direct += factor*angle*np.dot(weights*u*u, grid.potential(v*v, rank))
        angle = angular(left[1], left[2], right[1], right[2],
                        rank, left[2]-right[2])
        if angle:
            exchange += factor*angle**2*np.dot(weights*u*v, grid.potential(u*v, rank))
    dipole = np.dot(weights*r, u*v)
    angle = angular(left[1], left[2], right[1], right[2], 1, left[2]-right[2])
    recoil = 4*math.pi/3*angle**2*float(energy(left)-energy(right))**2*dipole**2
    return dict(direct=float(direct), exchange=float(exchange), recoil=float(recoil),
                total=float(direct+exchange-recoil),
                norm_left=float(np.dot(weights, u*u)),
                norm_right=float(np.dot(weights, v*v)))


class LinearProbes:
    """Passive growth bounds on an evolving 211/322 background.

    Positive parts of dipole horizon gains are added while their stabilizing
    contributions are dropped. Higher horizon multipoles and finite-cloud
    continuum corrections remain outside this diagnostic.
    """
    def __init__(self, states, *, dipoles=True, escape_scale=1.):
        self.states = tuple(states)
        self.escape_scale = escape_scale
        self.metadata = []
        for state in self.states:
            c = dict(state=state, cross_a=cross_energy(state, A),
                     cross_b=cross_energy(state, B), absorption=[], emission=[], dipoles=[])
            n, l, m = state
            if l == m and m in (2, 3, 4):
                pumps = {2: (A, A), 3: (A, B), 4: (B, B)}[m]
                c['absorption'].append(channel(pumps, state))
                c['emission'].append(dict(
                    kind='loss_ab', channel=channel((B, state), A)))
            if state == (1, 0, 0):
                for pumps in [(A, A), (A, B), (B, B)]:
                    c['emission'].append(dict(kind='gain', channel=channel(pumps, state)))
            if state in [(2, 0, 0), (2, 1, 0), (2, 1, -1)]:
                c['emission'].append(dict(kind='gain', channel=channel((B, B), state)))
            if dipoles:
                for pumps in [(A, A), (A, B), (B, B)]:
                    output_m = sum(s[2] for s in pumps)-m
                    if abs(output_m) > 1 or (sum(s[1] for s in pumps)+l)%2 != 1:
                        continue
                    if float(energy(pumps[0])+energy(pumps[1])-energy(state)) >= 0:
                        continue
                    value = multipole_origin(pumps, state)
                    c['dipoles'].append(value)
            self.metadata.append(c)
        self.ns = np.array([s[0] for s in states])
        self.ls = np.array([s[1] for s in states])
        self.ms = np.array([s[2] for s in states])
        self.ca = np.array([c['cross_a']['total'] for c in self.metadata])
        self.cb = np.array([c['cross_b']['total'] for c in self.metadata])
        self.prefactors = np.array([
            2**(4*l+1)*math.factorial(n+l)/(n**(2*l+4)*math.factorial(n-l-1))
            *(math.factorial(l)/(math.factorial(2*l)*math.factorial(2*l+1)))**2
            for n, l, _ in states])

    def __call__(self, alpha0, mass, j, a, b, background):
        alpha, spin, clock = alpha0*mass, j/mass**2, background['clock']
        rplus = 1+math.sqrt(1-spin*spin)
        omega_h = spin/(2*rplus)
        frequencies = 1-alpha**2/(2*self.ns**2)-alpha0**2*mass*(self.ca*a+self.cb*b)
        products = np.ones(len(self.states))
        delta = self.ms*spin-2*rplus*alpha
        for index in range(1, int(max(self.ls))+1):
            products *= np.where(self.ls >= index, index**2*(1-spin**2)+delta**2, 1.)
        growth = (4*rplus*self.prefactors*products*alpha**(4*self.ls+5)/mass
                  *(clock*self.ms*omega_h-alpha*frequencies))
        populations = {A: a, B: b}
        pump_frequencies = {A: background['wa'], B: background['wb']}
        for index, c in enumerate(self.metadata):
            for row in c['absorption']:
                population = math.prod(populations[tuple(p)] for p in row['pumps'])
                growth[index] += row['coefficient']*rplus*alpha**6*population/mass**3
            for row in c['emission']:
                if row['kind'] == 'loss_ab':
                    growth[index] -= self.escape_scale*row['channel']['coefficient']*alpha**3*a*b/mass**3
                else:
                    population = math.prod(populations[tuple(p)] for p in row['channel']['pumps'])
                    growth[index] += row['channel']['coefficient']*alpha**3*population/mass**3
            for row in c['dipoles']:
                wh = sum(pump_frequencies[tuple(p)] for p in row['pumps'])-frequencies[index]
                population = math.prod(populations[tuple(p)] for p in row['pumps'])
                gain = (horizon_coefficient(1, row['m'], alpha, spin, wh, clock)
                        *row['total']**2*population/mass**3)
                growth[index] += max(gain, 0.)
        return growth


@lru_cache(maxsize=32)
def shell_data(n, m=2, order=48):
    """First-order normal interaction within a degenerate hydrogenic shell."""
    if m not in (2, 3, 4) or not isinstance(n, int) or n <= m:
        raise ValueError('Require n>m and m=2,3,4')
    grid = RadialGrid(order)
    r = grid.r

    def sector(states):
        functions, derivatives = {}, {}
        with mp.workdps(50):
            for state in states:
                poly = hydrogen_polynomial(state)
                functions[state[1]] = (np.exp(-r/state[0])*sum(
                    float(c)*r**p for p, c in poly.items()))[:, None]
                derivatives[state[1]] = (np.exp(-r/state[0])*sum(
                    float(c)*(p*r**(p-1)-r**p/state[0]) for p, c in poly.items()))[:, None]
        ls = tuple(s[1] for s in states)
        return Sector(states[0][2], ls, 1, functions, derivatives,
                      float(energy(states[0]))*np.eye(len(ls)), 0., np.zeros(len(ls)))

    states = [(n, l, m) for l in range(m, n, 2)]
    target = sector(states)
    matrices = []
    for pump in [A, B]:
        parent = sector([pump])
        fields = {pump[1]: parent.functions[pump[1]][:, 0]}
        potentials = density_potential(grid, parent, fields)
        normal = hartree(grid, target, potentials)
        for rank in range(max(target.ls)+pump[1]+1):
            density = transition_density(target, parent, fields, rank)
            normal -= 4*math.pi/(2*rank+1)*density.T@(
                (grid.w*r*r)[:, None]*grid.potential(density, rank))
        momentum = momentum_projection(grid, target, parent,
            dict(fields=fields, coefficients=np.ones(1)))
        normal += np.real(momentum@momentum.conj().T)
        matrices.append((normal+normal.T)/2)
    channels = [channel((B, state), A) for state in states]
    output_ls = sorted({row['l'] for c in channels for row in c['partial']})
    emission = np.array([[next((p['total'] for p in c['partial'] if p['l'] == l), 0.)
                          for c in channels] for l in output_ls])
    pumps = {2: (A, A), 3: (A, B), 4: (B, B)}[m]
    growth = channel(pumps, (n, m, m))['coefficient']
    return dict(n=n, m=m, states=states, normal_a=matrices[0], normal_b=matrices[1],
                emission=emission, output_ls=output_ls, growth=growth)


def mixed_shell(n, alpha, spin, mother, daughter, *, m=2, order=48):
    """Fine-structure and first-order self-gravity select coherent shell modes."""
    data = shell_data(n, m, order=order)
    ls = np.array([s[1] for s in data['states']], dtype=float)
    fine = (alpha**2*(-1/(8*n**4)+(2*ls-3*n+1)/(n**4*(ls+.5)))
            +2*m*spin*alpha**3/(n**3*ls*(ls+.5)*(ls+1)))
    operator = np.diag(fine)+mother*data['normal_a']+daughter*data['normal_b']
    values, vectors = eigh(operator)
    gain = data['growth']*vectors[0]**2
    loss = 2*math.pi*np.sum((data['emission']@vectors)**2, axis=0)
    kH, kI = channel((A, A), B)['coefficient'], channel((B, B), A)['coefficient']
    ratio = np.divide(loss*kH, 2*kI*gain, out=np.full_like(gain, np.inf), where=gain > 1e-30)
    pump_population = {2: mother**2, 3: mother*daughter, 4: daughter**2}[m]
    rplus = 1+math.sqrt(1-spin*spin)
    physical_ratio = np.divide(loss*mother*daughter, gain*pump_population*rplus*alpha**3,
        out=np.full_like(gain, np.inf), where=gain*pump_population > 1e-30)
    return dict(n=n, m=m, alpha=alpha, spin=spin, mother=mother, daughter=daughter,
                shifts=values.tolist(), vectors=vectors.tolist(),
                gain=gain.tolist(), loss=loss.tolist(), equilibrium_loss_over_gain=ratio.tolist(),
                instantaneous_loss_over_gain=physical_ratio.tolist())


def coherent_loss_bound(n, m=2):
    """Minimum continuum loss at fixed leading monopole gain, over all phases."""
    data = shell_data(n, m)
    amplitudes = data['emission']
    norms = np.sqrt(np.sum(amplitudes**2, axis=0))
    normalized = amplitudes/norms
    gram = normalized.T@normalized
    inverse_column = np.linalg.solve(gram, np.eye(len(gram))[:, 0])
    minimum_loss = 2*math.pi*norms[0]**2/inverse_column[0]
    ratio = minimum_loss/data['growth']
    kH, kI = channel((A, A), B)['coefficient'], channel((B, B), A)['coefficient']
    answer = dict(n=n, m=m, states=data['states'], minimum_loss=minimum_loss,
                  normalized_gram_condition=float(np.linalg.cond(gram)))
    if m == 2:
        answer['equilibrium_loss_over_gain'] = ratio*kH/(2*kI)
    else:
        answer['critical_alpha_rplus_one_third'] = (
            ratio if m == 3 else ratio*2*kI/kH)**(1/(3 if m == 3 else 6))
    minimizing = inverse_column/inverse_column[0]*norms[0]/norms
    answer['minimizing_coefficients_at_unit_lowest_l'] = minimizing.tolist()
    answer['verified_minimum_loss'] = float(2*math.pi*np.sum((amplitudes@minimizing)**2))
    return answer


class CoherentShellProbes:
    """Norm-growth bounds independent of conservative mixing in each m=2 shell."""
    def __init__(self, ns):
        self.ns = tuple(ns)
        self.data = [shell_data(n) for n in ns]
        self.dipole_strength = []
        for data in self.data:
            self.dipole_strength.append(sum(multipole_origin((A, B), state)['total']**2
                for state in data['states'] if state[1] <= 4))
        self.loss = [2*math.pi*d['emission'].T@d['emission'] for d in self.data]
        self.kH = channel((A, A), B)['coefficient']

    def __call__(self, alpha0, mass, j, a, b, background):
        alpha, spin, clock = alpha0*mass, j/mass**2, background['clock']
        rplus = 1+math.sqrt(1-spin*spin)
        rates = []
        for index, data in enumerate(self.data):
            n, dimension = data['n'], len(data['states'])
            # A scalar upper bound on ordinary SR also covers arbitrary shell phases.
            ls = np.array([s[1] for s in data['states']], dtype=float)
            fine = (alpha**2*(-1/(8*n**4)+(2*ls-3*n+1)/(n**4*(ls+.5)))
                    +4*spin*alpha**3/(n**3*ls*(ls+.5)*(ls+1)))
            chemical = (float(energy((n, 2, 2)))+float(eigh(np.diag(fine)
                +a/mass*data['normal_a']+b/mass*data['normal_b'], eigvals_only=True)[0]))
            frequency = 1+alpha**2*chemical
            sr = max(0., max(4*rplus*detweiler_coefficient(alpha, spin, State(*state))
                *(clock*state[2]*spin/(2*rplus)-alpha*frequency)
                *alpha**(4*state[1]+5)/mass for state in data['states']))
            matrix = -alpha**3*a*b/mass**3*self.loss[index]
            matrix[0, 0] += data['growth']*rplus*alpha**6*a*a/mass**3
            wh = background['wa']+background['wb']-frequency
            dipole = max(0., horizon_coefficient(1, 1, alpha, spin, wh, clock))
            dipole *= self.dipole_strength[index]*a*b/mass**3
            rates.append(float(eigh(matrix, eigvals_only=True)[-1])+sr+dipole)
        # This last integral bounds very high n even when all escape is omitted.
        rates.append(self.kH*rplus*alpha**6*a*a/mass**3)
        return np.array(rates)
