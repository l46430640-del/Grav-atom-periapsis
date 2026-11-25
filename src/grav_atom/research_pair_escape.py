"""Gravity-induced 322+322 -> 211+continuum, including central recoil.

The continuum uses delta(E-E') normalization in hydrogenic units.
This leading weak-field coefficient does not include cloud-profile changes.
"""
import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_bvp, solve_ivp
from scipy.special import gammainc, gammaincc

from .research_absorbed_pair import DECAY, DELTA, DIPOLE, PRODUCT
from .research_collective_mode import angular


ENERGY = 1/72
PARENT_NORM = 4/(81*math.sqrt(30))


def coulomb_moment(l, power, damping, energy=ENERGY):
    """Integral r**power exp(-damping*r) R_El(r) dr."""
    if (not isinstance(l, int) or l < 0 or not isinstance(power, int)
            or power < 0 or not np.isfinite([damping, energy]).all()
            or damping <= 0 or energy <= 0):
        raise ValueError('Require nonnegative integer indices and positive scales')
    with mp.workdps(45):
        k = mp.sqrt(2*mp.mpf(energy))
        s = mp.mpf(damping)
        exponent = power+l+1
        value = (mp.sqrt(2/(mp.pi*k))*mp.coulombc(l, -1/k)*k**(l+1)
                 *mp.gamma(exponent)/(s+1j*k)**exponent
                 *mp.hyp2f1(l+1+1j/k, exponent, 2*l+2, 2j*k/(s+1j*k)))
        if abs(value.imag) > mp.mpf('1e-30')*max(1, abs(value.real)):
            raise ArithmeticError('The real Coulomb moment acquired an imaginary part')
        return float(value.real)


def angular_products(l, rank):
    """Density multipole times its projection on the outgoing scalar."""
    return angular(2, 2, 1, 1, rank, 1)*angular(l, 3, 2, 2, rank, 1)


def matrix_elements(energy=ENERGY):
    """Laplace-integrated direct, recoil and contact-interaction amplitudes.

    Off-shell energy dependence belongs to canonical recoil; the equivalent
    accelerated-frame dipole is compared only at the physical emitted energy.
    """
    channels = []
    for l in (3, 5):
        moment = lambda p, s: coulomb_moment(l, p, s, energy)
        parts = {}
        for rank in (1, 3):
            angle = 4*math.pi/(2*rank+1)*angular_products(l, rank)
            lower = PRODUCT*math.factorial(rank+5)/DECAY**(rank+6)
            upper = PRODUCT*math.factorial(4-rank)/DECAY**(5-rank)
            overlap = (lower*moment(3-rank, 1/3)
                -lower*sum(DECAY**j/math.factorial(j)*moment(3-rank+j, 7/6)
                           for j in range(rank+6))
                +upper*sum(DECAY**j/math.factorial(j)*moment(4+rank+j, 7/6)
                           for j in range(5-rank)))
            parts[f'direct_L{rank}'] = -angle*PARENT_NORM*overlap
        angle = 4*math.pi/3*angular_products(l, 1)
        recoil = angle*DELTA*(energy+1/18)*DIPOLE*PARENT_NORM*moment(5, 1/3)
        accelerated = angle*DELTA**2*DIPOLE*PARENT_NORM*moment(5, 1/3)
        contact = (sum(angular_products(l, rank) for rank in (1, 3))
                   *PARENT_NORM**2/math.sqrt(24)*moment(7, 7/6))
        total = sum(parts.values())+recoil
        channels.append(dict(l=l, **parts, recoil=recoil, accelerated=accelerated,
                             total=total, contact=contact,
                             coefficient=2*math.pi*total**2))
    return dict(energy=energy, k=math.sqrt(2*energy), channels=channels,
                coefficient=sum(row['coefficient'] for row in channels),
                direct_only_coefficient=2*math.pi*sum(
                    (row['direct_L1']+row['direct_L3'])**2 for row in channels),
                contact_coefficient_in_literature_units=math.pi/32*sum(
                    row['contact']**2 for row in channels))


def radial_quadrature(order=192, cutoff=100., energy=ENERGY):
    """Real-axis Coulomb quadrature, with recoil evaluated by a gradient."""
    if (not isinstance(order, int) or order < 24
            or not np.isfinite([cutoff, energy]).all() or cutoff <= 0 or energy <= 0):
        raise ValueError('Require at least 24 quadrature points and positive scales')
    x, weights = np.polynomial.legendre.leggauss(order)
    radius, weights = cutoff*(x+1)/2, cutoff*weights/2
    parent = PARENT_NORM*radius**2*np.exp(-radius/3)
    channels = []
    for l in (3, 5):
        with mp.workdps(30):
            k = mp.sqrt(2*mp.mpf(energy))
            continuum = np.array([float(
                mp.sqrt(2/(mp.pi*k))*mp.coulombf(l, -1/k, k*r)/r)
                for r in radius])
        measure = weights*radius**2*continuum*parent
        direct = 0.
        for rank in (1, 3):
            lower = (PRODUCT*math.factorial(rank+5)/DECAY**(rank+6)
                     *gammainc(rank+6, DECAY*radius)/radius**(rank+1))
            upper = (PRODUCT*math.factorial(4-rank)/DECAY**(5-rank)
                     *gammaincc(5-rank, DECAY*radius)*radius**rank)
            angle = 4*math.pi/(2*rank+1)*angular_products(l, rank)
            direct -= angle*np.dot(measure, lower+upper)
        # (H_3-E_322)(r R_32)=R_32/3, giving a local momentum source.
        recoil = 4*math.pi/3*angular_products(l, 1)*DELTA*DIPOLE/3*np.sum(measure)
        channels.append(dict(l=l, direct=float(direct), recoil=float(recoil),
                             total=float(direct+recoil)))
    return dict(order=order, cutoff=cutoff, energy=energy, channels=channels)


def radial_source(radius, l):
    """Canonical relative-coordinate source, before continuum projection."""
    radius = np.asarray(radius, dtype=float)
    if l not in (3, 5) or not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError('Require l=3 or 5 and positive finite radii')
    parent = PARENT_NORM*radius**2*np.exp(-radius/3)
    direct = np.zeros_like(radius)
    for rank in (1, 3):
        lower = (PRODUCT*math.factorial(rank+5)/DECAY**(rank+6)
                 *gammainc(rank+6, DECAY*radius)/radius**(rank+1))
        upper = (PRODUCT*math.factorial(4-rank)/DECAY**(5-rank)
                 *gammaincc(5-rank, DECAY*radius)*radius**rank)
        direct -= 4*math.pi/(2*rank+1)*angular_products(l, rank)*parent*(lower+upper)
    recoil = 4*math.pi/3*angular_products(l, 1)*DELTA*DIPOLE/3*parent
    return direct+recoil


def outgoing_response(l, outer=120., tolerance=1e-9):
    """Independent outgoing boundary-value problem and source-work check."""
    if l not in (3, 5) or not np.isfinite([outer, tolerance]).all() or outer < 60 or tolerance <= 0:
        raise ValueError('Require l=3 or 5, outer radius >=60 and positive tolerance')
    inner = .001
    with mp.workdps(35):
        k = mp.mpf(1)/6
        regular = lambda r: mp.coulombf(l, -1/k, k*r)
        outgoing = lambda r: mp.coulombg(l, -1/k, k*r)+1j*mp.coulombf(l, -1/k, k*r)
        inner_ratio = float(regular(inner)/mp.diff(regular, inner))
        outer_slope = complex(mp.diff(outgoing, outer)/outgoing(outer))
    radius = np.unique(np.concatenate([
        np.geomspace(inner, 1., 80), np.linspace(1., outer, 600)]))

    def equation(r, y):
        potential = l*(l+1)/r**2-2/r-2*ENERGY
        return np.vstack([y[1], potential*y[0]+2*r*radial_source(r, l),
                          y[3], potential*y[2]])

    def boundary(left, right):
        u, du = right[0]+1j*right[2], right[1]+1j*right[3]
        mismatch = du-outer_slope*u
        return np.array([left[0]-inner_ratio*left[1],
                         left[2]-inner_ratio*left[3], mismatch.real, mismatch.imag])

    solved = solve_bvp(equation, boundary, radius, np.zeros((4, len(radius))),
                       tol=tolerance, max_nodes=10000)
    if not solved.success:
        raise ArithmeticError(solved.message)
    right = solved.y[:, -1]
    flux = right[0]*right[3]-right[2]*right[1]
    work, error = quad(lambda r: -2*r*float(radial_source(r, l))*solved.sol(r)[2],
                       inner, outer, epsabs=1e-13, epsrel=1e-9, limit=120)
    return dict(l=l, outer=outer, tolerance=tolerance, nodes=len(solved.x),
                max_residual=float(np.max(solved.rms_residuals)),
                outward_flux=float(flux), source_work=work, source_work_error=error)


def reaction_parameters(alpha, absorption_origin, escape_coefficient):
    """Coefficients for d epsilon / d(t/M), using leading 211 saturation."""
    if (not np.isfinite([alpha, absorption_origin, escape_coefficient]).all()
            or not 0 < alpha < .5 or absorption_origin <= 0 or escape_coefficient <= 0):
        raise ValueError('Require 0<alpha<0.5 and positive coefficients')
    spin = 4*alpha/(1+4*alpha*alpha)
    rplus = 1+math.sqrt(1-spin*spin)
    absorption = 2*rplus*absorption_origin**2*alpha**6
    escape = escape_coefficient*alpha**3
    scale = absorption/escape
    ratio = scale/(1-scale+math.sqrt(1-scale+scale*scale))
    decay = 2*ratio-ratio**2/scale
    # A kinematic diagnostic for following the saturation curve after m=0
    # absorption. This term is NOT included in the fixed-background ODE.
    slope = 4*alpha*(3+4*alpha*alpha)/(1+4*alpha*alpha)**2
    restoring = alpha*slope/(1-alpha*slope)
    return dict(alpha=alpha, spin=spin, absorption=absorption, escape=escape,
                escape_over_absorption=1/scale, attracting_ratio=ratio,
                scaled_mother_decay=decay, restoring_211_per_absorbed_partner=restoring)


def reaction_trajectory(parameters, seed_log=-180., initial_ratio=None, tolerance=1e-10):
    """Two reactions on a fixed BH; log population resolves a tiny daughter seed.

    x=epsilon_211/epsilon_211(0), y=epsilon_322/epsilon_211(0).
    Time is A*epsilon_211(0)**2*t/M. H and I count absorbed/emitted quanta
    in the same population units. No superradiance or gravitational radiation.
    """
    if (not np.isfinite([seed_log, tolerance]).all() or not -700 < seed_log < 0
            or tolerance <= 0 or (initial_ratio is not None
                                 and (not np.isfinite(initial_ratio) or initial_ratio <= 0))):
        raise ValueError('Require a finite negative seed log and positive tolerance/ratio')
    beta = parameters['escape_over_absorption']
    ratio = parameters['attracting_ratio']
    y0 = math.exp(seed_log) if initial_ratio is None else initial_ratio

    def equation(t, y):
        x, log_b, _, _ = y
        b = math.exp(log_b)
        incoming, outgoing = x*x*b, beta*x*b*b
        return [-2*incoming+outgoing, x*x-2*beta*x*b, incoming, outgoing]

    def half_cloud(t, y):
        return y[0]-.5

    half_cloud.terminal = True
    half_cloud.direction = -1
    time_bound = 2*(abs(math.log(y0))+3/(2*parameters['scaled_mother_decay'])+100)
    solved = solve_ivp(equation, (0., time_bound), [1., math.log(y0), 0., 0.],
                       method='Radau', rtol=tolerance, atol=tolerance/100,
                       max_step=2., events=half_cloud, dense_output=True)
    if not solved.success or not len(solved.t_events[0]):
        raise ArithmeticError('The two-reaction model did not reach half population')
    times = np.linspace(0., solved.t[-1], 101)
    values = solved.sol(times)
    values[1] = np.exp(values[1])
    x, y, incoming, outgoing = values
    alpha = parameters['alpha']
    energies = 1+alpha*alpha*np.array([-1/8, -1/18, -7/36, 1/72])
    number = x+y+incoming+outgoing
    momentum = x+2*y+3*outgoing
    energy = energies@values
    final = values[:, -1]
    return dict(seed_log=math.log(y0), evaluations=solved.nfev,
                scaled_half_time=float(solved.t[-1]),
                final_daughter_to_mother=float(final[1]/final[0]),
                final_absorbed=float(final[2]), final_escaped=float(final[3]),
                number_error=float(np.max(np.abs(number-(1+y0)))),
                angular_momentum_error=float(np.max(np.abs(momentum-(1+2*y0)))),
                energy_error=float(np.max(np.abs(energy-(energies[0]+energies[1]*y0)))),
                samples=[dict(time=float(t), mother=float(a), daughter=float(b),
                              absorbed=float(h), escaped=float(i))
                         for t, a, b, h, i in zip(times, *values)])
