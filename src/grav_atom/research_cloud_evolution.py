"""Adiabatic two-cloud evolution with an energy-consistent moving black hole.

Cloud populations are rest masses in units of the initial BH mass. The
Hamiltonian retains hydrogenic binding and averaged Newtonian self-gravity.
Pair rates are leading dilute-cloud kernels; this is not a full field evolution.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import solve_ivp, quad_vec
from scipy.linalg import eigh
from scipy.optimize import root

from boson_ep.models import State
from boson_ep.spectrum import detweiler_coefficient, horizon_omega_M
from .research_absorbed_pair import origin_response
from .research_collective_mode import RadialGrid, make_sector, energy_functional
from .research_pair_escape import matrix_elements


@lru_cache(maxsize=1)
def self_gravity_coefficients():
    grid = RadialGrid(40)
    a = make_sector(grid, 1, (1,), 16)
    b = make_sector(grid, 2, (2,), 16)
    ca, cb = eigh(a.h0)[1][:, 0], eigh(b.h0)[1][:, 0]
    ea = energy_functional(grid, [(a, ca)], 1.)
    eb = energy_functional(grid, [(b, cb)], 1.)
    together = energy_functional(grid, [(a, ca), (b, cb)], 1.)
    rotated = energy_functional(grid, [(a, ca), (b, 1j*cb)], 1.)
    cross_direct_exchange = -(together['gravitational']-ea['gravitational']-eb['gravitational'])
    cross_recoil = together['recoil']-ea['recoil']-eb['recoil']
    return dict(aa=-2*ea['gravitational'], bb=-2*eb['gravitational'],
                ab=cross_direct_exchange-cross_recoil,
                ab_gravity=cross_direct_exchange, ab_recoil=cross_recoil,
                relative_phase_energy_error=abs(together['energy']-rotated['energy']))


def cloud_energy(alpha0, mass, a, b, *, self_gravity=True, model=None):
    """Total mass-energy and its three thermodynamic derivatives, in M0 units."""
    if model is not None:
        if not self_gravity:
            raise ValueError('A finite-cloud model includes self-gravity')
        return model.energy(alpha0, mass, a, b)
    c = self_gravity_coefficients() if self_gravity else dict(aa=0., ab=0., bb=0.)
    quadratic = c['aa']*a*a+2*c['ab']*a*b+c['bb']*b*b
    binding = alpha0**2*mass*mass*(a/8+b/18)
    interaction = .5*alpha0**2*mass*quadratic
    energy = mass+a+b-binding-interaction
    wa = 1-alpha0**2*mass*mass/8-alpha0**2*mass*(c['aa']*a+c['ab']*b)
    wb = 1-alpha0**2*mass*mass/18-alpha0**2*mass*(c['ab']*a+c['bb']*b)
    clock = 1-alpha0**2*mass*(a/4+b/9)-.5*alpha0**2*quadratic
    return dict(energy=energy, wa=wa, wb=wb, wh=2*wa-wb, we=2*wb-wa,
                clock=clock, binding=binding, interaction=interaction)


def instantaneous_rates(alpha0, mass, angular_momentum, a, b, *,
                        pair_scale=1., escape_scale=1., self_gravity=True, gw=True,
                        cloud_model=None):
    if (not np.isfinite([alpha0, mass, angular_momentum, a, b,
                         pair_scale, escape_scale]).all()
            or mass <= 0 or alpha0 <= 0 or min(pair_scale, escape_scale) < 0):
        raise ValueError('Require finite positive masses and nonnegative rate factors')
    alpha = alpha0*mass
    spin = angular_momentum/mass**2
    if not .001 <= alpha < .5 or not 0 <= spin < 1 or min(a, b) < 0:
        raise ValueError('State outside the small-alpha, positive-cloud model')
    e = cloud_energy(alpha0, mass, a, b, self_gravity=self_gravity, model=cloud_model)
    clock = e['clock']
    if clock <= 0:
        raise ArithmeticError('The adiabatic energy derivative must stay positive')
    if e['we'] <= 1 or e['wh'] <= 0:
        raise ValueError('Fixed pair kernels require open escape and positive horizon frequency')
    growth = []
    for state, w in [(State(2, 1, 1), e['wa']), (State(3, 2, 2), e['wb'])]:
        coefficient = detweiler_coefficient(alpha, spin, state)
        gamma = (4*(1+math.sqrt(1-spin*spin))*coefficient
                 *(clock*state.m*horizon_omega_M(spin)-alpha*w)
                 *alpha**(4*state.l+5)/mass)
        growth.append(gamma)
    parent_action = a if cloud_model is None else a+b
    if cloud_model is not None and cloud_model.ordinary_horizon is not None:
        factors = cloud_model.ordinary_horizon.factors(parent_action/mass, alpha, spin)
        growth = [growth[i]*factors[i] for i in (0, 1)]
    absorption_constant, escape_constant = (reaction_constants() if cloud_model is None else
                                            cloud_model.reaction_coefficients(parent_action/mass))
    absorption = (pair_scale*absorption_constant*(1+math.sqrt(1-spin*spin))*alpha**6
                  *parent_action**2*b/mass**3)
    escape = escape_scale*escape_constant*alpha**3*parent_action*b*b/mass**3
    annihilation = .02*alpha**14*a*a/mass**2 if gw else 0.
    sa, sb = growth[0]*a, growth[1]*b
    da = sa-2*absorption+escape-annihilation
    db = sb+absorption-2*escape
    dm = (-e['wa']*sa-e['wb']*sb+e['wh']*absorption)/clock
    dj = -(sa+2*sb)/alpha0
    return dict(**e, alpha=alpha, spin=spin, sa=sa, sb=sb,
                gamma_a=growth[0], gamma_b=growth[1],
                absorption=absorption, escape=escape, annihilation=annihilation,
                da=da, db=db, dm=dm, dj=dj,
                horizon_area_driver=dm-horizon_omega_M(spin)/mass*dj)


@lru_cache(maxsize=1)
def reaction_constants():
    return 2*origin_response()['total']**2, matrix_elements()['coefficient']


def uncoupled_saturation(alpha0, spin0=.9, *, self_gravity=True, cloud_model=None):
    """Endpoint from energy/J conservation and zero 211 horizon detuning."""
    guess = alpha0*(spin0-4*alpha0/(1+4*alpha0**2))
    if guess <= 0:
        raise ValueError('Initial BH must support 211 superradiance')

    def equations(x):
        mass, a = x
        spin = (spin0-a/alpha0)/mass**2
        if not 0 < spin < 1:
            return [mass+a-1, (spin-4*alpha0)]
        e = cloud_energy(alpha0, mass, a, 0., self_gravity=self_gravity, model=cloud_model)
        return [e['energy']-1, e['clock']*horizon_omega_M(spin)-alpha0*mass*e['wa']]

    answer = root(equations, [1-guess, guess], tol=1e-11)
    if not answer.success or np.linalg.norm(equations(answer.x)) > 1e-10:
        raise ArithmeticError('Uncoupled saturation equations failed')
    mass, a = answer.x
    return dict(mass=float(mass), mother=float(a),
                spin=float((spin0-a/alpha0)/mass**2))


def evolve(alpha0, *, spin0=.9, seed_log=-180., pair_scale=1., escape_scale=1.,
           self_gravity=True, gw=True, tolerance=2e-9, linear_rates=None, cloud_model=None,
           post_peak_points=80):
    """Follow formation and half loss, optionally integrating positive probe rates."""
    if (not np.isfinite([alpha0, spin0, seed_log, pair_scale, escape_scale, tolerance]).all()
            or not .005 <= alpha0 <= .2 or not 0 < spin0 < 1
            or not -650 < seed_log < -10 or pair_scale < 0 or escape_scale <= 0
            or tolerance <= 0):
        raise ValueError('Invalid evolution parameters')
    if (isinstance(post_peak_points, bool) or not isinstance(post_peak_points, int)
            or not 1 <= post_peak_points <= 100000):
        raise ValueError('post_peak_points must be an integer between 1 and 100000')
    options = dict(pair_scale=pair_scale, escape_scale=escape_scale,
                   self_gravity=self_gravity, gw=gw, cloud_model=cloud_model)
    seed = math.exp(seed_log)
    initial = np.array([seed_log, seed_log, 1., spin0, 0., 0., 0., 0., 0., 0., 0.])
    scale = instantaneous_rates(alpha0, 1., spin0, seed, seed, **options)['gamma_a']
    if scale <= 0:
        raise ValueError('Initial parent is not superradiant')

    def rates(y):
        a, b = np.exp(y[:2])
        return instantaneous_rates(alpha0, y[2], y[3], a, b, **options)

    def equation(t, y):
        a, b = np.exp(y[:2])
        v = rates(y)
        # Evolve the prescribed seeds without underflow; no spontaneous source is added.
        loga = v['gamma_a']-2*v['absorption']/a+v['escape']/a-v['annihilation']/a
        logb = v['gamma_b']+v['absorption']/b-2*v['escape']/b
        return np.array([loga, logb, v['dm'], v['dj'],
                         v['absorption'], v['escape'], v['we']*v['escape'],
                         v['wa']*v['annihilation'], v['annihilation']/alpha0,
                         v['sa'], v['sb']])/scale

    def maximum(t, y):
        return equation(t, y)[0]

    maximum.terminal, maximum.direction = True, -1
    first = solve_ivp(equation, (0., 2000.), initial, method='Radau',
                      rtol=tolerance, atol=tolerance/100, max_step=.25,
                      events=maximum, dense_output=True)
    if not first.success or not len(first.t_events[0]):
        raise ArithmeticError('Formation did not reach a first maximum')
    peak = first.y[:, -1]
    mother_peak = math.exp(peak[0])

    def half(t, y):
        return y[0]-math.log(mother_peak/2)

    half.terminal, half.direction = True, -1
    second = solve_ivp(equation, (first.t[-1], first.t[-1]+1e7), peak,
                       method='Radau', rtol=tolerance, atol=tolerance/100,
                       events=half, dense_output=True, first_step=.01)
    if not second.success or not len(second.t_events[0]):
        raise ArithmeticError('Cloud did not reach half its first peak')
    times1 = np.linspace(0., first.t[-1], 61)
    times2 = np.linspace(first.t[-1], second.t[-1], post_peak_points+1)[1:]
    times = np.concatenate([times1, times2])
    values = np.hstack([first.sol(times1), second.sol(times2)])
    energy0 = cloud_energy(alpha0, 1., seed, seed,
                           self_gravity=self_gravity, model=cloud_model)['energy']
    j0 = spin0+3*seed/alpha0
    rows, error_energy, error_j, error_a, error_b = [], 0., 0., 0., 0.
    area_min, escape_min = math.inf, math.inf
    for t, y in zip(times, values.T):
        a, b = np.exp(y[:2])
        v = rates(y)
        total_energy = v['energy']+y[6]+y[7]
        total_j = y[3]+(a+2*b+3*y[5])/alpha0+y[8]
        error_energy = max(error_energy, abs(total_energy-energy0))
        error_j = max(error_j, abs(total_j-j0))
        error_a = max(error_a, abs(a-seed-y[9]+2*y[4]-y[5]+alpha0*y[8]))
        error_b = max(error_b, abs(b-seed-y[10]-y[4]+2*y[5]))
        area_min = min(area_min, v['horizon_area_driver'])
        escape_min = min(escape_min, v['we']-1)
        rows.append(dict(time_M0=float(t/scale), mother=float(a), daughter=float(b),
                         mass=float(y[2]), spin=v['spin'], absorbed=float(y[4]),
                         escaped=float(y[5]), escaped_energy=float(y[6]),
                         GW_energy=float(y[7]), gamma_a_M0=v['gamma_a'],
                         gamma_b_M0=v['gamma_b'],
                         net_211_horizon_extraction=float(y[9]),
                         net_322_horizon_extraction=float(y[10]),
                         escape_kinetic_energy_over_mu=v['we']-1,
                         mother_derivative_M0=v['da'],
                         fraction_gain_from_pairs=v['absorption']/b,
                         horizon_area_driver=v['horizon_area_driver']))
    reference = uncoupled_saturation(alpha0, spin0,
                                     self_gravity=self_gravity, cloud_model=cloud_model)
    result = dict(alpha0=alpha0, spin0=spin0, seed_log=seed_log, tolerance=tolerance,
                pair_scale=pair_scale, escape_scale=escape_scale, self_gravity=self_gravity,
                gw=gw, time_scale=scale, evaluations=first.nfev+second.nfev,
                peak_time_M0=float(first.t[-1]/scale), peak_mother=mother_peak,
                peak_daughter=math.exp(peak[1]),
                peak_mass=float(peak[2]), peak_spin=float(peak[3]/peak[2]**2),
                half_time_M0=float(second.t[-1]/scale),
                time_after_peak_M0=float((second.t[-1]-first.t[-1])/scale),
                uncoupled_saturation=reference,
                peak_over_uncoupled=mother_peak/reference['mother'],
                energy_conservation_error=error_energy,
                angular_momentum_conservation_error=error_j,
                mother_reaction_count_error=error_a, daughter_reaction_count_error=error_b,
                minimum_escape_kinetic_energy_over_mu=escape_min,
                minimum_horizon_area_driver=area_min,
                maximum_post_peak_over_first_peak=float(
                    np.exp(np.max(second.y[0]))/mother_peak), samples=rows)
    if cloud_model is not None:
        result['cloud_model'] = cloud_model.label
    if linear_rates is not None:
        integrals, errors = [], []
        for solution in [first, second]:
            def positive_rate(t):
                y = solution.sol(t)
                a, b = np.exp(y[:2])
                value = np.asarray(linear_rates(alpha0, y[2], y[3], a, b, rates(y)))
                if value.ndim != 1 or not np.isfinite(value).all():
                    raise ValueError('Linear probe rates must be a finite vector')
                return np.maximum(value, 0.)/scale
            integral, error = quad_vec(positive_rate, solution.t[0], solution.t[-1],
                epsabs=2e-8, epsrel=2e-10, norm='max',
                points=solution.t[::max(1, len(solution.t)//20)])
            integrals.append(integral)
            errors.append(float(error))
        result['linear_positive_exponents'] = (integrals[0]+integrals[1]).tolist()
        result['linear_positive_formation'] = integrals[0].tolist()
        result['linear_positive_after_peak'] = integrals[1].tolist()
        result['linear_quadrature_error_estimates'] = errors
        result['linear_rate_at_half_M0'] = np.asarray(linear_rates(
            alpha0, values[2, -1], values[3, -1],
            math.exp(values[0, -1]), math.exp(values[1, -1]), rates(values[:, -1]))).tolist()
    return result
