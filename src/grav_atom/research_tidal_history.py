"""Three-mode cloud and circular companion in a weak-field Hamiltonian model.

The counterrotating 21-1 population is dilute. Its self-gravity uses the
leading exchange integral on the finite-parent energy model. Horizon and
reaction rates are adiabatic; this is not a relativistic field evolution.
"""
import math
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from boson_ep.models import State
from boson_ep.spectrum import detweiler_coefficient, horizon_omega_M
from .research_competing_levels import A, B, cross_energy
from .research_moving_resonance import EXCHANGE


class ThreeModeModel:
    def __init__(self, parent):
        self.parent = parent
        self.cross_difference = (cross_energy(B, (2, 1, -1))['total']
                                 -cross_energy(B, A)['total'])

    def energy(self, alpha0, mass, j, a, b, c):
        base = self.parent.energy(alpha0, mass, a+c, b)
        quadratic = EXCHANGE*a*c+self.cross_difference*b*c
        hf_action = a-c+16*b/135
        h = alpha0**6*j*mass**3/12
        hf = h*hf_action/alpha0
        energy = base['energy']-alpha0**2*mass*quadratic+hf
        wa = base['wa']-alpha0**2*mass*EXCHANGE*c+h/alpha0
        wb = base['wb']-alpha0**2*mass*self.cross_difference*c+16*h/(135*alpha0)
        wc = (base['wa']-alpha0**2*mass*(EXCHANGE*a+self.cross_difference*b)
              -h/alpha0)
        clock = base['clock']-alpha0**2*quadratic+3*hf/mass
        spin_derivative = alpha0**5*mass**3*hf_action/12
        return dict(energy=energy, wa=wa, wb=wb, wc=wc, wh=2*wa-wb,
                    we=2*wb-wa, clock=clock, spin_derivative=spin_derivative,
                    gap=2*h+alpha0**3*mass*(EXCHANGE*(a-c)+self.cross_difference*b),
                    hyperfine=h)

    def rates(self, alpha0, mass, j, a, b, c):
        if (not np.isfinite([alpha0, mass, j, a, b, c]).all()
                or min(alpha0, mass) <= 0 or min(a, b, c) < 0):
            raise ValueError('Require finite positive masses and nonnegative populations')
        alpha, spin = alpha0*mass, j/mass**2
        if not .001 < alpha < .5 or not 0 <= spin < 1:
            raise ValueError('State outside the weak-field positive-spin model')
        e = self.energy(alpha0, mass, j, a, b, c)
        if e['clock'] <= 0 or e['wh'] <= 0 or e['we'] <= 1:
            raise ValueError('Energy derivatives or reaction frequencies left the model domain')
        rplus = 1+math.sqrt(1-spin*spin)
        omega_h = horizon_omega_M(spin)
        gammas = []
        for state, name in [(State(2, 1, 1), 'wa'), (State(3, 2, 2), 'wb'),
                            (State(2, 1, -1), 'wc')]:
            detuning = (e['clock']*state.m*omega_h
                +mass*state.m*e['spin_derivative']-alpha*e[name])
            gammas.append(4*rplus*detweiler_coefficient(alpha, spin, state)*detuning
                          *alpha**(4*state.l+5)/mass)
        sa, sb, sc = np.array(gammas)*np.array([a, b, c])
        absorption, escape = self.parent.reaction_coefficients((a+b+c)/mass)
        horizon = absorption*rplus*alpha**6*(a+b)**2*b/mass**3
        infinity = escape*alpha**3*(a+b)*b*b/mass**3
        annihilation = .02*alpha**14*a*a/mass**2
        da, db, dc = sa-2*horizon+infinity-annihilation, sb+horizon-2*infinity, sc
        dj = -(sa+2*sb-sc)/alpha0
        dm = (-e['wa']*sa-e['wb']*sb-e['wc']*sc+e['wh']*horizon
              -e['spin_derivative']*dj)/e['clock']
        radiation = e['we']*infinity+e['wa']*annihilation
        return dict(**e, alpha=alpha, spin=spin, gamma_a=gammas[0], gamma_b=gammas[1],
            gamma_c=gammas[2], sa=sa, sb=sb, sc=sc, absorption=horizon, escape=infinity,
            annihilation=annihilation, da=da, db=db, dc=dc, dm=dm, dj=dj,
            radiation=radiation, radiation_j=(3*infinity+annihilation)/alpha0,
            horizon_area_driver=dm-omega_h*dj/mass)


def orbital_state(alpha0, mass, central, companion, angular, x, tidal_scale=1.):
    """Canonical circular-orbit Hamiltonian and leading quadrupole coupling."""
    total = central+companion
    radius = angular**2*total/(central*companion)**2
    omega = math.sqrt(total/radius**3)
    energy = -central*companion/(2*radius)
    energy_s = energy*(3/central-1/total)
    eta = tidal_scale*9*companion/(alpha0**3*mass**2*radius**3)
    eta_s = eta*(6/central-3/total)
    canonical_omega = omega-6*eta*x/(alpha0*angular)
    clock = 1+energy_s+eta_s*x/alpha0
    gw_power = 32/5*(central*companion)**2*total/radius**5
    specific_j = angular*companion/(central*total)
    radiation_redshift = 1+energy_s+omega*specific_j
    return dict(radius=radius, omega=omega, canonical_omega=canonical_omega,
        energy=energy, energy_s=energy_s, eta=eta, eta_s=eta_s, clock=clock,
        gw_power=gw_power, specific_j=specific_j, radiation_redshift=radiation_redshift)


def cloud_orbit(model, alpha0, mass, j, a, b, c, central, companion, angular, x,
                tidal_scale=1.):
    """Canonical orbit with an optional action-dependent cloud quadrupole."""
    factor, difference = (model.tidal_parameters(alpha0, mass, j, a, b, c)
        if hasattr(model, 'tidal_parameters') else (1., 0.))
    orbit = orbital_state(alpha0, mass, central, companion, angular, x, tidal_scale*factor)
    orbit['tidal_phase'] = orbit['eta']*x*difference/factor
    orbit['tidal_log_gradient'] = (model.tidal_log_gradient(alpha0, mass, j, a, b, c)
        if hasattr(model, 'tidal_log_gradient') else np.zeros(5))
    return orbit


def isolated_history(alpha0, model, *, spin0=.9, seed_log=-180., tolerance=2e-9):
    """Retain dense output for locating both passages through a moving spectrum."""
    seed = math.exp(seed_log)
    initial = np.array([seed_log, seed_log, 1., spin0])
    scale = model.rates(alpha0, 1., spin0, seed, seed, 0.)['gamma_a']

    def equation(t, y):
        a, b = np.exp(y[:2])
        r = model.rates(alpha0, y[2], y[3], a, b, 0.)
        return np.array([r['da']/a, r['db']/b, r['dm'], r['dj']])/scale

    def jacobian(t, y):
        # Keep exponentially dilute log-population columns inside the model
        # domain instead of letting adaptive differencing enlarge them freely.
        steps = 1e-7*np.maximum(1.,abs(y))
        return np.column_stack([(equation(t,y+np.eye(4)[i]*steps[i])
            -equation(t,y-np.eye(4)[i]*steps[i]))/(2*steps[i]) for i in range(4)])

    def peak(t, y):
        return equation(t, y)[0]

    peak.terminal, peak.direction = True, -1
    first = solve_ivp(equation, (0., 2000.), initial, method='Radau',jac=jacobian,
        rtol=tolerance, atol=tolerance/100, max_step=.25, events=peak, dense_output=True)
    if not first.success or not len(first.t_events[0]):
        raise ArithmeticError('No isolated-cloud peak')
    maximum = math.exp(first.y[0, -1])

    def half(t, y):
        return y[0]-math.log(maximum/2)

    half.terminal, half.direction = True, -1
    second = solve_ivp(equation, (first.t[-1], first.t[-1]+1e7), first.y[:, -1],
        method='Radau', jac=jacobian,rtol=tolerance, atol=tolerance/100, first_step=.01,
        events=half, dense_output=True)
    if not second.success or not len(second.t_events[0]):
        raise ArithmeticError('No isolated-cloud half-loss point')
    return dict(alpha0=alpha0, spin0=spin0, seed_log=seed_log, scale=scale,
                first=first, second=second, peak=maximum, model=model)


def crossing_setup(history, *, fraction=.7, mass_ratio=.001):
    model, alpha0, scale = history['model'], history['alpha0'], history['scale']
    first, second = history['first'], history['second']
    if not .5 <= fraction < 1 or not 0 < mass_ratio <= .1:
        raise ValueError('Require a post-peak fraction and a small companion')
    target = brentq(lambda t: second.sol(t)[0]-math.log(fraction*history['peak']),
                    second.t[0], second.t[-1])
    y = second.sol(target)
    a, b = np.exp(y[:2])
    r = model.rates(alpha0, y[2], y[3], a, b, 0.)
    companion = mass_ratio*y[2]
    omega = r['gap']/2
    seed = math.exp(history['seed_log'])
    initial_energy = model.energy(alpha0, 1., history['spin0'], seed, seed, 0.)['energy']
    initial_omega = omega*((initial_energy+companion)/(r['energy']+companion))**2
    angular = initial_energy*companion/(initial_energy+companion)**(1/3)*initial_omega**(-1/3)
    def frequency_difference(t):
        y = first.sol(t)
        aa, bb = np.exp(y[:2])
        e = model.energy(alpha0, y[2], y[3], aa, bb, 0.)
        orbital = initial_omega*((e['energy']+companion)/(initial_energy+companion))**2
        return 2*orbital-e['gap']
    growth_crossing = brentq(frequency_difference, first.t[0], first.t[-1])
    growth_state = first.sol(growth_crossing)
    ga, gb = np.exp(growth_state[:2])
    growth_rates = model.rates(alpha0, growth_state[2], growth_state[3], ga, gb, 0.)
    def sweep(y, v):
        aa, bb = np.exp(y[:2])
        mass, j = y[2:4]
        if hasattr(model, 'gap_gradient'):
            gradient = model.gap_gradient(alpha0,mass,j,aa,bb,0.)
            gap_dot = np.dot(gradient,[v['dm'],v['dj'],v['da'],v['db'],v['dc']])
        else:
            gap_dot = (alpha0**6/6*(mass**3*v['dj']+3*j*mass**2*v['dm'])
                +alpha0**3*(EXCHANGE*(mass*v['da']+aa*v['dm'])
                             +model.cross_difference*(mass*v['db']+bb*v['dm'])))
        return -2*v['gap']*v['radiation']/(v['energy']+companion)-gap_dot
    return dict(fraction=fraction, mass_ratio=mass_ratio, companion=companion,
        initial_omega=initial_omega, initial_angular=angular,
        target_time=target, target_time_M0=target/scale, target_rates=r,
        target_state=y.tolist(), growth_time=growth_crossing,
        growth_time_M0=growth_crossing/scale, growth_rates=growth_rates,
        growth_state=growth_state.tolist(), growth_sweep=sweep(growth_state, growth_rates),
        target_sweep=sweep(y, r))


def off_resonant_history(history, setup, initial_time, initial_state, *, window=40.,
                        tolerance=2e-9):
    """Propagate free coherence between isolated resonances, omitting weak forcing.

    Unlike evolving the fast complex ratio directly, its unwrapped phase is
    a smooth quadrature. Switching the small coupling off and on records
    explicit work; varying the window tests the approximation, not just the ODE.
    """
    model, alpha0, scale = history['model'], history['alpha0'], history['scale']
    companion = setup['companion']
    initial = np.array(initial_state, dtype=float)
    a, b = np.exp(initial[:2])
    ratio = complex(initial[4], initial[5])
    c = a*abs(ratio)**2
    if c <= 0:
        raise ValueError('The free coherent amplitude must be nonzero')
    e = model.energy(alpha0, initial[2], initial[3], a, b, c)
    x = 2*a*ratio.real
    o = cloud_orbit(model,alpha0,initial[2],initial[3],a,b,c,e['energy'],companion,initial[6],x)
    initial_detuning = 2*o['canonical_omega']-o['clock']*e['gap']
    if initial_detuning > -.5*window*math.sqrt(abs(setup['growth_sweep'])):
        raise ValueError('Growth passage has not left resonance; free propagation is invalid')
    initial[4:6] = [math.log(c), math.atan2(ratio.imag, ratio.real)]
    switch_off = -o['eta']*x/alpha0
    initial[12] += switch_off

    def properties(y):
        a, b, c = np.exp(y[[0, 1, 4]])
        v = model.rates(alpha0, y[2], y[3], a, b, c)
        o = orbital_state(alpha0, y[2], v['energy'], companion, y[6], 0., 0.)
        delta = 2*o['omega']-o['clock']*v['gap']
        return a, b, c, v, o, delta

    def equation(t, y):
        a, b, c, v, o, delta = properties(y)
        l_gw = -o['gw_power']/o['omega']
        l_mass = -o['specific_j']*v['radiation']
        return np.array([v['da']/a, v['db']/b, v['dm'], v['dj'],
            v.get('counter_fractional_rate', v['gamma_c']),
            -delta, l_gw+l_mass, o['radiation_redshift']*v['radiation'], v['radiation_j'],
            o['gw_power'], -l_gw, -l_mass, 0.])/scale

    def approach(t, y):
        return properties(y)[-1]+window*math.sqrt(setup['target_sweep'])

    approach.direction, approach.terminal = 1, True
    solution = solve_ivp(equation, (initial_time, history['second'].t[-1]*1.2), initial,
        method='Radau', rtol=tolerance, atol=tolerance/100, max_step=.05,
        dense_output=True, events=approach)
    if not solution.success or not len(solution.t_events[0]):
        raise ArithmeticError('The second resonance was not reached')
    energies, momenta, areas = [], [], []
    for y in solution.y.T:
        a, b, c, v, o, delta = properties(y)
        energies.append(v['energy']+o['energy']+y[7]+y[9]-y[12])
        momenta.append(y[3]+(a+2*b-c)/alpha0+y[6]+y[8]+y[10]+y[11])
        areas.append(v['horizon_area_driver'])
    peak_time = brentq(lambda t: properties(solution.sol(t))[3]['da'],
                      initial_time, solution.t[-1])
    maximum_mother = math.exp(solution.sol(peak_time)[0])
    final = solution.y[:, -1].copy()
    a, b, c, v, o, delta = properties(final)
    ratio = math.sqrt(c/a)*np.exp(1j*final[5])
    final[4:6] = [ratio.real, ratio.imag]
    x = 2*a*ratio.real
    full = cloud_orbit(model,alpha0,final[2],final[3],a,b,c,v['energy'],companion,final[6],x)
    switch_on = full['eta']*x/alpha0
    final[12] += switch_on
    return dict(start_time=initial_time, stop_time=float(solution.t[-1]),
        window=window, tolerance=tolerance, evaluations=solution.nfev,
        initial_energy_balance=energies[0], final_energy_balance=energies[-1],
        initial_angular_balance=momenta[0], final_angular_balance=momenta[-1],
        energy_error=max(abs(e-energies[0]) for e in energies),
        angular_momentum_error=max(abs(j-momenta[0]) for j in momenta),
        minimum_area_driver=min(areas),
        maximum_mother=maximum_mother, peak_time=peak_time,
        peak_over_isolated=maximum_mother/history['peak'],
        free_phase=float(solution.y[5, -1]), counter_over_mother=c/a,
        log_counter_over_mother=float(solution.y[4, -1]-solution.y[0, -1]),
        incoming_amplitude=abs(ratio), forced_amplitude_estimate=abs(full['eta']/delta),
        switch_off_work=switch_off, switch_on_work=switch_on,
        final_state=final.tolist()), solution


def binary_history(history, setup, *, tidal_scale=1., tolerance=2e-8,
                   method='Radau', start_time=None, initial_state=None, stop_time=None,
                   departure_window=None, departure_absolute=False, stop_fraction=.5,
                   max_response_fraction=None, daughter_limit=None, daughter_direction=0,
                   track_coherent_memory=False, track_reaction_cycle=False, seconds_limit=240.):
    """Follow the same orbit through cloud formation and depletion.

    Counterrotating amplitude is represented by its ratio to the mother,
    retaining both forced response and free coherence near each resonance.
    Energy exchange with the small tidal interaction energy is tracked as
    an explicit dissipative correction rather than silently discarded.
    The optional coherent-memory monitor assumes linear quantum gain/loss
    and a coherent entrance; enable it only during a dilute matching interval.
    Reaction quadratures measure transport without modifying the evolution.
    """
    model, alpha0, scale = history['model'], history['alpha0'], history['scale']
    companion = setup['companion']
    started, evaluations = time.monotonic(), [0]
    # log(a), log(b), M, J, Re(c/a), Im(c/a), L, radiated E/J, orbital GW E/J,
    # orbital angular momentum of lost mass, and dissipative tidal work.
    initial = np.r_[history['first'].y[:, 0], 0., 0., setup['initial_angular'], np.zeros(6)]

    def properties(y):
        a, b = np.exp(y[:2])
        ratio = complex(y[4], y[5])
        c = a*abs(ratio)**2
        x, yy = 2*a*y[4], 2*a*y[5]
        rates = model.rates(alpha0, y[2], y[3], a, b, c)
        orbital = cloud_orbit(model,alpha0,y[2],y[3],a,b,c,rates['energy'],companion,y[6],x,tidal_scale)
        delta = 2*orbital['canonical_omega']-orbital['clock']*rates['gap']
        return a, b, c, x, yy, ratio, rates, orbital, delta

    def equation(t, y):
        evaluations[0] += 1
        if evaluations[0] % 1000 == 0 and time.monotonic()-started > seconds_limit:
            raise TimeoutError(f'Coherent passage exceeded its wall-clock allowance at t={t:.9g} '
                               f'after {evaluations[0]} evaluations')
        a, b, c, x, yy, ratio, v, o, delta = properties(y)
        fa, fb = v['da']/a, v['db']/b
        eta = o['eta']
        gamma_c = v.get('counter_fractional_rate', v['gamma_c'])
        dratio = -1j*eta*(1-ratio**2)-1j*(delta+o['tidal_phase'])*ratio+(gamma_c-fa)*ratio/2
        l_gw = -o['gw_power']/o['omega']
        l_mass = -o['specific_j']*v['radiation']
        dl = -2*eta*yy/alpha0+l_gw+l_mass
        # The conservative Hamiltonian closes exactly; open amplitude rates
        # modify its tiny interaction term by this explicitly retained work.
        tidal_work = (x/alpha0*(-o['eta_s']*v['radiation']
            -2*eta*v['dm']/y[2]-6*eta*(l_gw+l_mass)/y[6])
            +eta/alpha0*(fa+gamma_c)*x/2)
        tidal_work += eta*x/alpha0*np.dot(o['tidal_log_gradient'],
            [v['dm'],v['dj'],v['da'],v['db'],v['dc']])
        result = [fa+2*eta*y[5], fb, v['dm'], v['dj'],
            dratio.real, dratio.imag, dl, o['radiation_redshift']*v['radiation'],
            v['radiation_j'], o['gw_power'], -l_gw, -l_mass, tidal_work]
        if track_coherent_memory:
            result.append(-v['quantum_fraction']*v['birth_rate']/b)
        if track_reaction_cycle:
            result.extend([v['absorption'], v.get('induced_flux', 0.), -v['sc'],
                           v['sb'], -eta*yy, v['escape']])
        return np.array(result)/scale

    if start_time is None:
        start_time = 0.
    if initial_state is None:
        a, b, c, x, yy, ratio, v, o, delta = properties(initial)
        damping = (v['da']/a-v.get('counter_fractional_rate', v['gamma_c']))/2
        forced = -1j*o['eta']/complex(damping, delta)
        initial[4:6] = [forced.real, forced.imag]
    else:
        initial = np.asarray(initial_state, dtype=float)
    if len(initial) != 13:
        raise ValueError('A coherent history requires the thirteen physical and balance variables')
    if track_coherent_memory:
        initial = np.r_[initial,0.]
    cycle_index = len(initial)
    if track_reaction_cycle:
        initial = np.r_[initial,np.zeros(6)]
    if stop_time is None:
        stop_time = history['second'].t[-1]*1.2
    if not 0 < stop_fraction < 1:
        raise ValueError('The population stop fraction must lie between zero and one')

    def half(t, y):
        return y[0]-math.log(history['peak']*stop_fraction)

    half.direction, half.terminal = -1, True
    events = [half]
    departure_index, response_index, daughter_index = None, None, None
    if departure_window is not None:
        def departure(t, y):
            detuning = properties(y)[-1]
            return (abs(detuning) if departure_absolute else detuning)-departure_window*math.sqrt(setup['target_sweep'])
        departure.direction, departure.terminal = 1, True
        departure_index = len(events)
        events.append(departure)
    if max_response_fraction is not None:
        def response_limit(t, y):
            a, b, c, x, yy, ratio, v, o, delta = properties(y)
            return v.get('induced_response_fraction', v.get('induced_occupation', 0.)/b)-max_response_fraction
        response_limit.direction, response_limit.terminal = 1, True
        response_index = len(events)
        events.append(response_limit)
    if daughter_limit is not None:
        if not np.isfinite(daughter_limit) or daughter_limit <= 0 or daughter_direction not in (-1,0,1):
            raise ValueError('Require a positive daughter threshold and a valid crossing direction')
        def daughter_crossing(t,y):
            return y[1]-math.log(daughter_limit)
        daughter_crossing.direction, daughter_crossing.terminal = daughter_direction, True
        daughter_index = len(events)
        events.append(daughter_crossing)
    def jacobian(t, y):
        # The spectator begins exponentially dilute. Adaptive differencing of
        # almost-zero columns can otherwise sample far outside the cloud table.
        matrix = np.zeros((len(y), len(y)))
        for column in range(7):
            step = 1e-7*max(1., abs(y[column]))
            offset = np.zeros(len(y))
            offset[column] = step
            matrix[:, column] = (equation(t, y+offset)-equation(t, y-offset))/(2*step)
        return matrix
    solver_options = dict(jac=jacobian) if method in ('Radau', 'BDF') else {}
    solution = solve_ivp(equation, (start_time, stop_time), initial, method=method,
        rtol=tolerance, atol=tolerance/100, dense_output=True, events=events,
        first_step=min(.0001, abs(stop_time-start_time)), max_step=.5, **solver_options)
    if not solution.success:
        raise ArithmeticError(solution.message)
    times = np.unique(np.r_[np.linspace(start_time, solution.t[-1], 161),
        solution.t[::max(1, len(solution.t)//320)],
        np.linspace(max(start_time, history['first'].t[-1]-.05), solution.t[-1], 161)])
    times = times[(times >= start_time) & (times <= solution.t[-1])]
    values = solution.sol(times)
    rows, energy_error, j_error, area_min = [], 0., 0., math.inf
    total_initial, j_initial = None, None
    minimum_radius = math.inf
    maximum_c = float(np.max(solution.y[4]**2+solution.y[5]**2))
    for t, y in zip(times, values.T):
        a, b, c, x, yy, ratio, v, o, delta = properties(y)
        total_energy = v['energy']+o['energy']+o['eta']*x/alpha0+y[7]+y[9]-y[12]
        total_j = y[3]+(a+2*b-c)/alpha0+y[6]+y[8]+y[10]+y[11]
        if total_initial is None:
            total_initial, j_initial = total_energy, total_j
        energy_error = max(energy_error, abs(total_energy-total_initial))
        j_error = max(j_error, abs(total_j-j_initial))
        area_min = min(area_min, v['horizon_area_driver'])
        minimum_radius = min(minimum_radius, o['radius']*alpha0**2*y[2])
        maximum_c = max(maximum_c, c/a)
        rows.append(dict(time_M0=float(t/scale), time=float(t), mother=float(a),
            daughter=float(b), counterrotating=float(c), mass=float(y[2]), spin=v['spin'],
            omega=o['omega'], resonance=v['gap']/2, detuning=delta, tidal_coupling=o['eta'],
            cloud_gap=v['gap'], intrinsic_mother_rate=v['da']/a,
            mother_horizon_rate=v['gamma_a'], mother_horizon_flux=v['sa'],
            counter_horizon_flux=v['sc'],
            counter_decay=-v['gamma_c'], radius_over_Bohr=o['radius']*alpha0**2*y[2],
            orbital_angular=float(y[6]), horizon_area_driver=v['horizon_area_driver']))
        if 'induced_flux' in v:
            rows[-1].update(induced_flux=v['induced_flux'], induced_occupation=v['induced_occupation'],
                daughter_gain=v['absorption']/b, daughter_induced_loss=v['induced_flux']/b,
                counter_fractional_rate=v['counter_fractional_rate'])
    late = [r for r in rows if r['time'] > start_time+.8*(solution.t[-1]-start_time)]
    compact = rows[::max(1, len(rows)//100)]
    if compact[-1] is not rows[-1]:
        compact.append(rows[-1])
    result = dict(alpha0=alpha0, seed_log=history['seed_log'], tidal_scale=tidal_scale,
        tolerance=tolerance, method=method, evaluations=solution.nfev, steps=len(solution.t),
        mass_ratio=setup['mass_ratio'], fraction=setup['fraction'],
        initial_omega=setup['initial_omega'], start_time=start_time, stop_time=float(solution.t[-1]),
        reached_half=bool(stop_fraction == .5 and len(solution.t_events[0])),
        reached_population_limit=bool(len(solution.t_events[0])), stop_fraction=stop_fraction,
        departed=bool(departure_index is not None and len(solution.t_events[departure_index])),
        response_limit_reached=bool(response_index is not None and len(solution.t_events[response_index])),
        daughter_limit_reached=bool(daughter_index is not None and len(solution.t_events[daughter_index])),
        daughter_limit=daughter_limit, daughter_direction=daughter_direction,
        departure_window=departure_window, seconds=time.monotonic()-started,
        departure_absolute=departure_absolute,
        energy_error=energy_error,
        initial_energy_balance=total_initial, final_energy_balance=total_energy,
        initial_angular_balance=j_initial, final_angular_balance=total_j,
        angular_momentum_error=j_error, minimum_area_driver=area_min,
        minimum_radius_over_Bohr=minimum_radius, maximum_counter_over_mother=maximum_c,
        final_tidal_dissipative_work=float(solution.y[12, -1]-initial[12]),
        final_frequency_ratio=rows[-1]['omega']/setup['initial_omega'],
        late_sampled_maximum_scaled_detuning=max(abs(r['detuning']) for r in late)
            /math.sqrt(setup['target_sweep']),
        final_state=solution.y[:13, -1].tolist(), samples=compact)
    if track_coherent_memory:
        result['log_coherent_memory_fraction'] = float(solution.y[13,-1])
        result['coherent_memory_fraction'] = float(math.exp(solution.y[13,-1]))
    if track_reaction_cycle:
        h, induced, direct, sb, tide, escape = solution.y[cycle_index:,-1]
        ai, bi = np.exp(initial[:2])
        ci = ai*(initial[4]**2+initial[5]**2)
        end = solution.y[:13,-1]
        af, bf = np.exp(end[:2])
        cf = af*(end[4]**2+end[5]**2)
        result['reaction_integrals'] = dict(
            parent_pair_absorption=float(h), induced_absorption=float(induced),
            direct_counter_absorption=float(direct), daughter_horizon=float(sb),
            tidal_transfer=float(tide), pair_escape=float(escape),
            daughter_change=float(bf-bi), counter_change=float(cf-ci),
            daughter_balance_error=float(bf-bi-sb-h+2*escape+induced),
            counter_balance_error=float(cf-ci-tide+induced+direct),
            orbital_balance_error=float(end[6]-initial[6]+end[10]-initial[10]
                                        +end[11]-initial[11]-2*tide/alpha0))
    return result, solution
