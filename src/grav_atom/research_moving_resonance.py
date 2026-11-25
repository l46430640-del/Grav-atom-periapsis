"""Local tidal crossing of the 211/21-1 doublet in a depleting cloud.

Frequencies use 1/M0, their derivatives 1/M0**2. The spectral shift is the
leading hydrogenic direct-plus-exchange result. Local passage freezes the
depletion drift and orbital coefficients; it is not a binary history.
"""
import math

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from boson_ep.models import State
from boson_ep.spectrum import detweiler_coefficient, horizon_omega_M


EXCHANGE = 27/1280


def resonance(alpha0, mass, angular_momentum, mother, rates):
    if (not np.isfinite([alpha0, mass, angular_momentum, mother]).all()
            or min(alpha0, mass, mother) <= 0):
        raise ValueError('Require finite positive masses and coupling')
    if not np.isfinite([rates[key] for key in ('da', 'dm', 'dj')]).all():
        raise ValueError('Require finite background rates')
    spin = angular_momentum/mass**2
    if not 0 <= spin < 1:
        raise ValueError('Require a subextremal nonnegative spin')
    h = angular_momentum*alpha0**6*mass**3/12
    interaction = EXCHANGE*mother*alpha0**3*mass
    hdot = alpha0**6/12*(mass**3*rates['dj']
                           +3*angular_momentum*mass**2*rates['dm'])
    udot = EXCHANGE*alpha0**3*(mass*rates['da']+mother*rates['dm'])
    return dict(omega=h+interaction/2, hyperfine=h, interaction=interaction,
                omega_dot=hdot+udot/2, hyperfine_dot=hdot, interaction_dot=udot)


def encounter(alpha0, mass, angular_momentum, mother, rates, mass_ratio):
    if not math.isfinite(mass_ratio) or not 0 < mass_ratio <= .3:
        raise ValueError('Require a mass ratio between zero and 0.3')
    result = resonance(alpha0, mass, angular_momentum, mother, rates)
    q, omega = mass_ratio, result['omega']
    alpha, spin, number = alpha0*mass, angular_momentum/mass**2, mother/alpha0
    momentum = q*mass**(5/3)*omega**(-1/3)/(1+q)**(1/3)
    coupling = 9*q/(1+q)*mass*omega**2/alpha**3
    gw_drift = 96/5*q/(1+q)**(1/3)*mass**(5/3)*omega**(11/3)
    sweep = 2*(gw_drift-result['omega_dot'])
    if sweep <= 0:
        raise ValueError('The local model requires positive relative sweep')
    backreaction = 6*omega*number/momentum
    feedback = backreaction-result['interaction']
    gamma_minus = (4*(1+math.sqrt(1-spin*spin))
        *detweiler_coefficient(alpha, spin, State(2, 1, -1))
        *(-horizon_omega_M(spin)-alpha*(1-alpha**2/8))*alpha**9/mass)
    result.update(alpha0=alpha0, mass=mass, spin=spin, mother=mother, mass_ratio=q,
        orbital_angular_momentum=momentum, cloud_number=number, tidal_coupling=coupling,
        orbital_frequency_drift=gw_drift, relative_sweep=sweep,
        drift_over_orbital=-result['omega_dot']/gw_drift,
        orbital_feedback=backreaction, net_feedback=feedback,
        feedback_ratio=backreaction/result['interaction'],
        frozen_LZ=coupling**2/(2*gw_drift), moving_LZ=coupling**2/sweep,
        scaled_coupling=coupling/math.sqrt(sweep), scaled_feedback=feedback/math.sqrt(sweep),
        minus_decay_over_sweep=abs(gamma_minus)/math.sqrt(sweep),
        cloud_fractional_loss_over_sweep=-rates['da']/mother/math.sqrt(sweep),
        radius_over_Bohr=((1+q)/(mass*omega)**2)**(1/3)*alpha**2)
    return result


def stationary_initial(coupling, feedback, time):
    """Continue the upper adiabatic state near the initially occupied pole."""
    def equation(theta):
        return ((time+feedback*(math.cos(theta)-1))*math.sin(theta)
                +2*coupling*math.cos(theta))
    angles = np.geomspace(1e-14, math.pi/2, 256)
    left = 0.
    for right in angles:
        if equation(right) < 0:
            theta = brentq(equation, left, right, xtol=1e-14)
            return np.array([math.sin(theta), 0., math.cos(theta)])
        left = right
    raise ArithmeticError('No initially polarized upper branch in the requested interval')


def local_passage(coupling, feedback=0., *, stop=30., preparation=1., tolerance=2e-10):
    """Integrate the local nonlinear LZ normal form, not a full conversion history.

    s=sqrt(relative_sweep)*t; delta=s+feedback*(z-1); dot z=2*coupling*y.
    The Hamiltonian work is included as an independent conservation check.
    """
    if (not np.isfinite([coupling, feedback, stop, preparation, tolerance]).all()
            or min(coupling, stop, preparation, tolerance) <= 0):
        raise ValueError('Require positive finite coupling, interval, and tolerance')
    start = -preparation*max(40., 8*(abs(feedback)*coupling**2)**(1/3), 8*coupling)
    initial = stationary_initial(coupling, feedback, start)

    def energy(s, state):
        x, _, z = state[:3]
        return coupling*x-s*z/2-feedback*(z-1)**2/4

    def equation(s, state):
        x, y, z = state[:3]
        delta = s+feedback*(z-1)
        return [delta*y, -delta*x-2*coupling*z, 2*coupling*y, -z/2]

    solution = solve_ivp(equation, (start, stop), np.r_[initial, 0.],
        method='DOP853', rtol=tolerance, atol=tolerance/100, max_step=.2, dense_output=True)
    if not solution.success:
        raise ArithmeticError(solution.message)
    times = np.linspace(start, stop, 1001)
    states = solution.sol(times)
    norms = np.sum(states[:3]**2, axis=0)
    energies = np.array([energy(s, y) for s, y in zip(times, states.T)])
    end = solution.y[:, -1]
    delta = stop+feedback*(end[2]-1)
    return dict(coupling=coupling, feedback=feedback, start=start, stop=stop,
        preparation=preparation, tolerance=tolerance, evaluations=solution.nfev,
        final_bloch=end[:3].tolist(), transferred=(1-end[2])/2,
        final_detuning=delta, norm_error=float(max(abs(norms-1))),
        work_error=float(max(abs(energies-energy(start, initial)-states[3]))),
        work_scale=float(max(1., max(abs(energies)), max(abs(states[3])))),
        samples=[dict(time=float(s), transferred=float((1-y[2])/2),
                      detuning=float(s+feedback*(y[2]-1)))
                 for s, y in zip(times[::50], states.T[::50])])


def absorbing_passage(coupling, orbital, interaction, decay, *, stop=30.,
                      preparation=1., tolerance=2e-10):
    """Local crossing with occupation decay of the counterrotating state.

    n and z are normalized to the initial particle number, not the surviving
    number. w records only tidal angular momentum exchange; absorption must
    not be counted as a torque on the orbit. Black-hole spin changes due to
    this extra absorption are beyond this leading local model.
    """
    parameters = [coupling, orbital, interaction, decay, stop, preparation, tolerance]
    if (not np.isfinite(parameters).all()
            or min(coupling, stop, preparation, tolerance) <= 0
            or min(orbital, interaction, decay) < 0):
        raise ValueError('Require finite nonnegative feedback and decay parameters')
    feedback = orbital-interaction
    start = -preparation*max(40., 8*(abs(feedback)*coupling**2)**(1/3), 8*coupling)
    initial = stationary_initial(coupling, feedback, start)

    def energy(s, state):
        x, _, z, _, w = state[:5]
        return coupling*x-s*z/2-orbital*w*w/4+interaction*(z-1)**2/4

    def equation(s, state):
        x, y, z, n, w = state[:5]
        delta = s+orbital*w-interaction*(z-1)
        loss = decay*(n-z)/2
        transfer = 2*coupling*y
        work = -z/2-decay*coupling*x/2+(-s+interaction*(z-1))*loss/2
        return [delta*y-decay*x/2, -delta*x-2*coupling*z-decay*y/2,
                transfer+loss, -loss, transfer, loss, work]

    state0 = np.r_[initial, 1., initial[2]-1, 0., 0.]
    solution = solve_ivp(equation, (start, stop), state0,
        method='DOP853', rtol=tolerance, atol=tolerance/100, max_step=.2, dense_output=True)
    if not solution.success:
        raise ArithmeticError(solution.message)
    times = np.linspace(start, stop, 1001)
    states = solution.sol(times)
    x, y, z, n, w, loss, work = states
    detuning = times+orbital*w-interaction*(z-1)
    energies = np.array([energy(s, row) for s, row in zip(times, states.T)])
    return dict(coupling=coupling, orbital=orbital, interaction=interaction, decay=decay,
        start=start, stop=stop, preparation=preparation, tolerance=tolerance,
        evaluations=solution.nfev, initial_bloch=initial.tolist(),
        final_state=states[:6, -1].tolist(), mother=float((n[-1]+z[-1])/2),
        daughter=float((n[-1]-z[-1])/2), absorbed=float(loss[-1]),
        tidal_transfer=float(-w[-1]/2), final_detuning=float(detuning[-1]),
        purity_error=float(max(abs(x*x+y*y+z*z-n*n))),
        number_error=float(max(abs(n+loss-1))),
        angular_momentum_error=float(max(abs(z-w-loss-1))),
        minimum_population=float(min(np.min((n+z)/2), np.min((n-z)/2))),
        work_error=float(max(abs(energies-energy(start, state0)-work))),
        work_scale=float(max(1., max(abs(energies)), max(abs(work)))),
        samples=[dict(time=float(s), mother=float((row[3]+row[2])/2),
            daughter=float((row[3]-row[2])/2), absorbed=float(row[5]),
            detuning=float(delta))
            for s, row, delta in zip(times[::50], states.T[::50], detuning[::50])])


def evolving_passage(snapshot, *, stop=1200., preparation=1., loss_power=3.,
                     tolerance=2e-9, tidal_scale=1.):
    """Reduced evolving cloud and orbit with a prescribed intrinsic loss law.

    Intrinsic loss is dA/ds=-d*A**loss_power. The initial snapshot fixes its
    rate, but this is not the three-level reaction history. Black-hole mass
    and spin are fixed. Orbital angular momentum, tidal coupling, and the
    Newtonian exchange gap evolve. Pure-state and angular-momentum balances
    provide checks independent of the integration tolerance.
    """
    if (not np.isfinite([stop, preparation, loss_power, tolerance, tidal_scale]).all()
            or min(stop, preparation, tolerance) <= 0 or loss_power < 1
            or tidal_scale < 0):
        raise ValueError('Require a positive interval and loss exponent at least one')
    root_sweep = math.sqrt(snapshot['relative_sweep'])
    omega_scale = snapshot['omega']/root_sweep
    eta0 = snapshot['scaled_coupling']*tidal_scale
    u = snapshot['interaction']/root_sweep
    b = snapshot['orbital_feedback']/root_sweep
    decay = snapshot['minus_decay_over_sweep']
    loss_rate = snapshot['cloud_fractional_loss_over_sweep']
    gw = snapshot['orbital_frequency_drift']/snapshot['relative_sweep']
    momentum = snapshot['orbital_angular_momentum']/snapshot['cloud_number']
    if loss_rate <= 0:
        raise ValueError('Require a depleting snapshot')
    start = -preparation*40.
    if loss_power == 1:
        initial_number = math.exp(-loss_rate*start)
    else:
        base = 1+(loss_power-1)*loss_rate*start
        if base <= 0:
            raise ValueError('Preparation precedes the finite-time loss-law domain')
        initial_number = base**(-1/(loss_power-1))
    polarization = (stationary_initial(eta0, b-u, start) if eta0 > 0
                    else np.array([0., 0., 1.]))
    x, y, z = initial_number*polarization
    initial_momentum = momentum*(1+gw*start/omega_scale)**(-1/3)
    initial_momentum -= z-initial_number
    initial = np.array([x, y, (initial_number+z)/2, (initial_number-z)/2,
                        initial_momentum, 0., 0., 0.])

    def equation(s, state):
        x, y, a, c, angular = state[:5]
        ratio = (momentum/angular)**3
        eta = eta0*ratio**2
        delta = 2*omega_scale*(ratio-1)-u*(a-c-1)
        intrinsic = loss_rate*a**loss_power
        absorption = decay*c
        attenuation = (loss_rate*a**(loss_power-1)+decay)/2
        transfer = eta*y
        gw_torque = angular*gw/(3*omega_scale)*ratio**(8/3)
        return [delta*y-attenuation*x,
                -delta*x-2*eta*(a-c)-attenuation*y,
                transfer-intrinsic, -transfer-absorption,
                -2*transfer-gw_torque, intrinsic, absorption, gw_torque]

    solution = solve_ivp(equation, (start, stop), initial, method='DOP853',
        rtol=tolerance, atol=tolerance/100, max_step=.5, dense_output=True)
    if not solution.success:
        raise ArithmeticError(solution.message)
    times = np.linspace(start, stop, 2001)
    states = solution.sol(times)
    x, y, a, c, angular, intrinsic, absorption, gw_loss = states
    ratio = (momentum/angular)**3
    delta = 2*omega_scale*(ratio-1)-u*(a-c-1)
    torque = -2*eta0*ratio**2*y
    total_angular = angular+a-c+intrinsic-absorption+gw_loss
    return dict(alpha0=snapshot['alpha0'], mass_ratio=snapshot['mass_ratio'],
        start=start, stop=stop, preparation=preparation, tolerance=tolerance,
        loss_power=loss_power, tidal_scale=tidal_scale,
        scaled_loss_rate=loss_rate, evaluations=solution.nfev,
        initial_number=initial_number, final_mother=float(a[-1]),
        final_daughter=float(c[-1]), intrinsic_loss=float(intrinsic[-1]),
        absorbed=float(absorption[-1]), final_frequency_ratio=float(ratio[-1]),
        final_detuning=float(delta[-1]),
        late_maximum_detuning=float(max(abs(delta[times > .8*stop]))),
        late_positive_orbital_torque_fraction=float(np.mean(torque[times > .8*stop] > 0)),
        purity_error=float(max(abs(x*x+y*y-4*a*c))),
        number_error=float(max(abs(a+c+intrinsic+absorption-initial_number))),
        angular_momentum_error=float(max(abs(total_angular-total_angular[0]))),
        samples=[dict(time=float(s), mother=float(row[2]), daughter=float(row[3]),
            intrinsic_loss=float(row[5]), absorbed=float(row[6]),
            frequency_ratio=float(r), detuning=float(d))
            for s, row, r, d in zip(times[::100], states.T[::100], ratio[::100], delta[::100])])
