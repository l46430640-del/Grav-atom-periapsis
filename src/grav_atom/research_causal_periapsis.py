"""Canonical slow cloud mode coupled to a migrating inclined orbit.

The negative-energy mode is normalized to one unit of relative cloud
angular momentum. Only its resolved horizon width is damped; the scalar
continuum and Peters fluxes remain local. Time is epsilon*alpha**3*t/M.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp

from .research_cloud_secular import anomaly_grid, axisymmetric_field, derivatives, hamiltonian
from .research_periapsis_migration import equilibrium, loss_derivatives
from .research_static_tide import StaticTideTable, angular_factors


class PoleTide(StaticTideTable):
    def __init__(self, axes, values, labels, frequency, width):
        super().__init__(axes, values, labels)
        if not np.isfinite([frequency, width]).all() or frequency <= 0 or width <= 0:
            raise ValueError('This model requires a decaying negative-energy pole')
        self.frequency, self.width = frequency, width
        self.magnetic = np.array([m for _, m in labels])
        self.groups = [self.magnetic == m for m in np.unique(self.magnetic)]

    def source(self, a, e, inclination, omega, node=0., averaged=False):
        if not (self.axes[0][0] <= a <= self.axes[0][-1] and self.axes[1][0] <= e <= self.axes[1][-1]):
            raise ValueError(f'Orbit outside pole source table: a={a}, e={e}')
        c = self.spline([a, e])
        da = self.spline([a, e], nu=(1, 0))
        de = self.spline([a, e], nu=(0, 1))
        d, di, dw = angular_factors(self.labels, inclination, omega)
        rows = np.array([c*d, da*d, de*d, c*di, c*dw])
        groups = self.groups if averaged else [np.ones(len(c), dtype=bool)]
        # Equal angular harmonics still interfere in the averaged comparison.
        grouped = np.array([np.sum(rows[:, mask], axis=1) for mask in groups]).T
        grouped *= self.frequency/np.sqrt(self.width)*np.exp(1j*node)
        return dict(zip(['f', 'a', 'e', 'inclination', 'omega'], grouped))


def conservative_gradient(a, e, omega, jz, cloud, alpha, spin, model):
    """H_A,H_j,H_jz,H_omega at fixed remaining reduced orbital variables."""
    j = np.sqrt(1-e*e)
    sini = np.sqrt(1-(jz/j)**2)
    cosine, sine, weight = anomaly_grid(96)
    measure = 1-e*cosine
    axis = (j*sine*np.cos(omega)+(cosine-e)*np.sin(omega))/measure
    field, radial, polar = axisymmetric_field(a*measure, sini*axis, model)
    ha = (cloud*np.dot(weight*measure**2, radial)+6*alpha**2/(a**3*j)
          -5*spin*alpha**3*jz/(a**3.5*j**3))
    hz = (cloud*np.dot(weight*measure, -jz/(j*j*sini)*axis*polar)
          +2*spin*alpha**3/(a**2.5*j**3))
    hj, hw = derivatives(a, e, omega, jz, cloud, alpha, spin, model=model)
    return np.array([ha+1/(2*a*a), hj, hz, hw])


def mixing_quantities(source, amplitude, a, j, jz, cloud, q, frequency):
    """Hamiltonian and its orbit gradient, holding the inertial cloud mode fixed."""
    e = np.sqrt(1-j*j)
    cosi = jz/j
    sini = np.sqrt(1-cosi*cosi)
    f = source['f']
    fj = -j/e*source['e']+cosi/(j*sini)*source['inclination']
    fz = -source['inclination']/(j*sini)
    inner = lambda values:np.vdot(amplitude, values)
    stored = float(np.vdot(amplitude, amplitude).real)
    work = inner(f)
    gradient = 2*cloud*q*np.array([inner(source['a']).real, inner(fj).real,
        inner(fz).real, inner(source['omega']).real, -work.imag])
    value = cloud*q*(-frequency*stored+2*work.real)
    return dict(value=float(value), gradient=gradient, action=stored,
                source_exchange=float(2*work.imag), overlap=float(work.real))


def coupled_history(ionization, pole, *, cloud, alpha, spin, model, q,
                    a0=3., afinal=2.85, jz0=.08715556843599832,
                    delta_e=0., delta_omega=0., averaged=False, initial='steady',
                    tolerance=2e-9, max_step=8., budget=240., gamma_scale=1.,
                    radiation=True, duration=None, sample_count=401, include_modes=False):
    """Propagate orbit, mode, terminal particle flows, and exact model balances.

    The averaged comparison has independent copies of the pole for distinct
    angular source harmonics. It removes their interference while retaining
    their individual causal phases, forces, and stored actions.
    """
    if (not np.isfinite([cloud, q, tolerance, max_step, budget, gamma_scale]).all()
            or min(cloud, q, tolerance, max_step, budget) <= 0 or gamma_scale < 0
            or (duration is not None and (not np.isfinite(duration) or duration <= 0))):
        raise ValueError('Require positive scales and a nonnegative damping factor')
    if duration is None and not a0 > afinal > 0:
        raise ValueError('An inward migration must have positive decreasing endpoints')
    if not isinstance(sample_count, int) or sample_count < 3:
        raise ValueError('At least three diagnostic times are required')
    started = time.perf_counter()
    point = equilibrium(a0, jz0, cloud, alpha, spin, model)
    e0, omega0 = point['e']+delta_e, np.pi/2+delta_omega
    j0 = np.sqrt(1-e0*e0)
    source = pole.source(a0, e0, np.arccos(jz0/j0), omega0, averaged=averaged)
    nu = conservative_gradient(a0, e0, omega0, jz0, cloud, alpha, spin, model)[2]/np.sqrt(a0)
    gamma, frequency = gamma_scale*pole.width, pole.frequency
    if initial == 'steady':
        b0 = source['f']/(frequency-nu+.5j*gamma)
    elif initial == 'zero':
        b0 = np.zeros_like(source['f'])
    else:
        raise ValueError('Choose steady or zero initial mode')
    count = len(b0)
    offset = 5+2*count
    initial_state = np.r_[a0, j0, jz0, omega0, 0., b0.real, b0.imag, np.zeros(6)]

    def unpack(state):
        return state[5:5+count]+1j*state[5+count:offset]

    def energy(state):
        a, j, jz, omega, node = state[:5]
        e, inclination = np.sqrt(1-j*j), np.arccos(jz/j)
        source = pole.source(a, e, inclination, omega, node, averaged)
        mix = mixing_quantities(source, unpack(state), a, j, jz, cloud, q, frequency)
        return -1/(2*a)+hamiltonian(a, e, omega, jz, cloud, alpha, spin, model=model)+mix['value']

    energy0 = energy(initial_state)
    angular0 = np.sqrt(a0)*jz0-cloud*q*np.vdot(b0, b0).real

    def rhs(_, state):
        if time.perf_counter()-started > budget:
            raise TimeoutError('Causal migration exceeded its bounded run')
        a, j, jz, omega, node = state[:5]
        e, inclination = np.sqrt(1-j*j), np.arccos(jz/j)
        amplitude = unpack(state)
        source = pole.source(a, e, inclination, omega, node, averaged)
        base = conservative_gradient(a, e, omega, jz, cloud, alpha, spin, model)
        mix = mixing_quantities(source, amplitude, a, j, jz, cloud, q, frequency)
        gradient = np.r_[base, 0.]+mix['gradient']
        orbit = np.array([0., -gradient[3], -gradient[4], gradient[1], gradient[2]])/(cloud*np.sqrt(a))
        external = np.zeros(5)
        ion_number, fast_horizon, outgoing_axis = 0., 0., 0.
        if radiation:
            loss, rates = loss_derivatives(a, j, jz, omega, ionization, alpha, cloud, averaged)
            external[:4] = q*loss
            external[4] = -q*rates['node_torque']/(np.sqrt(a)*j*np.sin(inclination))
            fast_horizon = rates.get('horizon_number', 0.)
            ion_number = rates['number']-fast_horizon
            # Separate the Peters contribution from the scalar orbit torque.
            full_axis = np.sqrt(a)*external[2]+jz*external[0]/(2*np.sqrt(a))
            gw_axis = full_axis+q*rates['axial_torque']
            outgoing_axis = q*(rates['axial_torque']+rates['number'])-gw_axis
        orbit += external
        bdot = ((1j*frequency-.5*gamma)*amplitude-1j*source['f'])/cloud
        horizon = q*q/cloud*(gamma*mix['action']+fast_horizon)
        escaped = q*q/cloud*ion_number
        damping_work = q*gamma*(frequency*mix['action']-mix['overlap'])
        external_work = gradient@external
        cycles = a**(-1.5)/(2*np.pi*cloud)
        return np.r_[orbit, bdot.real, bdot.imag, cycles, horizon, escaped,
                     damping_work+external_work, outgoing_axis, q*q/cloud*fast_horizon]

    def reached(_, state):
        return state[0]-afinal

    reached.terminal, reached.direction = True, -1
    solution = solve_ivp(rhs, [0., 100/q if duration is None else duration], initial_state,
        method='DOP853', rtol=tolerance, atol=tolerance/100, max_step=max_step,
        events=reached if duration is None else None, dense_output=True)
    if not solution.success or (duration is None and len(solution.t_events[0]) != 1):
        raise RuntimeError(solution.message)
    times = np.linspace(0, solution.t[-1], sample_count)
    states = solution.sol(times)
    rows = []
    for t, state in zip(times, states.T):
        a, j, jz, omega, node = state[:5]
        action = float(np.vdot(unpack(state), unpack(state)).real)
        cycles, absorbed, escaped, work, outgoing_j, fast_absorbed = state[offset:]
        energy_error = energy(state)-energy0-work
        angular = np.sqrt(a)*jz-cloud*q*action-cloud/q*(absorbed+escaped)
        angular_error = angular+outgoing_j-angular0
        rows.append(dict(s=float(t), a=float(a), e=float(np.sqrt(1-j*j)), jz=float(jz),
            omega=float(omega), node=float(node), inclination=float(np.arccos(jz/j)),
            cycles=float(cycles), horizon_fraction=float(absorbed),
            negative_horizon_fraction=float(fast_absorbed),
            static_horizon_fraction=float(absorbed-fast_absorbed),
            ionization_fraction=float(escaped), relative_excitation_action=float(q*q*action),
            energy_balance_error=float(energy_error), axial_balance_error=float(angular_error)))
        if include_modes:
            rows[-1]['mode_real'] = list(unpack(state).real)
            rows[-1]['mode_imag'] = list(unpack(state).imag)
    final = rows[-1]
    samples = rows[::2]
    if samples[-1] is not final:
        samples.append(final)
    return dict(q=q, averaged=averaged, initial=initial, a0=a0, afinal=afinal,
        cloud=cloud, alpha=alpha, spin=spin, jz0=jz0, delta_e=delta_e, delta_omega=delta_omega,
        tolerance=tolerance, max_step=max_step, gamma_scale=gamma_scale, radiation=radiation,
        diagnostic_sample_count=sample_count,
        frequency=frequency, width=gamma, final_s=final['s'],
        final_time_M=final['s']/(cloud*alpha**3), final_cycles=final['cycles'],
        horizon_fraction=final['horizon_fraction'], ionization_fraction=final['ionization_fraction'],
        negative_horizon_fraction=final['negative_horizon_fraction'],
        static_horizon_fraction=final['static_horizon_fraction'],
        total_cloud_loss_fraction=final['horizon_fraction']+final['ionization_fraction'],
        maximum_excitation_action=max(row['relative_excitation_action'] for row in rows),
        final_e=final['e'], final_inclination=final['inclination'],
        e_range=[min(row['e'] for row in rows), max(row['e'] for row in rows)],
        omega_range=[min(row['omega'] for row in rows), max(row['omega'] for row in rows)],
        maximum_energy_balance_error=max(abs(row['energy_balance_error']) for row in rows),
        maximum_axial_balance_error=max(abs(row['axial_balance_error']) for row in rows),
        nfev=solution.nfev, elapsed_seconds=time.perf_counter()-started, samples=samples)
