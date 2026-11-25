"""Coupled orbital, cloud-frame and black-hole directions in Bohr units.

The cloud frame is parallel transported, with zero angular velocity along its
axis. Angular momentum includes lost particles and negative-energy modes. The
equilibrium profile stays fixed; an optional quadrupolar shape reacts on the orbit.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp

from .research_axis_response import orbital_frame, vector_torque, continuum_vectors
from .research_causal_periapsis import conservative_gradient, mixing_quantities
from .research_cloud_secular import hamiltonian
from .research_periapsis_migration import equilibrium, loss_derivatives, peters_drift


Z = np.array([0., 0., 1.])


def cross_matrix(w):
    x, y, z = w
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def rotation_velocity(j, jz, node, angular_velocity):
    """Element derivatives from a rigid rotation of the entire orbital frame."""
    ci, si = jz/j, np.sqrt(1-(jz/j)**2)
    normal = np.array([si*np.sin(node), -si*np.cos(node), ci])
    ascending = np.array([np.cos(node), np.sin(node), 0.])
    idot = angular_velocity@ascending
    ndot = (angular_velocity[2]-ci*(angular_velocity@normal))/(si*si)
    wdot = angular_velocity@normal-ci*ndot
    return np.array([0., 0., -j*si*idot, wdot, ndot])


def spin_orbit(a, j, jz, node, spin, alpha):
    """Lense-Thirring energy and gradient at fixed cloud-frame spin vector."""
    ci, si, cn, sn = jz/j, np.sqrt(1-(jz/j)**2), np.cos(node), np.sin(node)
    normal = np.array([si*sn, -si*cn, ci])
    di = np.array([ci*sn, -ci*cn, -si])
    dn = np.array([si*cn, si*sn, 0.])
    factor = 2*alpha**3/(a**2.5*j*j)
    value = factor*(spin@normal)
    gradient = np.array([-2.5*value/a,
        -2*value/j+factor*(spin@di)*ci/(j*si),
        -factor*(spin@di)/(j*si), 0., factor*(spin@dn)])
    return dict(value=value, gradient=gradient, spin_gradient=factor*normal)


class MovingAxesSystem:
    def __init__(self, ionization, pole, *, cloud, alpha, spin, q, model, beta,
                 absorption=None, averaged=False, free_axes=True, radiation=True,
                 gamma_scale=1., shape=None):
        if min(cloud, alpha, spin, q, beta) <= 0 or gamma_scale < 0:
            raise ValueError('Require positive physical scales and nonnegative damping')
        self.ion, self.pole, self.absorption = ionization, pole, absorption
        self.cloud, self.alpha, self.spin, self.q = cloud, alpha, spin, q
        self.model, self.beta = model, beta
        self.averaged, self.free_axes, self.radiation = averaged, free_axes, radiation
        self.gamma = gamma_scale*pole.width
        self.cloud_j = cloud/alpha
        self.shape = shape
        self.pole_count = len(pole.groups) if averaged else 1
        shape_count = 0 if shape is None else (len(shape.groups) if averaged else 1)
        self.count = self.pole_count+shape_count
        self.charges = np.r_[np.ones(self.pole_count), 2*np.ones(shape_count)]
        self.frequencies = np.r_[np.full(self.pole_count, pole.frequency),
            np.full(shape_count, shape.frequency if shape is not None else 0.)]
        self.mode_end = 5+2*self.count
        self.frame_start = self.mode_end+3
        self.flow_start = self.frame_start+9

    def source(self, a, e, inclination, omega, node=0.):
        result = self.pole.source(a, e, inclination, omega, node, self.averaged)
        if self.shape is not None:
            extra = self.shape.source(a, e, inclination, omega, node, self.averaged)
            result = {key:np.r_[value, extra[key]] for key, value in result.items()}
        return result

    def damping(self):
        return np.r_[np.full(self.pole_count, self.gamma), np.zeros(self.count-self.pole_count)]

    def split(self, state):
        n, k, p = self.count, self.mode_end, self.frame_start
        return (state[:5], state[5:5+n]+1j*state[5+n:k], state[k:p],
                state[p:self.flow_start].reshape(3, 3), state[self.flow_start:])

    def initial(self, a=3.25, jz=.08715556843599832, delta_e=.008,
                delta_omega=.05, preparation='steady'):
        point = equilibrium(a, jz, self.cloud, self.alpha, self.spin, self.model)
        e, omega = point['e']+delta_e, np.pi/2+delta_omega
        j = np.sqrt(1-e*e)
        src = self.source(a, e, np.arccos(jz/j), omega)
        gradient = conservative_gradient(a, e, omega, jz, self.cloud, self.alpha, self.spin, self.model)
        nu = gradient[2]/np.sqrt(a)
        b = src['f']/(self.frequencies-self.charges*nu+.5j*self.damping())
        if preparation == 'zero':
            b *= 0
        elif preparation != 'steady':
            raise ValueError('Choose a steady or zero initial mode')
        return np.r_[a, j, jz, omega, 0., b.real, b.imag, self.spin*Z,
                     np.eye(3).ravel(), np.zeros(8)]

    def energy(self, state):
        orbit, b, spin, _, flow = self.split(state)
        a, j, jz, omega, node = orbit
        e = np.sqrt(1-j*j)
        src = self.source(a, e, np.arccos(jz/j), omega, node)
        delta = self.frequencies+self.charges*self.beta*(spin[2]-self.spin)
        mixing = self.cloud*self.q*(-np.dot(delta, abs(b)**2)+2*np.vdot(b, src['f']).real)
        base = -1/(2*a)+hamiltonian(a, e, omega, jz, self.cloud, self.alpha, model=self.model)
        c_base = self.cloud_j*(1-flow[0]-flow[1])
        return float(base+spin_orbit(a, j, jz, node, spin, self.alpha)['value']+mixing
                     +self.alpha*self.beta*c_base*spin[2]/self.q)

    def momentum(self, state):
        orbit, b, spin, frame, flow = self.split(state)
        a, j, jz, omega, node = orbit
        normal = orbital_frame(np.arccos(jz/j), omega, node)[:, 2]
        c = self.cloud_j*(1-flow[0]-flow[1]-self.q**2*np.dot(self.charges, abs(b)**2))
        return frame@(c*Z+spin+self.q/self.alpha*np.sqrt(a)*j*normal)+flow[5:8]

    def evaluate(self, state):
        orbit, b, spin, frame, flow = self.split(state)
        a, j, jz, omega, node = orbit
        e, inclination = np.sqrt(1-j*j), np.arccos(jz/j)
        eps, q, alpha = self.cloud, self.q, self.alpha
        c = self.cloud_j*(1-flow[0]-flow[1]-q*q*np.dot(self.charges, abs(b)**2))
        if c <= 0:
            raise ValueError('Cloud angular momentum was exhausted')
        delta = self.frequencies+self.charges*self.beta*(spin[2]-self.spin)
        gamma = self.damping()
        src = self.source(a, e, inclination, omega, node)
        mix = mixing_quantities(src, b, a, j, jz, eps, q, 0.)
        mix['gradient'][4] = -2*eps*q*np.vdot(b, self.charges*src['f']).imag
        base = np.r_[conservative_gradient(a, e, omega, jz, eps, alpha, 0., self.model), 0.]
        lt = spin_orbit(a, j, jz, node, spin, alpha)
        gradient = base+lt['gradient']+mix['gradient']

        def canonical(grad):
            return np.array([0., -grad[3], -grad[4], grad[1], grad[2]])/(eps*np.sqrt(a))

        def torque(velocity):
            return vector_torque(a, j, jz, node, velocity)

        vg, vs = canonical(base+mix['gradient']), canonical(lt['gradient'])
        external, ion_velocity, abs_velocity = np.zeros(5), np.zeros(5), np.zeros(5)
        outgoing, incoming, gw_velocity = np.zeros(3), np.zeros(3), np.zeros(5)
        ni, nh = 0., 0.
        if self.radiation:
            loss, rates = loss_derivatives(a, j, jz, omega, self.ion, alpha, eps, self.averaged, gw=False)
            ion_velocity = np.r_[q*loss, -q*rates['node_torque']/(np.sqrt(a)*j*np.sin(inclination))]
            gw_velocity = np.r_[q*peters_drift(a, j, jz, alpha, eps), 0.]
            ni = rates['number']
            outgoing, incoming = continuum_vectors(self.ion, a, e, inclination, omega, node, self.averaged)
            if self.absorption is not None:
                loss, rates = loss_derivatives(a, j, jz, omega, self.absorption, alpha, eps,
                                               self.averaged, gw=False)
                abs_velocity = np.r_[q*loss, -q*rates['node_torque']/(np.sqrt(a)*j*np.sin(inclination))]
                nh = rates['horizon_number']
            external = ion_velocity+gw_velocity+abs_velocity

        cloud_external = -q/alpha*torque(abs_velocity)+q*q/alpha*incoming
        cloud_torque = -q/alpha*torque(vg)+cloud_external
        c_velocity = (cloud_torque-(cloud_torque@Z)*Z)/c+self.beta/eps*np.cross(spin, Z)
        wframe = np.cross(Z, c_velocity) if self.free_axes else np.zeros(3)
        wexternal = np.cross(Z, cloud_external/c) if self.free_axes else np.zeros(3)
        spin_velocity = -q/alpha*torque(vs)+self.beta*c/eps*np.cross(Z, spin)
        if not self.free_axes:
            spin_velocity *= 0
        spin_velocity -= np.cross(wframe, spin)
        velocity = vg+vs+external+rotation_velocity(j, jz, node, -wframe)
        bdot = ((1j*delta-.5*gamma)*b-1j*src['f'])/eps
        horizon = q*q/eps*(np.dot(gamma, abs(b)**2)+nh)
        escaped = q*q/eps*ni
        radiated_j = frame@(q*q/alpha*outgoing-q/alpha*torque(gw_velocity))
        energy_spin_gradient = lt['spin_gradient']+alpha*self.beta*c/q*Z
        work = gradient@(external+rotation_velocity(j, jz, node, -wexternal))
        work += energy_spin_gradient@(-np.cross(wexternal, spin))
        work += q*np.dot(gamma, delta*abs(b)**2-(b.conj()*src['f']).real)
        work -= eps*self.beta*spin[2]/q*(horizon+escaped)
        result = np.r_[velocity, bdot.real, bdot.imag, spin_velocity,
            (frame@cross_matrix(wframe)).ravel(), horizon, escaped, q*q/eps*nh,
            work, a**(-1.5)/(2*np.pi*eps), radiated_j]
        expected_cdot = -self.cloud_j*(horizon+escaped+2*q*q*np.vdot(b, self.charges*bdot).real)
        return result, dict(cloud_momentum=c, frequency=delta,
            longitudinal_torque_error=float(cloud_torque[2]-expected_cdot),
            cloud_angular_velocity=wframe)


def moving_history(system, *, afinal=3.1, duration=None, tolerance=2e-9,
                   max_step=8., budget=240., samples=801, preparation='steady',
                   retain_samples=False, **initial_options):
    started = time.perf_counter()
    initial = system.initial(preparation=preparation, **initial_options)
    energy0, momentum0 = system.energy(initial), system.momentum(initial)

    def rhs(_, state):
        if time.perf_counter()-started > budget:
            raise TimeoutError('Moving-frame evolution exceeded its bounded run')
        return system.evaluate(state)[0]

    def reached(_, state):
        return state[0]-afinal

    reached.terminal, reached.direction = True, -1
    solution = solve_ivp(rhs, [0., 100/system.q if duration is None else duration], initial,
        method='DOP853', rtol=tolerance, atol=tolerance/100, max_step=max_step,
        events=reached if duration is None else None, dense_output=True)
    if not solution.success or (duration is None and len(solution.t_events[0]) != 1):
        raise RuntimeError(solution.message)
    rows = []
    for t, state in zip(np.linspace(0, solution.t[-1], samples),
                        solution.sol(np.linspace(0, solution.t[-1], samples)).T):
        orbit, b, spin, frame, flow = system.split(state)
        a, j, jz, omega, node = orbit
        axis = frame@Z
        hole = frame@spin/system.spin
        rows.append(dict(s=float(t), a=float(a), e=float(np.sqrt(1-j*j)), jz=float(jz),
            omega=float(omega), node=float(node), inclination=float(np.arccos(jz/j)),
            cloud_tilt=float(np.arccos(np.clip(axis[2], -1., 1.))),
            hole_tilt=float(np.arccos(np.clip(hole[2], -1., 1.))),
            relative_tilt=float(np.arccos(np.clip(spin[2]/system.spin, -1., 1.))),
            horizon_fraction=float(flow[0]), ionization_fraction=float(flow[1]),
            negative_horizon_fraction=float(flow[2]), cycles=float(flow[4]),
            energy_error=float(system.energy(state)-energy0-flow[3]),
            angular_error=float(np.linalg.norm(system.momentum(state)-momentum0)),
            frame_error=float(np.max(abs(frame.T@frame-np.eye(3)))),
            spin_norm_error=float(abs(np.linalg.norm(spin)/system.spin-1)),
            excitation=float(system.q**2*np.vdot(b, b).real),
            shape_excitation=float(system.q**2*np.sum(abs(b[system.pole_count:])**2)),
            mode_real=list(b.real), mode_imag=list(b.imag)))
    final = rows[-1]
    return dict(q=system.q, cloud=system.cloud, alpha=system.alpha, spin=system.spin,
        beta=system.beta, averaged=system.averaged, free_axes=system.free_axes,
        radiation=system.radiation, coupled_shape=system.shape is not None,
        preparation=preparation, final_s=final['s'],
        horizon_fraction=final['horizon_fraction'], ionization_fraction=final['ionization_fraction'],
        negative_horizon_fraction=final['negative_horizon_fraction'],
        total_loss=final['horizon_fraction']+final['ionization_fraction'],
        final_e=final['e'], final_cycles=final['cycles'], tolerance=tolerance, max_step=max_step,
        omega_range=[min(r['omega'] for r in rows), max(r['omega'] for r in rows)],
        e_range=[min(r['e'] for r in rows), max(r['e'] for r in rows)],
        maxima={key:max(r[key] for r in rows) for key in ['cloud_tilt', 'hole_tilt', 'relative_tilt',
            'angular_error', 'frame_error', 'spin_norm_error', 'excitation', 'shape_excitation']},
        maximum_energy_error=max(abs(r['energy_error']) for r in rows),
        nfev=solution.nfev, elapsed_seconds=time.perf_counter()-started,
        diagnostic_times=samples, samples=rows if retain_samples else rows[::max(1, (samples-1)//80)])
