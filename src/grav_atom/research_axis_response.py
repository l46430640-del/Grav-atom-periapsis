"""Linear cloud and hole directions driven by a resolved secular orbit.

The orbit is prescribed from a causal-mode migration history. Conservative
torques, selective continuum loss and the hole's reaction are all retained.
This diagnoses the frozen-axis approximation; it does not re-evolve the orbit.
"""

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline

from .research_causal_periapsis import conservative_gradient, mixing_quantities
from .research_inclined_ionization import angular_matrices, rotated_amplitudes
from .research_periapsis_migration import loss_derivatives


def orbital_frame(inclination, omega, node):
    ci, si, cw, sw = np.cos(inclination), np.sin(inclination), np.cos(omega), np.sin(omega)
    cn, sn = np.cos(node), np.sin(node)
    return np.array([[cn*cw-sn*ci*sw, -cn*sw-sn*ci*cw, sn*si],
                     [sn*cw+cn*ci*sw, -sn*sw+cn*ci*cw, -cn*si],
                     [si*sw, si*cw, ci]])


def vector_torque(a, j, jz, node, velocity):
    """d(sqrt(A)*j*n_orbit)/d sigma from reduced element derivatives."""
    ci, si = jz/j, np.sqrt(1-(jz/j)**2)
    sn, cn = np.sin(node), np.cos(node)
    normal = np.array([si*sn, -si*cn, ci])
    di = np.array([ci*sn, -ci*cn, -si])
    dn = np.array([si*cn, si*sn, 0.])
    adot, jdot, jzdot, _, ndot = velocity
    idot = (ci*jdot-jzdot)/(j*si)
    return ((j*adot/(2*np.sqrt(a))+np.sqrt(a)*jdot)*normal
            +np.sqrt(a)*j*(idot*di+ndot*dn))


def continuum_vectors(response, a, e, inclination, omega, node, averaged=False):
    amplitude, _ = response.amplitudes(a, e)
    loss = 2*np.pi*amplitude.T@amplitude

    def evaluate(angle):
        c = rotated_amplitudes(inclination, angle)
        wave = amplitude@c
        ladder = np.dot(response.raise_factor, wave[response.lower+1].conj()*wave[response.lower])
        outgoing = 2*np.pi*np.array([ladder.real, ladder.imag, np.dot(response.m, abs(wave)**2)])
        incoming = -np.array([np.vdot(c, matrix@loss@c).real for matrix in angular_matrices(1)])
        rotation = orbital_frame(inclination, angle, node)
        return np.array([rotation@outgoing, rotation@incoming])

    return np.mean([evaluate(w) for w in [0, np.pi/2, np.pi, 3*np.pi/2]], axis=0) if averaged else evaluate(omega)


def direction_forcing(row, pole, response, *, cloud, alpha, spin, model, absorptive_response=None):
    a, e, jz, omega, node = [row[key] for key in ['a', 'e', 'jz', 'omega', 'node']]
    j, q = np.sqrt(1-e*e), row['q']
    inclination = np.arccos(jz/j)
    base = conservative_gradient(a, e, omega, jz, cloud, 0., 0., model)
    gravity = np.array([0., -base[3], 0., base[1], base[2]])/(cloud*np.sqrt(a))
    source = pole.source(a, e, inclination, omega, node)
    b = np.array(row['mode_real'])+1j*np.array(row['mode_imag'])
    mix = mixing_quantities(source, b, a, j, jz, cloud, q, pole.frequency)['gradient']
    modal = np.array([0., -mix[3], -mix[4], mix[1], mix[2]])/(cloud*np.sqrt(a))
    spin_orbit = np.array([0., 0., 0., 0., 2*spin*alpha**3/(cloud*a**3*j**3)])
    scalar, rates = loss_derivatives(a, j, jz, omega, response, alpha, cloud, gw=False)
    all_loss, _ = loss_derivatives(a, j, jz, omega, response, alpha, cloud)
    ion = np.r_[q*scalar, -q*rates['node_torque']/(np.sqrt(a)*j*np.sin(inclination))]
    gw = np.r_[q*(all_loss-scalar), 0.]
    outgoing, incoming = continuum_vectors(response, a, e, inclination, omega, node)
    torque_g = vector_torque(a, j, jz, node, gravity+modal)
    torque_s = vector_torque(a, j, jz, node, spin_orbit)
    torque_ion = vector_torque(a, j, jz, node, ion)
    torque_gw = vector_torque(a, j, jz, node, gw)
    torque_abs = np.zeros(3)
    if absorptive_response is not None:
        loss, rates = loss_derivatives(a, j, jz, omega, absorptive_response, alpha, cloud, gw=False)
        velocity = np.r_[q*loss, -q*rates['node_torque']/(np.sqrt(a)*j*np.sin(inclination))]
        torque_abs = vector_torque(a, j, jz, node, velocity)
    # The local s-wave horizon sink carries no vector angular momentum.
    cloud_drive = -q/cloud*(torque_g+torque_abs)+q*q/cloud*incoming
    hole_drive = -q/(alpha*spin)*torque_s
    outward = q*q/alpha*outgoing-q/alpha*torque_gw
    total = (cloud/alpha*cloud_drive+spin*hole_drive+outward
             +q/alpha*(torque_g+torque_s+torque_ion+torque_gw+torque_abs))
    return dict(cloud=cloud_drive, hole=hole_drive, outward=outward,
                local_transverse_balance_error=float(np.linalg.norm(total[:2])))


def axis_response(history, pole, response, *, model, beta, stride=1,
                  tolerance=2e-10, max_step=8., budget=240., absorptive_response=None):
    """Small direction response about the original common cloud/hole axis."""
    started = time.perf_counter()
    if history['averaged'] or beta <= 0 or not isinstance(stride, int) or stride < 1:
        raise ValueError('Require a coherent history, positive coupling and positive integer stride')
    cloud, alpha, spin, q = [history[k] for k in ['cloud', 'alpha', 'spin', 'q']]
    cloud_j = cloud/alpha
    chosen = history['samples'][::stride]
    if chosen[-1]['s'] != history['samples'][-1]['s']:
        chosen.append(history['samples'][-1])
    times = np.array([r['s'] for r in chosen])
    drives = []
    local_error = 0.
    for row in chosen:
        if time.perf_counter()-started > budget:
            raise TimeoutError('Direction response exceeded its bounded run')
        result = direction_forcing(dict(row, q=q), pole, response, cloud=cloud,
            alpha=alpha, spin=spin, model=model, absorptive_response=absorptive_response)
        drives.append([result[key][0]-1j*result[key][1] for key in ['cloud', 'hole', 'outward']])
        local_error = max(local_error, result['local_transverse_balance_error'])
    drive = CubicSpline(times, np.array(drives))
    # eta=x-iy; positive spin-orbit frequency gives eta_dot=-i*frequency*eta.
    decay = alpha**7/12
    matrix = np.array([[-spin, spin], [cloud_j, -cloud_j]])*(1j*beta/cloud)
    matrix += np.array([[-1., 1.], [cloud_j/spin, -cloud_j/spin]])*(decay/cloud)

    def rhs(t, state):
        if time.perf_counter()-started > budget:
            raise TimeoutError('Direction response exceeded its bounded run')
        source = drive(t)
        return np.r_[matrix@state[:2]+source[:2], source[2]]

    solution = solve_ivp(rhs, [times[0], times[-1]], np.zeros(3, dtype=complex),
        method='DOP853', rtol=tolerance, atol=tolerance/100, max_step=max_step,
        t_eval=times)
    if not solution.success:
        raise RuntimeError(solution.message)
    initial = chosen[0]
    j0 = np.sqrt(1-initial['e']**2)
    initial_orbit = orbital_frame(initial['inclination'], initial['omega'], initial['node'])[:, 2]
    initial_j = q/alpha*np.sqrt(initial['a'])*j0*(initial_orbit[0]-1j*initial_orbit[1])
    rows = []
    for row, (eta_c, eta_s, j_out) in zip(chosen, solution.y.T):
        a, e, inclination, omega, node = [row[k] for k in ['a', 'e', 'inclination', 'omega', 'node']]
        frame = orbital_frame(inclination, omega, node)
        peri, normal = frame[:, 0], frame[:, 2]
        axis = np.array([eta_c.real, -eta_c.imag, 1.])
        axis /= np.linalg.norm(axis)
        relative_i = np.arccos(np.clip(axis@normal, -1., 1.))
        ascending = np.cross(axis, normal)
        ascending /= np.linalg.norm(ascending)
        relative_w = np.arctan2(peri@np.cross(normal, ascending), peri@ascending)
        old_rate = pole.rates(a, e, inclination, omega)['number']
        new_rate = pole.rates(a, e, relative_i, relative_w)['number']
        uniform = pole.rates(a, e, relative_i, relative_w, True)['number']
        orbit_j = q/alpha*np.sqrt(a)*np.sqrt(1-e*e)*(normal[0]-1j*normal[1])
        balance = cloud_j*eta_c+spin*eta_s+orbit_j+j_out-initial_j
        rows.append(dict(s=row['s'], cloud_tilt=float(abs(eta_c)), hole_tilt=float(abs(eta_s)),
            relative_tilt=float(abs(eta_c-eta_s)), relative_inclination=float(relative_i),
            relative_omega=float(relative_w), static_rate_before=old_rate,
            static_rate_after=new_rate, static_to_uniform=new_rate/uniform,
            transverse_balance_error=float(abs(balance))))
    return dict(scope='Linear direction response on a prescribed coherent orbit. Relative angles and static rates are posterior diagnostics; no orbit feedback is included.',
        q=q, cloud=cloud, alpha=alpha, spin=spin, beta=beta, stride=stride,
        tolerance=tolerance, max_step=max_step, forcing_points=len(times),
        relative_frequency=beta*(cloud_j+spin),
        maximum_cloud_tilt=max(r['cloud_tilt'] for r in rows),
        maximum_hole_tilt=max(r['hole_tilt'] for r in rows),
        maximum_relative_tilt=max(r['relative_tilt'] for r in rows),
        maximum_static_to_uniform=max(r['static_to_uniform'] for r in rows),
        relative_omega_range=[min(r['relative_omega'] for r in rows), max(r['relative_omega'] for r in rows)],
        maximum_transverse_balance_error=max(r['transverse_balance_error'] for r in rows),
        maximum_local_balance_error=local_error, elapsed_seconds=time.perf_counter()-started,
        samples=rows[::max(1, (len(rows)-1)//80)])
