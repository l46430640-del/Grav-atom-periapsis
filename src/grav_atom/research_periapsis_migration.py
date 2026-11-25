"""Slow migration through a conservative periapsis libration region.

Local fluxes and torques drive the orbit; cloud and black-hole parameters
are frozen. Time s=epsilon*alpha**3*t/M has no orbital factor.
"""

import time
from functools import lru_cache

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import NdBSpline, make_interp_spline
from scipy.optimize import brentq

from .research_cloud_secular import derivatives
from .research_inclined_ionization import angular_matrices, rotated_amplitudes
from .research_collective_mode import angular, make_sector
from .research_rydberg_storage import radial
from .research_tidal_continuum import source_grid
from .source_normalization import directed_companion_coefficient


class ResponseTable:
    """Tensor cubic interpolation of real amplitudes, not of positive rates."""

    def __init__(self, axes, projections):
        first = projections[0][0]
        labels = [(row['harmonic'], channel['ell'], channel['final_m'])
                  for row in first['harmonics'] for channel in row['channels']]
        arrays = []
        for line in projections:
            entries = []
            for point in line:
                local = [(row['harmonic'], c['ell'], c['final_m'])
                         for row in point['harmonics'] for c in row['channels']]
                if local != labels:
                    raise ValueError('Harmonic thresholds cannot lie inside a response table')
                entries.append([c['amplitudes'] for row in point['harmonics'] for c in row['channels']])
            arrays.append(entries)
        self._initialize(axes, labels, arrays)

    @classmethod
    def from_arrays(cls, axes, channels, amplitudes):
        """Load a grid stored with one shared (h, ell, m) axis."""
        obj = cls.__new__(cls)
        obj._initialize(axes, channels, amplitudes)
        return obj

    def _initialize(self, axes, labels, arrays):
        self.a, self.e = map(np.asarray, axes)
        self.values = np.asarray(arrays)
        if self.values.shape != (len(self.a), len(self.e), len(labels), 3):
            raise ValueError('Continuum data must resolve each grid point, channel and parent m')
        labels = list(map(tuple, labels))
        groups = list(dict.fromkeys((h,l) for h,l,m in labels))
        if labels != [(h,l,m) for h,l in groups for m in range(-l,l+1)]:
            raise ValueError('Each continuum multiplet must contain consecutive ascending m')
        sa = make_interp_spline(self.a, self.values, axis=0)
        se = make_interp_spline(self.e, sa.c, axis=1)
        self.spline = NdBSpline((sa.t, se.t), np.swapaxes(se.c, 0, 1), (3, 3), extrapolate=False)
        self.h = np.array([row[0] for row in labels])
        self.m = np.array([row[2] for row in labels])
        self.lower = np.array([i for i, (_, ell, m) in enumerate(labels) if m < ell])
        self.raise_factor = np.array([np.sqrt(ell*(ell+1)-m*(m+1))
                                      for _, ell, m in labels if m < ell])
        # Local drift and transverse radiation torque use the same vertices.
        self.amplitudes = lru_cache(maxsize=1)(self.amplitudes)

    def amplitudes(self, a, e):
        if not (self.a[0] <= a <= self.a[-1] and self.e[0] <= e <= self.e[-1]):
            raise ValueError(f'Orbit outside response table: a={a}, e={e}')
        return self.spline([a, e]), self.spline([a, e], nu=(0, 1))

    def rates(self, a, e, inclination, omega, averaged=False):
        amplitude, derivative_e = self.amplitudes(a, e)
        loss = 2*np.pi*amplitude.T@amplitude

        def evaluate(angle):
            c = rotated_amplitudes(inclination, angle)
            wave = amplitude@c
            probabilities = abs(wave)**2
            ladder = np.dot(self.raise_factor, wave[self.lower+1].conj()*wave[self.lower])
            outgoing = 2*np.pi*np.array([ladder.real, ladder.imag, np.dot(self.m, probabilities)])
            incoming = -np.array([np.vdot(c, matrix@loss@c).real for matrix in angular_matrices(1)])
            torque = outgoing+incoming
            axis = np.array([np.sin(inclination)*np.sin(angle),
                             np.sin(inclination)*np.cos(angle), np.cos(inclination)])
            generator = np.array([np.cos(angle), -np.sin(angle), 0])
            return np.array([2*np.pi*np.sum(probabilities),
                2*np.pi*np.dot(self.h, probabilities), torque[2], np.dot(axis, torque),
                2*np.pi*np.vdot(wave, derivative_e@c).imag, np.dot(generator, torque)])

        result = np.mean([evaluate(w) for w in [0, np.pi/2, np.pi, 3*np.pi/2]], axis=0) if averaged else evaluate(omega)
        return dict(zip(['number', 'energy', 'normal_torque', 'axial_torque',
                         'eccentric_connection', 'node_torque'], result))


def loss_derivatives(a, j, jz, omega, response, alpha, cloud, averaged=False, gw=True):
    """Dissipative d(a,j,jz,omega)/d(q*s), with fixed cloud axis."""
    e = np.sqrt(1-j*j)
    inclination = np.arccos(jz/j)
    rates = response.rates(a, e, inclination, omega, averaged)
    ce = rates['energy']
    out = np.array([-2*np.sqrt(a)*ce, (j*ce-rates['normal_torque'])/np.sqrt(a),
        (jz*ce-rates['axial_torque'])/np.sqrt(a),
        (j/e*rates['eccentric_connection']
         +np.cos(inclination)/(j*np.sin(inclination))*rates['node_torque'])/np.sqrt(a)])
    if gw:
        out += peters_drift(a, j, jz, alpha, cloud)
    return out, rates


def peters_drift(a, j, jz, alpha, cloud):
    e2 = 1-j*j
    adot = -64/5*alpha**5/cloud*(1+73*e2/24+37*e2**2/96)/(a**3*j**7)
    jdot = 304/15*alpha**5/cloud*e2*(1+121*e2/304)/(a**4*j**6)
    return np.array([adot, jdot, jz/j*jdot, 0.])


def equilibrium(a, jz, cloud, alpha, spin=0., model='full', bracket=(.1, .7)):
    fun = lambda e: derivatives(a, e, np.pi/2, jz, cloud, alpha, spin, model=model)[0]
    e = brentq(fun, *bracket, xtol=2e-12)
    j = np.sqrt(1-e*e)
    step = 2e-5
    hjj = (fun(np.sqrt(1-(j+step)**2))-fun(np.sqrt(1-(j-step)**2)))/(2*step)
    hww = (derivatives(a, e, np.pi/2+step, jz, cloud, alpha, spin, model=model)[1]
           -derivatives(a, e, np.pi/2-step, jz, cloud, alpha, spin, model=model)[1])/(2*step)
    return dict(e=e, j=j, hjj=hjj, hww=hww, stable=bool(hjj*hww > 0))


def dipole_s_sources(a, e, radius, harmonics, anomaly_order=64):
    """Complete recoil dipole exciting l=0, resolved in parent magnetic m."""
    r = np.asarray(radius)
    harmonics = np.asarray(harmonics)
    node, weight = np.polynomial.legendre.leggauss(anomaly_order)
    split = np.arccos(np.clip((1-r/a)/e, -1, 1))
    width = np.column_stack([split, np.pi-split])/2
    center = np.column_stack([split, np.pi+split])/2
    anomaly = (center[:, :, None]+width[:, :, None]*node).reshape(len(r), -1)
    quadrature = (width[:, :, None]*weight/np.pi).reshape(len(r), -1)
    measure = 1-e*np.cos(anomaly)
    orbit_r = a*measure
    mean = anomaly-e*np.sin(anomaly)
    true = np.arctan2(np.sqrt(1-e*e)*np.sin(anomaly), np.cos(anomaly)-e)
    profile = np.minimum(r[:, None], orbit_r)/np.maximum(r[:, None], orbit_r)**2-r[:, None]/orbit_r**2
    source = np.zeros((len(harmonics), len(r), 3))
    for m in [-1, 1]:
        kernel = np.einsum('hrk,rk->hr',
            np.cos(harmonics[:, None, None]*mean[None, :, :]-m*true[None, :, :]),
            quadrature*measure*profile)
        coefficient = directed_companion_coefficient(1, m)*angular(0, 0, 1, -m, 1, m)
        source[:, :, 1-m] = -coefficient.real*kernel*radial(2, 1, r)
    return source


def bound_absorption_screen(a, e, inclination, omega, alpha, spin, *, maximum_n=12,
                            order=40, anomaly_order=64, fine=True):
    """Finite ns basis screening, not a radially complete horizon response.

    A common matched s-wave sink combines the bound paths coherently.
    Fine shifts are the leading bare terms; cloud-induced shifts are omitted.
    """
    grid = source_grid(a, order, 100., breakpoints=(a*(1-e), a*(1+e)))
    r = grid.r
    harmonics = np.arange(-6, 1)
    source = dipole_s_sources(a, e, r, harmonics, anomaly_order)
    n = np.arange(1, maximum_n+1)
    waves = np.array([radial(int(number), 0, r) for number in n])
    coupling = np.einsum('nr,hrb,r->hnb', waves, source, grid.w*r*r)
    energies = -.5/n**2
    parent = -.125
    if fine:
        energies += alpha**2*(-1/(8*n**4)+(1-3*n)/(n**4*.5))
        parent -= 17/128*alpha**2
    origin = 2/n**1.5
    sink = 2*(1+np.sqrt(1-spin*spin))*alpha**3
    width_matrix = sink*np.outer(origin, origin)
    c = rotated_amplitudes(inclination, omega)
    rows = []
    for h, vertex in zip(harmonics, coupling):
        detuning = parent+h*a**(-1.5)-energies
        response = np.linalg.solve(np.diag(detuning)+.5j*width_matrix, vertex)
        final = np.sqrt(sink)*origin@response
        rate = abs(final@c)**2
        loss = np.outer(final.conj(), final)
        torque = -np.array([np.vdot(c, matrix@loss@c).real for matrix in angular_matrices(1)])
        axis = np.array([np.sin(inclination)*np.sin(omega), np.sin(inclination)*np.cos(omega), np.cos(inclination)])
        rows.append(dict(h=int(h), number=float(rate), energy=float(h*rate),
            normal_torque=float(torque[2]), axial_torque=float(axis@torque),
            ns_detunings=list(detuning), ground_population_width=float(width_matrix[0, 0]),
            source_norm=float(np.linalg.norm(vertex@c))))
    return dict(a=a, e=e, inclination=float(inclination), omega=float(omega), alpha=alpha,
        spin=spin, maximum_n=maximum_n, fine=fine, harmonics=rows,
        number=sum(row['number'] for row in rows), energy=sum(row['energy'] for row in rows),
        normal_torque=sum(row['normal_torque'] for row in rows),
        axial_torque=sum(row['axial_torque'] for row in rows))


def radial_absorption_screen(a, e, inclination, omega, alpha, spin, *, size=32,
                             order=48, anomaly_order=64, fine=True, fine_n=8):
    """Closed s-wave Coulomb resolvent including positive-energy basis states.

    Only the nearest negative-energy pole receives its matched width. This
    avoids ultraviolet resummation of an unrenormalized point absorber.
    Cloud-induced shifts, other partial waves, and the dispersive force are
    not included; this function tests the omission of horizon absorption.
    """
    grid = source_grid(a, order, 512., breakpoints=(a*(1-e), a*(1+e)))
    sector = make_sector(grid, 0, (0,), size, decay=.6)
    harmonics = np.arange(-6, 1)
    source = dipole_s_sources(a, e, grid.r, harmonics, anomaly_order)
    weights = grid.w*grid.r**2
    vertex = np.einsum('rn,hrb,r->hnb', sector.functions[0], source, weights)
    operator = sector.h0.copy()
    parent = -.125
    if fine:
        for n in range(1, fine_n+1):
            state = sector.functions[0].T@(weights*radial(n, 0, grid.r))
            operator += alpha**2*(-1/(8*n**4)+(1-3*n)/(n**4*.5))*np.outer(state, state)
        parent -= 17/128*alpha**2
    values, vectors = np.linalg.eigh(operator)
    origin = sector.origin@vectors
    vertex = np.einsum('nm,hnb->hmb', vectors, vertex)
    sink = 2*(1+np.sqrt(1-spin*spin))*alpha**3
    c = rotated_amplitudes(inclination, omega)
    rows = []
    for h, coupling in zip(harmonics, vertex):
        detuning = parent+h*a**(-1.5)-values
        nearest = int(np.argmin(np.where(values < 0, abs(detuning), np.inf)))
        broadened = detuning.astype(complex)
        broadened[nearest] += .5j*sink*origin[nearest]**2
        final = np.sqrt(sink)*origin@(coupling/broadened[:, None])
        rate = abs(final@c)**2
        loss = np.outer(final.conj(), final)
        torque = -np.array([np.vdot(c, matrix@loss@c).real for matrix in angular_matrices(1)])
        axis = np.array([np.sin(inclination)*np.sin(omega), np.sin(inclination)*np.cos(omega), np.cos(inclination)])
        first_order = sink*abs(origin@(coupling@c/detuning))**2
        rows.append(dict(h=int(h), number=float(rate), energy=float(h*rate),
            normal_torque=float(torque[2]), axial_torque=float(axis@torque),
            nearest_energy=float(values[nearest]), nearest_detuning=float(detuning[nearest]),
            nearest_width=float(sink*origin[nearest]**2), first_order_number=float(first_order)))
    return dict(a=a, e=e, inclination=float(inclination), omega=float(omega), alpha=alpha,
        spin=spin, size=size, order=order, anomaly_order=anomaly_order, fine=fine, fine_n=fine_n,
        overlap_error=sector.overlap_error, negative_states=int(sum(values < 0)),
        positive_states=int(sum(values > 0)), harmonics=rows,
        number=sum(row['number'] for row in rows), energy=sum(row['energy'] for row in rows),
        normal_torque=sum(row['normal_torque'] for row in rows),
        axial_torque=sum(row['axial_torque'] for row in rows))


def migration(response, *, cloud, alpha, spin, model, q, a0=3., afinal=2.85,
              jz0=np.sqrt(1-.002**2)*np.cos(np.deg2rad(85)),
              delta_e=0., delta_omega=0., averaged=False, tolerance=2e-9,
              max_step=8., budget=240.):
    """Finite-q orbit with energy, both actions, and dissipative periapsis drift."""
    start = time.perf_counter()
    point = equilibrium(a0, jz0, cloud, alpha, spin, model)
    e0 = point['e']+delta_e
    y0 = [a0, np.sqrt(1-e0*e0), jz0, np.pi/2+delta_omega, 0., 0., 0., 0., 0., 0.]

    def rhs(_, state):
        if time.perf_counter()-start > budget:
            raise TimeoutError('Migration exceeded its time budget')
        a, j, jz, omega = state[:4]
        e = np.sqrt(1-j*j)
        hj, hw = derivatives(a, e, omega, jz, cloud, alpha, spin, model=model)
        loss, rate = loss_derivatives(a, j, jz, omega, response, alpha, cloud, averaged)
        orbit = q*loss+np.array([0., -hw/(cloud*np.sqrt(a)), 0., hj/(cloud*np.sqrt(a))])
        # Cycles are Newtonian orbital cycles, not a detector waveform phase.
        cycles = a**(-1.5)/(2*np.pi*cloud)
        cloud_loss = q*q/cloud*rate['number']
        ion_energy = q*a**(-1.5)*rate['energy']
        total_energy = -orbit[0]/(2*a*a)
        axial_loss = -np.sqrt(a)*orbit[2]-jz*orbit[0]/(2*np.sqrt(a))
        static_loss = q*q/cloud*rate.get('static_number', 0.)
        return np.r_[orbit, cycles, cloud_loss, ion_energy, total_energy, axial_loss, static_loss]

    def reached(_, state):
        return state[0]-afinal

    reached.terminal = True
    reached.direction = -1
    solution = solve_ivp(rhs, [0, 100/q], y0, method='DOP853', rtol=tolerance,
        atol=tolerance/100, max_step=max_step, events=reached, dense_output=True)
    if not solution.success or len(solution.t_events[0]) != 1:
        raise RuntimeError(solution.message)
    times = np.linspace(0, solution.t[-1], 1001)
    state = solution.sol(times)
    e = np.sqrt(1-state[1]**2)
    inclination = np.arccos(state[2]/state[1])
    return dict(q=q, averaged=averaged, cloud=cloud, alpha=alpha, spin=spin,
        a0=a0, afinal=afinal, jz0=jz0, delta_e=delta_e, delta_omega=delta_omega,
        tolerance=tolerance, max_step=max_step, elapsed_seconds=time.perf_counter()-start,
        final_s=float(times[-1]), final_time_M=float(times[-1]/(cloud*alpha**3)),
        final_cycles=float(state[4, -1]),
        ionization_fraction=float(state[5, -1]-state[9, -1]),
        total_parent_loss_fraction=float(state[5, -1]),
        static_absorption_fraction=float(state[9, -1]),
        ionization_energy_specific=float(state[6, -1]),
        kepler_binding_change=float(1/(2*afinal)-1/(2*a0)),
        kepler_energy_balance_error=float(state[7, -1]-(1/(2*afinal)-1/(2*a0))),
        axial_angular_balance_error=float(state[8, -1]-(np.sqrt(a0)*jz0-np.sqrt(afinal)*state[2, -1])),
        e_range=[float(min(e)), float(max(e))],
        omega_range=list(map(float, [min(state[3]), max(state[3])])),
        final_e=float(e[-1]), final_inclination=float(inclination[-1]),
        samples=[dict(s=float(t), a=float(state[0, k]), e=float(e[k]), jz=float(state[2, k]),
            omega=float(state[3, k]), inclination=float(inclination[k]), cycles=float(state[4, k]))
            for k, t in enumerate(times) if k % 5 == 0], nfev=solution.nfev)
