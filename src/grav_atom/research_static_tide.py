"""Static tidal absorption in the self-consistent 211 cloud background.

The full m=2,0 conservative Hessian is retained. Horizon loss is evaluated
to first order from the s-wave origin, not by a point-sink resummation.
"""

from functools import lru_cache

import numpy as np
from scipy.interpolate import NdBSpline, make_interp_spline
from scipy.integrate import solve_ivp, trapezoid
from scipy.linalg import eig, lu_factor, lu_solve
from scipy.special import sph_harm_y

from .research_collective_mode import RadialGrid, angular, collective_operator, make_sector
from .research_inclined_ionization import angular_matrices
from .research_cloud_secular import anomaly_grid, axisymmetric_field, derivatives
from .research_rydberg_storage import radial


def fine_operator(grid, sector, alpha, spin, maximum_n=8):
    weights = grid.w*grid.r**2
    result = np.zeros_like(sector.h0)
    for index, ell in enumerate(sector.ls):
        block = sector.block(index)
        for n in range(ell+1, maximum_n+1):
            state = sector.functions[ell].T@(weights*radial(n, ell, grid.r))
            value = alpha**2*(-1/(8*n**4)+(2*ell-3*n+1)/(n**4*(ell+.5)))
            if ell:
                value += 2*sector.m*spin*alpha**3/(n**3*ell*(ell+.5)*(ell+1))
            result[block, block] += value*np.outer(state, state)
    return result


@lru_cache(maxsize=8)
def rotation_spectrum(rank):
    return np.linalg.eigh(angular_matrices(rank)[0])


def angular_factors(labels, inclination, omega, magnetic_row=-1):
    """D^L_{m,k}(R_x(I)R_z(omega)) and its angular derivatives."""
    result, derivative_i, derivative_w = [], [], []
    for rank, magnetic in labels:
        values, vectors = rotation_spectrum(rank)
        if abs(magnetic_row) > rank:
            raise ValueError('The magnetic row lies outside the multipole')
        left = vectors[rank+magnetic_row]*np.exp(-1j*inclination*values)
        right = vectors[rank+magnetic]
        phase = np.exp(-1j*magnetic*omega)
        value = np.dot(left, right)*phase
        result.append(value)
        derivative_i.append(np.dot(-1j*values*left, right)*phase)
        derivative_w.append(-1j*magnetic*value)
    return np.array(result), np.array(derivative_i), np.array(derivative_w)


def radial_tide_coefficients(radius, weights, labels, a, eccentricity, anomaly_order=64):
    e = eccentricity
    if not (a > 0 and 0 < e < .8):
        raise ValueError('Require a positive semimajor axis and moderate nonzero eccentricity')
    r = np.asarray(radius)
    node, weight = np.polynomial.legendre.leggauss(anomaly_order)
    split = np.arccos(np.clip((1-r/a)/e, -1, 1))
    widths = np.column_stack([split, np.pi-split])/2
    centers = np.column_stack([split, np.pi+split])/2
    anomaly = (centers[:, :, None]+widths[:, :, None]*node).reshape(len(r), -1)
    quadrature = (widths[:, :, None]*weight/np.pi).reshape(len(r), -1)
    measure = 1-e*np.cos(anomaly)
    orbital_r = a*measure
    true = np.arctan2(np.sqrt(1-e*e)*np.sin(anomaly), np.cos(anomaly)-e)
    profiles = {}
    for rank in weights:
        profile = np.minimum(r[:, None], orbital_r)**rank/np.maximum(r[:, None], orbital_r)**(rank+1)
        if rank == 1:
            profile -= r[:, None]/orbital_r**2
        profiles[rank] = quadrature*measure*profile
    result = []
    for rank, magnetic in labels:
        kernel = np.sum(profiles[rank]*np.cos(magnetic*true), axis=1)
        coefficient = -4*np.pi/(2*rank+1)*sph_harm_y(rank, magnetic, np.pi/2, 0).real
        result.append(float(coefficient*np.dot(weights[rank], kernel)))
    return np.array(result)


class StaticCloudTide:
    def __init__(self, cloud, alpha, spin, *, size=28, order=48,
                 parent_lmax=3, response_lmax=4, source_lmax=5, fine_n=8,
                 anomalous=True):
        self.cloud, self.alpha, self.spin = cloud, alpha, spin
        self.grid = grid = RadialGrid(order)
        self.parent = parent = make_sector(grid, 1, tuple(range(1, parent_lmax+1, 2)), size)
        self.upper = upper = make_sector(grid, 2, tuple(range(2, response_lmax+1, 2)), size)
        self.lower = lower = make_sector(grid, 0, tuple(range(0, response_lmax+1, 2)), size)
        self.operator = op = collective_operator(grid, parent, upper, lower, cloud)
        correction = -17/128*alpha**2+spin*alpha**3/12
        au = op['upper']+fine_operator(grid, upper, alpha, spin, fine_n)-correction*np.eye(len(upper.h0))
        av = op['lower']+fine_operator(grid, lower, alpha, spin, fine_n)-correction*np.eye(len(lower.h0))
        coupling = cloud*op['coupling'] if anomalous else np.zeros_like(op['coupling'])
        self.hessian = np.block([[au, coupling], [coupling.T, av]])
        self.factor = lu_factor(self.hessian)
        sink_vector = np.r_[np.zeros(len(upper.h0)), lower.origin]
        self.origin_response = lu_solve(self.factor, sink_vector)
        self.sink = 2*(1+np.sqrt(1-spin*spin))*alpha**3
        self.background = op['background']
        # Rank-one contact resummation is not used; this measures the finite-basis correction.
        contact = np.dot(sink_vector, self.origin_response)
        self.source_lmax = source_lmax
        self.weights = self.response_weights(self.origin_response)
        self.labels = [(rank, magnetic) for rank in self.weights for magnetic in range(-rank, rank+1, 2)]
        eigenvalues = np.linalg.eigvalsh(self.hessian)
        self.diagnostics = dict(cloud=cloud, alpha=alpha, spin=spin, size=size, order=order,
            parent_lmax=parent_lmax, response_lmax=response_lmax, source_lmax=source_lmax,
            fine_n=fine_n, anomalous=anomalous,
            stationary_residual=op['background']['residual'], chemical=op['background']['chemical']+correction,
            minimum_static_eigenvalue_absolute=float(min(abs(eigenvalues))),
            contact_resummation_parameter=float(.5*self.sink*contact),
            reciprocity_error=op['reciprocity_error'],
            overlap_error=max(s.overlap_error for s in [parent, upper, lower]))

    def response_weights(self, origin_response):
        grid, parent, upper, lower = self.grid, self.parent, self.upper, self.lower
        weights = {}
        for rank in range(1, self.source_lmax+1, 2):
            values = np.zeros_like(grid.r)
            for index, ell in enumerate(upper.ls):
                state = upper.functions[ell]@origin_response[upper.block(index)]
                for a in parent.ls:
                    values += state*angular(ell, 2, a, 1, rank, 1)*self.background['fields'][a]
            offset = len(upper.h0)
            for index, ell in enumerate(lower.ls):
                block = lower.block(index)
                state = lower.functions[ell]@origin_response[offset+block.start:offset+block.stop]
                for a in parent.ls:
                    values -= state*angular(ell, 0, a, 1, rank, -1)*self.background['fields'][a]
            weights[rank] = np.sqrt(self.sink)*grid.w*grid.r**2*values
        return weights

    def slow_pole(self):
        metric = np.r_[np.ones(len(self.upper.h0)), -np.ones(len(self.lower.h0))]
        values, vectors = eig(metric[:, None]*self.hessian)
        index = int(np.argmin(abs(values)))
        if max(abs(values.imag)) > 1e-10:
            raise ArithmeticError('The diagnostic assumes a real conservative spectrum')
        values, vectors = values.real, vectors.real
        left = np.linalg.inv(vectors)
        contact = np.r_[np.zeros(len(self.upper.h0)), self.lower.origin]
        row = (contact@vectors[:, index])*left[index]*metric/values[index]
        gamma = self.sink*(contact@vectors[:, index])*(left[index]@contact)
        return dict(frequency=float(values[index]), population_width=float(gamma),
            nearest_other_frequency=float(min(abs(np.delete(values, index)))),
            weights=self.response_weights(row))

    def shifted_response(self, a, e, inclination, omega, frequency):
        metric = np.r_[np.ones(len(self.upper.h0)), -np.ones(len(self.lower.h0))]
        contact = np.r_[np.zeros(len(self.upper.h0)), self.lower.origin]
        response = np.linalg.solve(self.hessian-frequency*np.diag(metric), contact)
        coefficients = self.coefficients(a, e, weights=self.response_weights(response))
        return coefficients@angular_factors(self.labels, inclination, omega)[0]

    def coefficients(self, a, eccentricity, anomaly_order=64, *, weights=None):
        response_weights = self.weights if weights is None else weights
        return radial_tide_coefficients(self.grid.r, response_weights, self.labels,
                                        a, eccentricity, anomaly_order)

    def static_response(self, a, e, inclination, omega, anomaly_order=64):
        coefficients = self.coefficients(a, e, anomaly_order)
        angles, di, dw = angular_factors(self.labels, inclination, omega)
        amplitude = coefficients@angles
        return dict(amplitude=[float(amplitude.real), float(amplitude.imag)],
            number=float(abs(amplitude)**2),
            normal_torque=float(-np.imag(amplitude.conjugate()*(coefficients@dw))),
            axial_torque=float(-abs(amplitude)**2),
            node_torque=float(-np.imag(amplitude.conjugate()*(coefficients@di))),
            coefficients=list(coefficients))

    def dipole_polarization(self, a, e, inclination, omega, anomaly_order=64):
        """Real induced Hamiltonian per epsilon*q, retaining the dipole source."""
        from .research_periapsis_migration import dipole_s_sources
        # Remove the bare radial factor to recover the potential coefficient.
        r = self.grid.r
        source = dipole_s_sources(a, e, r, [0], anomaly_order)[0, :, 2]
        parent_bare = radial(2, 1, r)
        profile = np.divide(source, parent_bare, out=np.zeros_like(source), where=parent_bare > 1e-100)
        profile /= angular(0, 0, 1, 1, 1, -1)
        weights = self.grid.w*r*r
        blocks = []
        for sector, sign in [(self.upper, -1), (self.lower, 1)]:
            for ell in sector.ls:
                wave = sum(angular(ell, sector.m, a, 1, 1, sector.m-1)*self.background['fields'][a]
                           for a in self.parent.ls)
                blocks.append(sign*sector.functions[ell].T@(weights*profile*wave))
        source_coefficients = np.concatenate(blocks)
        shape = np.cos(omega)+1j*np.cos(inclination)*np.sin(omega)
        real = -float(source_coefficients@lu_solve(self.factor, source_coefficients))*abs(shape)**2
        norm = float(np.linalg.norm(lu_solve(self.factor, source_coefficients))**2)*abs(shape)**2
        return dict(hamiltonian_per_cloud_q=real, response_norm_per_q_squared=norm)

    def harmonic_potential(self, a, e, inclination, omega, harmonic, anomaly_order=64):
        """Cloud-frame V_{L,-1}^{(h)} radial coefficients per companion mass."""
        r = self.grid.r
        node, weight = np.polynomial.legendre.leggauss(anomaly_order)
        split = np.arccos(np.clip((1-r/a)/e, -1, 1))
        widths = np.column_stack([split, np.pi-split])/2
        centers = np.column_stack([split, np.pi+split])/2
        anomaly = (centers[:, :, None]+widths[:, :, None]*node).reshape(len(r), -1)
        quadrature = (widths[:, :, None]*weight/np.pi).reshape(len(r), -1)
        measure = 1-e*np.cos(anomaly)
        distance = a*measure
        mean = anomaly-e*np.sin(anomaly)
        true = np.arctan2(np.sqrt(1-e*e)*np.sin(anomaly), np.cos(anomaly)-e)
        angles = angular_factors(self.labels, inclination, omega)[0]
        result = {rank:np.zeros_like(r, dtype=complex) for rank in self.weights}
        for (rank, magnetic), angle in zip(self.labels, angles):
            profile = np.minimum(r[:, None], distance)**rank/np.maximum(r[:, None], distance)**(rank+1)
            if rank == 1:
                profile -= r[:, None]/distance**2
            radial = np.sum(quadrature*measure*profile*np.cos(harmonic*mean-magnetic*true), axis=1)
            coefficient = -4*np.pi/(2*rank+1)*sph_harm_y(rank, magnetic, np.pi/2, 0).real
            result[rank] += coefficient*angle*radial
        return result

    def negative_harmonic(self, a, e, inclination, omega, harmonic, anomaly_order=64,
                          tolerance=2e-10):
        """Outgoing m=2 partner and closed m=0 tide; perturbative horizon loss.

        The outgoing continuum is Newtonian. Leading bare fine structure is
        retained in the closed partner and parent energy only.
        """
        from .research_cloud_scattering import field_from_sector, scattering_response
        if harmonic >= 0:
            raise ValueError('The harmonic must be negative')
        potential = self.harmonic_potential(a, e, inclination, omega, harmonic, anomaly_order)
        sources = []
        for sector in [self.upper, self.lower]:
            source = {ell:np.zeros_like(self.grid.r, dtype=complex) for ell in sector.ls}
            for ell in sector.ls:
                for rank, radial in potential.items():
                    wave = sum(angular(ell, sector.m, lp, 1, rank, sector.m-1)*self.background['fields'][lp]
                               for lp in self.parent.ls)
                    source[ell] += wave*(-radial.conj() if sector.m == 2 else radial)
            sources.append(source)
        chemical = self.diagnostics['chemical']
        sigma = -harmonic*a**(-1.5)
        driving = dict(parent=field_from_sector(self.parent, self.background['coefficients']),
            source=sources[0], lower_source=sources[1], energy=chemical+sigma,
            chemical=chemical, frequency=sigma/2, output_m=2)
        correction = fine_operator(self.grid, self.lower, self.alpha, self.spin,
                                   self.diagnostics['fine_n'])
        result = scattering_response((self.grid, self.parent, self.upper, self.lower), self.cloud,
            output_ls=self.upper.ls, closed_ls=self.lower.ls, driving=driving,
            closed_correction=correction, tolerance=tolerance)
        result['horizon_number_first_order'] = self.sink*result['closed_origin_squared']
        result['horizon_energy_first_order'] = harmonic*result['horizon_number_first_order']
        result['harmonic'] = harmonic
        return result


class StaticTideTable:
    def __init__(self, axes, values, labels):
        self.axes = list(map(np.asarray, axes))
        self.labels = [tuple(pair) for pair in labels]
        sa = make_interp_spline(self.axes[0], np.asarray(values), axis=0)
        se = make_interp_spline(self.axes[1], sa.c, axis=1)
        self.spline = NdBSpline((sa.t, se.t), np.swapaxes(se.c, 0, 1), (3, 3), extrapolate=False)

    def rates(self, a, e, inclination, omega, averaged=False):
        if not (self.axes[0][0] <= a <= self.axes[0][-1] and self.axes[1][0] <= e <= self.axes[1][-1]):
            raise ValueError('Orbit outside static response table')
        coefficients, de = self.spline([a, e]), self.spline([a, e], nu=(0, 1))

        def evaluate(angle):
            angular, di, dw = angular_factors(self.labels, inclination, angle)
            amplitude = coefficients@angular
            number = abs(amplitude)**2
            return np.array([number, 0., -np.imag(amplitude.conjugate()*(coefficients@dw)),
                -number, np.imag(amplitude.conjugate()*(de@angular)),
                -np.imag(amplitude.conjugate()*(coefficients@di))])

        if averaged:
            angles, di, dw = angular_factors(self.labels, inclination, 0.)
            magnetic = np.array([m for _, m in self.labels])
            values = np.zeros(6)
            # Equal Fourier harmonics interfere even after periapsis averaging.
            for m in np.unique(magnetic):
                mask = magnetic == m
                amplitude = coefficients[mask]@angles[mask]
                number = abs(amplitude)**2
                values += [number, 0., m*number, -number,
                    np.imag(amplitude.conjugate()*(de[mask]@angles[mask])),
                    -np.imag(amplitude.conjugate()*(coefficients[mask]@di[mask]))]
        else:
            values = evaluate(omega)
        result = dict(zip(['number', 'energy', 'normal_torque', 'axial_torque',
                           'eccentric_connection', 'node_torque'], values))
        result['static_number'] = result['number']
        return result


class CombinedResponse:
    def __init__(self, *responses):
        self.responses = responses

    def rates(self, *args, **kwargs):
        rows = [response.rates(*args, **kwargs) for response in self.responses]
        keys = list(dict.fromkeys(key for row in rows for key in row))
        return {key:sum(row.get(key, 0.) for row in rows) for key in keys}


def nodal_frequency(a, e, omega, jz, cloud, alpha, spin, model, order=96):
    """Node angular velocity in Bohr time alpha**3*t/M."""
    j = np.sqrt(1-e*e)
    cosi = jz/j
    sini = np.sqrt(1-cosi*cosi)
    cosine, sine, weight = anomaly_grid(order)
    measure = 1-e*cosine
    axis = (j*sine*np.cos(omega)+(cosine-e)*np.sin(omega))/measure
    gradient = axisymmetric_field(a*measure, sini*axis, model)[2]
    derivative = cloud*np.dot(weight*measure, gradient*(-cosi/(j*sini))*axis)
    derivative += 2*spin*alpha**3/(a**2.5*j**3)
    return float(derivative/np.sqrt(a))


def forced_libration(response, pole_response, pole, *, a, e0, omega0, jz,
                     cloud, alpha, spin, model, duration=6000., discard=3000.,
                     initial='steady', tolerance=2e-9, max_step=4.):
    """One retarded pole driven by a conservative orbit, without backreaction.

    Fast radial modes follow the source instantaneously. This diagnostic
    compares horizon flow; it does not replace the orbital source force.
    """
    if not (0 <= discard < duration and cloud > 0):
        raise ValueError('Require a positive cloud fraction and an ordered averaging interval')
    frequency, gamma = pole['frequency'], pole['population_width']

    def amplitudes(e, omega):
        inclination = np.arccos(jz/np.sqrt(1-e*e))
        angles = angular_factors(response.labels, inclination, omega)[0]
        full = response.spline([a, e])@angles
        slow = pole_response.spline([a, e])@angles
        return full, slow

    full, slow = amplitudes(e0, omega0)
    nu = nodal_frequency(a, e0, omega0, jz, cloud, alpha, spin, model)
    z0 = frequency*slow/(frequency-nu+.5j*gamma) if initial == 'steady' else 0j
    if initial not in ['steady', 'zero']:
        raise ValueError('Choose steady or zero pole preparation')

    def rhs(_, state):
        j, omega, x, y = state
        e = np.sqrt(1-j*j)
        hj, hw = derivatives(a, e, omega, jz, cloud, alpha, spin, model=model)
        nu = nodal_frequency(a, e, omega, jz, cloud, alpha, spin, model)
        slow = amplitudes(e, omega)[1]
        zdot = (1j*(frequency-nu)-.5*gamma)*(x+1j*y)/cloud-1j*frequency*slow/cloud
        return [-hw/(cloud*np.sqrt(a)), hj/(cloud*np.sqrt(a)), zdot.real, zdot.imag]

    initial_state = [np.sqrt(1-e0*e0), omega0, z0.real, z0.imag]
    solution = solve_ivp(rhs, [0., duration], initial_state, method='DOP853',
        rtol=tolerance, atol=tolerance/100, max_step=max_step, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    times = np.linspace(discard, duration, 3001)
    states = solution.sol(times)
    rates, frozen, instantaneous = [], [], []
    for j, omega, x, y in states.T:
        e = np.sqrt(1-j*j)
        full, slow = amplitudes(e, omega)
        nu = nodal_frequency(a, e, omega, jz, cloud, alpha, spin, model)
        rates.append(abs(x+1j*y+full-slow)**2)
        frozen.append(abs(full)**2)
        instantaneous.append(abs(frequency*slow/(frequency-nu+.5j*gamma)+full-slow)**2)
    average = lambda values:float(trapezoid(values, times)/(duration-discard))
    return dict(a=a, e0=e0, omega0=float(omega0), jz=jz, duration=duration, discard=discard,
        initial=initial, tolerance=tolerance, max_step=max_step, nfev=solution.nfev,
        mean_causal_number=average(rates), mean_frozen_number=average(frozen),
        mean_local_precessing_number=average(instantaneous),
        e_range=[float(min(np.sqrt(1-states[0]**2))), float(max(np.sqrt(1-states[0]**2)))],
        omega_range=[float(min(states[1])), float(max(states[1]))])
