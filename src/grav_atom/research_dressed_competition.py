"""Mixed scalar escape and absorption of dilute modes on a finite 211 cloud."""
import math

import mpmath as mp
import numpy as np
from scipy.linalg import eig, solve
from scipy.interpolate import CubicHermiteSpline, CubicSpline, PchipInterpolator

from .research_collective_mode import collective_operator
from .research_competing_levels import A, B, channel, hydrogen_polynomial, cross_energy
from .research_cloud_scattering import (
    field_from_sector, scaled, contraction, add_fields, scattering_response)


C = (4, 2, 2)


def bound_modes(configuration, fraction, states=(B, C)):
    """Track isolated positive-norm branches by overlap with bare bound states.

    The n=3 and n=4 even-parity m=2 shells each have one hydrogenic state.
    Degenerate higher shells need a subspace treatment, not this tracker.
    """
    if any(s not in (B, C) for s in states):
        raise ValueError('This isolated-branch calculation supports 322 and 422')
    grid, parent, upper, lower = configuration
    operator = collective_operator(grid, parent, upper, lower, fraction)
    values, vectors = eig(operator['matrix'])
    split = len(upper.h0)
    a = field_from_sector(parent, operator['background']['coefficients'])
    if np.dot(grid.w*grid.r**3*np.exp(-grid.r/2), a.values[1]) < 0:
        a = scaled(a, -1.)
    result = []
    for state in states:
        with mp.workdps(50):
            polynomial = hydrogen_polynomial(state)
            reference_wave = np.exp(-grid.r/state[0])*sum(
                float(c)*grid.r**p for p, c in polynomial.items())
        reference = np.zeros(split)
        reference[upper.block(upper.ls.index(state[1]))] = upper.functions[state[1]].T@(
            grid.w*grid.r**2*reference_wave)
        overlaps = abs(reference@vectors[:split])**2
        index = int(np.argmax(overlaps))
        value, vector = values[index], vectors[:, index]
        norm = float(np.vdot(vector[:split], vector[:split]).real
                     -np.vdot(vector[split:], vector[split:]).real)
        if norm <= 0 or abs(value.imag) > 1e-10 or overlaps[index] < .8:
            raise ArithmeticError('The isolated positive-norm branch was not identified')
        vector = np.real(vector)/math.sqrt(norm)
        if reference@vector[:split] < 0:
            vector *= -1
        u = field_from_sector(upper, vector[:split])
        v = field_from_sector(lower, vector[split:])
        absorption = (2*float(abs(lower.origin@vector[split:])**2)/fraction**2
                      if fraction else channel((A, A), state)['coefficient'])
        result.append(dict(state=state, upper=u, lower=v, frequency=float(value.real),
            absorption=absorption, reference_overlap=float(overlaps[index]),
            reference_projection_norm=float(reference@reference),
            lower_norm=float(np.dot(vector[split:], vector[split:])),
            residual=float(np.linalg.norm(operator['matrix']@vector-value.real*vector))))
    return dict(parent=a, chemical=operator['background']['chemical'], operator=operator,
                parent_energy=operator['background']['energy'], modes=result,
                overlap_error=max(s.overlap_error for s in [parent, upper, lower]))


def mixed_driving(configuration, background, first, second,
                  output_ls=(3, 5, 7), closed_ls=(1, 3, 5)):
    """Sum-frequency source; ordered contractions distinguish identical pumps."""
    grid = configuration[0]
    a = background['parent']
    pairs = [(first, second)] if first['state'] == second['state'] else [(first, second), (second, first)]
    sources = []
    for upper_key, lower_key, ls in [('upper', 'lower', output_ls), ('lower', 'upper', closed_ls)]:
        terms = []
        for i, j in pairs:
            u, v = i[upper_key], j[upper_key]
            partner = j[lower_key]
            terms.extend([contraction(grid, u, v, a, ls),
                          contraction(grid, a, u, partner, ls),
                          contraction(grid, u, a, partner, ls)])
        source = add_fields(*terms)
        if max(np.max(abs(s.imag)) for s in source.values()) > 1e-12:
            raise ArithmeticError('Real modes gave a complex mixed source')
        sources.append({l: s.real for l, s in source.items()})
    frequency = (first['frequency']+second['frequency'])/2
    return dict(parent=a, chemical=background['chemical'], frequency=frequency,
                energy=background['chemical']+2*frequency,
                source=sources[0], lower_source=sources[1])


def compare_channels(configuration, fraction, *, output_ls=(3, 5, 7), closed_ls=(1, 3, 5)):
    background = bound_modes(configuration, fraction)
    b, c = background['modes']
    responses = []
    for first, second in [(b, b), (b, c)]:
        driving = mixed_driving(configuration, background, first, second, output_ls, closed_ls)
        responses.append(scattering_response(configuration, fraction,
            output_ls=output_ls, closed_ls=closed_ls, driving=driving))
    main, mixed = responses
    return dict(fraction=fraction, main_escape=main, spectator_escape=mixed,
        main_absorption=b['absorption'], spectator_absorption=c['absorption'],
        equilibrium_loss_over_gain=mixed['rate']/c['absorption']*b['absorption']/(2*main['rate']),
        absorption_ratio=c['absorption']/b['absorption'],
        parent_chemical=background['chemical'], parent_energy=background['parent_energy'],
        overlap_error=background['overlap_error'],
        higher_gain_groups=higher_gain_groups(configuration, background, fraction),
        modes=[{k: v for k, v in mode.items() if k not in ['upper', 'lower']}
               for mode in background['modes']])


def higher_gain_groups(configuration, background, fraction, maximum_n=8):
    """Coherent gain traces within resolved higher principal-shell clusters.

    Only the even-parity m=2 sector in the supplied angular/radial basis is
    included. Different clusters require the usual secular separation.
    These finite-basis bounds are not an infinite-Rydberg completeness claim.
    """
    _, _, upper, lower = configuration
    op = background['operator']
    values, vectors = eig(op['matrix'])
    split = len(upper.h0)
    modes = []
    for frequency, vector in zip(values, vectors.T):
        norm = float(np.vdot(vector[:split], vector[:split]).real
                     -np.vdot(vector[split:], vector[split:]).real)
        if norm <= 0 or abs(frequency.imag) > 1e-10 or frequency.real <= 0:
            continue
        if background['chemical']+frequency.real >= 0:
            continue
        vector = np.real(vector)/math.sqrt(norm)
        if fraction:
            origin = float(lower.origin@vector[split:])/fraction
        else:
            response = -solve(op['lower']+frequency.real*np.eye(len(lower.h0)),
                              op['coupling'].T@vector[:split])
            origin = float(lower.origin@response)
        modes.append(dict(frequency=float(frequency.real), absorption=2*origin**2))
    modes.sort(key=lambda row: row['frequency'])
    groups = []
    start = 0
    for n in range(3, maximum_n+1):
        dimension = sum(l < n for l in upper.ls)
        selected = modes[start:start+dimension]
        if len(selected) != dimension:
            raise ArithmeticError('The radial basis does not resolve the requested bound clusters')
        groups.append(dict(n=n, dimension=dimension,
            minimum_frequency=selected[0]['frequency'], maximum_frequency=selected[-1]['frequency'],
            coherent_gain_trace=sum(r['absorption'] for r in selected), modes=selected))
        start += dimension
    return groups


class DressedModel:
    """Energy-consistent parent and dilute-mode actions from resolved radial data.

    N=a+b is total cloud action, b=J_cloud-N is the 322 excitation action.
    The finite-parent normal form is kept through first order in b, with
    the known dilute b**2 coefficient retained as a small correction.
    """
    label = 'Finite-parent Newtonian modes and scalar radiation'

    def __init__(self, rows, ordinary_horizon=None):
        from .research_cloud_evolution import self_gravity_coefficients
        self.ordinary_horizon = ordinary_horizon
        if ordinary_horizon is not None:
            self.label += '; finite-cloud ordinary horizon prefactors'
        x = np.array([r['fraction'] for r in rows])
        if x[0] != 0 or np.any(np.diff(x) <= 0):
            raise ValueError('Require ordered parent fractions starting at zero')
        self.maximum = float(x[-1])
        self.rate_values = np.array([[r['main_absorption'], r['main_escape']['rate'],
            r['spectator_absorption'], r['spectator_escape']['rate']] for r in rows])
        if not np.isfinite(self.rate_values).all() or np.any(self.rate_values <= 0):
            raise ValueError('Require finite positive resolved rates')
        self.rates = PchipInterpolator(x, self.rate_values, axis=0, extrapolate=False)
        chemical = np.array([r['parent_chemical'] for r in rows])
        energy = np.array([r['parent_energy'] for r in rows])
        self.parent = CubicHermiteSpline(x, x*energy, chemical, extrapolate=False)
        self.parent_derivative = self.parent.derivative()
        self.c = self_gravity_coefficients()
        caC, cbC = cross_energy(C, A)['total'], cross_energy(C, B)['total']
        frequencies = np.array([[r['modes'][0]['frequency'], r['modes'][1]['frequency']]
                                for r in rows])
        initial_slopes = np.array([self.c['aa']-self.c['ab'], self.c['aa']-caC])
        self.spectra = CubicSpline(x, frequencies, axis=0,
            bc_type=((1, initial_slopes), (2, np.zeros(2))), extrapolate=False)
        self.spectra_derivative = self.spectra.derivative()
        self.bb = self.c['ab']-(self.c['aa']+self.c['bb'])/2
        self.bc = self.c['ab']+caC-self.c['aa']-cbC
        traces = np.array([[g['coherent_gain_trace'] for g in r['higher_gain_groups'] if g['n'] >= 5]
                           for r in rows])
        self.higher_gains = PchipInterpolator(x, traces, axis=0, extrapolate=False)
        minimum_frequencies = np.array([r['higher_gain_groups'][2]['minimum_frequency'] for r in rows])
        self.higher_minimum = PchipInterpolator(x, minimum_frequencies, extrapolate=False)

    def check_fraction(self, fraction):
        if not np.isfinite(fraction) or not 0 <= fraction <= self.maximum:
            raise ValueError('Cloud action fraction lies outside the resolved interval')

    def reaction_coefficients(self, fraction):
        self.check_fraction(fraction)
        return self.rates(fraction)[:2]

    def energy(self, alpha0, mass, a, b):
        n = a+b
        x = n/mass
        self.check_fraction(x)
        f, df = float(self.parent(x)), float(self.parent_derivative(x))
        sigma, dsigma = float(self.spectra(x)[0]), float(self.spectra_derivative(x)[0])
        correction = alpha0**2*(mass**3*f+mass**2*b*sigma+mass*self.bb*b*b)
        wa = 1+alpha0**2*(mass**2*df+mass*b*dsigma)
        wb = wa+alpha0**2*(mass**2*sigma+2*mass*self.bb*b)
        clock = 1+alpha0**2*(mass**2*(3*f-x*df)
            +mass*b*(2*sigma-x*dsigma)+self.bb*b*b)
        binding = alpha0**2*mass**2*(a/8+b/18)
        return dict(energy=mass+n+correction, wa=wa, wb=wb,
                    wh=2*wa-wb, we=2*wb-wa, clock=clock,
                    binding=binding, interaction=-correction-binding)

    def spectator_rates(self, alpha0, mass, j, a, b, background):
        """422 net growth and the selected higher-shell no-escape envelope.

        Both use leading Kerr matching and omit finite daughter-density changes.
        Finite-parent ordinary amplitudes are included when profiles are supplied.
        """
        from boson_ep.models import State
        from boson_ep.spectrum import detweiler_coefficient, horizon_omega_M
        n = a+b
        x = n/mass
        self.check_fraction(x)
        _, _, h, escape = self.rates(x)
        alpha, spin, clock = alpha0*mass, j/mass**2, background['clock']
        rplus = 1+math.sqrt(1-spin*spin)
        frequency = (background['wa']+alpha**2*float(self.spectra(x)[1])
                     +alpha0**2*mass*self.bc*b)
        state = State(*C)
        sr = (4*rplus*detweiler_coefficient(alpha, spin, state)
              *(clock*state.m*horizon_omega_M(spin)-alpha*frequency)
              *alpha**(4*state.l+5)/mass)
        if self.ordinary_horizon is not None:
            sr *= self.ordinary_horizon.factors(x, alpha, spin)[2]
        gain = h*rplus*alpha**6*n*n/mass**3
        loss = escape*alpha**3*n*b/mass**3
        higher_state = State(5, 2, 2)
        higher_frequency = background['wa']+alpha**2*float(self.higher_minimum(x))
        higher_sr = (4*rplus*detweiler_coefficient(alpha, spin, higher_state)
            *max(0., clock*2*horizon_omega_M(spin)-alpha*higher_frequency)*alpha**13/mass)
        if self.ordinary_horizon is not None:
            higher_frequency = background['wa']+alpha**2*self.ordinary_horizon.minimum_frequency(x)
            higher_sr = (4*rplus/mass*self.ordinary_horizon.higher_weight(x, alpha, spin)
                *max(0., clock*2*horizon_omega_M(spin)-alpha*higher_frequency))
        higher_gain = float(max(self.higher_gains(x)))*rplus*alpha**6*n*n/mass**3
        return np.array([sr+gain-loss, higher_sr+higher_gain])
