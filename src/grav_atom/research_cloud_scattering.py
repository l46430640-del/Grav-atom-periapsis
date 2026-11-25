"""Scalar escape from a dressed 211 cloud and its cloud-mass derivative.

The emitted wave uses a Coulomb charge 1+epsilon. Only the short-range
normal interaction is treated in the Born derivative. A separate
iteration includes the closed conjugate partner for finite fractions.
Both calculations use the same conservative Newtonian energy functional.
"""
from dataclasses import dataclass
from functools import lru_cache
import math

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve

from .research_collective_mode import (
    angular, RadialGrid, make_sector, collective_operator, growing_branch)


@dataclass
class Field:
    m: int
    values: dict
    derivatives: dict | None


def field_from_sector(sector, coefficients):
    return Field(sector.m,
        {l: sector.functions[l]@coefficients[sector.block(i)]
         for i, l in enumerate(sector.ls)},
        {l: sector.derivatives[l]@coefficients[sector.block(i)]
         for i, l in enumerate(sector.ls)})


def scaled(field, scale):
    return Field(field.m, {l: scale*v for l, v in field.values.items()},
        None if field.derivatives is None else
        {l: scale*v for l, v in field.derivatives.items()})


def vector_angle(left_l, left_m, right_l, right_m):
    root = math.sqrt(2*math.pi/3)
    minus = angular(left_l, left_m, right_l, right_m, 1, -1)
    plus = angular(left_l, left_m, right_l, right_m, 1, 1)
    zero = angular(left_l, left_m, right_l, right_m, 1, 0)
    return np.array([root*(minus-plus), 1j*root*(minus+plus),
                     math.sqrt(4*math.pi/3)*zero])


def momentum(grid, left, right):
    """Cartesian matrix element <left|p|right>, with an adjoint fallback."""
    if right.derivatives is None:
        if left.derivatives is None:
            raise ValueError('One momentum leg must have a radial derivative')
        return momentum(grid, right, left).conj()
    value = np.zeros(3, dtype=complex)
    for l, wave in left.values.items():
        for a, radial in right.values.items():
            angle = vector_angle(l, left.m, a, right.m)
            if not np.any(angle):
                continue
            gradient = right.derivatives[a]+(a*(a+1)+2-l*(l+1))/2*radial/grid.r
            value += -1j*angle*np.vdot(wave, grid.w*grid.r**2*gradient)
    return value


def contraction(grid, i, j, final, output_ls, *, mass_product=1.):
    """One ordered canonical gravitational contraction, linear in i,j,final*.

    The constituent mass product multiplies gravity, not central recoil.
    """
    output_m = i.m+j.m-final.m
    result = {l: np.zeros(len(grid.r), dtype=complex) for l in output_ls}
    order = i.m-final.m
    for rank in range(abs(order), max(i.values)+max(final.values)+1):
        density = np.zeros(len(grid.r), dtype=complex)
        for li, vi in i.values.items():
            for lk, vk in final.values.items():
                density += angular(li, i.m, lk, final.m, rank, order)*vi*vk.conj()
        if not np.any(density):
            continue
        lower_integrand = grid.r**(rank+2)*density
        below = grid.cumulative(lower_integrand)[0]
        below[:grid.order] = (2*grid.half[0]*regular_origin_integral(grid.order, 2*rank+2)
                               @lower_integrand[:grid.order])
        above = grid.cumulative(grid.r**(1-rank)*density)[1]
        potential = -4*math.pi/(2*rank+1)*(below/grid.r**(rank+1)+grid.r**rank*above)
        for l in output_ls:
            for lj, vj in j.values.items():
                result[l] += mass_product*angular(l, output_m, lj, j.m, rank, order)*potential*vj
    dipole = momentum(grid, final, i)
    if np.any(dipole):
        if j.derivatives is None:
            raise ValueError('The local recoil leg needs a derivative')
        for l in output_ls:
            for lj, vj in j.values.items():
                coefficient = -1j*np.dot(dipole, vector_angle(l, output_m, lj, j.m))
                if coefficient:
                    gradient = j.derivatives[lj]+(lj*(lj+1)+2-l*(l+1))/2*vj/grid.r
                    result[l] += coefficient*gradient
    return result


def add_fields(*fields):
    return {l: sum(f[l] for f in fields) for l in fields[0]}


def quadratic_source(grid, parent, upper, lower, output_ls=(3, 5)):
    """Coefficient of exp(-2 i sigma t) in the nonlinear field equation."""
    return add_fields(contraction(grid, upper, upper, parent, output_ls),
                      contraction(grid, parent, upper, lower, output_ls),
                      contraction(grid, upper, parent, lower, output_ls))


def setup(size=28, order=40, maximum_parent_l=3, outer=512.):
    if outer not in (256., 512., 1024.):
        raise ValueError('Use a radial panel boundary for the outer radius')
    grid = RadialGrid(order)
    if outer == 1024.:
        nodes, weights = np.polynomial.legendre.leggauss(order)
        grid.half = np.append(grid.half, 256.)
        grid.r = np.concatenate([grid.r, 768.+256.*nodes])
        grid.w = np.concatenate([grid.w, 256.*weights])
    panels = int(np.sum(np.cumsum(2*grid.half) <= outer))
    grid.half = grid.half[:panels]
    grid.panels = panels
    grid.r, grid.w = grid.r[:panels*order], grid.w[:panels*order]
    parent = make_sector(grid, 1, tuple(range(1, maximum_parent_l+1, 2)), size)
    upper = make_sector(grid, 2, tuple(range(2, maximum_parent_l+2, 2)), size)
    lower = make_sector(grid, 0, tuple(range(0, maximum_parent_l+2, 2)), size)
    return grid, parent, upper, lower


def dressed_source(configuration, fraction, output_ls=(3, 5)):
    grid, parent, upper, lower = configuration
    operator = collective_operator(grid, parent, upper, lower, fraction)
    mode = growing_branch(operator, upper, lower)
    a = field_from_sector(parent, operator['background']['coefficients'])
    if np.dot(grid.w*grid.r**3*np.exp(-grid.r/2), a.values[1]) < 0:
        a = scaled(a, -1.)
    vector = np.real(mode['vector'])/math.sqrt(mode['norm'])
    split = len(upper.h0)
    u, v = field_from_sector(upper, vector[:split]), field_from_sector(lower, vector[split:])
    if np.dot(grid.w*grid.r**4*np.exp(-grid.r/3), u.values[2]) < 0:
        u, v = scaled(u, -1.), scaled(v, -1.)
    source = quadratic_source(grid, a, u, v, output_ls)
    for value in source.values():
        if np.max(np.abs(value.imag)) > 1e-12:
            raise ArithmeticError('Real bound profiles gave a complex escape source')
    return dict(parent=a, upper=u, lower=v,
        source={l: value.real for l, value in source.items()},
        direct_source={l: value.real for l, value in
                      contraction(grid, u, u, a, output_ls).items()},
        energy=operator['background']['chemical']+2*mode['value'],
        chemical=operator['background']['chemical'], frequency=mode['value'],
        bound_residual=mode['residual'],
        overlap_error=max(s.overlap_error for s in [parent, upper, lower]))


@lru_cache(maxsize=160)
def _regular_coulomb_solution(l, energy, charge, inner, outer, tolerance):
    """Reuse a bounded set of regular waves on different source grids."""
    k = math.sqrt(2*energy)
    def equation(r, y):
        return np.array([y[1], (l*(l+1)/r**2-2*charge/r-k*k)*y[0]])
    with mp.workdps(30):
        regular = lambda r: mp.coulombf(l, -charge/k, k*r)
        initial = float(regular(inner))
        slope = float(mp.diff(regular, inner)/regular(inner))
    solution = solve_ivp(equation, (inner, outer), [1., slope], method='DOP853',
        rtol=tolerance, atol=tolerance/100, dense_output=True)
    if not solution.success:
        raise ArithmeticError(solution.message)
    return initial, solution.sol


def coulomb_waves(grid, ls, energy, charge=1., *, outgoing=False, tolerance=2e-11):
    """Coulomb waves from endpoint special functions and a real radial ODE."""
    if energy <= 0 or charge <= 0:
        raise ValueError('This diagnostic requires an open attractive Coulomb channel')
    k = math.sqrt(2*energy)
    result = {}
    inner, outer = min(1e-5, grid.r[0]/2), float(2*np.sum(grid.half))
    for l in ls:
        def equation(r, y):
            return np.array([y[1], (l*(l+1)/r**2-2*charge/r-k*k)*y[0]])
        initial, solution = _regular_coulomb_solution(l, energy, charge, inner, outer, tolerance)
        f, df = initial*solution(grid.r)
        row = dict(regular=f, normalized=math.sqrt(2/(math.pi*k))*f/grid.r)
        if outgoing:
            with mp.workdps(30):
                irregular = lambda r: mp.coulombg(l, -charge/k, k*r)
                right, slope = float(irregular(outer)), float(mp.diff(irregular, outer))
            back = solve_ivp(equation, (outer, inner), [right, slope], method='DOP853',
                rtol=tolerance, atol=tolerance/100, dense_output=True)
            if not back.success:
                raise ArithmeticError(back.message)
            g, dg = back.sol(grid.r)
            # Propagating F inward would mix roundoff into the singular solution.
            o, do = g+1j*f, dg+1j*df
            row.update(outgoing=o,
                wronskian_relative_error=float(np.max(np.abs((f*do-df*o)/(-k)-1))))
        result[l] = row
    return result


@lru_cache(maxsize=32)
def regular_origin_integral(order, power):
    """Integrate x**power times a Gauss interpolant without endpoint cancellation."""
    nodes, weights = np.polynomial.legendre.leggauss(order)
    x, w = (nodes+1)/2, weights/2
    barycentric = (-1.)**np.arange(order)*np.sqrt((1-nodes**2)*weights)
    matrix = np.empty((order, order))
    for i, upper in enumerate(x):
        sample = upper*x
        cardinal = barycentric[None, :]/(sample[:, None]-x[None, :])
        cardinal /= np.sum(cardinal, axis=1)[:, None]
        matrix[i] = upper*(w*sample**power)@cardinal/x**power
    return matrix


def free_response(grid, source, waves, energy, *, m=3):
    k = math.sqrt(2*energy)
    result = {}
    for l, s in source.items():
        f, o = waves[l]['regular'], waves[l]['outgoing']
        integrand = f*grid.r*s
        below = grid.cumulative(integrand)[0]
        # Enforce the regular lower-limit power before multiplying the irregular wave.
        below[:grid.order] = (2*grid.half[0]*regular_origin_integral(grid.order, 2*l+1)
                               @integrand[:grid.order])
        above = grid.cumulative(o*grid.r*s)[1]
        result[l] = -2/k*(o*below+f*above)/grid.r
    return Field(m, result, None)


def continuum_projection(grid, source, waves):
    return {l: np.dot(grid.w*grid.r**2*waves[l]['normalized'], s)
            for l, s in source.items()}


def short_range_born(configuration, bare, *, tolerance=2e-11):
    grid = configuration[0]
    ls = tuple(bare['source'])
    waves = coulomb_waves(grid, ls, bare['energy'], outgoing=True, tolerance=tolerance)
    response = free_response(grid, bare['source'], waves, bare['energy'])
    a = bare['parent']
    correction = add_fields(contraction(grid, a, response, a, ls),
                            contraction(grid, response, a, a, ls))
    # The -epsilon/r tail is already included by charge=1+epsilon.
    correction = {l: value+response.values[l]/grid.r for l, value in correction.items()}
    projected = continuum_projection(grid, correction, waves)
    leading = continuum_projection(grid, bare['source'], waves)
    rate = 2*math.pi*sum(abs(value)**2 for value in leading.values())
    derivative = 4*math.pi*sum((leading[l].conjugate()*projected[l]).real for l in ls)
    return dict(leading=leading, amplitude_derivative=projected, rate=rate,
        relative_rate_derivative=derivative/rate,
        wronskian_relative_error=max(w['wronskian_relative_error'] for w in waves.values()),
        source_work=float(-2*sum(np.dot(grid.w*grid.r**2*bare['source'][l],
                                      response.values[l].imag) for l in ls)))


def scattering_response(configuration, fraction, *, tolerance=2e-11, iterations=40,
                        anomalous=True, output_ls=(3, 5, 7), closed_ls=(1, 3, 5),
                        driving=None, closed_correction=None):
    """Second harmonic of a bound Bogoliubov excitation, including its closed partner.

    The daughter amplitude is infinitesimal. The parent mass dependence is
    retained within the projected Newtonian model; there is no Kerr source
    correction or horizon damping in this conservative scattering solve.
    Supplied driving may specify output_m for a different conjugate pair.
    A Hermitian closed_correction may supply a resolved bound spectral shift.
    """
    grid = configuration[0]
    current = dressed_source(configuration, fraction, output_ls) if driving is None else driving
    if set(current['source']) != set(output_ls):
        raise ValueError('The source must match the outgoing angular truncation')
    a, source = current['parent'], current['source']
    output_m = current.get('output_m', 3)
    ls = tuple(source)
    wave = coulomb_waves(grid, ls, current['energy'], 1+fraction,
                          outgoing=True, tolerance=tolerance)
    effective = {l: value.astype(complex) for l, value in source.items()}
    closed = make_sector(grid, 2*a.m-output_m, closed_ls, configuration[1].size)
    closed_energy = current['chemical']-2*current['frequency']
    if closed_energy >= 0:
        raise ValueError('The conjugate partner is assumed below threshold')
    correction = np.zeros_like(closed.h0) if closed_correction is None else np.asarray(closed_correction)
    if correction.shape != closed.h0.shape or not np.allclose(correction, correction.conj().T):
        raise ValueError('The closed-channel correction must be a matching Hermitian matrix')
    factor = lu_factor(closed_energy*np.eye(len(closed.h0))-closed.h0-correction)
    lower_source = (current['lower_source'] if 'lower_source' in current else
                    quadratic_source(grid, a, current['lower'], current['upper'], closed_ls))
    if set(lower_source) != set(closed_ls):
        raise ValueError('The source must match the closed angular truncation')
    def project(values):
        return np.concatenate([closed.functions[l].T@(grid.w*grid.r**2*values[l])
                               for l in closed_ls])
    coefficients = np.zeros(len(closed.h0), dtype=complex)
    source_coefficients = project(lower_source)
    lower = field_from_sector(closed, coefficients)
    for iteration in range(iterations):
        response = free_response(grid, effective, wave, current['energy'], m=output_m)
        value = add_fields(contraction(grid, a, response, a, ls),
                           contraction(grid, response, a, a, ls))
        if anomalous:
            value = add_fields(value, contraction(grid, a, a, lower, ls))
        updated = {l: source[l]+fraction*(value[l]+response.values[l]/grid.r) for l in ls}
        residual = math.sqrt(sum(np.dot(grid.w*grid.r**2, abs(updated[l]-effective[l])**2)
                                 for l in ls))
        if anomalous:
            rhs = add_fields(contraction(grid, a, lower, a, closed_ls),
                             contraction(grid, lower, a, a, closed_ls),
                             contraction(grid, a, a, response, closed_ls))
            new_coefficients = lu_solve(factor, source_coefficients+fraction*project(rhs))
            residual = max(residual, float(np.linalg.norm(new_coefficients-coefficients)))
            coefficients = new_coefficients
            lower = field_from_sector(closed, coefficients)
        effective = updated
        if residual < tolerance/100:
            break
    else:
        raise ArithmeticError('Normal scattering iteration did not converge')
    response = free_response(grid, effective, wave, current['energy'], m=output_m)
    amplitude = continuum_projection(grid, effective, wave)
    rate = 2*math.pi*sum(abs(value)**2 for value in amplitude.values())
    upper_work = -2*sum(np.vdot(source[l], grid.w*grid.r**2*response.values[l]).imag for l in ls)
    lower_work = -2*np.vdot(source_coefficients, coefficients).imag
    work = upper_work-lower_work
    dispersive = sum(np.vdot(source[l], grid.w*grid.r**2*response.values[l]).real for l in ls)
    dispersive += np.vdot(source_coefficients, coefficients).real
    return dict(fraction=fraction, rate=rate, source_work=float(work),
                upper_source_work=float(upper_work), lower_source_work=float(lower_work),
                anomalous=anomalous, output_ls=list(output_ls), closed_ls=list(closed_ls),
                partial_rates={str(l): float(2*math.pi*abs(value)**2)
                               for l, value in amplitude.items()},
                emitted_energy=current['energy'], conjugate_energy=closed_energy,
                output_m=output_m, closed_m=closed.m,
                dispersive_response=float(dispersive),
                closed_origin_amplitude=[float((closed.origin@coefficients).real),
                                         float((closed.origin@coefficients).imag)],
                closed_origin_squared=float(abs(closed.origin@coefficients)**2),
                closed_response_norm=float(np.vdot(coefficients, coefficients).real),
                wronskian_relative_error=max(w['wronskian_relative_error'] for w in wave.values()),
                relative_work_error=float(work/rate-1), iterations=iteration+1,
                residual=residual)


def derivative_check(size=28, order=40, maximum_parent_l=3, outer=512.,
                     steps=(.001, .0005), decompose=False):
    configuration = setup(size, order, maximum_parent_l, outer)
    grid = configuration[0]
    bare = dressed_source(configuration, 0.)
    born = short_range_born(configuration, bare)
    rows = []
    fixed_waves = coulomb_waves(grid, tuple(bare['source']), bare['energy']) if decompose else None
    leading = born['leading']
    def rate_derivative(delta):
        return 4*math.pi*sum((leading[l].conjugate()*value).real
                             for l, value in delta.items())/born['rate']
    for fraction in steps:
        current = dressed_source(configuration, fraction)
        wave = coulomb_waves(grid, tuple(current['source']), current['energy'], 1+fraction)
        amplitude = continuum_projection(grid, current['source'], wave)
        rate = 2*math.pi*sum(abs(value)**2 for value in amplitude.values())
        row = dict(fraction=fraction, energy=current['energy'],
            profile_charge_rate=rate,
            profile_charge_relative_derivative=(rate/born['rate']-1)/fraction,
            total_relative_derivative=(rate/born['rate']-1)/fraction
                                      +born['relative_rate_derivative'],
            bound_residual=current['bound_residual'])
        if decompose:
            corrections = {}
            for name, values in [
                ('bound_profiles', {l: (current['direct_source'][l]-bare['source'][l])/fraction
                                     for l in amplitude}),
                ('conjugate_source', {l: (current['source'][l]-current['direct_source'][l])/fraction
                                       for l in amplitude})]:
                corrections[name] = rate_derivative(continuum_projection(grid, values, fixed_waves))
            for name, energy, charge in [('energy', current['energy'], 1.),
                                          ('coulomb_charge', bare['energy'], 1+fraction)]:
                shifted = coulomb_waves(grid, tuple(bare['source']), energy, charge)
                changed = continuum_projection(grid, bare['source'], shifted)
                corrections[name] = rate_derivative({l: (changed[l]-leading[l])/fraction
                                                      for l in amplitude})
            corrections['short_range_scattering'] = born['relative_rate_derivative']
            row['derivative_decomposition'] = corrections
        rows.append(row)
    return dict(radial_size=size, panel_order=order, maximum_parent_l=maximum_parent_l,
        outer=outer, frozen_energy=bare['energy'], overlap_error=bare['overlap_error'],
        frozen_rate=born['rate'], source_work=born['source_work'],
        wronskian_relative_error=born['wronskian_relative_error'],
        born_relative_derivative=born['relative_rate_derivative'],
        born_amplitude_derivative={str(l): [v.real, v.imag]
                                   for l, v in born['amplitude_derivative'].items()},
        steps=rows)
