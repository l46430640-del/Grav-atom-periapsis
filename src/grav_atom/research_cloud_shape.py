"""Leading non-rigid, odd-parity quadrupolar cloud response."""

import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline
from scipy.linalg import eig

from .research_collective_mode import RadialGrid, make_sector, angular
from .research_counter_mode import paired_operator
from .research_static_tide import fine_operator, radial_tide_coefficients, angular_factors, StaticTideTable
from .research_cloud_secular import derivatives


class ShapeTide(StaticTideTable):
    """Unit-Krein-normalized quadrupolar source, without horizon damping."""
    def __init__(self, shape, axes):
        super().__init__(axes, [[shape.coefficients(a, e) for e in axes[1]] for a in axes[0]], shape.labels)
        self.frequency = shape.frequency
        self.groups = [np.array([m == k for _, m in shape.labels])
                       for k in sorted(set(m for _, m in shape.labels))]

    def source(self, a, e, inclination, omega, node=0., averaged=False):
        c = self.spline([a, e])
        da, de = self.spline([a, e], nu=(1, 0)), self.spline([a, e], nu=(0, 1))
        d, di, dw = angular_factors(self.labels, inclination, omega, -2)
        rows = np.array([c*d, da*d, de*d, c*di, c*dw])
        groups = self.groups if averaged else [np.ones(len(c), dtype=bool)]
        grouped = np.array([np.sum(rows[:, mask], axis=1) for mask in groups]).T*np.exp(2j*node)
        return dict(zip(['f', 'a', 'e', 'inclination', 'omega'], grouped))


class QuadrupoleShape:
    def __init__(self, cloud, alpha, spin, *, size=28, order=48, maximum_l=5, source_lmax=6):
        self.grid = grid = RadialGrid(order)
        parent = make_sector(grid, 1, tuple(range(1, maximum_l+1, 2)), size)
        upper = make_sector(grid, 3, tuple(range(3, maximum_l+1, 2)), size)
        lower = make_sector(grid, -1, tuple(range(1, maximum_l+1, 2)), size)
        self.parent, self.upper, self.lower = parent, upper, lower
        op = paired_operator(grid, parent, upper, lower, cloud)
        self.background = op['background']
        parent_shift = -17/128*alpha**2+spin*alpha**3/12
        au = op['upper']+fine_operator(grid, upper, alpha, spin)-parent_shift*np.eye(len(upper.h0))
        av = op['lower']+fine_operator(grid, lower, alpha, spin)-parent_shift*np.eye(len(lower.h0))
        self.hessian = np.block([[au, cloud*op['coupling']], [cloud*op['coupling'].T, av]])
        metric = np.r_[np.ones(len(upper.h0)), -np.ones(len(lower.h0))]
        values, vectors = eig(metric[:, None]*self.hessian)
        index = int(np.argmin(abs(values)))
        if max(abs(values.imag)) > 1e-10:
            raise ArithmeticError('Shape screen requires a stable conservative spectrum')
        vector = vectors[:, index].real
        norm = vector@(metric*vector)
        if norm >= 0 or values[index].real <= 0:
            raise ArithmeticError('The nearest shape mode is not the negative-energy branch')
        vector /= np.sqrt(-norm)
        ground = lower.functions[1].T@(grid.w*grid.r**2*self.background['fields'][1])
        if ground@vector[len(upper.h0):len(upper.h0)+size] < 0:
            vector *= -1
        self.frequency = float(values[index].real)
        self.weights = {}
        self.source_lmax = source_lmax
        self.labels = [(rank, m) for rank in range(2, source_lmax+1, 2) for m in range(-rank, rank+1, 2)]
        self.source_weights = []
        for sector in [upper, lower]:
            source_weights = {}
            for rank in range(2, source_lmax+1, 2):
                blocks = []
                for ell in sector.ls:
                    product = sum(angular(ell, sector.m, lp, 1, rank, sector.m-1)
                        *self.background['fields'][lp] for lp in parent.ls)
                    blocks.append(sector.functions[ell].T*(grid.w*grid.r**2*product)[None, :])
                source_weights[rank] = np.concatenate(blocks)
            self.source_weights.append(source_weights)
        for rank in range(2, source_lmax+1, 2):
            projected = np.concatenate([w[rank] for w in self.source_weights])
            self.weights[rank] = vector@projected
        self.norm = float(vector@vector)
        self.diagnostics = dict(frequency=self.frequency, euclidean_norm=self.norm,
            nearest_other_frequency=float(min(abs(np.delete(values.real, index)))),
            parent_residual=op['background']['residual'], size=size, order=order,
            maximum_l=maximum_l, source_lmax=source_lmax,
            eigen_residual=float(np.linalg.norm(metric*(self.hessian@vector)-self.frequency*vector)))

    def coefficients(self, a, e):
        return radial_tide_coefficients(self.grid.r, self.weights, self.labels, a, e)

    def static_response(self, a, e, inclination, omega):
        coefficients = self.coefficients(a, e)
        d, _, dw = angular_factors(self.labels, inclination, omega, -2)
        dww = -np.array([m*m for _, m in self.labels])*d
        f, fw, fww = coefficients@d, coefficients@dw, coefficients@dww
        # Construct the full source by applying the same radial integral to each basis row.
        blocks = []
        for sector_weights in self.source_weights:
            matrix = np.array([radial_tide_coefficients(self.grid.r,
                {rank:w[k] for rank, w in sector_weights.items()}, self.labels, a, e)
                for k in range(len(next(iter(sector_weights.values()))))])
            blocks.extend(matrix@d.conj())
        source = np.array(blocks)
        full = -np.vdot(source, np.linalg.solve(self.hessian, source)).real
        slow = abs(f)**2/self.frequency
        return dict(source=[float(f.real), float(f.imag)], induced_energy=float(slow),
            full_induced_energy=float(full), fast_relative_difference=float(full/slow-1),
            occupation_per_q_squared=float(self.norm*abs(f/self.frequency)**2),
            omega_curvature=float(2*(abs(fw)**2+np.vdot(f, fww).real)/self.frequency))


def shape_history(shape, history, axes, *, model, preparation='steady', stride=1,
                  max_step=8., tolerance=2e-10, budget=240.):
    started = time.perf_counter()
    values = [[shape.coefficients(a, e) for e in axes[1]] for a in axes[0]]
    table = StaticTideTable(axes, values, shape.labels)
    chosen = history['samples'][::stride]
    if chosen[-1]['s'] != history['samples'][-1]['s']:
        chosen.append(history['samples'][-1])
    times = np.array([r['s'] for r in chosen])
    source = []
    curvature, radial_curvature, eccentricity_factor = [], [], []
    for r in chosen:
        c = table.spline([r['a'], r['e']])
        ce = table.spline([r['a'], r['e']], nu=(0, 1))
        d, di, dw = angular_factors(shape.labels, r['inclination'], r['omega'], -2)
        dww = -np.array([m*m for _, m in shape.labels])*d
        phase = np.exp(2j*r['node'])
        j, ci, si = np.sqrt(1-r['e']**2), np.cos(r['inclination']), np.sin(r['inclination'])
        fj = -j/r['e']*(ce@d)+ci/(j*si)*(c@di)
        source.append(phase*np.array([c@d, c@dw, c@dww, fj]))
        args = (r['a'], r['e'], r['omega'], r['jz'], history['cloud'], history['alpha'])
        plus = derivatives(*args[:2], args[2]+1e-4, *args[3:], model=model)[1]
        minus = derivatives(*args[:2], args[2]-1e-4, *args[3:], model=model)[1]
        curvature.append((plus-minus)/2e-4)
        hj = [derivatives(r['a'], np.sqrt(1-(j+shift)**2), r['omega'],
            r['jz'], history['cloud'], history['alpha'], model=model)[0]
            for shift in [1e-5, -1e-5]]
        radial_curvature.append((hj[0]-hj[1])/2e-5)
        eccentricity_factor.append(j/r['e'])
    source = np.array(source)
    spline = CubicSpline(times, source)
    node_rate = CubicSpline(times, [r['node'] for r in chosen])(times[0], 1)*history['cloud']
    initial = source[0, 0]/(shape.frequency-2*node_rate)
    if preparation == 'zero':
        initial = 0j
    elif preparation != 'steady':
        raise ValueError('Choose a steady or zero shape preparation')

    def rhs(t, state):
        if time.perf_counter()-started > budget:
            raise TimeoutError('Forced shape evolution exceeded its bounded run')
        return (1j*shape.frequency*state-1j*spline(t)[0])/history['cloud']

    solution = solve_ivp(rhs, [times[0], times[-1]], [initial], method='DOP853',
        t_eval=times, rtol=tolerance, atol=tolerance/100, max_step=max_step)
    if not solution.success:
        raise RuntimeError(solution.message)
    b, q, eps = solution.y[0], history['q'], history['cloud']
    force = 2*eps*q*(b.conj()*source[:, 1]).real
    induced_curvature = 2*eps*q*(b.conj()*source[:, 2]).real
    radial_force = 2*eps*q*(b.conj()*source[:, 3]).real
    return dict(q=q, preparation=preparation, frequency=shape.frequency, stride=stride,
        forcing_times=len(times), maximum_occupation=float(q*q*shape.norm*max(abs(b)**2)),
        maximum_equivalent_angle_shift=float(max(abs(force/curvature))),
        maximum_relative_curvature=float(max(abs(induced_curvature/curvature))),
        maximum_equivalent_eccentricity_shift=float(max(abs(
            radial_force*np.array(eccentricity_factor)/radial_curvature))),
        maximum_physical_amplitude=float(q*max(abs(b))),
        initial_node_frequency=float(node_rate), elapsed_seconds=time.perf_counter()-started,
        scope='One odd-parity m=-1 shape mode forced by a prescribed moving-axis orbit; no shape feedback.',
        samples=[dict(s=float(t), amplitude=[float(z.real), float(z.imag)])
                 for t, z in zip(times[::max(1, (len(times)-1)//80)], b[::max(1, (len(times)-1)//80)])])
