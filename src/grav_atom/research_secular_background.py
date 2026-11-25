"""Self-consistent Newtonian 211 density for conservative test-orbit motion.

The stationary wavefunction is normalized to one. The potential returned
here is per unit cloud mass fraction, in the Bohr units of cloud_secular.
"""

import math

import numpy as np
from scipy.interpolate import BPoly, CubicSpline
from scipy.special import eval_legendre

from .research_collective_mode import (
    RadialGrid, angular, make_sector, stationary_parent)
from .research_rydberg_storage import radial


class StationaryCloud:
    def __init__(self, fraction, size=24, order=40, angular_max=3, *, daughter_ratio=0.):
        if angular_max not in (1, 3, 5, 7):
            raise ValueError("Require odd angular maximum from 1 to 7")
        if not np.isfinite(daughter_ratio) or daughter_ratio < 0:
            raise ValueError("Require a nonnegative dilute daughter ratio")
        grid = RadialGrid(order)
        sector = make_sector(grid, 1, tuple(range(1, angular_max+1, 2)), size)
        state = stationary_parent(grid, sector, fraction)
        self.name = "stationary_211" if not daughter_ratio else "stationary_211_plus_bare_322"
        self.fraction = fraction
        self.radius = grid.r
        self.splines = {}
        r = grid.r
        dipole = state['fields'][1]
        sign = np.sign(np.dot(grid.w*r*r*radial(2, 1, r), dipole))
        self.dipole_radial = CubicSpline(np.r_[0., r], np.r_[0., sign*dipole], extrapolate=False)
        for rank in range(0, max(2*angular_max, 4 if daughter_ratio else 0)+1, 2):
            density = sum(angular(left, 1, right, 1, rank, 0)
                          *state['fields'][left]*state['fields'][right]
                          for left in sector.ls for right in sector.ls)
            if daughter_ratio:
                # Sensitivity check only; the parent is not resolved in the daughter field.
                density += daughter_ratio*angular(2, 2, 2, 2, rank, 0)*radial(3, 2, r)**2
            lower = grid.cumulative(r**(rank+2)*density)[0]
            upper = grid.cumulative(r**(1-rank)*density)[1]
            coefficient = -math.sqrt(4*math.pi/(2*rank+1))
            value = coefficient*(lower/r**(rank+1)+r**rank*upper)
            first = coefficient*(-(rank+1)*lower/r**(rank+2)
                                 +rank*r**(rank-1)*upper)
            second = coefficient*((rank+1)*(rank+2)*lower/r**(rank+3)
                +rank*(rank-1)*r**(rank-2)*upper-(2*rank+1)*density)
            self.splines[rank] = BPoly.from_derivatives(
                r, np.column_stack([value, first, second]), extrapolate=False)
        norm = sum(np.dot(grid.w*r*r, field*field) for field in state['fields'].values())
        self.diagnostics = dict(fraction=fraction, daughter_ratio=daughter_ratio, size=size, radial_order=order,
            angular_max=angular_max, norm=float(norm), chemical=state['chemical'],
            stationary_residual=state['residual'], overlap_error=sector.overlap_error,
            iterations=state['iterations'], angular_weights={str(l):float(np.dot(
                grid.w*r*r, state['fields'][l]**2)) for l in sector.ls})

    def evaluate(self, radius, cosine):
        """Potential and partial radial/angular derivatives, cosine=cos(theta)."""
        r, z = np.broadcast_arrays(np.asarray(radius), np.asarray(cosine))
        if (np.any(r < self.radius[0]) or np.any(r > self.radius[-1])
                or np.any(np.abs(z) > 1+1e-12)):
            raise ValueError("Orbit outside the tabulated cloud domain")
        result = [np.zeros_like(r, dtype=float) for _ in range(3)]
        for rank, spline in self.splines.items():
            polynomial = eval_legendre(rank, z)
            # P'_l = sum_{k=l-1,l-3,...} (2k+1)P_k also holds at the poles.
            derivative = sum((2*k+1)*eval_legendre(k, z)
                             for k in range(rank-1, -1, -2))
            value = spline(r)
            result[0] += value*polynomial
            result[1] += spline(r, nu=1)*polynomial
            result[2] += value*derivative
        return tuple(result)
