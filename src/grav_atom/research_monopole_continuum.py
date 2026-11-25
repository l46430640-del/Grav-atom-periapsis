"""Continuum sensitivity to the finite parent and its spherical Hartree field."""
from functools import lru_cache

import mpmath as mp
import numpy as np
from scipy.integrate import solve_ivp


def exterior_basis(energy, charge, radius, ell_max):
    """Unit-amplitude Coulomb pair and radial derivatives; DLMF 33.4.2, 33.4.4."""
    k = np.sqrt(2*energy)
    eta, rho = -charge/k, k*radius
    def s(l):
        return l/rho+eta/l
    def r(l):
        return np.hypot(1., eta/l)
    with mp.workdps(30):
        x = [np.array([float(fn(l, eta, rho)) for fn in (mp.coulombf, mp.coulombg)]) for l in [0,1]]
    for l in range(1, ell_max+1):
        x.append(((s(l)+s(l+1))*x[-1]-r(l)*x[-2])/r(l+1))
    values = np.array(x[:-1])
    slopes = k*np.array([s(l+1)*x[l]-r(l+1)*x[l+1] for l in range(ell_max+1)])
    if np.max(abs((values[:,0]*slopes[:,1]-values[:,1]*slopes[:,0])/k+1)) > 2e-10:
        raise ArithmeticError('Exterior Coulomb basis lost its Wronskian')
    return np.stack([values, slopes], axis=1)


class MonopoleContinuum:
    """Open test waves in -1/x + epsilon*u0 with Coulomb charge 1+epsilon at infinity."""
    def __init__(self, profile, match=110.):
        self.profile, self.match = profile, match
        if abs(profile.splines[0](match)*match+1) > 1e-9:
            raise ValueError('Matching surface is still inside the cloud')
        self.solution = lru_cache(maxsize=40)(self._solve)

    def _solve(self, energy, ell_max, tolerance):
        if energy <= 0:
            raise ValueError('The continuum energy must be positive')
        ell = np.arange(ell_max+1)
        inner = 1e-5
        eps = self.profile.fraction
        potential = self.profile.splines[0]
        rmin = self.profile.radius[0]
        def rhs(r, y):
            u, up = y.reshape(2,-1)
            v = -1/r+eps*float(potential(max(r,rmin)))
            return np.r_[up, (ell*(ell+1)/r**2+2*(v-energy))*u]
        y0 = np.r_[np.ones_like(ell), (ell+1)/inner-1/(ell+1)]
        sol = solve_ivp(rhs, [inner,self.match], y0, rtol=tolerance, atol=tolerance/100,
                        method='DOP853', dense_output=True)
        if not sol.success:
            raise ArithmeticError(sol.message)
        fg = exterior_basis(energy, 1+eps, self.match, ell_max)
        coefficients = np.linalg.solve(fg, sol.y[:,-1].reshape(2,-1).T[:,:,None])[:,:,0]
        norm = np.sqrt(2/(np.pi*np.sqrt(2*energy)))/np.linalg.norm(coefficients,axis=1)
        return sol.sol, norm

    def waves(self, grid, ls, energy, *, tolerance=2e-11):
        if min(grid.r) < 1e-5 or max(grid.r) > self.match:
            raise ValueError('Source grid is outside the matched wave solution')
        sol, norm = self.solution(energy, max(ls), tolerance)
        values = sol(grid.r)[:len(norm)]*norm[:,None]/grid.r
        return {l:dict(normalized=values[l]) for l in ls}
