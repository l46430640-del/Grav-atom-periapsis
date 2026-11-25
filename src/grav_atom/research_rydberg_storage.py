"""Finite-history Rydberg storage in a prescribed, weak tidal frequency sweep.

Time is t/M, radius is r/a_B, and mode amplitudes are relative to the parent.
The diagonal density diagnostic is not a coherent gravitational waveform.
"""
from functools import lru_cache
import math

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import (eval_genlaguerre, eval_legendre, fresnel, gamma,
                           gammainc, gammaln, roots_genlaguerre)

from boson_ep.spectrum import horizon_omega_M, horizon_radius_M
from .research_collective_mode import angular
from .source_normalization import directed_companion_coefficient


def radial(n, ell, radius):
    if not isinstance(n, int) or not isinstance(ell, int) or not 0 <= ell < n:
        raise ValueError('Require integer n > ell >= 0')
    r = np.asarray(radius)
    if np.any(r < 0):
        raise ValueError('Require nonnegative radius')
    norm = math.exp(math.log(2)-2*math.log(n)
                    +.5*(gammaln(n-ell)-gammaln(n+ell+1)))
    x = 2*r/n
    return norm*x**ell*np.exp(-x/2)*eval_genlaguerre(n-ell-1, 2*ell+1, x)


@lru_cache(maxsize=2048)
def tidal_amplitude(n, ell, radius):
    """Return eta/(q alpha^3) for 211 -> (n,ell,-1), harmonic -2."""
    if ell < 1 or ell % 2 != 1 or radius <= 0:
        raise ValueError('Require a positive odd ell and positive orbit radius')
    result, error = 0j, 0.
    for rank in (ell-1, ell+1):
        if rank < 2:
            continue
        coefficient = directed_companion_coefficient(rank, -2)*angular(ell, -1, 1, 1, rank, -2)
        def integrand(r):
            return (r*r*radial(n, ell, r)*radial(2, 1, r)
                    *min(r, radius)**rank/max(r, radius)**(rank+1))
        a, ea = quad(integrand, 0., radius, epsabs=1e-13, epsrel=1e-10)
        b, eb = quad(integrand, radius, np.inf, epsabs=1e-13, epsrel=1e-10)
        result -= coefficient*(a+b)
        error += abs(coefficient)*(ea+eb)
    return complex(result), float(error)


def loss_coefficient(alpha, chi, ell, m):
    """Return lim n^3 lambda_n, where lambda is a POPULATION absorption rate."""
    if not 0 < alpha < .5 or not 0 <= chi < 1 or ell < 0 or abs(m) > ell:
        raise ValueError('Invalid weak-field mode or black-hole parameters')
    rp, oh = horizon_radius_M(chi), horizon_omega_M(chi)
    if alpha <= m*oh:
        raise ValueError('This storage model requires a passive channel')
    factor = (2**(4*ell+1)
        *(math.factorial(ell)/(math.factorial(2*ell)*math.factorial(2*ell+1)))**2
        *math.prod(j*j*(1-chi*chi)+(chi*m-2*rp*alpha)**2 for j in range(1, ell+1)))
    return 4*rp*factor*(alpha-m*oh)*alpha**(4*ell+5)


def mode_loss(alpha, chi, n, ell, m):
    """Finite n correction, with the real frequency kept to Bohr order."""
    if not isinstance(n, int) or n <= ell:
        raise ValueError('Require integer n > ell')
    correction = math.prod(1-j*j/n**2 for j in range(1, ell+1))
    oh = horizon_omega_M(chi)
    correction *= (alpha*(1-alpha**2/(2*n*n))-m*oh)/(alpha-m*oh)
    return loss_coefficient(alpha, chi, ell, m)*correction/n**3


def finite_band_storage(rate, chirp, band, alpha, loss):
    """Continuum high-n births over detuning (-band,0), at constant rate.

    No daughter initially. Q is occupation/parent, H is the terminal horizon
    current/parent, and excitation = Q + absorbed over the finite history.
    """
    if min(chirp, band, alpha) <= 0 or min(rate, loss) < 0:
        raise ValueError('Require positive scales and nonnegative rates')
    coefficient = loss*(2/alpha**3)**1.5
    depth = coefficient*band**2.5/chirp
    if depth < 1e-4:
        fraction = sum((-depth)**k/(math.factorial(k)*(1+2.5*k)) for k in range(8))
    else:
        fraction = .4*gamma(.4)*gammainc(.4, depth)/depth**.4
    excited = rate*band/chirp
    stored = excited*fraction
    horizon = rate*.4*(-math.expm1(-depth))
    return dict(stored=stored, excited=excited, absorbed=excited-stored,
                horizon=horizon, depth=depth, survival_fraction=fraction)


def finite_chirp_integral(detuning, chirp, start_band):
    """Exact lossless integral int[-D/gamma,0] exp[-i(delta*t+gamma*t^2/2)] dt.

    Source is switched on at a specified finite time. This tests preparation
    and threshold rounding without interpreting occupations as local density.
    """
    if chirp <= 0 or start_band <= 0:
        raise ValueError('Require positive chirp and start band')
    scale = math.sqrt(math.pi*chirp)
    s1, c1 = fresnel(detuning/scale)
    s0, c0 = fresnel((detuning-start_band)/scale)
    return math.sqrt(math.pi/chirp)*np.exp(.5j*detuning**2/chirp)*(c1-c0-1j*(s1-s0))


def finite_chirp_tail(chirp, start_band, tail_band):
    """Average threshold rounding over a small, uniformly weighted tail.

    Replace its remote source turn-on by -infinity with an explicit error
    bound. For x>64 use the averaged large-x limit, again with an error bound.
    This avoids resolving irrelevant endpoint oscillations in a tiny tail.
    """
    if min(chirp, tail_band) <= 0 or start_band <= tail_band:
        raise ValueError('Require positive scales and a tail inside the start band')
    extent = tail_band/math.sqrt(chirp)
    separation = (start_band-tail_band)/math.sqrt(chirp)
    if separation < 10:
        raise ValueError('The omitted source endpoint must be remote')
    def integrand(x):
        s, c = fresnel(x/math.sqrt(math.pi))
        return .5*((c+.5)**2+(s+.5)**2)
    stop = min(64., extent)
    integral, error = quad(integrand, 0, stop, epsabs=1e-10, epsrel=1e-10, limit=2000)
    ratio = (integral+extent-stop)/extent
    error = error/extent+3/separation
    if extent > stop:
        error += 2/(stop*extent)
    return float(ratio), float(error)


def radial_moment(n, ell, power, extra_order=0):
    """Exact-degree Gaussian quadrature of <r^power>, for n <= 128."""
    exponent = 2*ell+2+power
    if exponent <= -1 or n > 128 or extra_order < 0:
        raise ValueError('Moment must exist, with n <= 128 and nonnegative extra order')
    x, w = roots_genlaguerre(n-ell+1+extra_order, exponent)
    polynomial = eval_genlaguerre(n-ell-1, 2*ell+1, x)
    keep = (w > 0) & (polynomial != 0)
    log_prefactor = (math.log(4)-4*math.log(n)+gammaln(n-ell)
                     -gammaln(n+ell+1)+(3+power)*math.log(n/2))
    return float(np.exp(log_prefactor+np.log(w[keep])+2*np.log(abs(polynomial[keep]))).sum())


def equatorial_force(n, ell, m, radius):
    """R^2 dPhi/dR per unit cloud mass for an incoherent single eigenstate.

    A central point mass gives 1. Both interior and exterior even multipoles
    of |Y_lm|^2 are retained; this is not merely the enclosed spherical mass.
    """
    if radius <= 0 or abs(m) > ell:
        raise ValueError('Require positive radius and |m| <= ell')
    result = 0.
    for rank in range(0, 2*ell+1, 2):
        shape = (4*math.pi/(2*rank+1)*math.sqrt((2*rank+1)/(4*math.pi))
                 *eval_legendre(rank, 0.)*angular(ell, m, ell, m, rank, 0))
        inside = quad(lambda r:r**(rank+2)*radial(n, ell, r)**2,
                      0, radius, epsabs=1e-13)[0]
        outside = 0.
        if rank:
            lower = quad(lambda r:r**(1-rank)*radial(n, ell, r)**2,
                         0, radius, epsabs=1e-13)[0]
            outside = radial_moment(n, ell, -rank-1)-lower
        result += shape*((rank+1)*inside/radius**rank-rank*radius**(rank+1)*outside)
    return float(result)


def local_sweep(alpha, q, cloud_fraction, total_coefficient):
    """Prescribed local sweep using GW plus averaged g=-2 tidal work only."""
    if min(alpha, q, total_coefficient) <= 0 or cloud_fraction < 0:
        raise ValueError('Invalid sweep parameters')
    u = alpha**3/16
    gw = 2*(96/5)*q*u**(11/3)
    tidal = 12*cloud_fraction*total_coefficient*q*alpha**2*u**(4/3)
    return dict(omega=u, gw=gw, tidal=tidal, chirp=gw+tidal)


def sinking_sweep(couplings, offsets, feedback, duration=80., rtol=2e-10):
    """Conservative few-mode sweep, x=t+feedback*sum|b|^2.

    All frequencies and time are scaled by sqrt(external chirp). Offsets are
    daughter energies relative to a chosen resonance. Decay is omitted.
    Finite-duration oscillations must be checked, not mistaken for asymptotic
    Landau-Zener probabilities. This is the known sinking-resonance model.
    """
    couplings = np.asarray(couplings, dtype=complex)
    offsets = np.asarray(offsets, dtype=float)
    if couplings.ndim != 1 or offsets.shape != couplings.shape or feedback < 0 or duration <= 0:
        raise ValueError('Require matching vectors and nonnegative feedback')
    def rhs(t, y):
        occupation = float(np.vdot(y[1:], y[1:]).real)
        detuning = offsets-t-feedback*occupation
        return -1j*np.r_[np.vdot(couplings, y[1:]), couplings*y[0]+detuning*y[1:]]
    initial = np.r_[1.+0j, np.zeros(len(couplings), complex)]
    solution = solve_ivp(rhs, (-duration, duration), initial, method='DOP853',
                         rtol=rtol, atol=rtol*.01, dense_output=True)
    if not solution.success:
        raise RuntimeError(solution.message)
    # Endpoint beats are suppressed by a final interval average, still at finite T.
    samples = solution.sol(np.linspace(.8*duration, duration, 501))
    populations = abs(samples[1:])**2
    norms = np.sum(abs(solution.y)**2, axis=0)
    return dict(mean_populations=populations.mean(axis=1).tolist(),
                endpoint_populations=populations[:, -1].tolist(),
                endpoint_ranges=np.ptp(populations, axis=1).tolist(),
                norm_error=float(max(abs(norms-1))), evaluations=solution.nfev)
