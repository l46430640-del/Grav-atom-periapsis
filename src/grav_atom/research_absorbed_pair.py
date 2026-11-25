"""Leading Coulomb response for gravity-assisted 211+211 -> 322+BH.

Lengths and energies use a_B and E_B. The source includes the acceleration
of the central body. Horizon matching is leading order in alpha only.
"""
import math

import mpmath as mp
import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.special import gammainc, gammaincc


ENERGY = -7/36
DELTA = 5/72
ANGULAR = math.sqrt(2/15)
DECAY = 5/6
PRODUCT = 4/(81*math.sqrt(30)*math.sqrt(24))
DIPOLE = PRODUCT*math.factorial(6)/DECAY**7
ACCELERATION = PRODUCT*math.factorial(3)/DECAY**4


def sources(radius):
    """Real radial sources in the accelerated and canonical relative frames."""
    radius = np.asarray(radius, dtype=float)
    if not np.isfinite(radius).all() or np.any(radius <= 0):
        raise ValueError('Require positive finite radii')
    parent = radius*np.exp(-radius/2)/math.sqrt(24)
    lower = DIPOLE*gammainc(7, DECAY*radius)/radius**2
    upper = radius*ACCELERATION*gammaincc(4, DECAY*radius)
    direct = -ANGULAR*parent*(lower+upper)
    indirect = ANGULAR*parent*ACCELERATION*radius
    # This form avoids subtracting two leading dipoles close to the center.
    physical = ANGULAR*parent*(
        radius*ACCELERATION*gammainc(4, DECAY*radius)-lower)
    canonical = -ANGULAR*DELTA*DIPOLE*np.exp(-radius/2)/math.sqrt(24)*(
        radius/2-3)
    return dict(direct=direct, indirect=indirect, physical=physical,
                canonical=canonical)


def origin_response():
    """Exact Laplace moments of the decaying Coulomb Green function."""
    with mp.workdps(45):
        energy = -mp.mpf(7)/36
        kappa = mp.sqrt(-2*energy)
        nu = 1/kappa
        decay = mp.mpf(5)/6
        product = 4/(81*mp.sqrt(30)*mp.sqrt(24))
        dipole = product*mp.factorial(6)/decay**7
        acceleration = product*mp.factorial(3)/decay**4
        angular = mp.sqrt(mp.mpf(2)/15)

        def moment(power, damping):
            return (2*kappa*mp.gamma(power+2)*mp.gamma(power+1)
                    /mp.gamma(power+2-nu)/(damping+kappa)**(power+2)
                    *mp.hyp2f1(power+2, 1-nu, power+2-nu,
                               (damping-kappa)/(damping+kappa)))

        full_decay = mp.mpf(4)/3
        direct = -angular/mp.sqrt(24)*(
            dipole*moment(0, mp.mpf('.5'))
            -dipole*sum(decay**j/mp.factorial(j)*moment(j, full_decay)
                        for j in range(7))
            +product*(moment(6, full_decay)/decay
                      +3*moment(5, full_decay)/decay**2
                      +6*moment(4, full_decay)/decay**3
                      +6*moment(3, full_decay)/decay**4))
        indirect = angular*acceleration/mp.sqrt(24)*moment(3, mp.mpf('.5'))
        canonical = -angular*(mp.mpf(5)/72)*dipole/mp.sqrt(24)*(
            moment(2, mp.mpf('.5'))/2-3*moment(1, mp.mpf('.5')))
        factor = -2*mp.gamma(1-nu)
        result = {key: float(factor*value) for key, value in dict(
            direct=direct, indirect=indirect, canonical=canonical,
            total=direct+indirect).items()}
        contact = (factor*mp.sqrt(mp.mpf(3)/40)/mp.pi
                   *4/(81*mp.sqrt(30))/24*moment(5, full_decay))
        result.update(contact_origin=float(contact),
                      contact_rate_coefficient=float(contact**2/32))
        return result


def green_quadrature(order=96, cutoff=60.):
    """Independent Whittaker integration; no truncation of the Coulomb spectrum."""
    if not isinstance(order, int) or order < 24 or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError('Require at least 24 nodes and positive cutoff')
    x, weights = np.polynomial.legendre.leggauss(order)
    radius, weights = cutoff*(x+1)/2, cutoff*weights/2
    with mp.workdps(30):
        kappa = mp.sqrt(mp.mpf(7)/18)
        nu = 1/kappa
        waves = np.array([float(mp.whitw(nu, .5, 2*kappa*r)) for r in radius])
        factor = float(-2*mp.gamma(1-nu))
    source = sources(radius)
    return dict(order=order, cutoff=cutoff, **{
        key: float(factor*np.dot(weights, waves*radius*source[key]))
        for key in ['direct', 'indirect', 'physical', 'canonical']})


def radial_response(*, inner=1e-4, outer=60., tolerance=1e-9, absorption=0.):
    """Independent BVP with a weak absorbing inner boundary as a flux check.

    absorption is a dimensionless imaginary contact length, not an exact
    Kerr boundary condition. Its zero limit verifies leading-order matching.
    """
    if (not np.isfinite([inner, outer, tolerance, absorption]).all()
            or not 0 < inner < outer or outer < 10
            or tolerance <= 0 or absorption < 0):
        raise ValueError('Require an ordered radial interval and nonnegative absorption')
    coefficients = [1., -1.]
    for n in range(2, 7):
        coefficients.append(-2*(coefficients[-1]+ENERGY*coefficients[-2])/(n*(n+1)))
    regular = sum(a*inner**(n+1) for n, a in enumerate(coefficients))
    derivative = sum((n+1)*a*inner**n for n, a in enumerate(coefficients))
    boundary_ratio = regular/derivative
    radius = np.unique(np.concatenate([
        np.geomspace(inner, 1., 100), np.linspace(1., outer, 400)]))

    def equation(r, y):
        potential = -2*(ENERGY+1/r)
        forcing = 2*r*sources(r)['physical']
        return np.vstack([y[1], potential*y[0]+forcing,
                          y[3], potential*y[2]])

    def boundary(left, right):
        # u=(boundary_ratio+i*absorption)u' gives inward probability flux.
        return np.array([left[0]-boundary_ratio*left[1]+absorption*left[3],
                         left[2]-boundary_ratio*left[3]-absorption*left[1],
                         right[0], right[2]])

    solved = solve_bvp(equation, boundary, radius, np.zeros((4, len(radius))),
                       tol=tolerance, max_nodes=12000)
    if not solved.success:
        raise ArithmeticError(solved.message)
    left = solved.y[:, 0]
    origin = complex(left[1], left[3])/derivative
    current = left[0]*left[3]-left[2]*left[1]
    source_work, error = quad(
        lambda r: -2*r*float(sources(r)['physical'])*solved.sol(r)[2],
        inner, outer, epsabs=1e-14, epsrel=1e-8, limit=150)
    return dict(inner=inner, outer=outer, tolerance=tolerance,
                absorption=absorption, nodes=len(solved.x),
                regular_derivative=derivative,
                max_residual=float(np.max(solved.rms_residuals)),
                origin_real=origin.real, origin_imag=origin.imag,
                inward_flux=float(-current), source_work=source_work,
                source_work_error=error)


def occupation_gain_M(alpha, spin, cloud_fraction, origin):
    """Leading horizon probability loss, equal to the partner's stimulated gain."""
    if (not np.isfinite([alpha, spin, cloud_fraction, origin]).all()
            or alpha <= 0 or not 0 <= spin < 1 or cloud_fraction < 0):
        raise ValueError('Require positive alpha, subextremal spin and nonnegative cloud')
    rplus = 1+math.sqrt(1-spin*spin)
    return 2*rplus*cloud_fraction**2*alpha**6*origin**2
