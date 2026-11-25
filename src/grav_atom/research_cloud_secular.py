"""Test-particle secular dynamics in a fixed, hydrogenic 211 cloud.

Radii are in M/alpha**2. The averaged specific Hamiltonian is divided by
alpha**2; its canonical time is alpha**3*t/(M*sqrt(a)).
"""

from functools import lru_cache
import time

import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import gammainc, gammaincc, roots_legendre


def potential(radius, model="full"):
    """Return monopole and P_2 coefficients per unit cloud mass fraction."""
    r = np.asarray(radius, dtype=float)
    if np.any(r <= 0):
        raise ValueError("Radius must be positive")
    if model == "inner":
        return np.zeros_like(r), r*r/120
    if model == "outer":
        return -1/r, 6/r**3
    u0 = -gammainc(5, r)/r-gammaincc(4, r)/4
    u2 = 6*gammainc(7, r)/r**3+r*r*gammaincc(2, r)/120
    if model == "spherical":
        u2 = np.zeros_like(r)
    elif model != "full":
        raise ValueError("Unknown potential model")
    return u0, u2


@lru_cache(maxsize=8)
def anomaly_grid(order):
    node, weight = roots_legendre(order)
    anomaly = np.pi*(node+1)
    return np.cos(anomaly), np.sin(anomaly), weight/2


def axisymmetric_field(radius, cosine, model="full"):
    """Potential and partial r, cos(theta) derivatives for an axial cloud."""
    if hasattr(model, "evaluate"):
        return model.evaluate(radius, cosine)
    u0, u2 = potential(radius, model)
    u0p, u2p = potential_derivative(radius, model)
    p2 = (3*cosine**2-1)/2
    return u0+u2*p2, u0p+u2p*p2, 3*u2*cosine


def hamiltonian(a, eccentricity, omega, jz, cloud, alpha=0.0, spin=0.0,
                order=96, model="full"):
    """Orbit average at fixed semimajor axis and jz=sqrt(1-e**2)*cos(I)."""
    e = float(eccentricity)
    j2 = 1-e*e
    if not (a > 0 and 0 <= e < 1 and jz*jz <= j2+1e-14):
        raise ValueError("Unphysical orbit")
    j = np.sqrt(j2)
    cosine, sine, weight = anomaly_grid(order)
    measure = 1-e*cosine
    cosf = (cosine-e)/measure
    sinf = j*sine/measure
    sintheta_axis = sinf*np.cos(omega)+cosf*np.sin(omega)
    cos_theta = np.sqrt(max(0.0, 1-jz*jz/j2))*sintheta_axis
    field = axisymmetric_field(a*measure, cos_theta, model)[0]
    average = np.dot(weight*measure, field)
    return (cloud*average-3*alpha**2/(a*a*j)
            +2*spin*alpha**3*jz/(a**2.5*j**3))


def linear_coefficients(a, inclination, cloud, alpha=0.0, spin=0.0,
                        step=0.004, order=96, model="full"):
    """Coefficients of e**2*cos(omega)**2 and e**2*sin(omega)**2.

    Richardson extrapolation removes the leading finite-e error. The e**2
    coefficients equal those of canonical Cartesian variables at the origin.
    """
    jz = np.cos(inclination)
    result = []
    for omega in (0, np.pi/2):
        base = hamiltonian(a, 0, omega, jz, cloud, alpha, spin, order, model)
        values = [(hamiltonian(a, e, omega, jz, cloud, alpha, spin,
                               order, model)-base)/e**2
                  for e in (step, step/2)]
        result.append((4*values[1]-values[0])/3)
    return np.array(result)


def growth_rate(a, alpha, coefficients):
    """Positive instability exponent M*lambda; zero for elliptic motion."""
    cx, cy = coefficients
    return 2*alpha**3/np.sqrt(a)*np.sqrt(max(0.0, -cx*cy))


def exact_coefficients(a, inclination, cloud, alpha=0.0, spin=0.0):
    """Analytic circular-orbit Hessian of the full hydrogenic potential."""
    z = np.cos(inclination)
    cx = (9*(1-5*z*z)*gammainc(7, a)/(4*a**3)
          -a*a*gammaincc(2, a)/80
          +a**4*np.exp(-a)*(3+z*z)/256)
    difference = (1-z*z)*np.exp(-a)*a*a*(4+4*a-a*a)/128
    common = -1.5*alpha**2/a**2+3*spin*alpha**3*z/a**2.5
    return cloud*np.array([cx, cx+difference])+common


def potential_derivative(radius, model="full"):
    r = np.asarray(radius, dtype=float)
    if model == "inner":
        return np.zeros_like(r), r/60
    if model == "outer":
        return 1/r**2, -18/r**4
    u0p = gammainc(5, r)/r**2
    u2p = -18*gammainc(7, r)/r**4+r*gammaincc(2, r)/60
    if model == "spherical":
        u2p = np.zeros_like(r)
    elif model != "full":
        raise ValueError("Unknown potential model")
    return u0p, u2p


def derivatives(a, e, omega, jz, cloud, alpha, spin=0.0,
                order=96, model="full"):
    """Return partial derivatives with respect to j and omega at fixed jz."""
    j2 = 1-e*e
    j = np.sqrt(j2)
    cosine, sine, weight = anomaly_grid(order)
    measure = 1-e*cosine
    cosf, sinf = (cosine-e)/measure, j*sine/measure
    dc = -sine**2/measure**2
    ds = -e*sine/(j*measure)+j*sine*cosine/measure**2
    axis = sinf*np.cos(omega)+cosf*np.sin(omega)
    axis_e = ds*np.cos(omega)+dc*np.sin(omega)
    axis_w = -sinf*np.sin(omega)+cosf*np.cos(omega)
    si2, si2_e = 1-jz*jz/j2, -2*jz*jz*e/j2**2
    si = np.sqrt(max(0.0, si2))
    field, radial_gradient, angular_gradient = axisymmetric_field(a*measure, si*axis, model)
    if si > 0:
        angular_term = angular_gradient*(si*axis_e+si2_e*axis/(2*si))
    else:
        # The even axial potential has a finite limit at zero inclination.
        curvature = axisymmetric_field(a*measure, np.full_like(axis, 1e-6), model)[2]/1e-6
        angular_term = curvature*si2_e*axis**2/2
    he = cloud*np.dot(weight, -cosine*field
        +measure*(-a*cosine*radial_gradient+angular_term))
    hj = -j*he/e+3*alpha**2/(a*a*j2)
    hj -= 6*spin*alpha**3*jz/(a**2.5*j2**2)
    hw = cloud*np.dot(weight*measure, angular_gradient*si*axis_w)
    return hj, hw


def secular_history(a, inclination, cloud, alpha, spin=0.0, e0=0.002,
                    omega0=np.pi/4, duration=1200.0, order=96,
                    tolerance=2e-10, model="full", samples=401):
    """Short conservative history in s=cloud*alpha**3*t/(M*sqrt(a))."""
    j0 = np.sqrt(1-e0*e0)
    jz = j0*np.cos(inclination)
    rho0 = np.sqrt(2*(1-j0))
    state = rho0*np.array([np.cos(omega0), np.sin(omega0)])

    def rhs(_, xy):
        x, y = xy
        rho2 = x*x+y*y
        j = 1-rho2/2
        e = np.sqrt(max(0.0, 1-j*j))
        hj, hw = derivatives(a, e, np.arctan2(y, x), jz,
                             cloud, alpha, spin, order, model)
        return np.array([-y*hj+x*hw/rho2, x*hj+y*hw/rho2])/cloud

    solution = solve_ivp(rhs, [0, duration], state, method="DOP853",
                         rtol=tolerance, atol=tolerance/100,
                         t_eval=np.linspace(0, duration, samples), max_step=5)
    if not solution.success:
        raise RuntimeError(solution.message)
    x, y = solution.y
    j = 1-(x*x+y*y)/2
    eccentricity = np.sqrt(1-j*j)
    omega = np.arctan2(y, x)
    energy = np.array([hamiltonian(a, e, w, jz, cloud, alpha, spin,
                                  order, model)
                       for e, w in zip(eccentricity, omega)])
    return dict(a=a, inclination=float(inclination), cloud=cloud, alpha=alpha,
                spin=spin, e0=e0, omega0=float(omega0), model=getattr(model, "name", model), order=order,
                tolerance=tolerance, jz=float(jz), time_s=list(solution.t),
                eccentricity=list(eccentricity), omega=list(omega),
                inclination_history=list(np.arccos(np.clip(jz/j, -1, 1))),
                energy_drift_over_cloud=float(np.ptp(energy)/cloud),
                maximum_eccentricity=float(max(eccentricity)),
                time_M_per_s=float(np.sqrt(a)/(cloud*alpha**3)),
                function_evaluations=solution.nfev)


def direct_history(a, inclination, cloud, alpha, e0=0.02,
                   omega0=np.pi/4, orbits=2000, tolerance=2e-10,
                   budget=240):
    """Unaveraged canonical 1PN test orbit, with a fixed Newtonian cloud.

    This conservative truncated Hamiltonian omits cloud-PN cross terms and
    black-hole spin. It tests secular averaging, not relativistic accuracy.
    """
    start = time.perf_counter()
    peri = np.array([np.cos(omega0), np.sin(omega0)*np.cos(inclination),
                     np.sin(omega0)*np.sin(inclination)])
    tangent = np.array([-np.sin(omega0), np.cos(omega0)*np.cos(inclination),
                        np.cos(omega0)*np.sin(inclination)])
    initial = np.r_[a*(1-e0)*peri,
                    np.sqrt((1+e0)/(a*(1-e0)))*tangent]

    def rhs(_, state):
        if time.perf_counter()-start > budget:
            raise TimeoutError("Direct orbit exceeded time budget")
        r, p = state[:3], state[3:]
        radius = np.linalg.norm(r)
        p2 = np.dot(p, p)
        z = r[2]/radius
        _, u2 = potential(radius)
        u0p, u2p = potential_derivative(radius)
        gradient = (u0p+u2p*(3*z*z-1)/2)*r/radius
        gradient += 3*u2*z*(np.array([0, 0, 1])-z*r/radius)/radius
        velocity = p*(1-alpha*alpha*(p2/2+3/radius))
        force = -r/radius**3-cloud*gradient
        force += alpha*alpha*(r/radius**4-1.5*p2*r/radius**3)
        return np.r_[velocity, force]

    period = 2*np.pi*a**1.5
    sample_t = np.linspace(0, orbits*period, 16*orbits+1)
    solution = solve_ivp(rhs, [0, sample_t[-1]], initial, method="DOP853",
                         rtol=tolerance, atol=tolerance/100, t_eval=sample_t,
                         max_step=period/8)
    if not solution.success:
        raise RuntimeError(solution.message)
    r, p = solution.y[:3].T, solution.y[3:].T
    radius = np.linalg.norm(r, axis=1)
    p2 = np.sum(p*p, axis=1)
    angular = np.cross(r, p)
    kepler = p2/2-1/radius
    eccentricity = np.sqrt(np.maximum(0, 1+2*kepler*np.sum(angular**2, axis=1)))
    u0, u2 = potential(radius)
    energy = (kepler+cloud*(u0+u2*(3*(r[:, 2]/radius)**2-1)/2)
              +alpha**2*(-p2**2/8-1.5*p2/radius+0.5/radius**2))
    # Bin ten nominal periods to remove orbital oscillations in osculating e.
    block = 160
    count = (len(sample_t)-1)//block
    means = eccentricity[:count*block].reshape(count, block).mean(axis=1)
    times = sample_t[:count*block].reshape(count, block).mean(axis=1)
    return dict(a=a, inclination=float(inclination), cloud=cloud, alpha=alpha,
                e0=e0, omega0=float(omega0), orbits=orbits, tolerance=tolerance,
                time_s=list(times*cloud/np.sqrt(a)),
                mean_eccentricity=list(means),
                maximum_mean_eccentricity=float(max(means)),
                energy_drift_over_cloud=float(np.ptp(energy)/cloud),
                jz_relative_drift=float(np.ptp(angular[:, 2])/np.sqrt(a)),
                function_evaluations=solution.nfev,
                elapsed_seconds=time.perf_counter()-start)
