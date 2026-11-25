"""Local, dipolar horizon response for negative orbital harmonics."""

import numpy as np

from .research_static_tide import StaticTideTable, angular_factors


def pole_factored_spline(spline, harmonic, frequency):
    def evaluate(point, nu=(0, 0)):
        if nu[0] != 0:
            raise ValueError('Semimajor-axis derivatives require differentiating the pole factor')
        return spline(point, nu=nu)/(frequency+harmonic*point[0]**(-1.5))
    return evaluate


class NegativeTideTable(StaticTideTable):
    def __init__(self, axes, values, harmonic, dispersion=None, reference_frequency=None):
        if not isinstance(harmonic, int) or harmonic >= 0:
            raise ValueError('A negative integer harmonic is required')
        values = np.asarray(values)
        if values.shape[-2:] != (2, 2):
            raise ValueError('Each point must contain two complex amplitudes as real-imaginary pairs')
        amplitudes = values[..., 0]+1j*values[..., 1]
        if reference_frequency is not None:
            if not np.isfinite(reference_frequency) or reference_frequency <= 0:
                raise ValueError('The reference frequency must be finite and positive')
            factor = reference_frequency+harmonic*np.asarray(axes[0])**(-1.5)
            if np.any(factor == 0) or np.any(factor*factor[0] <= 0):
                raise ValueError('A local response table must not straddle the reference pole')
            amplitudes = amplitudes*factor[:, None, None]
        super().__init__(axes, amplitudes, [(1, -1), (1, 1)])
        self.harmonic = harmonic
        if reference_frequency is not None:
            self.spline = pole_factored_spline(self.spline, harmonic, reference_frequency)
            if dispersion is not None:
                dispersion = np.asarray(dispersion)*factor[:, None, None, None]
        self.dispersion = None if dispersion is None else StaticTideTable(
            axes, dispersion, [(1, -1), (1, 1)])
        if self.dispersion is not None and reference_frequency is not None:
            self.dispersion.spline = pole_factored_spline(
                self.dispersion.spline, harmonic, reference_frequency)

    def rates(self, *args, **kwargs):
        result = super().rates(*args, **kwargs)
        result.pop('static_number')
        result['horizon_number'] = result['number']
        result['energy'] = self.harmonic*result['number']
        return result

    def induced_energy(self, a, e, inclination, omega):
        """Real source response per epsilon*q, without a slow-action reduction."""
        if self.dispersion is None:
            raise ValueError('No dispersive source response was supplied')
        matrix = self.dispersion.spline([a, e])
        d, _, dw = angular_factors(self.labels, inclination, omega)
        dww = -d
        value = float(np.vdot(d, matrix@d).real)
        curvature = float(2*np.vdot(dw, matrix@dw).real+2*np.vdot(d, matrix@dww).real)
        return dict(value=value, omega_curvature=curvature)
