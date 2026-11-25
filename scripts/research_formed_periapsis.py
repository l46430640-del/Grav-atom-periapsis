"""Bounded periapsis checks on a cloud selected from its formation history."""

from pathlib import Path
import argparse
import gzip
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np

from grav_atom.research_causal_periapsis import PoleTide, coupled_history
from grav_atom.research_axis_response import axis_response
from grav_atom.research_inclined_ionization import eccentric_projection, coherent_flux, torque_coefficients
from grav_atom.research_periapsis_migration import ResponseTable, equilibrium
from grav_atom.research_secular_background import StationaryCloud
from grav_atom.research_relative_tilt import tilted_cloud, inspect_tilt
from grav_atom.research_static_tide import StaticCloudTide, CombinedResponse, nodal_frequency
from grav_atom.research_negative_tide import NegativeTideTable

OUT = ROOT/'runs/formed'
REFERENCE = ROOT/'data/formed_periapsis.json'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def parameters():
    selected = read(REFERENCE)['grid']
    return selected['parameters'], selected['formation']


def response_grid():
    rerun = OUT/'grid.json'
    if rerun.is_file():
        return read(rerun)
    data = read(REFERENCE)['grid']
    with gzip.open(ROOT/'data/ionization_projections.json.gz', 'rt', encoding='utf-8') as stream:
        data['projections'] = json.load(stream)
    return data


def profile(params):
    return StationaryCloud(params['cloud'], 28, 56, 5)


def grid():
    started = time.perf_counter()
    params, formation = parameters()
    model = StaticCloudTide(params['cloud'], params['alpha'], params['spin'])
    pole = model.slow_pole()
    axes = [np.linspace(3.08, 3.27, 5), np.linspace(.43, .63, 7)]
    values = [[list(model.coefficients(a, e, weights=pole['weights'])) for e in axes[1]]
              for a in axes[0]]
    projections = []
    for a in axes[0]:
        line = []
        for e in axes[1]:
            if time.perf_counter()-started > 230.:
                raise TimeoutError('Response grid exceeded its bounded run')
            line.append(eccentric_projection(float(a), float(e), maximum_harmonic=16,
                ell_max=8, order=24, anomaly_order=32, cutoff=80., tolerance=2e-11))
        projections.append(line)
        print(json.dumps(dict(a=float(a), seconds=time.perf_counter()-started)), flush=True)
    return dict(parameters=params, formation=formation, axes=[list(a) for a in axes],
        values=values, labels=model.labels, pole={k:v for k, v in pole.items() if k != 'weights'},
        projections=projections, elapsed_seconds=time.perf_counter()-started)


def models():
    data = response_grid()
    pole = PoleTide(data['axes'], data['values'], data['labels'],
        data['pole']['frequency'], data['pole']['population_width'])
    ion = ResponseTable(data['axes'], data['projections'])
    return data['parameters'], pole, ion, profile(data['parameters'])


def checks():
    params, pole, ion, cloud_profile = models()
    tide = StaticCloudTide(params['cloud'], params['alpha'], params['spin'])
    rows = []
    for a in [3.25, 3.175, 3.1]:
        point = equilibrium(a, params['jz0'], params['cloud'], params['alpha'], params['spin'], cloud_profile)
        e, inclination = point['e'], np.arccos(params['jz0']/point['j'])
        projected = eccentric_projection(a, e, maximum_harmonic=32, ell_max=12,
            order=40, anomaly_order=64, cutoff=100., tolerance=2e-12)
        node = nodal_frequency(a, e, np.pi/2, params['jz0'], params['cloud'],
                               params['alpha'], params['spin'], cloud_profile)
        row = dict(a=a, e=e, inclination=float(inclination), equilibrium=point,
            node_frequency=node, libration_frequency=float(np.sqrt(point['hjj']*point['hww']/a)),
            static_locked=tide.static_response(a, e, inclination, np.pi/2),
            static_aligned=tide.static_response(a, e, inclination, 0.), comparisons=[])
        for omega in [np.pi/2, np.pi/2+.06]:
            reference = coherent_flux(projected, inclination, omega)
            torque = torque_coefficients(projected, inclination, omega)
            table = ion.rates(a, e, inclination, omega)
            row['comparisons'].append(dict(omega=float(omega), direct_flux=reference,
                                          direct_torque=torque, table=table))
        rows.append(row)
        print(json.dumps(dict(a=a, e=e, energy_relative_error=
            row['comparisons'][0]['table']['energy']/row['comparisons'][0]['direct_flux']['energy_coefficient']-1)), flush=True)
    return dict(parameters=params, points=rows)


def negative():
    params, _, ion, cloud_profile = models()
    tide = StaticCloudTide(params['cloud'], params['alpha'], params['spin'])
    rows = []
    for a in [3.25, 3.175, 3.1]:
        point = equilibrium(a, params['jz0'], params['cloud'], params['alpha'], params['spin'], cloud_profile)
        e, inclination = point['e'], np.arccos(params['jz0']/point['j'])
        harmonics = [tide.negative_harmonic(a, e, inclination, np.pi/2, h) for h in range(-6, 0)]
        row = dict(a=a, e=e, inclination=float(inclination), harmonics=harmonics,
            energy=sum(r['horizon_energy_first_order'] for r in harmonics),
            number=sum(r['horizon_number_first_order'] for r in harmonics),
            ionization=ion.rates(a, e, inclination, np.pi/2))
        rows.append(row)
        print(json.dumps(dict(a=a, horizon_over_ion_energy=row['energy']/row['ionization']['energy'])), flush=True)
    return dict(parameters=params, scope='First-order horizon rates with finite-cloud outgoing partners; point checks, not an evolved negative-harmonic response.', points=rows)


def negative_grid(harmonic):
    started = time.perf_counter()
    data = response_grid()
    params = data['parameters']
    tide = StaticCloudTide(params['cloud'], params['alpha'], params['spin'], source_lmax=1)
    values, dispersion = [], []
    for a in data['axes'][0]:
        line, real_line = [], []
        for e in data['axes'][1]:
            points = []
            for inclination in [0., np.pi, np.pi/2]:
                if time.perf_counter()-started > 230.:
                    raise TimeoutError('Negative-harmonic table exceeded its bounded run')
                points.append(tide.negative_harmonic(a, e, inclination, 0., harmonic))
            amplitudes = [np.sqrt(tide.sink)*complex(*p['closed_origin_amplitude'])
                          for p in points[:2]]
            line.append([[amplitudes[0].real, amplitudes[0].imag],
                         [-amplitudes[1].real, -amplitudes[1].imag]])
            p0, pp, pm = [p['dispersive_response'] for p in points]
            cross = (p0+pp-4*pm)/2
            real_line.append([[p0, cross], [cross, pp]])
        values.append(line)
        dispersion.append(real_line)
        print(json.dumps(dict(a=a, seconds=time.perf_counter()-started)), flush=True)
    return dict(parameters=params, harmonic=harmonic, axes=data['axes'], values=values,
                dispersion=dispersion, elapsed_seconds=time.perf_counter()-started)


def negative_model():
    tables = []
    archived = {row['harmonic']:row for row in read(REFERENCE)['negative_grids']}
    for h in [-2, -1]:
        path = OUT/f'negative-grid-{h}.json'
        data = read(path) if path.is_file() else archived[h]
        if 'reference_frequency' not in data:
            raise ValueError('Regenerate both negative grids, then run regularize before use.')
        tables.append(NegativeTideTable(data['axes'], data['values'], h, data['dispersion'],
                                      data.get('reference_frequency')))
    return CombinedResponse(*tables)


def regularize():
    params, _ = parameters()
    tide = StaticCloudTide(params['cloud'], params['alpha'], params['spin'])
    size = len(tide.upper.h0)
    values, vectors = np.linalg.eigh(tide.hessian[size:, size:])
    reference = dict(frequency=float(-values[0]),
                     width=float(tide.sink*(tide.lower.origin@vectors[:, 0])**2))
    for h in [-2, -1]:
        path = OUT/f'negative-grid-{h}.json'
        data = read(path)
        data['reference_frequency'] = reference['frequency']
        data['closed_ground_reference'] = reference
        write(path, data)
    return reference


def negative_checks():
    params, pole, ion, cloud_profile = models()
    full = StaticCloudTide(params['cloud'], params['alpha'], params['spin'])
    dipole = StaticCloudTide(params['cloud'], params['alpha'], params['spin'], source_lmax=1)
    responses = negative_model().responses
    rows = []
    for a in [3.25, 3.175, 3.1]:
        point = equilibrium(a, params['jz0'], params['cloud'], params['alpha'], params['spin'], cloud_profile)
        e, inclination = point['e'], np.arccos(params['jz0']/point['j'])
        for response in responses:
            for omega in [np.pi/2, np.pi/2+.06]:
                direct = dipole.negative_harmonic(a, e, inclination, omega, response.harmonic)
                physical = full.negative_harmonic(a, e, inclination, omega, response.harmonic)
                table = response.rates(a, e, inclination, omega)
                real = response.induced_energy(a, e, inclination, omega)
                rows.append(dict(a=a, e=e, inclination=float(inclination), omega=float(omega),
                    harmonic=response.harmonic, direct=direct, full=physical, table=table,
                    dispersion=real, relative_curvature=params['cloud']*1e-4*real['omega_curvature']/point['hww']))
        print(json.dumps(dict(a=a, completed=len(rows))), flush=True)
    return dict(parameters=params, points=rows)


def resolution():
    params, _, _, cloud_profile = models()
    coarse = StaticCloudTide(params['cloud'], params['alpha'], params['spin'])
    fine = StaticCloudTide(params['cloud'], params['alpha'], params['spin'],
        size=32, order=64, parent_lmax=5, response_lmax=6, source_lmax=7, fine_n=10)
    rows = []
    for tide in [coarse, fine]:
        pole = tide.slow_pole()
        points = []
        for a in [3.25, 3.1]:
            point = equilibrium(a, params['jz0'], params['cloud'], params['alpha'], params['spin'], cloud_profile)
            e, inclination = point['e'], np.arccos(params['jz0']/point['j'])
            points.append(dict(a=a, static=tide.static_response(a, e, inclination, np.pi/2+.06),
                negative=tide.negative_harmonic(a, e, inclination, np.pi/2+.06, -2)))
        rows.append(dict(configuration=tide.diagnostics, pole={k:v for k, v in pole.items() if k != 'weights'},
                         points=points))
    return dict(parameters=params, comparisons=rows)


def directions(q, fine=False, absorption=False):
    params, pole, ion, cloud_profile = models()
    suffix = '-absorption' if absorption else ''
    history = read(OUT/f'migration-{q:g}-0-0-0{suffix}.json')
    calibration = tilted_cloud(params['alpha'], params['spin'], params['cloud'], size=20, order=48)
    result = axis_response(history, pole, ion, model=cloud_profile, beta=calibration['beta'],
        stride=1 if fine else 2, tolerance=2e-11 if fine else 2e-10,
        max_step=4. if fine else 8., absorptive_response=negative_model() if absorption else None)
    result['calibration'] = inspect_tilt(calibration)
    result['negative_harmonics'] = [-2, -1] if absorption else []
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['grid', 'checks', 'negative', 'negative-grid', 'regularize', 'negative-checks', 'resolution', 'migrate', 'directions', 'assemble'])
    parser.add_argument('--harmonic', type=int, default=-2)
    parser.add_argument('--q', type=float, default=1e-4)
    parser.add_argument('--averaged', action='store_true')
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--absorption', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if args.action == 'grid':
        result, target = grid(), OUT/'grid.json'
    elif args.action == 'negative-grid':
        result, target = negative_grid(args.harmonic), OUT/f'negative-grid-{args.harmonic}.json'
    elif args.action == 'negative-checks':
        result, target = negative_checks(), OUT/'negative-checks.json'
    elif args.action == 'regularize':
        result, target = regularize(), OUT/'ground-reference.json'
    elif args.action == 'resolution':
        result, target = resolution(), OUT/'resolution.json'
    elif args.action in ['checks', 'negative']:
        result = checks() if args.action == 'checks' else negative()
        target = OUT/f'{args.action}.json'
    elif args.action == 'directions':
        result = directions(args.q, args.fine, args.absorption)
        suffix = '-absorption' if args.absorption else ''
        target = OUT/f'directions-{args.q:g}-{int(args.fine)}{suffix}.json'
    elif args.action == 'assemble':
        data = read(OUT/'grid.json')
        data.pop('projections')
        cases = []
        for path in sorted(OUT.glob('migration-*.json')):
            case = read(path)
            case['source_file'] = path.name
            case.setdefault('negative_harmonics', [])
            case.setdefault('static_horizon_fraction', case['horizon_fraction'])
            case.setdefault('negative_horizon_fraction', 0.)
            stride = max(1, (len(case['samples'])-1)//40)
            case['samples'] = case['samples'][::stride]
            cases.append(case)
        result = dict(scope='Frozen cloud selected from a specified formation history, single retarded static pole and canonical forces, Born ionization and Peters radiation. Cases flagged with negative harmonics also contain local first-order s-wave horizon loss. Axis response is a linear posterior diagnostic without feedback.',
            grid=data, checks=read(OUT/'checks.json'), negative=read(OUT/'negative.json'), cases=cases,
            negative_grids=[read(OUT/f'negative-grid-{h}.json') for h in [-2, -1]],
            negative_checks=read(OUT/'negative-checks.json'),
            resolution=read(OUT/'resolution.json'),
            directions=[read(path) for path in sorted(OUT.glob('directions-*.json'))])
        target = OUT/'formed_periapsis.json'
    else:
        params, pole, ion, cloud_profile = models()
        if args.absorption:
            ion = CombinedResponse(ion, negative_model())
        result = coupled_history(ion, pole, model=cloud_profile, q=args.q, a0=3.25, afinal=3.1,
            delta_e=.008, delta_omega=.05, averaged=args.averaged,
            initial='zero' if args.zero else 'steady', tolerance=2e-10 if args.fine else 2e-9,
            max_step=4. if args.fine else 8., sample_count=4001 if args.q >= 1e-4 else 12001,
            include_modes=not args.averaged, **params)
        result['negative_harmonics'] = [-2, -1] if args.absorption else []
        suffix = '-absorption' if args.absorption else ''
        target = OUT/f'migration-{args.q:g}-{int(args.averaged)}-{int(args.zero)}-{int(args.fine)}{suffix}.json'
    write(target, result)
    summary = {key:result[key] for key in ['final_s', 'final_e', 'final_cycles',
        'horizon_fraction', 'total_cloud_loss_fraction', 'e_range', 'omega_range',
        'maximum_energy_balance_error', 'maximum_axial_balance_error',
        'maximum_cloud_tilt', 'maximum_hole_tilt', 'maximum_relative_tilt',
        'maximum_static_to_uniform', 'maximum_transverse_balance_error'] if key in result}
    print(json.dumps(dict(action=args.action, seconds=time.perf_counter()-started,
                         bytes=target.stat().st_size, summary=summary)), flush=True)


if __name__ == '__main__':
    main()
