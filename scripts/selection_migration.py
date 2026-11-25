"""Extend the archived eccentricity grid and follow off-center migrating orbits."""
import argparse
import gzip
import json
import os
from pathlib import Path

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

from research_formed_periapsis import (ROOT, REFERENCE, CombinedResponse, NegativeTideTable,
    PoleTide, ResponseTable, StaticCloudTide, eccentric_projection, profile, read)
from grav_atom.research_moving_axes import MovingAxesSystem, moving_history


EXTRA_E = np.array([.001, .035, .075, .12, .17, .23, .29, .35, .39, .66, .70, .74, .78])
DATA = ROOT/'data'
RUNS = ROOT/'runs'/'selection'
GRID = DATA/'selection_grid.npz'
MONO = DATA/'selection_monopole.npz'
TRACE_COLUMNS = ('s','a','e','jz','omega','node','horizon_fraction','ionization_fraction',
                 'negative_horizon_fraction','cycles','cloud_tilt','hole_tilt','excitation','shape_excitation')


def vertex(projection):
    return np.array([c['amplitudes'] for h in projection['harmonics'] for c in h['channels']])


def labels(projection):
    return [(h['harmonic'], c['ell'], c['final_m'])
            for h in projection['harmonics'] for c in h['channels']]


def build_grid():
    ref = read(REFERENCE)['grid']
    p = ref['parameters']
    args = (p['cloud'], p['alpha'], p['spin'])
    full, dip = StaticCloudTide(*args), StaticCloudTide(*args, source_lmax=1)
    weights = full.slow_pole()['weights']
    runs = RUNS/'grid-v1'
    runs.mkdir(exist_ok=True, parents=True)
    with gzip.open(DATA/'ionization_projections.json.gz', 'rt') as f:
        channels = labels(json.load(f)[0][0])
    rows = []
    for i, a in enumerate(ref['axes'][0]):
        row = []
        for j, e in enumerate(EXTRA_E):
            path = runs/f'{i}-{j}.npz'
            if not path.exists():
                proj = eccentric_projection(a, e, maximum_harmonic=16, ell_max=8,
                                           order=24, anomaly_order=32, cutoff=80.)
                if labels(proj) != channels:
                    raise ValueError('A continuum threshold or channel ordering changed')
                negative = []
                for h in [-2, -1]:
                    amplitudes = []
                    for inc, sign in [(0., 1.), (np.pi, -1.)]:
                        val = dip.negative_harmonic(a, e, inc, 0., h)
                        amplitudes.append(sign*np.sqrt(dip.sink)*np.array(val['closed_origin_amplitude']))
                    negative.append(amplitudes)
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, a=a, e=e, ion=vertex(proj), neg=negative,
                                    pole=full.coefficients(a, e, weights=weights))
                tmp.replace(path)
            with np.load(path) as v:
                if a != v['a'] or e != v['e']:
                    raise ValueError('Stale cached node')
                row.append({k:v[k] for k in ('ion','neg','pole')})
        rows.append(row)
        print(f'Completed response row A={a:g}', flush=True)
    tmp = GRID.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, a=ref['axes'][0], e=EXTRA_E, channels=channels,
                        **{k:np.array([[n[k] for n in r] for r in rows]) for k in rows[0][0]})
    tmp.replace(GRID)


def build_monopole():
    from grav_atom.research_monopole_continuum import MonopoleContinuum
    ref = read(REFERENCE)['grid']
    model = profile(ref['parameters'])
    waves = MonopoleContinuum(model)
    axes = [ref['axes'][0], np.sort(np.r_[ref['axes'][1], EXTRA_E])]
    values = []
    for a in axes[0]:
        row = []
        for e in axes[1]:
            proj = eccentric_projection(a, e, maximum_harmonic=16, ell_max=8, order=24,
                anomaly_order=32, cutoff=80., binding_energy=model.diagnostics['chemical'],
                parent_radial=model.dipole_radial, continuum_waves=waves.waves)
            row.append(vertex(proj))
        values.append(row)
        print(f'Completed spherical continuum row A={a:g}', flush=True)
    tmp = MONO.with_suffix('.tmp.npz')
    np.savez_compressed(tmp, a=axes[0], e=axes[1], channels=labels(proj), ion=values,
                        binding=model.diagnostics['chemical'], match=waves.match)
    tmp.replace(MONO)


def system(q, averaged, *, shape=False, monopole=False):
    from grav_atom.research_cloud_shape import QuadrupoleShape, ShapeTide
    ref = read(REFERENCE)
    grid = ref['grid']
    p = grid['parameters']
    with gzip.open(DATA/'ionization_projections.json.gz', 'rt') as f:
        old = json.load(f)
    with np.load(GRID) as ext:
        axes = [np.array(grid['axes'][0]), np.r_[grid['axes'][1], ext['e']]]
        if not np.array_equal(axes[0], ext['a']) or not np.array_equal(labels(old[0][0]), ext['channels']):
            raise ValueError('Extended and archived grids differ')
        perm = np.argsort(axes[1])
        axes[1] = axes[1][perm]
        def join(original, extra):
            return np.concatenate([original, extra], axis=1)[:,perm]
        vertices = join([[vertex(v) for v in line] for line in old], ext['ion'])
        ion = ResponseTable.from_arrays(axes, ext['channels'], vertices)
        if monopole:
            with np.load(MONO) as mod:
                if not all(np.array_equal(x, mod[k]) for x,k in
                           [(axes[0],'a'),(axes[1],'e'),(ext['channels'],'channels')]):
                    raise ValueError('Spherical continuum has incompatible axes')
                ion = ResponseTable.from_arrays(axes, mod['channels'], mod['ion'])
        pole = PoleTide(axes, join(grid['values'], ext['pole']), grid['labels'],
            grid['pole']['frequency'], grid['pole']['population_width'])
        ng = {g['harmonic']:g for g in ref['negative_grids']}
        tables = [NegativeTideTable(axes, join(ng[h]['values'], ext['neg'][:,:,i]), h,
                  reference_frequency=ng[h]['reference_frequency']) for i,h in enumerate([-2,-1])]
    sh = ShapeTide(QuadrupoleShape(p['cloud'], p['alpha'], p['spin']), axes) if shape else None
    return MovingAxesSystem(ion, pole, cloud=p['cloud'], alpha=p['alpha'], spin=p['spin'], q=q,
        model=profile(p), beta=ref['directions'][0]['beta'], absorption=CombinedResponse(*tables),
        averaged=averaged, shape=sh)


def run_name(angle,q,averaged,fine,shape,monopole=False,reference=False,initial_e=None,zero=False):
    entry = 'reference' if reference else f'{angle:g}'
    if initial_e is not None:
        entry += f'-e{initial_e:g}'
    return (f'migration-{entry}-{q:g}-{"D" if averaged else "C"}'
            + ('-fine' if fine else '') + ('-shape' if shape else '') + ('-monopole' if monopole else '')
            + ('-zero' if zero else ''))


def migrate(angle, q, averaged, fine, shape, monopole=False, reference=False, initial_e=None, trace=False, zero=False):
    s = system(q, averaged, shape=shape, monopole=monopole)
    de, dw = (.008,.05) if reference else (0.,np.radians(angle))
    if initial_e is not None:
        if reference:
            raise ValueError('Choose a reference orbit or an absolute initial eccentricity')
        from grav_atom.research_periapsis_migration import equilibrium
        center = equilibrium(3.25, read(REFERENCE)['grid']['parameters']['jz0'],
                             s.cloud,s.alpha,s.spin,s.model)
        de = initial_e-center['e']
    r = moving_history(s, delta_e=de, delta_omega=dw,
                       preparation='zero' if zero else 'steady',
                       tolerance=2e-10 if fine else 2e-9, max_step=4. if fine else 8.,
                       budget=450., samples=2001, retain_samples=True)
    r.update(initial_offset_deg=np.degrees(dw), initial_delta_e=de, monopole=monopole)
    RUNS.mkdir(exist_ok=True, parents=True)
    filename = run_name(angle,q,averaged,fine,shape,monopole,reference,initial_e,zero)
    path = RUNS/(filename+'.json')
    rows = r['samples']
    if trace:
        np.savez_compressed(RUNS/(filename+'.npz'),columns=TRACE_COLUMNS,
                            samples=[[v[k] for k in TRACE_COLUMNS] for v in rows])
    r['radial_bins'] = []
    for lo,hi in zip(np.linspace(3.1,3.25,6)[:-1],np.linspace(3.1,3.25,6)[1:]):
        subset = [v for v in rows if lo-1e-10 <= v['a'] <= hi+1e-10]
        if len(subset) < 2:
            raise ArithmeticError('Insufficient trajectory samples in radial bin')
        omega = np.array([v['omega'] for v in subset])
        r['radial_bins'].append(dict(a=[lo,hi], e=[min(v['e'] for v in subset),max(v['e'] for v in subset)],
            omega=[min(omega),max(omega)], net_apsidal_turns=(omega[-1]-omega[0])/(2*np.pi)))
    r['samples'] = rows[::25]
    path.write_text(json.dumps(r, allow_nan=False)+'\n', encoding='utf-8')
    print(filename, r['final_cycles'], r['horizon_fraction'], r['omega_range'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['grid','monopole-grid','migrate'])
    parser.add_argument('--angles', type=float, nargs='+', default=[20., 47.5, 60.])
    parser.add_argument('--q', type=float, default=1e-4)
    parser.add_argument('--models', nargs='+', choices=['C','D'], default=['C','D'])
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--shape', action='store_true')
    parser.add_argument('--monopole', action='store_true')
    parser.add_argument('--reference', action='store_true', help='Use the manuscript offsets')
    parser.add_argument('--initial-e', type=float, help='Absolute eccentricity, instead of the center value')
    parser.add_argument('--trace', action='store_true', help='Save all 2001 diagnostic rows for this history')
    parser.add_argument('--zero', action='store_true', help='Set the initial slow amplitudes to zero')
    args = parser.parse_args()
    if args.action == 'grid':
        build_grid()
    elif args.action == 'monopole-grid':
        build_monopole()
    else:
        for angle in ([0.] if args.reference else args.angles):
            for mode in args.models:
                migrate(angle, args.q, mode=='D', args.fine, args.shape, args.monopole, args.reference,
                        args.initial_e,args.trace,args.zero)
