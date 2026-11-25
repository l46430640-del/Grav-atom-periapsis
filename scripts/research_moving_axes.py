"""Bounded two-way cloud-direction tests on the selected formed background."""

from pathlib import Path
import argparse
import json
import os
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))

from research_formed_periapsis import REFERENCE, models, negative_model, read, write
from grav_atom.research_moving_axes import MovingAxesSystem, moving_history
from grav_atom.research_cloud_shape import QuadrupoleShape, ShapeTide, shape_history
from grav_atom.research_periapsis_migration import equilibrium
import numpy as np

OUT = ROOT/'runs/moving'


def physical_system(q, averaged=False, fixed=False, shape=False):
    params, pole, ion, model = models()
    reference = read(REFERENCE)
    beta = reference['directions'][0]['beta']
    shape_source = None
    if shape:
        mode = QuadrupoleShape(params['cloud'], params['alpha'], params['spin'])
        shape_source = ShapeTide(mode, reference['grid']['axes'])
    return MovingAxesSystem(ion, pole, cloud=params['cloud'], alpha=params['alpha'],
        spin=params['spin'], q=q, model=model, beta=beta, absorption=negative_model(),
        averaged=averaged, free_axes=not fixed, shape=shape_source)


def controls():
    result = []
    for radiation, damping in [(False, 0.), (False, 1.), (True, 1.)]:
        system = physical_system(3e-5)
        system.radiation = radiation
        system.gamma = damping*system.pole.width
        case = moving_history(system, duration=200., max_step=4., tolerance=2e-10)
        result.append(case)
    return dict(cases=result)


def shapes(q, fine=False, zero=False):
    system = physical_system(q)
    shape = QuadrupoleShape(system.cloud, system.alpha, system.spin,
        size=32 if fine else 28, order=64 if fine else 48,
        maximum_l=7 if fine else 5, source_lmax=8 if fine else 6)
    points = []
    for a in [3.25, 3.175, 3.1]:
        point = equilibrium(a, .08715556843599832, system.cloud, system.alpha, system.spin, system.model)
        e, inclination = point['e'], np.arccos(.08715556843599832/point['j'])
        response = shape.static_response(a, e, inclination, np.pi/2)
        points.append(dict(a=a, e=e, inclination=float(inclination), response=response,
            relative_curvature=system.cloud*q*response['omega_curvature']/point['hww']))
    history = read(OUT/f'migration-{q:g}-0-0-1-0.json')
    axes = read(REFERENCE)['grid']['axes']
    evolution = shape_history(shape, history, axes, model=system.model,
        preparation='zero' if zero else 'steady', stride=1 if fine else 2)
    return dict(q=q, fine=fine, mode=shape.diagnostics, points=points, evolution=evolution)


def comparisons(cases):
    result = []
    for q in sorted(set(c['q'] for c in cases)):
        chosen = {}
        for averaged in [False, True]:
            rows = [c for c in cases if c['q'] == q and c['averaged'] == averaged
                    and c['free_axes'] and c['preparation'] == 'steady']
            chosen[averaged] = min(rows, key=lambda c:c['tolerance'])
        coherent, average = chosen[False], chosen[True]
        result.append(dict(q=q, horizon_ratio=coherent['horizon_fraction']/average['horizon_fraction'],
            loss_ratio=coherent['total_loss']/average['total_loss'],
            cycle_change=coherent['final_cycles']/average['final_cycles']-1,
            omitted_alignment_exponent=coherent['alpha']**7/12
                *(1+coherent['cloud']/coherent['alpha']/coherent['spin'])
                *coherent['final_s']/coherent['cloud']))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['migrate', 'controls', 'shapes', 'assemble'])
    parser.add_argument('--q', type=float, default=3e-5)
    parser.add_argument('--averaged', action='store_true')
    parser.add_argument('--fixed', action='store_true')
    parser.add_argument('--fine', action='store_true')
    parser.add_argument('--zero', action='store_true')
    parser.add_argument('--shape', action='store_true')
    args = parser.parse_args()
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    if args.action == 'controls':
        result, target = controls(), OUT/'controls.json'
    elif args.action == 'shapes':
        result = shapes(args.q, args.fine, args.zero)
        target = OUT/f'shapes-{args.q:g}-{int(args.fine)}-{int(args.zero)}.json'
    elif args.action == 'assemble':
        cases = [read(p) for p in sorted(OUT.glob('migration-*.json'))]
        shape_cases = [read(p) for p in sorted(OUT.glob('shape-migration-*.json'))]
        for case in cases+shape_cases:
            case['samples'] = case['samples'][::max(1, (len(case['samples'])-1)//80)]
        result = dict(scope='Two-way cloud and hole directions, parallel-transported cloud frame, causal static mode and local negative-harmonic absorption. Fixed equilibrium radial profile and Born ionization. cases use rigid directions; shapes are forced diagnostics; shape_migrations also couple the leading quadrupolar shape to the orbit.',
            units=dict(time='sigma=cloud*alpha^3*t/M', energy='q*M*alpha^2',
                       angular_momentum='M^2', loss='initial mother-cloud particle number'),
            cases=cases,
            comparisons=comparisons(cases),
            shape_migrations=shape_cases,
            shapes=[read(p) for p in sorted(OUT.glob('shapes-*.json'))],
            controls=read(OUT/'controls.json'))
        target = OUT/'moving_axes.json'
    else:
        system = physical_system(args.q, args.averaged, args.fixed, args.shape)
        result = moving_history(system, tolerance=2e-10 if args.fine else 2e-9,
            max_step=4. if args.fine else 8., budget=280., samples=1201 if args.q >= 1e-4 else 3601,
            preparation='zero' if args.zero else 'steady', retain_samples=True)
        prefix = 'shape-migration' if args.shape else 'migration'
        target = OUT/f'{prefix}-{args.q:g}-{int(args.averaged)}-{int(args.fixed)}-{int(args.fine)}-{int(args.zero)}.json'
    write(target, result)
    summary = {k:result[k] for k in ['final_s', 'final_cycles', 'total_loss', 'horizon_fraction',
        'omega_range', 'maxima', 'maximum_energy_error'] if k in result}
    print(json.dumps(dict(action=args.action, seconds=time.perf_counter()-started,
        bytes=target.stat().st_size, summary=summary)), flush=True)


if __name__ == '__main__':
    main()
