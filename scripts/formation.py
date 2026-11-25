"""Recompute the cloud state selected for the periapsis calculation."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from scipy.integrate import solve_ivp

from grav_atom.research_cloud_evolution import cloud_energy, evolve, instantaneous_rates
from grav_atom.research_cloud_scattering import setup
from grav_atom.research_dressed_competition import compare_channels, DressedModel


def formation_check(nodes, alpha0=.04, target_ratio=4.7, tolerance=2e-11):
    model = DressedModel(nodes)
    history = evolve(alpha0, cloud_model=model, linear_rates=model.spectator_rates)
    last = history['samples'][-1]
    started = time.perf_counter()
    initial = np.r_[np.log([last['mother'], last['daughter']]), last['mass'],
        last['spin']*last['mass']**2, np.zeros(6)]
    scale = abs(last['mother_derivative_M0']/last['mother'])

    def evaluate(state):
        a, b = np.exp(state[:2])
        return instantaneous_rates(alpha0, state[2], state[3], a, b, cloud_model=model)

    def rhs(_, state):
        if time.perf_counter()-started > 240.:
            raise TimeoutError('Formation extension exceeded its bounded run')
        a, b = np.exp(state[:2])
        value = evaluate(state)
        probes = model.spectator_rates(alpha0, state[2], state[3], a, b, value)
        return np.r_[value['da']/a, value['db']/b, value['dm'], value['dj'],
            value['we']*value['escape'], value['wa']*value['annihilation'],
            value['annihilation']/alpha0, value['escape'], np.maximum(probes, 0.)]/scale

    def reached(_, state):
        return np.exp(state[0])/(alpha0**2*state[2]**3)-target_ratio

    reached.terminal, reached.direction = True, -1
    result = solve_ivp(rhs, [0., 1000.], initial, method='Radau', rtol=tolerance,
        atol=tolerance/100, events=reached, dense_output=True, max_step=.1)
    if not result.success or len(result.t_events[0]) != 1:
        raise RuntimeError('Formation continuation did not reach the requested ratio')
    energy0 = cloud_energy(alpha0, initial[2], *np.exp(initial[:2]), model=model)['energy']
    angular0 = initial[3]+(np.exp(initial[0])+2*np.exp(initial[1]))/alpha0
    rows = []
    for t, state in zip(np.linspace(0., result.t[-1], 101),
                        result.sol(np.linspace(0., result.t[-1], 101)).T):
        a, b = np.exp(state[:2])
        value = evaluate(state)
        exponents = np.array(history['linear_positive_exponents'])+state[8:]
        rows.append(dict(time_M0=float(last['time_M0']+t/scale), mother=float(a),
            daughter=float(b), mass=float(state[2]), alpha=float(alpha0*state[2]),
            spin=float(value['spin']), cloud=float(a/state[2]),
            ratio=float(a/(alpha0**2*state[2]**3)),
            local_depletion_time_M0=float(a/abs(value['da'])),
            positive_probe_exponents=list(exponents),
            log_probe_upper_over_mother=list(history['seed_log']+exponents-np.log(a)),
            energy_balance_error=float(value['energy']+state[4]+state[5]-energy0),
            axial_balance_error=float(state[3]+(a+2*b+3*state[7])/alpha0+state[6]-angular0)))
    return dict(scope='Unperturbed two-cloud formation continuation. Positive probe envelopes omit stabilizing escape for n=5..8 and are sufficient bounds only.',
        alpha0=alpha0, target_ratio=target_ratio, tolerance=tolerance,
        seed_log=history['seed_log'], start=last, final=rows[-1],
        maximum_energy_balance_error=max(abs(r['energy_balance_error']) for r in rows),
        maximum_axial_balance_error=max(abs(r['axial_balance_error']) for r in rows),
        elapsed_seconds=time.perf_counter()-started, samples=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recompute-rates', action='store_true',
                        help='solve the radial model instead of loading the saved rates')
    args = parser.parse_args()
    nodes = json.loads((ROOT/'data/formation_rates.json').read_bytes())
    if args.recompute_rates:
        config = setup(36, 64, 5)
        nodes = [compare_channels(config, node['fraction'], output_ls=(3, 5, 7, 9),
                                  closed_ls=(1, 3, 5, 7)) for node in nodes]
    result = formation_check(nodes)
    out = ROOT/'runs/formation.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result['final'], indent=2))
