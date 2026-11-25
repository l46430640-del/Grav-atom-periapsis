"""Assemble paired histories; --rerun-missing reproduces the manifest sequentially."""
import argparse
import numpy as np
from scipy.integrate import trapezoid

from research_formed_periapsis import read, write
from selection_migration import DATA, RUNS, TRACE_COLUMNS, migrate, run_name
from selection_open_horizon import vertex_spline


def manifest():
    def item(angle, **options):
        return dict(angle=angle, **(dict(q=1e-4,averaged=False,fine=False,shape=False,
                    monopole=False,reference=False,initial_e=None,trace=False)|options))
    for av in (False,True):
        yield item(0.,reference=True,averaged=av)
        yield item(0.,reference=True,averaged=av,monopole=True)
        yield item(0.,reference=True,averaged=av,shape=True,trace=True)
        for angle in (3.,20.,47.5,60.):
            yield item(angle,averaged=av)
            if angle!=3.:
                yield item(angle,averaged=av,monopole=True)
        yield item(47.5,averaged=av,fine=True,trace=not av)
        yield item(47.5,averaged=av,shape=True)
        yield item(47.5,q=3e-5,averaged=av,trace=True)
        yield item(-45.,initial_e=.02,averaged=av,trace=not av)
    for angle in (20.,60.):
        yield item(angle,fine=True,trace=True)
    for angle in (-90.,0.,45.):
        yield item(angle,initial_e=.02,trace=True)
    for angle in (3.,60.):
        yield item(angle,zero=True,trace=True)
        yield item(angle,zero=True,monopole=True,trace=True)


def trace_open_counts(arr, idx, spline, q, eps, averaged):
    t,a,e,z,w = (arr[:,idx[k]] for k in ['s','a','e','jz','omega'])
    ci = z/np.sqrt(1-e*e)
    parent = np.array([-(1-ci)*np.exp(-1j*w)/2,1j*np.sqrt((1-ci*ci)/2),
                       (1+ci)*np.exp(1j*w)/2]).T
    vertices = spline(np.column_stack([a,e]))
    amp = vertices*parent[:,None,:]
    count = np.sum(abs(amp)**2,axis=(1,2)) if averaged else np.sum(abs(amp.sum(axis=2))**2,axis=1)
    integrals = [q*q/eps*trapezoid(count[::n],x=t[::n]) for n in (1,2)]
    return dict(increment=integrals[0],half_sampling_difference=abs(integrals[1]-integrals[0]))


def assemble(rerun=False):
    with np.load(DATA/'selection_open_vertices.npz') as saved:
        spline = vertex_spline(saved['a'],saved['e'],saved['vertices'])
    upper = read(DATA/'selection_open_horizon.json')['h1_h16_norm2_bound']
    idx = {k:i for i,k in enumerate(TRACE_COLUMNS)}
    all_rows, traces, pairs = {}, {}, []
    for job in manifest():
        name = run_name(**{k:v for k,v in job.items() if k!='trace'})
        path = RUNS/(name+'.json')
        trace_path = RUNS/(name+'.npz')
        if rerun and (not path.exists() or (job['trace'] and not trace_path.exists())):
            migrate(**job)
        row = read(path)
        samples = row.pop('samples')
        if abs(samples[0]['a']-3.25)>1e-12 or abs(samples[-1]['a']-3.1)>1e-12:
            raise ValueError('Migration did not span the required interval')
        row['initial_e'] = samples[0]['e']
        if row['e_range'][0]<.001 or row['e_range'][1]>.78:
            raise ValueError('Migration left the computed eccentricity grid')
        if max(row['maximum_energy_error'],row['maxima']['angular_error'])>1e-9:
            raise ArithmeticError('Integrated balance needs review')
        row.pop('elapsed_seconds')
        bound_key = 'bare_continuum_open_H_upper' if job['monopole'] else 'open_H_upper'
        row[bound_key] = row['q']**2/row['cloud']*row['final_s']*upper
        if job['trace']:
            with np.load(trace_path) as src:
                if list(src['columns'])!=list(TRACE_COLUMNS):
                    raise ValueError('Wrong diagnostic columns')
                arr = src['samples']
            if not np.isclose(arr[-1,idx['horizon_fraction']],row['horizon_fraction'],rtol=1e-11,atol=0):
                raise ValueError('Trace and history differ')
            traces[name] = arr
            # The open-horizon vertices were calculated for the Coulomb continuum.
            if not job['monopole']:
                row['postprocessed_open_H'] = trace_open_counts(arr,idx,spline,row['q'],row['cloud'],row['averaged'])
                if row['postprocessed_open_H']['increment']>row['open_H_upper']:
                    raise ArithmeticError('Positive-harmonic flux exceeds the interpolated envelope')
        all_rows[name] = row
    for job in manifest():
        if not job['averaged']:
            continue
        spec = {k:v for k,v in job.items() if k!='trace'}
        dname = run_name(**spec)
        cname = run_name(**(spec|dict(averaged=False)))
        c,d = all_rows[cname],all_rows[dname]
        pairs.append(dict(C=cname,D=dname,tidal_horizon_ratio=c['horizon_fraction']/d['horizon_fraction'],
            total_loss_ratio=c['total_loss']/d['total_loss'],cycles_change=c['final_cycles']/d['final_cycles']-1,
            cycles_difference=c['final_cycles']-d['final_cycles']))
    write(DATA/'selection_migrations.json',dict(
        scope='A=3.25 to 3.1; fractions relative to initial parent number, time s=epsilon*alpha^3*t/M. Positive harmonics and intrinsic formation counts are not fed back. Each radial bin reports sampled extrema and net angular advance.',
        pairs=pairs,runs=all_rows,trace_columns=TRACE_COLUMNS))
    np.savez_compressed(DATA/'selection_traces.npz',columns=TRACE_COLUMNS,**traces)
    print(f'{len(all_rows)} histories, {len(pairs)} pairs and {len(traces)} dense traces.')


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rerun-missing',action='store_true')
    assemble(parser.parse_args().rerun_missing)
