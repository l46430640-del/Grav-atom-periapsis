"""Bound the integrated, omitted positive-frequency s-wave horizon term on the response grid."""
from types import SimpleNamespace

import numpy as np
from scipy.interpolate import NdBSpline, make_interp_spline

from selection_migration import DATA, EXTRA_E
from research_formed_periapsis import REFERENCE, read, write
from grav_atom.research_cloud_scattering import coulomb_waves
from grav_atom.research_periapsis_migration import dipole_s_sources
from grav_atom.research_tidal_continuum import source_grid


def vertices(a,es,hs,sink):
    grids = [source_grid(a,32,90.,breakpoints=(a*(1-e),a*(1+e))) for e in es]
    # Only the evaluation radii change with eccentricity; integrate each wave once.
    combined = SimpleNamespace(r=np.concatenate([g.r for g in grids]),half=grids[0].half)
    split = np.cumsum([len(g.r) for g in grids[:-1]])
    sources = [dipole_s_sources(a,e,g.r,hs,64) for e,g in zip(es,grids)]
    matrix = np.zeros((len(es),len(hs),3),complex)
    for hi,h in enumerate(hs):
        energy = -.125+h*a**(-1.5)
        k = np.sqrt(2*energy)
        out = coulomb_waves(combined,[0],energy,outgoing=True)[0]['outgoing']
        c0 = np.sqrt((2*np.pi/k)/(-np.expm1(-2*np.pi/k)))
        for i,(s,g,o) in enumerate(zip(sources,grids,np.split(out,split))):
            matrix[i,hi] = -2*np.sqrt(sink)*c0*((g.w*g.r*o)@s[hi])
    return matrix


def norm2(m):
    return np.linalg.svd(m,compute_uv=False)[...,0]**2


def vertex_spline(ax,es,vals):
    sa = make_interp_spline(ax,vals)
    se = make_interp_spline(es,sa.c,axis=1)
    return NdBSpline((sa.t,se.t),np.swapaxes(se.c,0,1),(3,3),extrapolate=False)


def main():
    grid = read(REFERENCE)['grid']
    p = grid['parameters']
    sink = 2*(1+np.sqrt(1-p['spin']**2))*p['alpha']**3
    ax = grid['axes'][0]
    es = np.sort(np.r_[grid['axes'][1],EXTRA_E])
    hs = np.arange(1,17)
    vals = np.array([vertices(a,es,hs,sink) for a in ax])
    np.savez_compressed(DATA/'selection_open_vertices.npz',a=ax,e=es,h=hs,vertices=vals)
    spline = vertex_spline(ax,es,vals)
    # The tensor B-spline basis is nonnegative and sums to one. The spectral
    # norm is convex, giving a bound for every normalized parent orientation.
    bound = float(np.max(norm2(spline.c)))
    tail = float(np.max(norm2(spline.c[:,:,8:])))
    checks = []
    for a,e in [(3.231,.105),(3.175,.552),(3.119,.712)]:
        direct = vertices(a,[e],np.arange(1,25),sink)[0]
        tab = spline([a,e])
        error = np.linalg.norm(tab-direct[:16])/np.linalg.norm(direct[:16])
        if error>.01:
            raise ArithmeticError(f'Open-horizon interpolation differs by {error}')
        checks.append(dict(a=a,e=e,vertex_relative_error=error,
                           direct_tail_h17_h24_norm2=norm2(direct[16:])))
    write(DATA/'selection_open_horizon.json',dict(
        scope='Bare Coulomb outgoing s wave with leading horizon matching. A bound on h=1..16 within the interpolated grid, not on all physical horizon channels.',
        a_range=[min(ax),max(ax)],e_range=[min(es),max(es)],h1_h16_norm2_bound=bound,
        h9_h16_norm2_bound=tail,point_checks=checks))


if __name__=='__main__':
    main()
