"""Repeat the conditional single-mode test at earlier times in the same formation model."""
import numpy as np

from formation import formation_check
from research_formed_periapsis import ROOT, profile, read, write
from selection_orbits import OrbitFamily, StaticCloudTide


def main():
    nodes = read(ROOT/'data/formation_rates.json')
    z = read(ROOT/'data/formed_periapsis.json')['grid']['parameters']['jz0']
    rows = []
    for ratio in [5.6,5.1]:
        stage = formation_check(nodes,target_ratio=ratio)['final']
        p = {k:stage[k] for k in ('cloud','alpha','spin')}
        tide = StaticCloudTide(p['cloud'],p['alpha'],p['spin'])
        pole = tide.slow_pole()
        family = OrbitFamily(3.25,z,p,profile(p),tide,pole)
        comparison = family.measure(.05,.008,verify=True)
        rows.append(dict(ratio=ratio,time_M0=stage['time_M0'],parameters=p,
            daughter_parent_ratio=stage['daughter']/stage['mother'],a=family.a,z=z,
            center_e=family.e0,width_deg_at_center_e=np.degrees(family.width),
            pole_frequency=pole['frequency'],pole_width=pole['population_width'],
            reference_offsets=comparison, moderate_offset=family.measure(np.radians(20.),verify=True)))
    write(ROOT/'data/selection_formation.json',dict(scope='Same isolated two-cloud history at two earlier stages. Finite stationary profile and pole recomputed at each stage. Fixed-orbit comparisons, not additional migrations.',stages=rows))


if __name__=='__main__':
    main()
