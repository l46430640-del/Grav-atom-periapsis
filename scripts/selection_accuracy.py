"""Off-grid checks of angular transport at eccentricities visited in migration."""
import numpy as np

from selection_migration import DATA, system
from research_formed_periapsis import REFERENCE, StaticCloudTide, read, write
from grav_atom.research_inclined_ionization import (coherent_flux, eccentric_projection,
    rotated_amplitudes, torque_coefficients)
from grav_atom.research_cloud_scattering import coulomb_waves
from grav_atom.research_periapsis_migration import dipole_s_sources
from grav_atom.research_tidal_continuum import source_grid


def open_s_wave(a,e,inc,w,tide):
    """First-order horizon matching of the open Coulomb s wave for h=1,...,4."""
    grid = source_grid(a,32,90.,breakpoints=(a*(1-e),a*(1+e)))
    hs = range(1,5)
    vertices = dipole_s_sources(a,e,grid.r,list(hs),64)
    c = rotated_amplitudes(inc,w)
    out = []
    for h,v in zip(hs,vertices):
        energy = -.125+h*a**(-1.5)
        k = np.sqrt(2*energy)
        wave = coulomb_waves(grid,[0],energy,outgoing=True)[0]
        c0 = np.sqrt((2*np.pi/k)/(-np.expm1(-2*np.pi/k)))
        origin = -2*c0*((grid.w*grid.r*wave['outgoing'])@v)
        out.append(tide.sink*abs(origin@c)**2)
    return out


def main():
    s = system(1e-4,False)
    m = system(1e-4,False,monopole=True)
    tide = StaticCloudTide(s.cloud,s.alpha,s.spin)
    z = read(REFERENCE)['grid']['parameters']['jz0']
    points = []
    for a,e in [(3.231,.105),(3.175,.552),(3.119,.712)]:
        inc = np.arccos(z/np.sqrt(1-e*e))
        direct = eccentric_projection(a,e,maximum_harmonic=32,ell_max=12,order=40,
                                      anomaly_order=64,cutoff=100.,tolerance=2e-12)
        for w in (0., np.pi/2):
            ion = s.ion.rates(a,e,inc,w)
            diag = s.ion.rates(a,e,inc,w,True)
            mod = m.ion.rates(a,e,inc,w)
            mod_d = m.ion.rates(a,e,inc,w,True)
            fine = coherent_flux(direct,inc,w)
            tor = torque_coefficients(direct,inc,w)
            nh = s.absorption.rates(a,e,inc,w)['horizon_number']
            closed = [tide.negative_harmonic(a,e,inc,w,h)['horizon_number_first_order']
                      for h in range(-6,0)]
            positive = open_s_wave(a,e,inc,w,tide)
            pts = dict(a=a,e=e,omega=w,inclination=inc,table_CE=ion['energy'],table_CE_D=diag['energy'],
                refined_CE=fine['energy_coefficient'],refined_CE_D=fine['averaged_energy_coefficient'],
                refined_CN=fine['population_coefficient'],table_CN=ion['number'],
                normal_torque_table=ion['normal_torque'],normal_torque_refined=tor['total'][2],
                axial_torque_table=ion['axial_torque'],axial_torque_refined=tor['cloud_axis_projection'],
                monopole_CE=mod['energy'],monopole_CE_D=mod_d['energy'],
                retained_negative_table=nh,negative_h_minus6_to_minus1=closed,positive_s_h1_to_h4=positive,
                full_static_h0=tide.static_response(a,e,inc,w)['number'])
            points.append(pts)
        print(f'Checked independent fluxes at A={a}, e={e}',flush=True)
    write(DATA/'selection_accuracy.json',dict(scope='Local, not uniform truncation estimates. Open positive horizon harmonics use bare Coulomb waves; static h=0 is not a causal history.',points=points))


if __name__=='__main__':
    main()
