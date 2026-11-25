"""Measure causal horizon selection along closed secular orbits at fixed A and z."""
import numpy as np
from scipy.integrate import solve_ivp, trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import brentq

from research_formed_periapsis import ROOT, REFERENCE, profile, read, write
from grav_atom.research_causal_periapsis import conservative_gradient
from grav_atom.research_cloud_evolution import instantaneous_rates
from grav_atom.research_cloud_secular import derivatives, hamiltonian, linear_coefficients
from grav_atom.research_dressed_competition import DressedModel
from grav_atom.research_periapsis_migration import equilibrium
from grav_atom.research_static_tide import StaticCloudTide, angular_factors


W0 = np.pi/2


def periodic_response(forcing, period, nodal_rate, frequency, width):
    """Power ratio and sideband weights of a damped pole. The common vertex scale cancels."""
    freqs = nodal_rate+2*np.pi*np.fft.fftfreq(len(forcing), period/len(forcing))
    b = np.fft.fft(forcing, axis=0)/(frequency-freqs+.5j*width)[:,None]
    coherent = abs(b.sum(axis=1))**2
    diagonal = np.sum(abs(b)**2, axis=1)
    return coherent.sum()/diagonal.sum(), coherent/coherent.sum(), diagonal/diagonal.sum()


def time_domain(forcing, period, nodal_rate, frequency, width):
    x = np.linspace(0., 1., len(forcing)+1)
    f = CubicSpline(x,np.vstack([forcing,forcing[0]]),bc_type='periodic')
    exponent = period*(1j*(frequency-nodal_rate)-width/2)
    def rhs(t,b):
        return exponent*b-1j*f(t)
    kw = dict(method='DOP853',rtol=1e-10,atol=1e-12)
    first = solve_ivp(rhs,[0.,1.],np.zeros(forcing.shape[1],complex),**kw)
    b0 = -first.y[:,-1]/np.expm1(exponent)
    check = solve_ivp(rhs,[0.,1.],b0,t_eval=x,**kw)
    if not first.success or not check.success:
        raise ArithmeticError('Periodic forced solution failed')
    b = check.y[:,:-1]
    return np.sum(abs(b.sum(axis=0))**2)/np.sum(abs(b)**2)


def section(t,y):
    return np.sin(y[1]-W0)


section.direction = 1
section.terminal = 2


class OrbitFamily:
    def __init__(self,a,z,p,model,tide,pole):
        self.a, self.z, self.model, self.tide, self.pole = a,z,model,tide,pole
        self.params = (p['cloud'],p['alpha'],p['spin'])
        pt = equilibrium(a,z,*self.params,model)
        self.e0 = pt['e']
        self.nu = np.sqrt(pt['hjj']*pt['hww']/a)
        if not pt['stable']:
            raise ValueError('No elliptic center at this point')
        self.H = lambda e,w: hamiltonian(a,e,w,z,*self.params,model=model)
        hj = lambda e: derivatives(a,e,0.,z,*self.params, model=model)[0]
        self.saddle = brentq(hj,.0001,self.e0) if hj(.0001)*hj(self.e0)<0 else 0.
        self.hs, self.hc = self.H(self.saddle,0.),self.H(self.e0,W0)
        self.width = W0-brentq(lambda w:self.H(self.e0,w)-self.hs,0.,W0)
        self.knots = np.linspace(.001,.789,81)
        values = [tide.coefficients(a,e,weights=pole['weights']) for e in self.knots]
        self.radial = CubicSpline(self.knots,values,extrapolate=False)
        self.coarse = CubicSpline(self.knots[::2],values[::2],extrapolate=False)
        mags = np.array([m for _,m in tide.labels])
        self.groups = [mags==m for m in np.unique(mags)]

    def separatrix_action(self,order):
        """Positive enclosed Delaunay area / (2*pi) for the island around +pi/2."""
        def axis(e):
            return self.H(e,W0)-self.hs
        emin = brentq(axis,0.,self.e0) if self.saddle else 0.
        emax = brentq(axis,self.e0,min(.98,np.sqrt(1-self.z*self.z)-1e-5))
        cuts = sorted({emin,emax,*([self.saddle] if emin<self.saddle<emax else [])})
        x,g = np.polynomial.legendre.leggauss(order)
        area = 0.
        for l,r in zip(cuts,cuts[1:]):
            # The cosine substitution regularizes the tips of the contour.
            t = np.pi*(x+1)/2
            ecc = l+(r-l)*(1-np.cos(t))/2
            for e,weight in zip(ecc,g*np.pi*(r-l)*np.sin(t)/4):
                boundary = brentq(lambda w:self.H(e,w)-self.hs,0.,W0)
                area += weight*(W0-boundary)*e/np.sqrt(1-e*e)
        return np.sqrt(self.a)*area/np.pi

    def rhs(self,t,y):
        j,w,node = y
        hj,hz,hw = conservative_gradient(self.a,np.sqrt(1-j*j),w,self.z,*self.params,self.model)[1:4]
        return np.array([-hw,hj,hz])/(np.sqrt(self.a)*self.nu)

    def measure(self,angle,de=0.,*,verify=False):
        e = self.e0+de
        sol = solve_ivp(self.rhs,[0.,150.],[np.sqrt(1-e*e),W0+angle,0.],events=section,
                        method='DOP853',rtol=3e-10,atol=2e-12,max_step=.22,dense_output=True)
        if not sol.success or len(sol.t_events[0])!=2:
            raise ArithmeticError('Did not resolve two crossings of the same orientation')
        t1,t2 = sol.t_events[0]
        t = np.linspace(t1,t2,513)
        j,w,node = sol.sol(t)
        if abs(j[-1]-j[0])>2e-8 or abs(np.exp(1j*w[-1])-np.exp(1j*w[0]))>2e-8:
            raise ArithmeticError('The apsidal orbit did not close')
        ecc = np.sqrt(1-j*j)
        if min(ecc)<self.knots[0] or max(ecc)>self.knots[-1]:
            raise ValueError('Periodic orbit outside radial overlap grid')
        period = (t2-t1)/self.nu
        nodal_rate = (node[-1]-node[0])/period
        phase = np.exp(1j*(node-np.linspace(node[0],node[-1],len(node))))
        rot = np.array([angular_factors(self.tide.labels,np.arccos(self.z/g),v)[0] for g,v in zip(j,w)])
        def source(spline):
            amp = (rot*spline(ecc)*phase[:,None])[:-1]
            return np.column_stack([amp[:,ix].sum(axis=1) for ix in self.groups])
        f = source(self.radial)
        args = (period,nodal_rate,self.pole['frequency'],self.pole['population_width'])
        r,c,d = periodic_response(f,*args)
        err = max(abs(periodic_response(f[::2],*args)[0]-r),
                  abs(periodic_response(source(self.coarse),*args)[0]-r))
        if err>1e-4:
            raise ArithmeticError(f'Refine radial or temporal sampling: {err}')
        winding = (w[-1]-w[0])/(2*np.pi)
        lib = abs(winding)<.5
        row = dict(initial_angle_deg=np.degrees(angle),delta_e=de, e_range=[min(ecc),max(ecc)],
            energy_depth=(self.hc-self.H(e,W0+angle))/(self.hc-self.hs),
            winding=winding, period=period, nodal_rate=nodal_rate, libration=bool(lib),
            libration_action=abs(trapezoid(np.sqrt(self.a)*(j-j[0]),x=w))/(2*np.pi) if lib else None,
            peak_offset_deg=np.degrees(max(abs(w-W0))) if lib else None,
            slow_ratio=r, unfiltered_ratio=np.sum(abs(f.sum(axis=1))**2)/np.sum(abs(f)**2),
            carrier_fraction_coherent=c[0], carrier_fraction_diagonal=d[0], refinement=err)
        if verify:
            row['time_domain_difference'] = abs(time_domain(f,*args)-r)
            if row['time_domain_difference']>1e-7:
                raise ArithmeticError('Fourier and time-domain comparisons disagree')
        return row


def intrinsic(formed):
    f = formed['grid']['formation']
    m, n = f['mass'],f['mother']
    rates = instantaneous_rates(.04,m,f['spin']*m*m,n,f['daughter'],
                    cloud_model=DressedModel(read(ROOT/'data/formation_rates.json')))
    cases = read(ROOT/'data/moving_axes.json')['cases']
    out = []
    for q in sorted({r['q'] for r in cases}):
        counts = []
        for avg in [False,True]:
            row = min((r for r in cases if r['q']==q and r['averaged']==avg),key=lambda r:r['tolerance'])
            time = row['final_s']*m/(f['cloud']*f['alpha']**3)
            pair = time*rates['absorption']/n
            ordinary = -time*(rates['sa']+rates['sb'])/n
            counts.append(dict(averaged=avg,time_M0=time,tidal_H=row['horizon_fraction'],
                pair_H=pair,ordinary_net_H=ordinary,net_H=row['horizon_fraction']+pair+ordinary,
                intrinsic_parent_depletion=-time*rates['da']/n))
        c,d = counts
        out.append(dict(q=q,counts=counts,pair_added_ratio=(c['tidal_H']+c['pair_H'])/(d['tidal_H']+d['pair_H']),
                        net_added_ratio=c['net_H']/d['net_H']))
    return out


def main():
    ref = read(REFERENCE)
    p = ref['grid']['parameters']
    model = profile(p)
    tide = StaticCloudTide(p['cloud'],p['alpha'],p['spin'])
    pole = tide.slow_pole()
    z = p['jz0']
    out = dict(scope='Fixed A,z, stationary spin and cloud axis. Periodic causal h=0 mode only.',
               intrinsic_counts=intrinsic(ref),families=[])
    def cx(a):
        return linear_coefficients(a,np.arccos(z),p['cloud'],p['alpha'],p['spin'],step=.003,model=model)[0]
    out['circular_bifurcation_A_at_reference_z'] = brentq(cx,3.175,3.25)
    for a, zz in [(3.25,z),(3.175,z),(3.1,z),(3.25,.045),(3.25,.14)]:
        family = OrbitFamily(a,zz,p,model,tide,pole)
        # Angles are taken on e=e_*; do not integrate on the separatrix itself.
        angles = np.degrees(family.width)*np.array([.08,.18,.30,.42,.54,.66,.78,.86,.91,.95,.98,.99])
        if a==3.25 and zz==z:
            angles = np.unique(np.r_[angles,20.,40.,47.5,60.])
        rows = [family.measure(np.radians(v),verify=i%4==0) for i,v in enumerate(angles)]
        changes = []
        for l,r in zip(rows,rows[1:]):
            if l['libration'] and r['libration'] and (l['slow_ratio']-1)*(r['slow_ratio']-1)<0:
                changes.append(brentq(lambda deg:family.measure(np.radians(deg))['slow_ratio']-1,
                               l['initial_angle_deg'],r['initial_angle_deg'],xtol=.002))
        area = family.separatrix_action(24)
        area_error = abs(area-family.separatrix_action(16))
        if area_error>3e-5:
            raise ArithmeticError('Separatrix area needs greater quadrature order')
        out['families'].append(dict(a=a,z=zz,center_e=family.e0,saddle_e=family.saddle,
            separatrix_action=area,action_refinement=area_error,
            width_deg_at_center_e=np.degrees(family.width),crossings_deg_at_center_e=changes,
            crossing_actions=[family.measure(np.radians(v))['libration_action'] for v in changes],
            reference_offsets=family.measure(.05,.008,verify=True),orbits=rows))
        print(f'A={a}, z={zz:g}, crossings={changes}',flush=True)
    write(ROOT/'data/selection_orbits.json',out)


if __name__=='__main__':
    main()
