"""Relative cloud/black-hole tilt in a conservative weak-field model.

The leading spin-orbit interaction is matched to the scalar hyperfine term.
The black-hole spin obeys its angular-momentum Poisson bracket. This restores
the common-rotation zero mode while retaining an internal precession mode.
Horizon dissipation and an evolving binary are not included in this module.
"""
from dataclasses import replace
import math

import numpy as np
from scipy.linalg import eig

from .research_collective_mode import (
    RadialGrid, make_sector, collective_operator, angular)


def radial_operator(grid, sector, power):
    result = np.zeros_like(sector.h0)
    for index, ell in enumerate(sector.ls):
        f = sector.functions[ell]
        result[sector.block(index), sector.block(index)] = (
            f.T@((grid.w*grid.r**(2+power))[:, None]*f))
    return result


def tilted_cloud(alpha, spin, fraction, *, size=16, order=40, maximum_l=3):
    """Return the coupled field/spin operator in units E_B=alpha**3/M.

    M=1, S=spin, C=fraction/alpha; the field wavefunction has unit norm.
    Coordinates are (u_m=2, v_m=0, delta S_-), with time dependence exp(-iwt).
    """
    if not 0 < alpha < .3 or not 0 < spin < 1 or not 0 < fraction <= .1:
        raise ValueError('Require positive weak-field coupling, spin and cloud fraction')
    if maximum_l not in [1,3,5]:
        raise ValueError('The odd-parity parent uses maximum l=1,3,5')
    grid = RadialGrid(order)
    parent = make_sector(grid, 1, tuple(range(1,maximum_l+1,2)), size)
    # l=3 is retained even for a purely l=1 parent, for its quadrupolar response.
    upper = make_sector(grid, 2, tuple(range(3,max(3,maximum_l)+1,2)), size)
    lower = make_sector(grid, 0, tuple(range(1,maximum_l+1,2)), size)
    sectors = [parent,upper,lower]
    sectors = [replace(s,h0=s.h0+2*spin*alpha**3*s.m*radial_operator(grid,s,-3))
               for s in sectors]
    parent,upper,lower = sectors
    op = collective_operator(grid,parent,upper,lower,fraction)
    background = op['background']
    coefficients, fields = background['coefficients'], background['fields']
    cloud = fraction/alpha
    overlap = sum(float(grid.w@(field*field/grid.r)) for field in fields.values())
    beta = 2*alpha**3*overlap
    rotation_u,rotation_v,torque_u,torque_v = [],[],[],[]
    for sector,rotation,torque,sign in [(upper,rotation_u,torque_u,-1.),
                                       (lower,rotation_v,torque_v,1.)]:
        for ell in sector.ls:
            if ell not in parent.ls:
                rotation.extend(np.zeros(size))
                torque.extend(np.zeros(size))
                continue
            ladder = math.sqrt(ell*(ell+1)-(2 if sector.m == 2 else 0))
            block = coefficients[parent.block(parent.ls.index(ell))]
            rotation.extend(sign*.5*ladder*block)
            torque.extend(ladder*sector.functions[ell].T@(grid.w*fields[ell]/grid.r))
    ru,rv,tu,tv = map(np.asarray,[rotation_u,rotation_v,torque_u,torque_v])
    field_size = len(ru)+len(rv)
    matrix = np.zeros((field_size+1,field_size+1))
    matrix[:field_size,:field_size] = op['matrix']
    matrix[:field_size,-1] = alpha**3*np.r_[tu,-tv]
    matrix[-1,:field_size] = -2*cloud*spin*alpha**3*np.r_[tu,tv]
    matrix[-1,-1] = beta*cloud
    rotation = np.r_[ru,rv,spin]
    signature = np.r_[np.ones(len(ru)),-np.ones(len(rv)),-1/(2*cloud*spin)]
    angular_momentum = np.r_[-2*cloud*ru,2*cloud*rv,1.]

    # O_+ = (x+iy)z = -sqrt(8*pi/15) r^2 Y_21; O_- is its adjoint.
    probes = []
    for sector,sign in [(upper,-1.),(lower,1.)]:
        blocks = []
        for ell in sector.ls:
            radial = sum(angular(ell,sector.m,a,1,2,sector.m-1)*fields[a]
                         for a in parent.ls)
            blocks.extend(sign*math.sqrt(8*math.pi/15)*sector.functions[ell].T@(
                grid.w*grid.r**4*radial))
        probes.append(np.array(blocks))
    qu,qv = probes
    drive = np.r_[qu/2,-qv/2,0.]
    readout = np.r_[qu,qv,0.]
    qtilt = float(qu@ru+qv@rv)
    total_frame = readout-qtilt*angular_momentum/(cloud+spin)
    hole_frame = readout.copy()
    hole_frame[-1] -= qtilt/spin
    return dict(matrix=matrix,signature=signature,rotation=rotation,drive=drive,readout=readout,
        angular_momentum=angular_momentum,total_frame_readout=total_frame,
        hole_frame_readout=hole_frame,
        beta=beta,cloud=cloud,spin=spin,quadrupole_tilt=qtilt,
        parent_residual=background['residual'],radial_overlap=overlap,
        relative_frequency=beta*(spin+cloud),fixed_frequency=beta*spin,
        coefficients=coefficients,background=background,
        configuration=dict(alpha=alpha,spin=spin,fraction=fraction,size=size,
            order=order,maximum_l=maximum_l,energy_unit='alpha**3/M'))


def pole_data(matrix,drive,readout):
    values,left,right = eig(matrix,left=True,right=True)
    residues = np.array([(readout@r)*(np.vdot(l,drive))/np.vdot(l,r)
                         for l,r in zip(left.T,right.T)])
    return values,residues,right


def rigid_operator(beta,cloud,spin,quadrupole):
    """Leading rigid tilt on (cloud angle, transverse BH angular momentum).

    beta has inverse-angular-momentum/time units. quadrupole and the forcing
    use the same per-particle convention as tilted_cloud. No damping is added.
    """
    if beta <= 0 or cloud <= 0 or spin <= 0:
        raise ValueError('Require positive coupling and angular momenta')
    matrix = beta*np.array([[spin,-1.],[-cloud*spin,cloud]])
    drive = np.array([-quadrupole,0.])
    readout = np.array([quadrupole,0.])
    momentum = np.array([cloud,1.])
    return dict(matrix=matrix,drive=drive,readout=readout,angular_momentum=momentum,
        rotation=np.array([1.,spin]),
        total_frame_readout=readout-quadrupole*momentum/(cloud+spin),
        hole_frame_readout=np.array([quadrupole,-quadrupole/spin]))


def inspect_tilt(model):
    m = model
    matrix,rotation,metric = m['matrix'],m['rotation'],m['signature']
    values,residues,vectors = pole_data(matrix,m['drive'],m['readout'])
    zero = int(np.argmin(abs(values)))
    frequency = m['relative_frequency']
    internal = int(np.argmin(abs(values-frequency)))
    if internal == zero or max(abs(values[[zero,internal]].imag)) > 1e-9*frequency:
        raise ArithmeticError('The two conservative tilt modes were not identified')
    fixed,fixed_residues,_ = pole_data(matrix[:-1,:-1],m['drive'][:-1],m['readout'][:-1])
    fixed_index = int(np.argmin(abs(fixed-m['fixed_frequency'])))
    cloud,spin,q = m['cloud'],m['spin'],m['quadrupole_tilt']
    geometric = dict(zero_residue=-q*q*cloud/(spin+cloud),
        internal_residue=-q*q*spin/(spin+cloud),fixed_residue=-q*q,
        frequency_ratio=1+cloud/spin,internal_static_ratio=(spin/(spin+cloud))**2)
    def real(z):
        if abs(z.imag) > 1e-8*max(1.,abs(z.real)):
            raise ArithmeticError('Conservative pole has a complex residue')
        return float(z.real)
    selected = dict(zero_frequency=real(values[zero]),internal_frequency=real(values[internal]),
        fixed_frequency=real(fixed[fixed_index]),zero_residue=real(residues[zero]),
        internal_residue=real(residues[internal]),fixed_residue=real(fixed_residues[fixed_index]))
    selected['frequency_ratio'] = selected['internal_frequency']/selected['fixed_frequency']
    selected['internal_static_ratio'] = (selected['internal_residue']/selected['internal_frequency']
        /(selected['fixed_residue']/selected['fixed_frequency']))
    frames = {}
    for name in ['total_frame_readout','hole_frame_readout']:
        _,weights,_ = pole_data(matrix,m['drive'],m[name])
        frames[name] = dict(zero_residue=real(weights[zero]),
            internal_residue=real(weights[internal]),
            internal_static_ratio=real(weights[internal])/selected['internal_frequency']
                /(selected['fixed_residue']/selected['fixed_frequency']))
    keep = np.ones(len(values),dtype=bool)
    keep[[zero,internal]] = False
    regular_static = real(-sum(residues[keep]/values[keep]))
    return dict(configuration=m['configuration'],cloud_spin_ratio=cloud/spin,
        radial_overlap=m['radial_overlap'],radial_over_hydrogen=24*m['radial_overlap'],
        parent_residual=m['parent_residual'],quadrupole_tilt=q,
        common_rotation_residual=float(np.linalg.norm(matrix@rotation)/np.linalg.norm(rotation)),
        hessian_symmetry_error=float(np.max(abs(metric[:,None]*matrix-(metric[:,None]*matrix).T))),
        angular_momentum_residual=float(np.linalg.norm(m['angular_momentum']@matrix)),
        external_torque_error=float(m['angular_momentum']@m['drive']+cloud*q),
        frames=frames,regular_static_response=regular_static,
        regular_over_internal_static=regular_static/(-selected['internal_residue']/selected['internal_frequency']),
        geometric_prediction=geometric,measured=selected,
        internal_frequency_relative_error=selected['internal_frequency']/frequency-1,
        fixed_frequency_relative_error=selected['fixed_frequency']/m['fixed_frequency']-1,
        maximum_imaginary_eigenvalue=float(max(abs(values.imag))))
