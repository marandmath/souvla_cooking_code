"""
Simple Model with Robin boundary conditions in the souvla cooking paper.

Usage: 
    simple_model_robin.py --omega_value=<omega_value> --stop_time=<stop_time> --cap_dt=<capture_dt>
    
Options:
    -h, --help

"""

from docopt import docopt
from numpy import exp, sin, cos, arctan, sqrt, tanh, abs, sign, real, max, piecewise
import numpy as np
from scipy.integrate import quad
from scipy.special import jn_zeros, j0, j1, erf
import dedalus.public as d3
from dedalus.tools import post
from dedalus.tools.parallel import Sync
import matplotlib.pyplot as plt
import logging
logger = logging.getLogger(__name__)
import csv
import zipfile
import io
import os
from datetime import datetime
from itertools import pairwise
import warnings
warnings.filterwarnings("ignore")

# COMMAND TO EXECUTE THIS FILE: 
# mpiexec -n 4 python3 simple_model_robin.py --omega_value= --stop_time= --cap_dt=

def main(omega_value, stop_time, capture_dt):
    """
    Execute simulation using the input value for ω, the terminating simulation time, and whether to
    use the CFL condition on the gradient of the temperature field as to take adaptive time steps for
    stability reasons.
    """
    
    # Problem Parameters
    # Units: Length: m, Time: s, Mass: kg, Temperature: °C, Power: W, Energy: J
    # Time scale defined via the the ratio of the characteristic length and tangential velocity which is
    # the angular period of our problem instead of the ratio of the square of the characteristic length 
    # and the meat's thermal diffusivity. NEED TO CHANGE TO THE LATTER IN THE PAPER.
    # -------------------------------------------------------------------------------------------------
    R_m = 0.05 # (m)
    R_f = 0.3 # (m)
    h = R_f-R_m # Length scale (m)
    len_scale = h # Length scale (m)
    λ = R_m/R_f # d'less
    ndr_m = λ/(1-λ)
    ndr_f = 1/(1-λ)
    ω = omega_value # (s⁻¹) # Original value: 2.1 
    T_H = 200 # (°C)
    T_inf = 25 # (°C)
    ΔT = T_H-T_inf # (°C)
    T_cook = (65-T_inf)/ΔT
    g = 9.81 # (ms⁻²)
    π = np.pi

    # Meat's thermal properties
    κ_m = 0.442 # (Wm⁻¹°C⁻¹)
    ρ_m = 1105.25 # (kgm⁻³)
    Cp_m = 3143.44 # (Jkg⁻¹°C⁻¹)
    D_m = κ_m/(ρ_m*Cp_m) # (m²s⁻¹)

    # time_scale = len_scale/vel_scale # (s) - Angular Period or Rotation Time scale
    time_scale = len_scale**2/D_m # (s) - Diffusion Time scale
    # vel_scale = ω*R_m # (ms⁻¹) - Tangential velocity scale
    vel_scale = D_m/len_scale # (ms⁻¹)

    # Air's thermal and fluid properties
    κ_a = 0.03095 # (Wm⁻¹°C⁻¹)
    ρ_a = 0.94577 # (kgm⁻³)
    Cp_a = 1009 # (Jkg⁻¹°C⁻¹)
    D_a = κ_a/(ρ_a*Cp_a) # (m²s⁻¹)
    ν = 2.3058e-5 # (m²s⁻¹)
    ρ_0 = 1.225 # (kgm⁻³)
    α = 1/T_inf # Ideal gas law (C⁻¹)

    # Dimensionless quantities
    diff_ratio = D_a/D_m # d'less
    Pr_a = ν/D_a # d'less
    Pe_m = len_scale*vel_scale/D_m # d'less
    Pe_a = len_scale*vel_scale/D_a # d'less
    Fr_sq = vel_scale**2/(g*len_scale) # d'less
    Ra = (α*g*ΔT*len_scale**3)/(ν*D_m) # d'less
    Ro = (ω/(2*π))*len_scale/vel_scale # d'less
    Re_a = vel_scale*len_scale/ν # d'less
    Re_r = (2*ω*R_m**2)/ν

    # Average Nusselt Number
    H_avg = (κ_m/h)*0.535*Re_r**(0.55)*Pr_a**(0.37)
    # -------------------------------------------------------------------------------------------------

    # Dedalus Parameters
    # -------------------------------------------------------------------------------------------------
    Nθ, Nρ = 160, 192 # Resolution # 128, 192
    dealias = 3/2 # There's only a second order non-linearity
    stop_sim_time = stop_time # Total run time
    timestepper = d3.SBDF2 # Choices: SBDF2, CNLF2, RK222, SBDF3, SBDF4, RK443
    Δt = 1e-5
    max_timestep = 1e-4
    dtype = np.float64
    flag1, flag2 = True, True
    # -------------------------------------------------------------------------------------------------

    # Bases
    # -------------------------------------------------------------------------------------------------
    coords = d3.PolarCoordinates('θ', 'ρ')
    dist = d3.Distributor(coords, dtype=dtype)
    disk_basis = d3.DiskBasis(coords, shape=(Nθ,Nρ), radius=R_f, dealias=(dealias, dealias), dtype=dtype)
    θbasis = disk_basis.S1_basis(radius=R_f) # Or disk_basis.azimuth_basis or disk_basis.edge equivalently
    ρbasis = disk_basis.radial_basis
    # -------------------------------------------------------------------------------------------------

    # Fields
    # -------------------------------------------------------------------------------------------------
    # t = dist.Field()
    T_m = dist.Field(name='T_m', bases=disk_basis)
    T_a = dist.Field(name='T_a', bases=disk_basis)
    u = dist.VectorField(coords, name='u', bases=disk_basis)
    p = dist.Field(name='p', bases=disk_basis)
    # For the disk, only one tau term is needed for enforcing the one boundary condition
    # Based on the docs https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html,
    # Tau polynomials via the built-in Lift operator and from the original basis are more stable
    # compared to tau variable times a radial polynomial from the Chebyshev-U basis
    tau_T_m = dist.Field(name='tau_T_m', bases=θbasis)
    tau_T_a = dist.Field(name='tau_T_a', bases=θbasis)
    tau_u = dist.VectorField(coords, name='tau_u', bases=θbasis)
    tau_p = dist.Field(name='tau_p')
    lift = lambda A: d3.Lift(A, disk_basis, -1)
    t = dist.Field()
    # -------------------------------------------------------------------------------------------------

    # Substitutions
    # -------------------------------------------------------------------------------------------------
    θ, ρ = dist.local_grids(disk_basis)
    eθ = dist.VectorField(coords, name='eθ', bases=disk_basis) # Need to put disk_basis here since azimuthial NCCs are not yet implemented
    eθ['g'][0] = 1
    eρ = dist.VectorField(coords, name='eρ', bases=ρbasis)
    eρ['g'][1] = 1
    
    # Boundary condition for T on the surface of the grill
    η = 1e-3
    φ_η = dist.Field(name='φ_η', bases=θbasis)
    φ_η_func = lambda az: T_inf+ΔT*(1/2-arctan(sin(az)/η)/π)
    φ_η['g'] = φ_η_func(θ)
    
    # Mask function for boundary conditions on the surface of the meat
    δ_stability = 1e-6
    η_τ = 1e-2
    def normalised_mask(x): return piecewise(x, [x<=-1,(x>-1)&(x<1),x>=1],
                                           [lambda x: 1 + δ_stability,
                                            lambda x: (1-tanh(3*x/sqrt(1-x**2)))/2 + δ_stability,
                                            lambda x: δ_stability])
    Γ = dist.Field(name='Γ', bases=ρbasis)
    Γ['g'] = normalised_mask((ρ-R_m)/η_τ)
    η_a = 1e-3
    
    # Gravitational accelaration vector in polar coordinates
    ey = dist.VectorField(coords, name='g', bases=θbasis)
    ey['g'][0] = cos(θ)
    ey['g'][1] = sin(θ)

    # Zeroth order approximation - (5.5) in the paper
    T_ε = dist.Field(name='T_ε', bases=disk_basis)
    T_ε_acc = 10 # Can also use mpmath.nsum() for adaptive accuracy of infinite series instead of a fixed one but this should work for now
    a_n = jn_zeros(0, T_ε_acc)/R_m
    trap_integral = dist.Field(name='trap_integral')
    abs_error = dist.Field(name='error', bases=disk_basis)
    T_ε.change_scales(dealias)
    abs_error.change_scales(dealias)
    T_ε_list = [np.copy(T_ε['g'])]
    abs_error_list = [np.copy(abs_error['g'])] 
    # -------------------------------------------------------------------------------------------------

    # Cartesian coordinates
    # -------------------------------------------------------------------------------------------------
    x = ρ*np.cos(θ)
    y = ρ*np.sin(θ)
    # -------------------------------------------------------------------------------------------------

    # Problem PDEs
    # -------------------------------------------------------------------------------------------------
    # Setting up the problem
    problem = d3.IVP([T_m, T_a, u, p, tau_T_m, tau_T_a, tau_u, tau_p], time=t, namespace=locals())
    problem.add_equation("dt(T_m) - D_m*div(Γ*grad(T_m)) + η_τ*H_avg*(T_m-T_a)*(grad(Γ)@grad(Γ))/(ρ_m*Cp_m) + lift(tau_T_m) = - Γ*ω*eθ@grad(T_m)")
    problem.add_equation("dt(T_a) - D_a*div((1-Γ)*grad(T_a)) - η_τ*κ_m*eρ@grad(T_m)*(grad(1-Γ)@grad(1-Γ))/(ρ_a*Cp_a) + lift(tau_T_a) = - (1-Γ)*ω*eθ@grad(T_a) - (1-Γ)*u@grad(T_a)") 
    problem.add_equation("dt(u) - ν*lap(u) + (1/ρ_0)*grad(p) + lift(tau_u) = - u@grad(u) - (1-α*(T_a-T_inf))*g*ey - Γ*(u-ω*R_m*eθ)/η_a - ω*eθ@grad(u)")  
    problem.add_equation("div(u) + tau_p = 0") # Incompressibility - Conservation of mass
    # -------------------------------------------------------------------------------------------------

    # Problem Boundary conditions
    # -------------------------------------------------------------------------------------------------
    problem.add_equation("T_a(ρ=R_f)=φ_η")
    problem.add_equation("u(ρ=R_f)=0")
    problem.add_equation("(eρ@grad(T_m))(ρ=R_f)=0") # No penetration; A Dirichlet BC will have a global effect which is undesired,
                                                    # thus a Neumann/Natural BC is the better alternative
    problem.add_equation("integ(p)=0") # Pressure gauge to avoid non-singularity of the sparse linear system in the LHS
    # -------------------------------------------------------------------------------------------------

    # Solver
    # -------------------------------------------------------------------------------------------------
    solver = problem.build_solver(timestepper, ncc_cutoff=1e-10)
    solver.stop_sim_time = stop_sim_time
    # -------------------------------------------------------------------------------------------------

    # Problem Initial Conditions
    # -------------------------------------------------------------------------------------------------
    T_m['g'] = T_inf
    T_a['g'] = T_inf
    u['g'] = 0
    # -------------------------------------------------------------------------------------------------

    # Analysis
    # -------------------------------------------------------------------------------------------------
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=capture_dt, max_writes=100)
    snapshots.add_task(T_m, scales=dealias, name='T_m')
    snapshots.add_task(T_a, scales=dealias, name='T_a')
    snapshots.add_task(eθ@u, scales=dealias, name='u_θ')
    snapshots.add_task(eρ@u, scales=dealias, name='u_ρ')
    snapshots.add_task(-d3.div(d3.skew(u)), scales=dealias, name='Vorticity')
    snapshots.add_task(p, scales=dealias, name='Pressure')
    
    profiles = solver.evaluator.add_file_handler('profiles', sim_dt=capture_dt, max_writes=10)
    profiles.add_task(d3.Average(-κ_m*eρ@d3.grad(T_m), coords[0]), name='Flux_T_m_azimuth_average')
    profiles.add_task(d3.Average(-κ_a*eρ@d3.grad(T_a), coords[0]), name='Flux_T_a_azimuth_average')

    scalars = solver.evaluator.add_file_handler('scalars', sim_dt=capture_dt, max_writes=1e6)
    scalars.add_task(d3.Average(T_a(ρ=R_m), coords[0]), name='<T_a>_θ_boundary')

    file_handlers = [snapshots, profiles, scalars]
    # -------------------------------------------------------------------------------------------------

    # CFL Condition for stability
    # -------------------------------------------------------------------------------------------------
    CFL = d3.CFL(
        solver, initial_dt=Δt, cadence=10, safety=0.2, threshold=0.01,
        min_change=0.5, max_dt=max_timestep
    )
    CFL.add_velocity((1-Γ)*u)
    # -------------------------------------------------------------------------------------------------

    # Flow properties
    # -------------------------------------------------------------------------------------------------
    flow = d3.GlobalFlowProperty(solver, cadence=10)
    flow.add_property(u@u, name='u2')
    flow.add_property(d3.Average(T_a(ρ=R_m), coords[0]), name='<T_a>_θ_boundary')
    # -------------------------------------------------------------------------------------------------

    # Main loop
    # -------------------------------------------------------------------------------------------------
    time_samples = [d3.Average(T_a(ρ=R_m), coords[0]).evaluate()]
    times = [0]
    trap_integral_op = lambda n, time, pr_int, t_1, t_2, T_1, T_2: pr_int+(t_2-t_1)*(T_1*exp(D_m*(t_1-time)*a_n[n]**2)+T_2*exp(D_m*(t_2-time)*a_n[n]**2))/2
    T_ε_n = [a_n[n]*j0(ρ*a_n[n])/j1(R_m*a_n[n]) for n in range(T_ε_acc)]
    prev_ints = T_ε_acc*[0]
    sim_div = 0.0
    try:
        logger.info('Starting main loop')
        while solver.proceed:
            Δt = CFL.compute_timestep()
            solver.step(Δt)
            if (solver.iteration-1) % 100 == 0:
                max_u = sqrt(flow.max('u2'))
                logger.info("Iteration=%i, Time=%e, dt=%e, max(u)=%e,  <T_a>_θ(ρ=R_m)=%e" %(solver.iteration, solver.sim_time, Δt, max_u, flow.max('<T_a>_θ_boundary')))
            if sim_div < solver.sim_time // capture_dt and solver.sim_time < stop_sim_time:
                sim_div += 1
                time_samples.append(d3.Average(solver.state[1](ρ=R_m), coords[0]).evaluate())
                times.append(solver.sim_time)
                for n in range(T_ε_acc):
                    with Sync() as sync:
                        if sync.comm.rank == 0:
                            trap_integral = trap_integral_op(n, solver.sim_time, prev_ints[n], times[-2], times[-1], time_samples[-2], time_samples[-1]).evaluate()
                            prev_ints[n] = trap_integral['g'][0][0]
                T_ε.change_scales(1)
                T_ε['g'] = T_inf+(2*D_m/R_m)*sum([prev_ints[n]*T_ε_n[n] for n in range(T_ε_acc)])
                T_ε.change_scales(dealias)
                abs_error['g'] = solver.state[0]['g'] - T_ε['g']
                T_ε_list.append(np.copy(T_ε['g']))
                abs_error_list.append(np.copy(abs_error['g']))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
    return T_ε_list, abs_error_list
    
if __name__ == '__main__':
    args = docopt(__doc__)
    omega_value = float(args['--omega_value'])
    stop_time = float(args['--stop_time'])
    capture_dt = float(args['--cap_dt'])
    T_ε_list, abs_error_list = main(omega_value, stop_time, capture_dt)
    T_ε_list_no_reps = []
    abs_error_list_no_reps = []
    for T_ε_arr in T_ε_list:
        if not any(np.array_equal(T_ε_arr, A) for A in T_ε_list_no_reps):
            T_ε_list_no_reps.append(T_ε_arr)
    for error_arr in abs_error_list:
        if not any(np.array_equal(error_arr, A) for A in abs_error_list_no_reps):
            abs_error_list_no_reps.append(error_arr)
    with Sync() as sync:
        if sync.comm.rank == 0:
            # Saving frames for zeroth-order approximation and error
            # -------------------------------------------------------------------------------------------------
            np.savez_compressed("T_ε.npz", *T_ε_list_no_reps)
            np.savez_compressed("abs_error.npz", *abs_error_list_no_reps)
            # -------------------------------------------------------------------------------------------------

            

