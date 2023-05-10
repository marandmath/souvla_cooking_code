"""
Simple Model with Dirichlet boundary conditions in the souvla cooking paper.

Usage: 
    simple_model_dirichlet.py --omega_value=<omega_value> --stop_time=<stop_time> --CFL_cond=<CFL_cond> --cap_dt=<capture_dt>
    
Options:
    -h, --help

"""

# COMMAND TO EXECUTE THIS FILE: 
# mpiexec -n 4 python3 simple_model_dirichlet.py --omega_value= --stop_time= --CFL_cond= --cap_dt=

from docopt import docopt
from numpy import exp, sin, arctan, sqrt, abs, sign, real, max
import numpy as np
from scipy.integrate import quad
from scipy.special import j0, j1, jn_zeros
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
import warnings
warnings.filterwarnings("ignore")

def main(omega_value, stop_time, CFL_cond, capture_dt):
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
    R = λ/(1-λ)
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
    H_avg = 0.535*Re_r**(0.55)*Pr_a**(0.37)
    # -------------------------------------------------------------------------------------------------

    # Dedalus Parameters
    # -------------------------------------------------------------------------------------------------
    Nθ, Nρ = 128, 128 # Resolution
    dealias = 1 # There are no non-linearities needed to dealias
    stop_sim_time = stop_time # Total run time
    timestepper = d3.SBDF2 # Other choices: SBDF3, SBDF4, RK222, RK443
    Δt = 1e-4
    max_timestep = 1e-2
    dtype = np.float64
    CFL_on = CFL_cond # Whether to use adaptive time step via CFL condition
    flag1, flag2 = True, True
    # -------------------------------------------------------------------------------------------------

    # Bases
    # -------------------------------------------------------------------------------------------------
    coords = d3.PolarCoordinates('θ', 'ρ')
    dist = d3.Distributor(coords, dtype=dtype)
    disk_basis = d3.DiskBasis(coords, shape=(Nθ,Nρ), radius=R, dealias=(dealias, dealias), dtype=dtype)
    θbasis = disk_basis.S1_basis(radius=R) # Or disk_basis.azimuth_basis or disk_basis.edge equivalently
    ρbasis = disk_basis.radial_basis
    # -------------------------------------------------------------------------------------------------

    # Fields
    # -------------------------------------------------------------------------------------------------
    # t = dist.Field()
    T_m = dist.Field(name='T_m', bases=disk_basis)
    # For the disk, only one tau term is needed for enforcing the one boundary condition
    # Based on the docs https://dedalus-project.readthedocs.io/en/latest/pages/tau_method.html,
    # Tau polynomials via the built-in Lift operator and from the original basis are more stable
    # compared to tau variable times a radial polynomial from the Chebyshev-U basis
    tau_T_m = dist.Field(name='tau_T_m', bases=θbasis)
    lift = lambda A: d3.Lift(A, disk_basis, -1)
    # -------------------------------------------------------------------------------------------------

    # Substitutions
    # -------------------------------------------------------------------------------------------------
    ε = 1/ω
    θ, ρ = dist.local_grids(disk_basis)
    eρ = dist.VectorField(coords, bases=ρbasis)
    eρ['g'][1] = 1
    eθ = dist.VectorField(coords, bases=disk_basis) # Need to put disk_basis here since azimuthial NCCs are not yet implemented
    eθ['g'][0] = 1
    η = 1e-3
    φ_η = dist.Field(name='φ_η', bases=θbasis)
    φ_η_func = lambda az: 1/2-arctan(sin(az)/η)/π
    φ_η['g'] = φ_η_func(θ)

    # Zeroth order approximation - (4.10) in the paper
    T_ε = dist.Field(name='T_ε', bases=disk_basis)
    T_C = dist.Field(name='T_C', bases=disk_basis)
    T_B = dist.Field(name='T_B', bases=disk_basis)
    T_C_acc = 11 # Can also use mpmath.nsum() for adaptive accuracy of infinite series instead of a fixed one but this should work for now
    T_B_acc = 10 # Can also use mpmath.nsum() for adaptive accuracy of infinite series   instead of a fixed one but this should work for now

    φ_m = lambda m: quad(lambda az: φ_η_func(az)*exp(-1j*m*az), 0, 2*π)[0]/(2*π)
    T_C_m = lambda m, az: φ_m(m)*exp(-(1+sign(m)*1j)*sqrt(ω*abs(m)/2)*(R-ρ))*exp(1j*m*az)
    T_C['g'] = real(sum([T_C_m(m,θ) for m in range(-round(T_C_acc/2)+1, round(T_C_acc/2))]))

    a_n = jn_zeros(0, T_B_acc)/R
    T_B_n = lambda n, time, rho: -(2*φ_m(0)/R)*exp(-time*a_n[n]**2)*j0(rho*a_n[n])/(a_n[n]*j1(R*a_n[n]))
    T_B['g'] = real(sum([T_B_n(n, 0, ρ) for n in range(T_B_acc)]))

    T_ε_op = T_C + T_B
    T_ε = T_ε_op.evaluate()

    # Checking if the field is well resolved, by inspecting the decay of the amplitude of the highest
    # wavenumber modes, under the Riemann-Lebesgue Lemma
    # plt.imshow(T_ε['c'], cmap='viridis', interpolation='nearest')
    # plt.colorbar()
    # plt.show()

    flux = dist.Field(name='flux', bases=disk_basis)
    flux = -κ_m*d3.dot(d3.grad(T_m), eρ)

    error = dist.Field(name='error', bases=disk_basis)
    error_op = T_m-T_ε
    error = error_op.evaluate()
    
    T_ε_list = [np.copy(T_ε['g'])]
    error_list = [np.copy(error['g'])]
    # -------------------------------------------------------------------------------------------------

    # Cartesian coordinates
    # -------------------------------------------------------------------------------------------------
    x = ρ*np.cos(θ)
    y = ρ*np.sin(θ)
    # -------------------------------------------------------------------------------------------------

    # Problem PDE(s)
    # -------------------------------------------------------------------------------------------------
    # Setting up the problem
    problem = d3.IVP([T_m, tau_T_m], namespace=locals())
    problem.add_equation("dt(T_m)-lap(T_m)+lift(tau_T_m)=-ω*dot(grad(T_m), eθ)")
    # -------------------------------------------------------------------------------------------------

    # Problem Boundary conditions
    # -------------------------------------------------------------------------------------------------
    problem.add_equation("T_m(ρ=R)=φ_η")
    # -------------------------------------------------------------------------------------------------

    # Solver
    # -------------------------------------------------------------------------------------------------
    solver = problem.build_solver(timestepper)
    solver.stop_sim_time = stop_sim_time
    # -------------------------------------------------------------------------------------------------

    # Problem Initial Conditions
    # -------------------------------------------------------------------------------------------------
    T_m['g'] = 0
    # -------------------------------------------------------------------------------------------------

    # Analysis
    # -------------------------------------------------------------------------------------------------
    snapshots = solver.evaluator.add_file_handler('snapshots', sim_dt=capture_dt, max_writes=50)
    snapshots.add_task(T_m, name='T_m')
    snapshots.add_task(T_ε, name='T_ε')
    snapshots.add_task(error, name='Error')

    azimuth_average = lambda A: d3.Average(A, coords[0])
    profiles = solver.evaluator.add_file_handler('profiles', sim_dt=capture_dt, max_writes=50)
    profiles.add_task(azimuth_average(flux), name='Heat_flux_azimuth_average')
    profiles.add_task(abs(azimuth_average(T_m)-azimuth_average(φ_η)), name='Absolute_SS_error')

    disk_average = lambda A: d3.Integrate(A, coords)
    scalars = solver.evaluator.add_file_handler('scalars', sim_dt=capture_dt, max_writes=1e6)
    scalars.add_task(disk_average(abs(error)), name='Average_abs_error')

    file_handlers = [snapshots, profiles, scalars]
    # -------------------------------------------------------------------------------------------------

    # CFL Condition via the gradient of the temperature field for stability
    # -------------------------------------------------------------------------------------------------
    CFL = d3.CFL(solver, initial_dt=Δt, cadence=5, safety=0.5, threshold=0.05,
                max_change=1.5, min_change=0.5, max_dt=max_timestep)
    CFL.add_velocity(d3.grad(T_m))
    # -------------------------------------------------------------------------------------------------

    # Temperature field properties
    # -------------------------------------------------------------------------------------------------
    BC_avg = azimuth_average(φ_η)
    temp = d3.GlobalFlowProperty(solver, cadence=100)
    temp.add_property(abs(azimuth_average(T_m)-BC_avg), name='abs_SS_dev')
    temp.add_property(abs(error), name='abs_error')
    # -------------------------------------------------------------------------------------------------

    # Main loop
    # -------------------------------------------------------------------------------------------------
    try:
        logger.info('Starting main loop')
        sim_div = 0.0
        while solver.proceed:
            if CFL_on:
                Δt = CFL.compute_timestep()
            solver.step(Δt)
            t = solver.sim_time
            T_C['g'] = real(sum([T_C_m(m, θ-ω*t) for m in range(-round(T_C_acc/2)+1, round(T_C_acc/2))]))
            T_B['g'] = real(sum([T_B_n(n, t, ρ) for n in range(T_B_acc)]))
            T_ε_op = T_C + T_B
            T_ε = T_ε_op.evaluate()
            error_op = T_m-T_ε
            error = error_op.evaluate()
            if T_ε(ρ=0).evaluate().allgather_data('g')[0,0] >= T_cook and flag1:
                t_cookthrough_ε = t
                flag1 = False
            if T_m(ρ=0).evaluate().allgather_data('g')[0,0] >= T_cook and flag2:
                t_cookthrough_m = t
                flag2 = False
            if (solver.iteration-1) % 100 == 0:
                logger.info("Iteration=%i, Time=%e, dt=%e, abs_SS_dev=%e, abs_error=%e" %(solver.iteration, t, Δt, temp.max('abs_SS_dev'), temp.max('abs_error')))
            if sim_div < t // capture_dt and t < stop_sim_time:
                sim_div += 1
                T_ε_list.append(np.copy(T_ε['g']))
                error_list.append(np.copy(error['g']))
    except:
        logger.error('Exception raised, triggering end of main loop.')
        raise
    finally:
        solver.log_stats()
    csv_data = [datetime.now().strftime('%d-%m-%Y %H:%M:%S'), ω, t_cookthrough_m, t_cookthrough_ε, error.allreduce_data_norm(order=np.inf)]
    
    return T_ε_list, error_list, csv_data
    
if __name__ == '__main__':
    args = docopt(__doc__)
    omega_value = float(args['--omega_value'])
    stop_time = float(args['--stop_time'])
    CFL_cond = bool(args['--CFL_cond'])
    capture_dt = float(args['--cap_dt'])
    T_ε_list, error_list, csv_data = main(omega_value, stop_time, CFL_cond, capture_dt)
    T_ε_list_no_reps = []
    error_list_no_reps = []
    for T_ε_arr in T_ε_list:
        if not any(np.array_equal(T_ε_arr, A) for A in T_ε_list_no_reps):
            T_ε_list_no_reps.append(T_ε_arr)
    for error_arr in error_list:
        if not any(np.array_equal(error_arr, A) for A in error_list_no_reps):
            error_list_no_reps.append(error_arr)
    with Sync() as sync:
        if sync.comm.rank == 0:
            # Saving frames for zeroth-order approximation and error
            # -------------------------------------------------------------------------------------------------
            np.savez_compressed("T_ε.npz", *T_ε_list_no_reps)
            np.savez_compressed("error.npz", *error_list_no_reps)
            # -------------------------------------------------------------------------------------------------
            
            # Post-solution analysis metrics stored to CSV file in serial
            # -------------------------------------------------------------------------------------------------
            with open("run_data.csv", 'a') as csvfile:
                writer_obj = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                writer_obj.writerow(
                    [
                        f"Datetime_of_sim = {csv_data[0]}", 
                        f"ω = {csv_data[1]}", 
                        f"t_cookthrough_m = {csv_data[2]:.12f}", 
                        f"t_cookthrough_ε = {csv_data[3]:.12f}", 
                        f"Max_Absolute_Error = {csv_data[4]:.12f}"
                    ]
                )
                csvfile.close()
            # -------------------------------------------------------------------------------------------------
            