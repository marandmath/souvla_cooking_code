"""
Plot distribution of heat in zeroth order approximation and error via the snapshots of simple_model_dirichlet.py.

Usage:
    plot_appr_error.py --omega_value=<omega_val> --cap_dt=<capture_dt>

Options:
    -h, --help

"""

# COMMANDS TO EXECUTE THIS FILE: 
# mpiexec -n 1 python3 plot_appr_error.py --omega_value= --cap_dt=

from docopt import docopt
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import numpy as np

def main(ω, capture_dt):
    
    T_ε_loaded = np.load('T_ε.npz')
    T_ε_list = [T_ε_loaded[f'arr_{i}.npy'] for i in range(len(T_ε_loaded.files))]
    error_loaded = np.load('error.npz')
    error_list = [error_loaded[f'arr_{i}.npy'] for i in range(len(error_loaded.files))]

    fig, axs = plt.subplots(1, 2, figsize=(16,8), subplot_kw={'projection': 'polar'})
    R_m = 0.05 # (m)
    R_f = 0.3 # (m)
    λ = R_m/R_f # d'less
    R = λ/(1-λ)
    rho = np.linspace(0, R, 32) # Nρ/4 since Chebyshev non-periodic basis excludes the nodes
    theta = np.linspace(0, 2*np.pi, 128)
    ρ, θ = np.meshgrid(rho, theta)

    quad1 = axs[0].pcolormesh(θ, ρ, T_ε_list[0], cmap=plt.cm.rainbow, shading='gouraud')
    fig.colorbar(quad1, ax=axs[0])
    axs[0].grid()
    axs[0].set_title(r"$T_ε(ρ,θ)$")
    axs[0].arrow(0, 0, ω*0*capture_dt,  rho[-2])

    quad2 = axs[1].pcolormesh(θ, ρ, error_list[0], cmap=plt.cm.seismic, shading='gouraud')
    fig.colorbar(quad2, ax=axs[1])
    axs[1].grid()
    axs[1].set_title(r"$Error(ρ,θ)$")
    axs[1].arrow(0, 0, ω*0*capture_dt,  rho[-2])

    def init():
        quad1.set_array([])
        quad2.set_array([])
        return quad1, quad2
        
    def animate(i):
        axs[0].cla()
        axs[1].cla()
        quad1 = axs[0].pcolormesh(θ, ρ, T_ε_list[i], cmap=plt.cm.rainbow, shading='gouraud')
        quad2 = axs[1].pcolormesh(θ, ρ, error_list[i], cmap=plt.cm.seismic, shading='gouraud')
        fig.suptitle(f"t = {i*capture_dt:.3f}s")
        # quad1.set_array(T_ε_list[i].ravel())
        # quad2.set_array(error_list[i].ravel())
        axs[0].arrow(0, 0, ω*i*capture_dt,  rho[-2])
        axs[1].arrow(0, 0, ω*i*capture_dt,  rho[-2])
        return quad1, quad2, 

    fig.tight_layout()

    anim = animation.FuncAnimation(fig, animate, interval=100, frames=len(T_ε_list), blit=True, repeat=False)
    anim.save('plots.mp4')
    
if __name__ == '__main__': 
    args = docopt(__doc__)
    omega_value = float(args['--omega_value'])
    capture_dt = float(args['--cap_dt'])
    main(omega_value, capture_dt)