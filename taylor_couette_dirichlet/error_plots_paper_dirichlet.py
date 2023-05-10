import numpy as np
import matplotlib.pyplot as plt

def abline(ax, slope, intercept):
    """Plot a line from slope and intercept"""
    x_vals = np.array(ax.get_xlim())
    y_vals = intercept + slope * x_vals
    ax.plot(x_vals, y_vals, 'k--')

time_scale = 4.91264e5
ω = np.array([2.1, 75, 200, 400])
errors = np.array([
    0.317467015613,
    0.196844541665,
    0.132517311445,
    0.102849301491
])
appr_cook_through = time_scale*np.array([
    0.007437992532,
    0.007438127752,
    0.007438078557,
    0.007438061971
])/60
simulation_cook_through = time_scale*np.array([
    0.007439113698,
    0.007438988089,
    0.007438998075,
    0.007438914717
])/60
real_cookthrough = time_scale*0.00748492/60

fig, axs = plt.subplots(nrows=1, ncols=2)

axs[0].loglog(ω, errors, 'b-o')
axs[0].set_xlabel(r'$ω$', fontsize=14)
axs[0].set_ylabel(r'$\|T_m-T_0\|_{\infty}$', fontsize=17)
axs[0].set_title('Error between $T_m$ and $T_0$ against $ω$,\n in a loglog scale', fontsize=18)
abline(axs[0], -0.5, 0.3)
axs[0].grid()

axs[1].plot(ω, simulation_cook_through, 'b-o', label=r'Dedalus $T_m$ cookthrough')
axs[1].plot(ω, appr_cook_through, 'r-o', label=r'$T_0$ cookthrough')
axs[1].axhline(real_cookthrough, color='k', linestyle='--', label=f'Analytical cookthrough time: \n{real_cookthrough:.2f} mins')
axs[1].set_xlabel(r'$ω$', fontsize=14)
axs[1].set_ylabel(r'$t$ (mins)', fontsize=14)
axs[1].set_title('Cookthrough times (in mins)\n against $ω$', fontsize=18)
axs[1].legend()

fig.tight_layout()
fig.set_size_inches(10.5,7)
fig.savefig("error_plots.png", dpi='figure')
