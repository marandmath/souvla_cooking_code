"""
Plot distribution of heat in meat slab via the snapshots of simple_model_dirichlet.py.

Usage:
    plot_simple_model_dirichlet.py <files>... [--omega_value=<omega_value>] [--output=<dir>]

Options:
    -h, --help
    --output=<dir>  Output directory [default: ./frames]
    --omega_value=<omega_val>  ω value [default: 2.1]

"""

# COMMANDS TO EXECUTE THIS FILE: 
# mpiexec -n 4 python3 plot_simple_model_dirichlet.py snapshots/*.h5 --omega_value=
# mpiexec -n 4 python3 plot_simple_model_dirichlet.py profiles/*.h5
# mpiexec -n 4 python3 plot_simple_model_dirichlet.py scalars/*.h5

import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dedalus.extras import plot_tools
import ffmpeg
from time import sleep

def main(filename, start, count, output, omega_value):
    """Save plot of specified tasks for given range of analysis writes."""

    R_m = 0.05 # (m)
    R_f = 0.3 # (m)
    λ = R_m/R_f # d'less
    R = λ/(1-λ)
    ω = omega_value # (s⁻¹) # Original value: 2.1
    savename_func = lambda write: 'write_{:06}.png'.format(write)
    title_func = lambda sim_time: '$t = {:.3f}$'.format(sim_time)
    dpi = 200
    func = lambda θ, ρ, data: (ρ*np.cos(θ), ρ*np.sin(θ), data) # Transition between Cartesian and polar coordinates

    if "snapshots" in filename:
        tasks = ['T_m', 'T_ε', 'Error']
        
        # Layout
        nrows, ncols = 1, 3
        image = plot_tools.Box(1, 2)
        pad = plot_tools.Frame(0.2, 0, 0, 0)
        margin = plot_tools.Frame(0.2, 0.1, 0.1, 0.1)
        scale = 2

        # Create figure
        mfig = plot_tools.MultiFigure(nrows, ncols, image, pad, margin, scale)
        fig = mfig.figure

        # Plot writes
        with h5py.File(filename, mode='r') as file:
            for index in range(start, start+count):
                for n, task in enumerate(tasks):
                    # Build subfigure axes
                    i, j = divmod(n, ncols)
                    axes = mfig.add_axes(i, j, [0, 0, 1, 1])
                    # Call 3D plotting helper, slicing in time
                    dset = file['tasks'][task]
                    if task == 'Error':
                        cmap = plt.cm.seismic
                    else:
                        cmap = plt.cm.rainbow
                    paxes, caxes = plot_tools.plot_bot_3d(dset, 0, index, axes=axes, title=task, even_scale=True, visible_axes=False, func=func, cmap=cmap)
                    sim_time = file['scales/sim_time'][index]
                    paxes.arrow(0, 0, R*np.cos(ω*sim_time),  R*np.sin(ω*sim_time))
                    caxes.set_xlim(np.nanmin(dset), np.nanmax(dset))
                # Add time title
                title = title_func(file['scales/sim_time'][index])
                title_height = 1 - 0.5 * mfig.margin.top / mfig.fig.y
                fig.suptitle(title, x=0.4, y=title_height, ha='left')
                # Save figure
                savename = savename_func(file['scales/write_number'][index])
                savepath = output.joinpath(savename)
                fig.savefig(str(savepath), dpi=dpi)
                fig.clear()
        plt.close(fig)
        
    elif "profiles" in filename:
        
        tasks = ['Heat_flux_azimuth_average', 'Absolute_SS_error']
        from plotpal.profiles import RolledProfilePlotter
        plotter = RolledProfilePlotter(".", file_dir="profiles", out_name="instantaneous_profiles", roll_writes=10, start_file=1, n_files=None)
        plotter.setup_grid(num_rows=1, num_cols=2, col_inch=3, row_inch=2, pad_factor=15)
        plotter.add_line('ρ', 'Heat_flux_azimuth_average', grid_num=0)
        plotter.add_line('ρ', 'Absolute_SS_error', grid_num=1)
        plotter.plot_lines(dpi=dpi)

    else:
        
        tasks = ['Average_abs_error']
        from plotpal.scalars import ScalarFigure, ScalarPlotter
        figs = []

        # Nu vs time
        fig1 = ScalarFigure(num_rows=1, num_cols=1, col_inch=6, fig_name='Disk_average_abs_error')
        fig1.add_field(0, 'Average_abs_error', color='orange')
        figs.append(fig1)

        # Load in figures and make plots
        plotter = ScalarPlotter(".", file_dir="scalars", out_name="traces", start_file=1, n_files=None, roll_writes=None)
        plotter.load_figures(figs)
        plotter.plot_figures(dpi=200)
        plotter.plot_convergence_figures(dpi=200)

# Creating plots and snapshots/profiles/scalars
# -------------------------------------------------------------------------------------------------
if __name__ == "__main__":

    import pathlib
    import pandas as pd
    from docopt import docopt # Note: See SO, question #31901138 for docopt usage caveat
    from dedalus.tools import logging
    from dedalus.tools import post
    from dedalus.tools.parallel import Sync

    args = docopt(__doc__)

    output_path = pathlib.Path(args['--output']).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    post.visit_writes(args['<files>'], main, output=output_path, omega_value=float(args['--omega_value']))
    # -------------------------------------------------------------------------------------------------
    
    if "snapshots" in args['<files>'][0]:
    # Combining snapshots to an .mp4 movie via ffmpeg in serial
    # -------------------------------------------------------------------------------------------------
        (
            ffmpeg
            .input("frames/*.png", pattern_type="glob")
            .filter("fps", fps=10, round="down")
            .filter("scale", size="hd1080", force_original_aspect_ratio="increase")
            .output("frames/out.mp4", crf=20, pix_fmt="yuv420p")
            .run()
        )
    # -------------------------------------------------------------------------------------------------
    
    elif "profiles" in args['<files>'][0]:
    # Combining profiles to an .mp4 movie via ffmpeg in serial
    # -------------------------------------------------------------------------------------------------
        (
            ffmpeg
            .input("instantaneous_profiles/*.png", pattern_type="glob")
            .filter("fps", fps=10, round="down")
            .filter("scale", width='-2', height='720')
            .output("instantaneous_profiles/out.mp4", crf=20, pix_fmt="yuv420p")
            .run()
        )
    # -------------------------------------------------------------------------------------------------