import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

from pocketplot import PlotParams, Plot2PanelMap
from stress_rt_reduction_to_npz import load_rt_data

import datalib_gr_ks as ks

import sys
import os

import pplotutil as pu

plt.rc("text", usetex=True)

class Plot2PanelBetaNdens(Plot2PanelMap):

    def get_axkey(self, idx):
        return [
            "Beta_pl",
            "Ndens",
        ][idx]

    def plotthis(self, localdata, axkey = None):
        # 'axkey' should correspond to the axis on top of which you're plotting
        # as supplied in 'get_mosaic'
        if axkey == "Beta_pl":
            beta = localdata.plasma_beta
            idxreg = np.where(~np.isnan(beta) & ~np.isinf(beta))
            maxb = np.max(beta[idxreg])
            R = localdata._rv * np.sin(localdata._thetav)
            idxreg = np.where(~np.isnan(beta) & ~np.isinf(beta) & (beta > 0))
            maxr = np.max(R[idxreg])
            print("Max beta = %.3g" % maxb)
            print("Max torus R = %.3g" % maxr)
            return localdata.plasma_beta
        elif axkey == "Ndens":
            return (-localdata.Rho_e + localdata.Rho_p) / \
                localdata.conf["torus_max_density"]
        else:
            print("I didn't recognize axkey: " + str(axkey))
            print("We may be about to experience an error")

    def textlabel(self, axkey = None):
        if axkey == "Beta_pl":
            return r"$\beta_{\rm plasma}$"
        elif axkey == "Ndens":
            return r"$n_{\rm tot.} / n_0$"
        else:
            return ""

    def _default_cmap(self, axkey = None):
        if axkey == "Beta_pl":
            return "hot_and_cold"
        elif axkey == "Ndens":
            return "inferno"

    def make_colorbar(self, fig, ax, mapdict):
        ax_divider = make_axes_locatable(ax)     
        cax1 = ax_divider.append_axes("left", size="5%", pad="0%")
        cax2 = ax_divider.append_axes("right", size="5%", pad="0%")
        cb1 = fig.colorbar(mapdict["Beta_pl"], cax=cax1)
        cb2 = fig.colorbar(mapdict["Ndens"], cax=cax2)
        cb1.ax.yaxis.set_ticks_position("left")
        cb1.ax.tick_params(labelsize=self.plotparams.tick_size)
        cb2.ax.tick_params(labelsize=self.plotparams.tick_size)
        return cb1, cb2

    def fnamelead(self):
        return "Betapl-Ndens"

    def _norm_dispatch(self, vmin, vmax, axkey = None):
        if axkey == "Beta_pl":
            return colors.LogNorm(vmin, vmax)
        elif axkey == "Ndens":
            return colors.LogNorm(vmin, vmax)

    def _default_vminmax(self, axkey = None):
        if axkey == "Beta_pl":
            return 0.01, 100
        elif axkey == "Ndens":
            return 1e-4, 1e0

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("usage: mpirun -np <nproc> python %s <aperture data dir> [it0 [it1 [step]]]" % sys.argv[0])
        print("  Generate plots in parallel using <nproc> processes")
        print("  If it0 given, process dumps up through dump it0 [exclusive]")
        print("  If it0 and it1 given, process dumps it0 [inclusive] to it1 [exclusive]")
        print("  If step given, process every <step> dumps from it0 [inclusive] to it1 [exclusive]") 
        print("  cinematic = False(default)/True -- true suppresses labels")
        print("    for a more film-like plot (e.g., for movies for public)")
        print("  savedir = ./ (default) -- directory where plots saved")
        print("  xmax/ymax = 15/10 (default)")
        print("    The max data limits of the x/y-axes of the plot")
        print("    The min data limtes are always -xmax and -ymax")
        sys.exit(0)

    args, kwargs = pu.getArgsAndKwargs(sys.argv[1:])
    # aperture_dir, step = args
    # step = int(step)
    aperture_dir = args[0]

    data = ks.DataKerrSchild(aperture_dir)
    steps = data.fld_steps

    it0 = 0
    it1 = len(steps)
    step = 1
    if len(args) == 2:
        it1 = int(args[1])
    if len(args) == 3:
        it0 = int(args[1])
        it1 = int(args[2])
    elif len(args) == 4:
        it0 = int(args[1])
        it1 = int(args[2])
        step = int(args[3])

    steps = data.fld_steps[it0:it1:step]

    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    mysteps = steps[rank::size]

    if rank == 0:
        print("Generating %d plots on %d processes." % (len(steps), size))

    cinematic = False
    if "cinematic" in kwargs:
        cinematic = kwargs["cinematic"]
    xmax = 15
    if "xmax" in kwargs:
        xmax = kwargs["xmax"]
    ymax = 0.75 * xmax
    if "ymax" in kwargs:
        ymax = kwargs["ymax"]

    plotparams = PlotParams()
    plotparams.xlim = (-xmax, xmax)
    plotparams.ylim = (-ymax, ymax)
    plotparams.label_size = 22
    plotparams.tick_size = 18
    if cinematic:
        plotparams.xaxis_visible = False
        plotparams.title_visible = False
    # plotparams.label_size = 12
    plotter = Plot2PanelBetaNdens(plotparams)

    for step in mysteps:  # Overriding 'step' variable here
        print("Working at dump %d" % step)
        fig = plotter.make_plot(step, data)

        fname = plotter.fnamebase() + "_" + str(step) + ".png"
        directory = "."
        if "savedir" in kwargs:
            directory = kwargs["savedir"]
            fname = os.path.join(directory, fname)
        fig.savefig(
            fname,
            bbox_inches="tight",
            dpi=300,
        )

    if rank == 0:
        directory = "."
        if "savedir" in kwargs:
            directory = kwargs["savedir"]
        pu.output_plot_command(
            os.path.join(directory, "chainplot_phid_ndens_cmd.txt")
        )

