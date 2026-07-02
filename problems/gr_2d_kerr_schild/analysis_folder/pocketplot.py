import numpy as np
import matplotlib
# matplotlib.rc("text", usetex=True)
matplotlib.rc("font", family="serif")
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

import pplotutil as pu

pu.register_hotcold_cmap(plt)


class PlotParams(object):

    def __init__(self):
        self.cmap = None
        self.norm = None
        self.xlim = None
        self.ylim = None
        self.vmin = None
        self.vmax = None
        self.fnameaddon = ""
        self.label_size = 20
        self.tick_size = 14
        self.xaxis_visible = True
        self.yaxis_visible = True
        self.title_visible = True


class PlotSpatialMap(object):
    
    def __init__(self, plotparams = None):
        if plotparams is None:
            plotparams = PlotParams()
        self.plotparams = plotparams
        self._Tmunu_prepped = False
        self.time = None

    def _get_Tmunu(self, localdata):
        if not self._Tmunu_prepped:
            self._Tmuunud = calc_Tmunu_pl(localdata) + calc_Tmunu_em(localdata)
            self._Tmunu_prepped = False  # Just always recalculate for now...
        return self._Tmuunud
    
#     def get_mosaic_pattern(self):
#         return [["Ttuphid", "Ttuphid_PL", "Ttuphid_EM"]]
    
    def get_axkey(self, rowidx, colidx):
        return [
            ["Ttuphid", "Ttuphid_PL", "Ttuphid_EM",],
            ["Ttutd", "Ttutd_PL", "Ttutd_EM",],
        ][rowidx][colidx]
    
    def get_mosaic(self):
        fig = plt.figure(figsize = (12.0, 10.0))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols = (2, 3),
            axes_pad = 0.0,
            share_all = True,
            cbar_location = "right",
            cbar_mode = "edge",
            cbar_size = "7%",
            cbar_pad = 0.35,
            label_mode = "1",
            direction = "column",
        )
        return fig, grid
        
    def plotthis(self, localdata, axkey = None):
        # 'axkey' should correspond to the axis on top of which you're plotting
        # as supplied in 'get_mosaic'
        # Plot total, plasma, and EM angular momentum density
        gsqrt = ks.gsqrt(localdata._rv, localdata._thetav, localdata.a)
        if axkey == "Ttuphid":
            return gsqrt * smoothField2d(self._get_Tmunu(localdata)[:,:,0,3])
        elif axkey == "Ttuphid_PL":
            return gsqrt * smoothField2d(calc_Tmunu_pl(localdata)[:,:,0,3])
        elif axkey == "Ttuphid_EM":
            return gsqrt * smoothField2d(calc_Tmunu_em(localdata)[:,:,0,3])
        elif axkey == "Ttutd":
            return -gsqrt * smoothField2d(self._get_Tmunu(localdata)[:,:,0,0])
        elif axkey == "Ttutd_PL":
            return -gsqrt * smoothField2d(calc_Tmunu_pl(localdata)[:,:,0,0])
        elif axkey == "Ttutd_EM":
            return -gsqrt * smoothField2d(calc_Tmunu_em(localdata)[:,:,0,0])
        else:
            print("I didn't recognize axkey: " + str(axkey))
            print("We may be about to experience an error")
            
    def textlabel(self, axkey = None):
        if axkey == "Ttuphid":
            return r"$\sqrt{-g} (T^t_\phi)_{\rm PL+EM}$"
        elif axkey == "Ttuphid_PL":
            return r"$\sqrt{-g} (T^t_\phi)_{\rm PL}$"
        elif axkey == "Ttuphid_EM":
            return r"$\sqrt{-g} (T^t_\phi)_{\rm EM}$"
        elif axkey == "Ttutd":
            return r"$-\sqrt{-g} (T^t_t)_{\rm PL+EM}$"
        elif axkey == "Ttutd_PL":
            return r"$-\sqrt{-g} (T^t_t)_{\rm PL}$"
        elif axkey == "Ttutd_EM":
            return r"$-\sqrt{-g} (T^t_t)_{\rm EM}$"
        else:
            return ""

    def plottitle(self, axkey = None):
        if axkey == "Ttuphid_PL":
            return r"$t = %.1fr_g/c$" % self.get_time()
        else:
            return ""
        
    def make_colorbar(self, fig, axdict, mapdict):
        cb1 = axdict["Ttuphid_EM"].cax.colorbar(mapdict["Ttuphid_EM"])
        cb2 = axdict["Ttutd_EM"].cax.colorbar(mapdict["Ttutd_EM"])
        return cb1, cb2
        # return axdict[-1].cax.colorbar(mapdict[list(mapdict.keys())[-1]])
    
    def fnamelead(self):
        return "EnAngMomDens"
    
    def fnamebase(self):
        return self.fnamelead() + self.plotparams.fnameaddon

    def _default_cmap(self, axkey = None):
        return "inferno"
    
    def cmap(self, axkey = None):
        if self.plotparams.cmap is None:
            mycmap = plt.get_cmap(self._default_cmap(axkey = axkey))
            mycmap.set_bad((0,0,0))
            self.plotparams.cmap = mycmap
        return self.plotparams.cmap
    
    def _norm_dispatch(self, vmin, vmax, axkey = None):
        return colors.LogNorm(vmin, vmax)
    
    def _default_vminmax(self, axkey = None):
        if axkey.startswith("Ttuphid"):
            return 1e3, 1e6
        else:
            return 1e3, 3e5
    
    def norm(self, axkey = None):
        if self.plotparams.norm is None:
            vmin, vmax = self._default_vminmax(axkey = axkey)
            if self.plotparams.vmin is not None:
                vmin = self.plotparams.vmin
            if self.plotparams.vmax is not None:
                vmax = self.plotparams.vmax
            norm = \
                self._norm_dispatch(vmin, vmax, axkey = axkey)
            return norm
        return self.plotparams.norm
    
    def xlim(self, data, axkey = None):
        if self.plotparams.xlim is None:
            self.plotparams.xlim = (0, np.exp(data.conf["size"][0]))
        return self.plotparams.xlim
    
    def ylim(self, data, axkey = None):
        if self.plotparams.ylim is None:
            xmin, xmax = self.xlim(data)
            self.plotparams.ylim = (-0.5*xmax, 0.5*xmax)
        return self.plotparams.ylim
    
    def xlabel(self, axkey = None):
        if axkey == "Ttuphid_PL":
            return r"$\rho/r_g$"
        else:
            return ""
    
    def ylabel(self, axkey = None):
        if axkey == "Ttuphid":
            return r"$z/r_g$"
        else:
            return ""
    
    def get_time(self):
        return self.time
    
    def set_time(self, step, localdata):
        time = step * localdata.conf["fld_output_interval"] * localdata.conf['dt']
        self.time = time
        localdata.load(step)
        
    def plot_contours(self, ax, localdata):
        Bp = localdata.conf["Bp"]
        clevels = np.linspace(0.0, 100, 101)**2 * Bp / 2.0
        return ax.contour(
            localdata.x1, localdata.x2, localdata.fluxB,
            clevels,
            colors='green',
            linewidths=0.5
        )
    
    def add_BH_to(self, ax, localdata):
        a = localdata.a

        rH = 1 + np.sqrt(1.0 - a*a)

        th_ergo = np.linspace(0.0, np.pi, 256)
        r_ergo = np.sqrt(1.0 - a**2 * np.cos(th_ergo)**2) + 1.0
        x_ergo = r_ergo * np.sin(th_ergo)
        z_ergo = r_ergo * np.cos(th_ergo)
        
        bh_horizon = plt.Circle((0.0, 0.0), rH, color='k',ec='w',zorder=10)
        ax.add_artist(bh_horizon)
        ax.plot(x_ergo, z_ergo, 'w--')
    
    def make_plot(self, step, localdata, textlabel = None, verbose = False):
        self.set_time(step, localdata)

        fig, axs = self.get_mosaic()
        
        tick_size = self.plotparams.tick_size
        label_size = self.plotparams.label_size
        
        a = localdata.a

        rH = 1 + np.sqrt(1.0 - a*a)
        
        mappables = {}
        axdict    = {}
        
        for i in range(len(axs)):
            # ax = axdict[key]
            nrows = len(axs.axes_column)
            rowidx = i // nrows
            colidx = i %  nrows
            ax = axs.axes_row[rowidx][colidx]
            key = self.get_axkey(rowidx, colidx)
            axdict[key] = ax
            
            # print("row, col, key = " + str((rowidx, colidx, key)))
            
            norm = self.norm(axkey = key)
            cmap = self.cmap(axkey = key)
        
            plotthis = self.plotthis(localdata, key)
            if verbose:
                lo, hi = plotthis.min(), plotthis.max()
                print(key, lo, hi)
                if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
                    lo = plotthis[~np.isnan(plotthis) & ~np.isinf(plotthis)].min()
                    hi = plotthis[~np.isnan(plotthis) & ~np.isinf(plotthis)].max()
                    print("Ignoring NaNs and Infs:", key, lo, hi)

            try:
                plot = ax.pcolormesh(localdata.x1, localdata.x2, plotthis, cmap=cmap,
                    shading = "nearest", norm=norm, rasterized=True)
            except:
                import pdb; pdb.set_trace()
            mappables[key] = plot
                
            title = self.plottitle(key)
            if title is not None:
                ax.set_title(title, fontsize=label_size)

            ax.set_aspect('equal')
            ax.set_xlim(*self.xlim(localdata, key))
            ax.set_ylim(*self.ylim(localdata, key))
            ax.tick_params(labelsize=tick_size)
            ax.set_xlabel(self.xlabel(key), fontsize=label_size)
            ax.set_ylabel(self.ylabel(key), fontsize=label_size)

            self.plot_contours(ax, localdata)

            textlabel = self.textlabel(key)
            if textlabel is not None:
                xmin, xmax = self.xlim(localdata, key)
                ymin, ymax = self.ylim(localdata, key)
                xbuf = 0.03 * (xmax - xmin)
                ybuf = 0.03 * (ymax - ymin)
                buffer = max(xbuf, ybuf)
                ax.text(
                    xmax - buffer,
                    ymax - buffer,
                    textlabel,
                    fontsize=label_size,
                    # transform=ax.transAxes,
                    transform=ax.transData,
                    ha = "right", va = "top",
                    bbox=dict(
                        boxstyle="round",
                        fc="w", ec="k",
                        alpha = 0.8,
                    ),
                )
            
            # bh_horizon = plt.Circle((0.0, 0.0), rH, color='k',ec='w',zorder=10)
            # ax.add_artist(bh_horizon)
            # ax.plot(x_ergo, z_ergo, 'w--')
            self.add_BH_to(ax, localdata)
        
        self.make_colorbar(fig, axdict, mappables)
            
        return fig


# Reflect about z-axis to create the impression of a fuller plot
class Plot2PanelMap(object):
    
    def __init__(self, plotparams = None):
        if plotparams is None:
            plotparams = PlotParams()
        self.plotparams = plotparams
        self._Tmunu_prepped = False
        self.time = None

    def _get_Tmunu(self, localdata):
        if not self._Tmunu_prepped:
            self._Tmuunud = calc_Tmunu_pl(localdata) + calc_Tmunu_em(localdata)
            self._Tmunu_prepped = False  # Just always recalculate for now...
        return self._Tmuunud
    
#     def get_mosaic_pattern(self):
#         return [["Ttuphid", "Ttuphid_PL", "Ttuphid_EM"]]
    
    # idx == 0 <--> left side
    # idx == 1 <--> right side
    def get_axkey(self, idx):
        return [
            "Hphid",
            "Ndens",
        ][idx]
    
    def get_mosaic(self):
        fig = plt.figure(figsize = (12.0, 10.0))
        return fig, fig.gca()
        
    def plotthis(self, localdata, axkey = None):
        # 'axkey' should correspond to the axis on top of which you're plotting
        # as supplied in 'get_mosaic'
        # Plot total, plasma, and EM angular momentum density
        # gsqrt = ks.gsqrt(localdata._rv, localdata._thetav, localdata.a)
        if axkey == "Hphid":
            return localdata.Hd3 / \
                (localdata._rv * np.sin(localdata._thetav)) / \
                localdata.conf["Bp"]
        elif axkey == "Ndens":
            return (-localdata.Rho_e + localdata.Rho_p) / \
                localdata.conf["torus_max_density"]
        else:
            print("I didn't recognize axkey: " + str(axkey))
            print("We may be about to experience an error")
            
    def textlabel(self, axkey = None):
        if axkey == "Hphid":
            return r"$H_\phi / (r \sin \theta) / B_0$"
        elif axkey == "Ndens":
            return r"$n_{\rm tot.} / n_0$"
        else:
            return ""

    def plottitle(self):
        return r"$t = %.1fr_g/c$" % self.get_time()
        
    def make_colorbar(self, fig, ax, mapdict):
        ax_divider = make_axes_locatable(ax)     
        cax1 = ax_divider.append_axes("left", size="5%", pad="0%")
        cax2 = ax_divider.append_axes("right", size="5%", pad="0%")
        cb1 = fig.colorbar(mapdict["Hphid"], cax=cax1)
        cb2 = fig.colorbar(mapdict["Ndens"], cax=cax2)
        cb1.ax.yaxis.set_ticks_position("left")
        cb1.ax.tick_params(labelsize=self.plotparams.tick_size)
        cb2.ax.tick_params(labelsize=self.plotparams.tick_size)
        return cb1, cb2
        # return axdict[-1].cax.colorbar(mapdict[list(mapdict.keys())[-1]])
    
    def fnamelead(self):
        return "Hphid-Ndens"
    
    def fnamebase(self):
        return self.fnamelead() + self.plotparams.fnameaddon

    def _default_cmap(self, axkey = None):
        if axkey == "Hphid":
            return "hot_and_cold"
        elif axkey == "Ndens":
            return "inferno"
    
    def cmap(self, axkey = None):
        if self.plotparams.cmap is None:
            mycmap = plt.get_cmap(self._default_cmap(axkey = axkey))
            mycmap.set_bad((0,0,0))
            return mycmap
        return self.plotparams.cmap
    
    def _norm_dispatch(self, vmin, vmax, axkey = None):
        if axkey == "Hphid":
            return colors.Normalize(vmin, vmax)
        elif axkey == "Ndens":
            return colors.LogNorm(vmin, vmax)
    
    def _default_vminmax(self, axkey = None):
        if axkey == "Hphid":
            return -10.0, 10.0
        elif axkey == "Ndens":
            return 1e-4, 1e0
    
    def norm(self, axkey = None):
        if self.plotparams.norm is None:
            vmin, vmax = self._default_vminmax(axkey = axkey)
            if self.plotparams.vmin is not None:
                vmin = self.plotparams.vmin
            if self.plotparams.vmax is not None:
                vmax = self.plotparams.vmax
            norm = \
                self._norm_dispatch(vmin, vmax, axkey = axkey)
            return norm
        return self.plotparams.norm
    
    def xlim(self, data, axkey = None):
        if self.plotparams.xlim is None:
            xmax = np.exp(data.conf["size"][0])
            self.plotparams.xlim = (-xmax, xmax)
        return self.plotparams.xlim
    
    def ylim(self, data, axkey = None):
        if self.plotparams.ylim is None:
            xmin, xmax = self.xlim(data)
            self.plotparams.ylim = (-xmax, xmax)
        return self.plotparams.ylim
    
    def xlabel(self, axkey = None):
        return r"$x / r_g$"
    
    def ylabel(self, axkey = None):
        return r"$z / r_g$"
    
    def get_time(self):
        return self.time
    
    def set_time(self, step, localdata):
        time = step * localdata.conf["fld_output_interval"] * localdata.conf['dt']
        self.time = time
        localdata.load(step)
        
    def plot_contours(self, ax, localdata):
        Bp = localdata.conf["Bp"]
        # clevels = np.linspace(0.0, 100, 101)**2 * Bp / 2.0
        clevels = np.linspace(1.0, 100, 100)**2 * Bp / 2.0
        c1 = ax.contour(
            localdata.x1, localdata.x2, localdata.fluxB,
            clevels,
            colors='green',
            linewidths=0.5
        )
        c2 = ax.contour(
            -localdata.x1, localdata.x2, localdata.fluxB,
            clevels,
            colors='green',
            linewidths=0.5
        )
        return c1, c2
    
    def add_BH_to(self, ax, localdata):
        a = localdata.a

        rH = 1 + np.sqrt(1.0 - a*a)

        th_ergo = np.linspace(0.0, np.pi, 256)
        r_ergo = np.sqrt(1.0 - a**2 * np.cos(th_ergo)**2) + 1.0
        x_ergo = r_ergo * np.sin(th_ergo)
        z_ergo = r_ergo * np.cos(th_ergo)
        
        bh_horizon = plt.Circle((0.0, 0.0), rH, color='k',ec='w',zorder=10)
        ax.add_artist(bh_horizon)
        ax.plot( x_ergo, z_ergo, 'w--')
        ax.plot(-x_ergo, z_ergo, 'w--')

    def get_xydata(self, localdata):
        xdata = localdata.x1
        ydata = localdata.x2
        return xdata, ydata
    
    def make_plot(self, step, localdata, textlabel = None, verbose = False):
        self.set_time(step, localdata)

        fig, ax = self.get_mosaic()
        
        tick_size = self.plotparams.tick_size
        label_size = self.plotparams.label_size
        
        mappables = {}
        
        for i in range(2):
            # ax = axdict[key]
            key = self.get_axkey(i)
            
            # print("row, col, key = " + str((rowidx, colidx, key)))
            
            norm = self.norm(axkey = key)
            cmap = self.cmap(axkey = key)
        
            plotthis = self.plotthis(localdata, key)
            if verbose:
                lo, hi = plotthis.min(), plotthis.max()
                print(key, lo, hi)
                if np.isnan(lo) or np.isnan(hi) or np.isinf(lo) or np.isinf(hi):
                    lo = plotthis[~np.isnan(plotthis) & ~np.isinf(plotthis)].min()
                    hi = plotthis[~np.isnan(plotthis) & ~np.isinf(plotthis)].max()
                    print("Ignoring NaNs and Infs:", key, lo, hi)

            try:
                xdata, ydata = self.get_xydata(localdata)
                if i == 0:
                    xdata = -xdata
                plot = ax.pcolormesh(xdata, ydata, plotthis, cmap=cmap,
                    shading = "nearest", norm=norm, rasterized=True)
            except:
                import pdb; pdb.set_trace()
            mappables[key] = plot
                
            if self.plotparams.title_visible:
                title = self.plottitle()
                if title is not None:
                    ax.set_title(title, fontsize=label_size)

            ax.set_aspect('equal')
            ax.set_xlim(*self.xlim(localdata, key))
            ax.set_ylim(*self.ylim(localdata, key))
            ax.tick_params(labelsize=tick_size)
            ax.set_xlabel(self.xlabel(key), fontsize=label_size)
            ax.set_ylabel(self.ylabel(key), fontsize=label_size)

            if not self.plotparams.xaxis_visible:
                ax.axes.get_xaxis().set_visible(False)
            # if not self.plotparams.yaxis_visible:

            ax.axes.get_yaxis().set_visible(False)

            self.plot_contours(ax, localdata)

            textlabel = self.textlabel(key)
            if textlabel is not None:
                xmin, xmax = self.xlim(localdata, key)
                ymin, ymax = self.ylim(localdata, key)
                xbuf = 0.03 * (xmax - xmin)
                ybuf = 0.03 * (ymax - ymin)
                buffer = max(xbuf, ybuf)
                xpos = xmax - buffer
                ypos = ymax - buffer
                ha = "right"
                if i == 0:
                    ha = "left"
                    xpos = xmin + buffer
                ax.text(
                    xpos, ypos,
                    textlabel,
                    fontsize=label_size,
                    # transform=ax.transAxes,
                    transform=ax.transData,
                    ha = ha, va = "top",
                    bbox=dict(
                        boxstyle="round",
                        fc="w", ec="k",
                        alpha = 0.8,
                    ),
                )
            
            # bh_horizon = plt.Circle((0.0, 0.0), rH, color='k',ec='w',zorder=10)
            # ax.add_artist(bh_horizon)
            # ax.plot(x_ergo, z_ergo, 'w--')
            self.add_BH_to(ax, localdata)
        
        self.make_colorbar(fig, ax, mappables)
            
        return fig


class PlotSpacetime(object):

    def __init__(self, plotparams = None):
        if plotparams is None:
            plotparams = PlotParams()
        self.plotparams = plotparams
    
    def get_mosaic(self):    
        fig = plt.figure(figsize = (12.0, 10.0))
        grid = ImageGrid(
            fig, 111,
            nrows_ncols = (1, 3),
            axes_pad = 0.0,
            share_all = True,
            cbar_location = "right",
            cbar_mode = "edge",
            cbar_size = "7%",
            cbar_pad = 0.35,
            label_mode = "1",
            direction = "column",
            aspect = False,
        )
        return fig, grid
    
    def getkeys(self):
        return ["Ntu", "Ttutd", "Ttuphid"]
        # return ["Ttutd"]
    
    def plotthis(self, key, data, rtdata):
        r = rtdata["r"]
        rh = 1.0 + np.sqrt(1.0 - data.a**2)
        rmid = 0.5 * (r[1:] + r[:-1])
        if key == "Ntu":
            plotthis = np.diff(rtdata[key], axis=1) / np.diff(r)[np.newaxis,:]
        elif key == "Ttutd":
            raw = rtdata["Ttutd_EM"] + rtdata["Ttutd_PL"]
            plotthis = np.diff(raw, axis = 1) / np.diff(r)[np.newaxis,:]
        elif key == "Ttuphid":
            raw = rtdata["Ttuphid_EM"] + rtdata["Ttuphid_PL"]
            plotthis = np.diff(raw, axis = 1) / np.diff(r)[np.newaxis,:]
        return plotthis / np.max(plotthis[:, rmid>1.5*rh])
    
    def textlabel(self, key):
        if key == "Ntu":
            return r"$2\pi \int_0^\pi N^t(r,\theta) \sqrt{-g}\, d \theta$"
        elif key == "Ttutd":
            return r"$-2\pi \int_0^\pi T^t_t(r,\theta) \sqrt{-g}\, d \theta$"
        elif key == "Ttuphid":
            return r"$2\pi \int_0^\pi T^t_\phi(r,\theta) \sqrt{-g}\, d \theta$"
        else:
            print("Did not recognize key: " + key)
            sys.exit(0)
            
    def get_title(self, key, data, rtdata):
        return None
            
    def get_rt(self, key, r, t):
        rmid = 0.5 * (r[1:] + r[:-1])
        return rmid, t
    
    def get_cmap(self, key):
        mycmap = plt.get_cmap("inferno")
        mycmap.set_bad((0,0,0))
        return mycmap
    
    def get_norm(self, key):
        return colors.LogNorm(1e-2, 1e0)
    
    def make_colorbar(self, axdict, mappables):
        key = self.getkeys()[-1]
        return axdict[key].cax.colorbar(mappables[key])

    def make_plot(self, data, rtdata,
        longways = False):
        fig, grid = self.get_mosaic()
        label_size = self.plotparams.label_size
        tick_size = self.plotparams.tick_size
        mappables = {}
        axdict    = {}
        r = rtdata["r"]
        t = rtdata["t"]
        for ax, key in zip(grid, self.getkeys()):
            print("Making plot for key: " + key)
            ax.tick_params(labelsize=tick_size)
            cmap = self.get_cmap(key)
            norm = self.get_norm(key)
            plotthis = self.plotthis(key, data, rtdata)
            r_plot, t_plot = self.get_rt(key, r, t)

            xcoord, ycoord = r_plot, t_plot
            xlab = r"$r$" 
            ylab = r"$ct/r_g$" 
            if longways:
                xcoord, ycoord = t_plot, r_plot
                plotthis = np.transpose(plotthis)
                xlab, ylab = ylab, xlab

            ax.set_xlabel(xlab, fontsize=label_size)
            ax.set_ylabel(ylab, fontsize=label_size)

            pm = ax.pcolormesh(
                xcoord, ycoord, plotthis,
                cmap = cmap,
                norm = norm,
            )
            xmin, xmax = xcoord[0], xcoord[-1]
            ymin, ymax = ycoord[0], ycoord[-1]
            xbuf = 0.03 * (xmax - xmin)
            ybuf = 0.03 * (ymax - ymin)
            if longways:
                ybuf = 0.05 * (ymax - ymin)
            textlabel = self.textlabel(key)
            if textlabel is not None:
                ax.text(
                    xmax - xbuf,
                    ymax - ybuf,
                    textlabel,
                    fontsize=label_size,
                    # transform=ax.transAxes,
                    transform=ax.transData,
                    ha = "right", va = "top",
                    bbox=dict(
                        boxstyle="round",
                        fc="w", ec="k",
                        alpha = 0.8,
                    ),
                )
            
            title = self.get_title(key, data, rtdata)
            if title is not None:
                ax.set_title(title, fontsize=label_size)
                
            mappables[key] = pm
            axdict[key] = ax
                
        self.make_colorbar(axdict, mappables)
            
        return fig
