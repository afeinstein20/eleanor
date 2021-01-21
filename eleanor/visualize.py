import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import warnings, os, requests
import lightkurve as lk
from bs4 import BeautifulSoup
from pylab import *
from astropy.timeseries import LombScargle
from astropy.wcs import WCS
import astropy.units as u
from astropy.coordinates import SkyCoord, Angle

from .ffi import use_pointing_model, load_pointing_model
from .mast import *

__all__ = []


class Visualize(object):
    """
    The main class for creating figures, movies, and interactive plots.
    Allows the user to have a grand ole time playing with their data!

    Parameters
    ----------
    obj :
        Object must have minimum attributes of 2D array of flux.
        Will allow for plotting of both postcards & tpfs.
    obj_type :
        Object type can be set to "tpf" or "postcard". Default is "tpf".
    """

    def __init__(self, object, obj_type="tpf"):
        self.obj      = object
        self.obj_type = obj_type.lower()

        if self.obj_type == "tpf":
            self.flux   = self.obj.tpf
            self.center = (np.nanmedian(self.obj.centroid_xs),
                             np.nanmedian(self.obj.centroid_ys))
            self.dimensions = self.obj.tpf[0].shape
        else:
            self.flux   = self.obj.flux
            self.center = self.obj.center_xy
            self.dimensions = self.obj.dimensions


    def aperture_contour(self, aperture=None, ap_color='w', ap_linewidth=4, ax=None, **kwargs):
        """
        Overplots the countour of an aperture on a target pixel file.
        Contribution from Gijs Mulders.

        Parameters
        ----------
        aperture : np.2darray, optional
            A 2D mask the same size as the target pixel file. Default
            is the eleanor default aperture.
        ap_color : str, optional
            The color of the aperture contour. Takes a matplotlib color.
            Default is red.
        ap_linewidth : int, optional
            The linewidth of the aperture contour. Default is 4.
        ax : matplotlib.axes._subplots.AxesSubplot, optional
            Axes to plot on.
        """
        if ax == None:
            fig, ax = plt.subplots(nrows=1)

        if aperture is None:
            aperture = self.obj.aperture

        ax.imshow(self.obj.tpf[0], **kwargs)

        f = lambda x,y: aperture[int(y),int(x) ]
        g = np.vectorize(f)

        x = np.linspace(0,aperture.shape[1], aperture.shape[1]*100)
        y = np.linspace(0,aperture.shape[0], aperture.shape[0]*100)
        X, Y= np.meshgrid(x[:-1],y[:-1])
        Z = g(X[:-1],Y[:-1])

        ax.contour(Z, [0.05], colors=ap_color, linewidths=[ap_linewidth],
                    extent=[0-0.5, x[:-1].max()-0.5,0-0.5, y[:-1].max()-0.5])

        if ax == None:
            return fig



    def pixel_by_pixel(self, colrange=None, rowrange=None, cmap='viridis',
                       data_type="corrected", mask=None, xlim=None,
                       ylim=None, color_by_pixel=False, color_by_aperture=True,
                       freq_range=[1/20., 1/0.1], aperture=None, ap_color='r',
                       ap_linewidth=2):
        """
        Creates a pixel-by-pixel light curve using the corrected flux.
        Contribution from Oliver Hall.

        Parameters
        ----------
        colrange : np.array, optional
             A list of start column and end column you're interested in
             zooming in on.
        rowrange : np.array, optional
             A list of start row and end row you're interested in zooming
             in on.
        cmap : str, optional
             Name of a matplotlib colormap. Default is 'viridis'.
        data_type : str, optional
             The type of flux used. Either: 'raw', 'corrected', 'amplitude',
             or 'periodogram'. If not, default set to 'corrected'.
        mask : np.array, optional
             Specifies the cadences used in the light curve. If not, default
             set to good quality cadences.
        xlim : np.array, optional
             Specifies the xlim on the subplots. If not, default is set to
             the entire light curve.
        ylim : np.array, optional
             Specifies the ylim on the subplots, If not, default is set to
             the entire light curve flux range.
        color_by_pixel : bool, optional
             Colors the light curve given the color of the pixel. If not,
             default is set to False.
        freq_range : list, optional
             List of minimum and maximum frequency to search in Lomb Scargle
             periodogram. Only used if data_type = 'periodogram'. If None,
             default = [1/20., 1/0.1].
        """
        if self.obj.lite:
            print('This is an eleanor-lite object. No pixel_by_pixel visualization can be created.')
            print('Please create a regular eleanor.TargetData object (lite=False) to use this tool.')
            return

        if colrange is None:
            colrange = [0, self.dimensions[1]]

        if rowrange is None:
            rowrange = [0, self.dimensions[0]]

        nrows = int(np.round(colrange[1]-colrange[0]))
        ncols = int(np.round(rowrange[1]-rowrange[0]))

        if (colrange[1] > self.dimensions[1]) or (rowrange[1] > self.dimensions[0]):
            raise ValueError("Asking for more pixels than available in the TPF.")


        figure = plt.figure(figsize=(20,8))
        outer = gridspec.GridSpec(1,2, width_ratios=[1,4])

        inner = gridspec.GridSpecFromSubplotSpec(ncols, nrows, hspace=0.1, wspace=0.1,
                                                 subplot_spec=outer[1])

        i, j = rowrange[0], colrange[0]

        if mask is None:
            q = self.obj.quality == 0
        else:
            q = mask == 0


        ## PLOTS TARGET PIXEL FILE ##

        ax = plt.subplot(outer[0])

        if aperture is None:
            aperture = self.obj.aperture

        plotflux = np.nanmedian(self.flux[:, rowrange[0]:rowrange[1],
                                          colrange[0]:colrange[1]], axis=0)
        c = ax.imshow(plotflux,
                      vmax=np.percentile(plotflux, 95),
                      cmap=cmap)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.15)
        plt.colorbar(c, cax=cax, orientation='vertical')

        f = lambda x,y: aperture[int(y),int(x) ]
        g = np.vectorize(f)

        x = np.linspace(colrange[0],colrange[1], nrows*100)
        y = np.linspace(rowrange[0],rowrange[1], ncols*100)
        X, Y= np.meshgrid(x[:-1],y[:-1])
        Z = g(X[:-1],Y[:-1])

        ax.contour(Z, [0.05],
                   colors=ap_color, linewidths=[ap_linewidth],
                    extent=[0-0.5, nrows-0.5,0-0.5, ncols-0.5])

        ## PLOTS PIXEL LIGHT CURVES ##
        for ind in range( int(nrows * ncols) ):
            if ind == 0:
                ax = plt.Subplot(figure, inner[ind])
                origax = ax
            else:
                ax = plt.Subplot(figure, inner[ind], sharex=origax)

            flux = self.flux[:,i,j]
            time = self.obj.time
            corr_flux = self.obj.corrected_flux(flux=flux)

            if data_type.lower() == 'corrected':
                y = corr_flux[q]/np.nanmedian(corr_flux[q])
                x = time[q]

            elif data_type.lower() == 'amplitude':
                lc = lk.LightCurve(time=time, flux=corr_flux)
                pg = lc.normalize().to_periodogram()
                x = pg.frequency.value
                y = pg.power.value

            elif data_type.lower() == 'raw':
                y = flux[q]/np.nanmedian(flux[q])
                x = time[q]

            elif data_type.lower() == 'periodogram':
                freq, power = LombScargle(time, corr_flux).autopower(minimum_frequency=freq_range[0],
                                                                     maximum_frequency=freq_range[1],
                                                                     method='fast')
                y = power
                x = 1/freq

            if color_by_pixel is False:
                color = 'k'
            else:
                rgb = c.cmap(c.norm(self.flux[100,i,j]))
                color = matplotlib.colors.rgb2hex(rgb)

            ax.plot(x, y, c=color)

            if color_by_aperture and aperture[i,j] > 0:
                for iax in ['top', 'bottom', 'left', 'right']:
                    ax.spines[iax].set_color(ap_color)
                    ax.spines[iax].set_linewidth(ap_linewidth)

            j += 1
            if j == colrange[1]:
                i += 1
                j  = colrange[0]

            if ylim is None:
                ax.set_ylim(np.percentile(y, 1), np.percentile(y, 99))
            else:
                ax.set_ylim(ylim[0], ylim[1])

            if xlim is None:
                ax.set_xlim(np.min(x)-0.1, np.max(x)+0.1)
            else:
                ax.set_xlim(xlim[0], xlim[1])

            if data_type.lower() == 'amplitude':
                ax.set_yscale('log')
                ax.set_xscale('log')
                ax.set_ylim(y.min(), y.max())
                ax.set_xlim(np.min(x),
                            np.max(x))

            ax.set_xticks([])
            ax.set_yticks([])

            figure.add_subplot(ax)

        return figure


    def tess_the_movie(self):
        """
        Opens the link to Ethan Kruse's TESS: The Movie YouTube videos for
        the sector your target is observed in.

        Parameters
        ----------

        Attributes
        ----------
        movie_url : str

        """
        def type_of_script():
            try:
                ipy_str = str(type(get_ipython()))
                if 'zmqshell' in ipy_str:
                    return 'jupyter'
                if 'terminal' in ipy_str:
                    return 'ipython'
            except:
                return 'terminal'

        sector = self.obj.source_info.sector

        base="https://www.youtube.com/results?search_query="
        query="TESS+the+movie+sector+{0}+ethankruse".format(sector)

        soup = BeautifulSoup(requests.get(base+query).text, "html.parser").find_all('script')[26]

        items = soup.text
        items = items.split('\n')[1].split('title')

        good_sector=0

        for subitem in items:
            j = subitem.find('Sector')
            if j > 0 and 'TESS: The Movie' in subitem:

                sect = subitem[j:j+100].split(',')[0].split(' ')[-1]

                if int(sect) == int(sector):
                    i = subitem.find('/watch?v')
                    ext = subitem[i:i+100].split('"')[0]
                    good_sector=1
                    break

        if good_sector == 1:
            self.movie_url = 'https://www.youtube.com{0}'.format(ext)

            call_location = type_of_script()

            if (call_location == 'terminal') or (call_location == 'ipython'):
                os.system('python -m webbrowser -t "{0}"'.format(self.movie_url))

            elif (call_location == 'jupyter'):
                from IPython.display import YouTubeVideo
                id = self.movie_url.split('=')[-1]
                return YouTubeVideo(id=id, width=900, height=500)

        else:
            print('No movie is available yet.')
            return


    def plot_gaia_overlay(self, tic=None, tpf=None, magnitude_limit=18):
        """Check if the source is contaminated."""

        if tic is None:
            tic = self.obj.source_info.tic

        if tpf is None:
            tpf = lk.search_tesscut(f'TIC {tic}')[0].download(cutout_size=(self.obj.tpf.shape[1],
                                                                           self.obj.tpf.shape[2]))

        fig = tpf.plot(show_colorbar=False, title='TIC {0}'.format(tic))
        fig = self._add_gaia_figure_elements(tpf, fig, magnitude_limit=magnitude_limit)

        return fig

    def _add_gaia_figure_elements(self, tpf, fig, magnitude_limit=18):
        """Make the Gaia Figure Elements"""
        # Get the positions of the Gaia sources
        c1 = SkyCoord(tpf.ra, tpf.dec, frame='icrs', unit='deg')
        # Use pixel scale for query size
        pix_scale = 21.0
        # We are querying with a diameter as the radius, overfilling by 2x.
        from astroquery.vizier import Vizier
        Vizier.ROW_LIMIT = -1
        result = Vizier.query_region(c1, catalog=["I/345/gaia2"],
                                     radius=Angle(np.max(tpf.shape[1:]) * pix_scale, "arcsec"))
        no_targets_found_message = ValueError('Either no sources were found in the query region '
                                              'or Vizier is unavailable')
        too_few_found_message = ValueError('No sources found brighter than {:0.1f}'.format(magnitude_limit))
        if result is None:
            raise no_targets_found_message
        elif len(result) == 0:
            raise too_few_found_message
        result = result["I/345/gaia2"].to_pandas()
        result = result[result.Gmag < magnitude_limit]
        if len(result) == 0:
            raise no_targets_found_message
        radecs = np.vstack([result['RA_ICRS'], result['DE_ICRS']]).T
        coords = tpf.wcs.all_world2pix(radecs, 0)
        try:
            year = ((tpf.time[0].jd - 2457206.375) * u.day).to(u.year)
        except:
            year = ((tpf.astropy_time[0].jd - 2457206.375) * u.day).to(u.year)
        pmra = ((np.nan_to_num(np.asarray(result.pmRA)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
        pmdec = ((np.nan_to_num(np.asarray(result.pmDE)) * u.milliarcsecond/u.year) * year).to(u.arcsec).value
        result.RA_ICRS += pmra
        result.DE_ICRS += pmdec

        # Gently size the points by their Gaia magnitude
        sizes = 10000.0 / 2**(result['Gmag']/2)

        target = tpf.wcs.world_to_pixel(c1)
        plt.scatter(target[0]+tpf.column, target[1]+tpf.row, s=50, zorder=1000, c='k', marker='x')

        plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='firebrick', alpha=0.5, edgecolors='r', s=sizes)
        plt.scatter(coords[:, 0]+tpf.column, coords[:, 1]+tpf.row, c='None', edgecolors='r', s=sizes)
        plt.xlim([tpf.column-0.5, tpf.column+tpf.shape[1]-0.5])
        plt.ylim([tpf.row-0.5, tpf.row+tpf.shape[2]-0.5])

        return fig

