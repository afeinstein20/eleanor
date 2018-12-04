import matplotlib as mpl
import matplotlib.animation as animation
import numpy as np
import bokeh.io
import bokeh.models
from astropy.io import fits

from astropy.wcs import WCS
from .ffi import use_pointing_model


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
    """

    def __init__(self, object):
        self.obj = object
        self.header = self.obj.header

        if self.obj.tpf is not None:
            self.flux   = self.obj.tpf
            self.center = (np.nanmedian(self.obj.centroid_xs),
                             np.nanmedian(self.obj.centroid_ys))
            self.pointing_model = self.obj.pointing_model
        else:
            self.flux   = self.obj.flux
            self.center = self.obj.center_xy




    def movie(self, output_fn=None, plot_lc=False, pointing_model=False, aperture=False, **kwargs):
        """
        This function allows the user to create a movie of a TPF or postcard
        
        Parameters
        ----------
        output_fn : str, optional 
            Output filename to save the movie to.
        plot_lc : bool, optional 
            If True, plot the light curve and track location along
            light curve with time-dependent movie (Defaults to False).
        pointing_model : bool, optional 
            If True, plot the movement of the pointing_model
            for a given source; only applicable on TPFs (Defaults to False).
        aperture : bool, optional 
            If True, overplot the aperture on the TPF movie.
        **kwargs : 
            Passed to matplotlib.pyplot.imshow.
        
        Returns
        -------
        ani : matplotlib.animation.Animation
            Movie.
        """
        if pointing_model==True:
            if self.obj.centroid_xs is None:
                print("Sorry, you can only track the pointing model on TPFs, not postcards\n",
                      "Please set pointing_model=False or remove when calling .movie()")
            else:
                xs = self.obj.centroid_xs-self.center[0]+4.0
                ys = self.obj.centroid_ys-self.center[1]+4.0


        if 'vmax' not in kwargs:
            kwargs['vmax'] = np.max(self.flux[0])
        if 'vmin' not in kwargs:
            kwargs['vmin'] = np.min(self.flux[0])
        if 'cmap' not in kwargs:
            kwargs['cmap'] = 'viridis' # the superior colormap


        def animate(i):
            nonlocal xs, ys, scats, line
            ax.imshow(self.flux[i], origin='lower', **kwargs)
            if pointing_model==True:
                for c in scats:
                    c.remove()
                scats=[]
                scats.append(ax.scatter(xs[i], ys[i], s=16, c='k', **kwargs))
            if plot_lc==True:
                for l in line:
                    l.remove()
                line=[]
                line.append(ax1.scatter(self.obj.time[i], self.obj.corr_flux[i], c='r', s=16))
            time_text.set_text('Frame {}'.format(i))


        line, scats = [], []

        # Creates 2 subplots if user requests lc to be plotted
        if plot_lc==True:
            fig  = mpl.pyplot.figure(figsize=(18,5))
            spec = mpl.gridspec.GridSpec(ncols=3, nrows=1)
            ax   = fig.add_subplot(spec[0,2])
            ax1  = fig.add_subplot(spec[0,0:2])
            ax1.plot(self.obj.time, self.obj.corr_flux,'k-')
            ax1.set_ylabel('Corrected Flux')
            ax1.set_xlabel('Time')
            ax1.set_xlim([np.min(self.obj.time)-0.05, np.max(self.obj.time)+0.05])
            ax1.set_ylim([np.min(self.obj.corr_flux)-0.05, np.max(self.obj.corr_flux)+0.05])

        # Creates 1 subplot is user just wants a pixel movie
        else:
            fig  = mpl.pyplot.figure()
            spec = mpl.gridspec.GridSpec(ncols=1, nrows=1)
            ax   = fig.add_subplot(spec[0,0])
            ax.set_xlabel('Pixel Column')
            ax.set_ylabel('Pixel Row')

        if aperture==True:
            print("Aperture has not been implemented yet. Sorry friend, we're working on it!")
            return

        # Plots axes in correct (x,y) coordinate space
        if self.obj.centroid_ys is not None:
            mid_x = int(np.round(self.center[0], 0))
            mid_y = int(np.round(self.center[1], 0))
            ax.set_yticklabels([str(e) for e in np.arange(mid_x-4, mid_x+6,1)])
            ax.set_xticklabels([str(e) for e in np.arange(mid_y-4, mid_y+6,1)])

        # Sets text for pixel movie frames
        time_text = ax.text(5.5, -0.25, '', color='white', fontweight='bold')
        time_text.set_text('')

        # Allows movie to be saved as mp4
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=20, metadata=dict(artist='Adina D. Feinstein'), bitrate=1800)

        ani = animation.FuncAnimation(fig, animate, frames=len(self.flux))

        mpl.pyplot.colorbar(ax.imshow(self.flux[0], **kwargs), ax=ax)

        mpl.pyplot.tight_layout()

        return ani


    def mark_gaia_sources(self):
        """Mark Gaia sources within a given TPF or postcard.
        
        Hover over the points to reveal the source's TIC ID, Gaia ID, Tmag, and Gmag. 
        Also crossmatches with TIC and identifies closest TIC object.
        """

        from bokeh.models import (ColumnDataSource, HoverTool, BasicTicker,
                          Slider, Button, Label, LinearColorMapper, Span,
                          ColorBar)
        from bokeh.plotting import figure, show, output_file
        from bokeh.palettes import Viridis256

        def create_labels(gaia_ra, gaia_dec):
            ticLabel, tmagLabel = np.zeros(len(gaia_id)), np.zeros(len(gaia_id))
            for i in range(len(gaia_ra)):
                tic, tmag, sep = tic_from_coords([gaia_ra[i], gaia_dec[i]])
                print(tic)
                if sep < 1.0:
                    ticLabel[i]  = tic
                    tmagLabel[i] = tmag[0]
            return ticLabel, tmagLabel

        center = [self.header['CEN_RA'], self.header['CEN_DEC']]
        center_xy = WCS(self.header).all_world2pix(center[0], center[1], 1)

        sources    = cone_search(pos=center, r=0.5, service='Mast.Catalogs.GaiaDR2.Cone')
        sources_xy = WCS(self.header).all_world2pix(sources['ra'], sources['dec'], 1)
        new_coords = []
        for i in range(len(sources_xy[0])):
            xy = [sources_xy[0][i], sources_xy[1][i]]
            coords_corr = use_pointing_model(xy, self.pointing_model[0])
            new_coords.append([coords_corr[0][0],coords_corr[0][1]])
        new_coords = np.array(new_coords)

        # Finds sources that lie in the frame
        in_frame = np.where( (new_coords[:,0] >= center_xy[0]-self.obj.dimensions[1]/2.) &
                             (new_coords[:,0] <= center_xy[0]+self.obj.dimensions[1]/2.) &
                             (new_coords[:,1] >= center_xy[1]-self.obj.dimensions[2]/2.) &
                             (new_coords[:,1] <= center_xy[1]+self.obj.dimensions[2]/2.) )
        gaia_pos_x = new_coords[:,1][in_frame] - center_xy[1] + self.obj.dimensions[1]/2.
        gaia_pos_y = new_coords[:,0][in_frame] - center_xy[0] + self.obj.dimensions[2]/2.
        gaia_g_mag = sources['phot_g_mean_mag'][in_frame]
        gaia_id    = sources['source_id'][in_frame]

        ticLabel, tmagLabel = create_labels(sources['ra'][in_frame], sources['dec'][in_frame])

        # Creates hover label
        hover = HoverTool()
        hover.tooltips = [
            ('TIC', ''),
            ('T_mag', ''),
            ('Gaia ID', ''),
            ('G_mag', '')
            ]

        x_range = (-0.5,self.obj.dimensions[1]-0.5)
        y_range = (-0.5,self.obj.dimensions[2]-0.5)
        p = figure(x_range=x_range, y_range=y_range, toolbar_location='above',
                   plot_width=500, plot_height=450)

        color_mapper = LinearColorMapper(palette='Viridis256', low=np.min(self.flux[0]),
                                         high=np.max(self.flux[0]))
        tpf_img = p.image(image=[self.flux[0]], x=-0.5, y=-0.5, dw=self.obj.dimensions[1],
                          dh=self.obj.dimensions[2], color_mapper=color_mapper)

        # Sets the placement of the tick labels
        p.xaxis.ticker = np.arange(0, self.obj.dimensions[1])
        p.yaxis.ticker = np.arange(0, self.obj.dimensions[2])

        # Sets the tick labels
        x_overrides, y_overrides = {}, {}
        x_list = np.arange(int(center_xy[0]-self.obj.dimensions[1]/2),
                           int(center_xy[0]+self.obj.dimensions[1]/2), 1)
        y_list = np.arange(int(center_xy[1]-self.obj.dimensions[2]/2),
                           int(center_xy[1]+self.obj.dimensions[2]/2), 1)

        for i in range(9):
            ind = str(i)
            x_overrides[i] = str(x_list[i])
            y_overrides[i] = str(y_list[i])

        p.xaxis.major_label_overrides = x_overrides
        p.yaxis.major_label_overrides = y_overrides
        p.xaxis.axis_label = 'Pixel Column Number'
        p.yaxis.axis_label = 'Pixel Row Number'

        # Sets the color bar
        color_bar = ColorBar(color_mapper=color_mapper, location=(0,0), border_line_color=None,
                             ticker=BasicTicker(), title='Intensity', title_text_align='left')
        p.add_layout(color_bar, 'right')

        # Adds points onto image
        source = ColumnDataSource(data=dict(x=gaia_pos_x, y=gaia_pos_y,
                                            tic=ticLabel, tmag=tmagLabel,
                                            gaia=gaia_id, gmag=gaia_g_mag))

        s = p.circle('x', 'y', size=8, source=source, line_color=None,
                     selection_color='red', nonselection_fill_alpha=0.0, nonselection_line_alpha=0.0,
                     nonselection_line_color=None, fill_color='black', hover_alpha=0.9, hover_line_color='white')

        # Activates hover feature
        p.add_tools(HoverTool( tooltips=[("TIC ID", "@tic")  , ("TESS Tmag", "@tmag"),
                                         ("Gaia ID", "@gaia"), ("Gaia Gmag", "@gmag")
                                         ], renderers=[s], mode='mouse', point_policy='snap_to_data') )

        show(p)
        return



    def click_aperture(self, path=None):
        """Interactively set aperture."""

        def click_pixels():
            nonlocal tpf
            """Creates a rectangle over a pixel when that pixel is clicked."""
            coords, rectList = [], []

            fig, ax = mpl.pyplot.subplots()
            ax.imshow(tpf[0], origin='lower')

            def onclick(event):
                """Update figure canvas."""
                nonlocal coords, rectList
                x, y = int(np.round(event.xdata,0)), int(np.round(event.ydata,0))

                # Highlights pixel 
                rect = mpl.patches.Rectangle((x-0.5, y-0.5), 1.0, 1.0)
                rect.set_color('white')
                rect.set_alpha(0.4)
                # Adds pixel if not previously clicked
                if [x,y] not in coords:
                    coords.append([x,y])
                    rectList.append(rect)
                    ax.add_patch(rect)
                fig.canvas.draw()
            cid = fig.canvas.mpl_connect('button_press_event', onclick)
            mpl.pyplot.show()
            mpl.pyplot.close()
            return coords, rectList


        def check_pixels():
            nonlocal tpf, coords, rectList
            """Presents a figure for the user to approve of selected pixels."""
            fig, ax = mpl.pyplot.subplots(1)
            ax.imshow(tpf[0], origin='lower')

            # Recreates patches for confirmation
            for i in range(len(coords)):
                x, y = coords[i][0], coords[i][1]
                rect = mpl.patches.Rectangle((x-0.5, y-0.5), 1.0, 1.0)
                rect.set_color('red')
                rect.set_alpha(0.4)
                ax.add_patch(rect)

            # Make Buttons         
            mpl.pyplot.text(-3.5, 5.5, 'Are you happy\nwith this\naperture?', fontsize=8)
            axRadio  = mpl.pyplot.axes([0.05, 0.45, 0.10, 0.15])
            butRadio = mpl.widgets.RadioButtons(axRadio, ('Yes', 'No'), activecolor='red')
            good=True

            # Checks if user is happy with custom aperture
            def get_status(value):
                nonlocal good
                if value == 'Yes':
                    good=True
                else:
                    good=False

            butRadio.on_clicked(get_status)
            mpl.pyplot.show()
            return good

        hdu = self.header
        tpf = self.flux
        coords, rectList = click_pixels()
        check = check_pixels()

        custlc = []
        if check==True:
            if len(coords) == 0:
                print("You have not selected any pixels. No photometry will be completed.")
            else:
                for f in range(len(tpf)):
                    cadence = []
                    for i in range(len(coords)):
                        cadence.append(tpf[f][coords[i][0], coords[i][1]])
                    custlc.append(np.sum(cadence))
                custlc = np.array(custlc) / np.nanmedian(custlc)
                return custlc

        else:
            self.click_aperture()
            
        return
