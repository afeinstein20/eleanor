import matplotlib as mpl
import numpy as np
import bokeh.io
import bokeh.models


__all__ = []


class Visualize(object):
    """
    The main class for creating figures, movies, and interactive plots
    Allows the user to have a grand ole time playing with their data!

    Args:
       obj: object must have minimum attributes of 2D array of flux
            will allow for plotting of both postcards & tpfs
    """

    def __init__(self, object):
        self.obj = object
        if self.obj.tpf is not None:
            self.flux   = self.obj.tpf
            self.center = (np.nanmedian(self.obj.centroid_xs), 
                             np.nanmedian(self.obj.centroid_ys))
        else:
            self.flux   = self.obj.flux
            self.center = self.obj.center_xy



    def movie(self, output_fn=None, plot_lc=False, pointing_model=False, aperture=False, **kwargs):
        """
        This function allows the user to create a movie of a TPF or postcard
        Parameters
        ----------
            output_fn: Allows the user to set an output filename to save the movie to
            plot_lc  : Allows the user to plot the light curve and track location along
                       light curve with time-dependent movie (Defaults to False)
            pointing_model: Allows the user to plot the movement of the pointing_model
                            for a given source; only applicable on TPFs (Defaults to False)
            aperture: Allows the user to overplot the aperture on the TPF movie
        """
        import matplotlib.animation as animation

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
        """
        Allows the user to mark other Gaia sources within a given TPF or postcard
        Hover over the points to reveal the source's TIC ID, Gaia ID, Tmag, and Gmag
        """
        return
