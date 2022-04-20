import os, sys

from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
import warnings
import pandas as pd
import copy
from astropy.stats import SigmaClip
from photutils import MMMBackground
from .mast import crossmatch_by_position
from urllib.request import urlopen


__all__ = ['Postcard']

class Postcard(object):
    """TESS FFI data for one postcard across one sector.
    
    A postcard is an rectangular subsection cut out from the FFIs. 
    It's like a TPF, but bigger. 
    The Postcard object contains a stack of these cutouts from all available 
    FFIs during a given sector of TESS observations.
    
    Parameters
    ----------
    filename : str
        Filename of the downloaded postcard.
    location : str, optional
        Filepath to `filename`.
    
    Attributes
    ----------
    dimensions : tuple
        (`x`, `y`, `time`) dimensions of postcard.
    flux, flux_err : numpy.ndarray
        Arrays of shape `postcard.dimensions` containing flux or error on flux 
        for each pixel.
    time : float
        ?
    header : dict
        Stored header information for postcard file.
    center_radec : tuple
        RA & Dec coordinates of the postcard's central pixel.
    center_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of 
        the postcard's central pixel on the FFI.
    origin_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of 
        the postcard's (0,0) pixel on the FFI.
    background2d : np.ndarray
        The 2D modeled background array.
    """
    def __init__(self, filename, background, filepath):
        self.filename = os.path.join(filepath, filename)
        self.local_path = copy.copy(self.filename)
        self.hdu = fits.open(self.local_path)
        self.background2d = fits.open(os.path.join(filepath, background))[1].data

    def __repr__(self):
        return "eleanor postcard ({})".format(self.filename)

    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        """Plots a single frame of a postcard.
        
        Parameters
        ----------
        frame : int, optional
            Index of frame. Default 0.
        
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot. Creates a new object by default.
        
        scale : str
            Scaling for colorbar; acceptable inputs are 'linear' or 'log'.
            Default 'linear'.
        
        **kwargs : passed to matplotlib.pyplot.imshow
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale == 'log':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dat = np.log10(self.flux[:, :, frame])
                dat[~np.isfinite(dat)] = np.nan
        else:
            dat = self.flux[:, :, frame]

        if ('vmin' not in kwargs) & ('vmax' not in kwargs):
            kwargs['vmin'] = np.nanpercentile(dat, 1)
            kwargs['vmax'] = np.nanpercentile(dat, 99)

        im = ax.imshow(dat, **kwargs)
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        cbar = plt.colorbar(im, ax=ax)
        if scale == 'log':
            cbar.set_label('log$_{10}$ Flux')
        else:
            cbar.set_label('Flux')

        # Reset the x/y ticks to the position in the ACTUAL FFI.
        xticks = ax.get_xticks() + self.center_xy[0]
        yticks = ax.get_yticks() + self.center_xy[1]
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        return ax

    def find_sources(self):
        """Finds the cataloged sources in the postcard and returns a table.

        Returns
        -------
        result : astropy.table.Table
            All the sources in a postcard with TIC IDs or Gaia IDs.
        """
        result = crossmatch_by_position(self.center_radec, 0.5, 'Mast.Tic.Crossmatch').to_pandas()
        result = result[['MatchID', 'MatchRA', 'MatchDEC', 'pmRA', 'pmDEC', 'Tmag']]
        result.columns = ['TessID', 'RA', 'Dec', 'pmRA', 'pmDEC', 'Tmag']
        return result


    @property
    def header(self):
        return self.hdu[1].header

    @property
    def center_radec(self):
        return(self.header['CEN_RA'], self.header['CEN_DEC'])

    @property
    def center_xy(self):
        return (self.header['CEN_X'],  self.header['CEN_Y'])

    @property
    def origin_xy(self):
        return (self.header['POSTPIX1'], self.header['POSTPIX2'])

    @property
    def flux(self):
        return self.hdu[2].data

    @property
    def dimensions(self):
        return self.flux.shape

    @property
    def flux_err(self):
        return self.hdu[3].data

    @property
    def time(self):
        return (self.hdu[1].data['TSTOP'] + self.hdu[1].data['TSTART'])/2

    @property
    def wcs(self):
        return WCS(self.header)

    @property
    def quality(self):
        return self.hdu[1].data['QUALITY']

    @property
    def bkg(self):
        return self.hdu[1].data['BKG']
    
    @property 
    def barycorr(self):
        return self.hdu[1].data['BARYCORR']
    
    
    @property
    def ffiindex(self):
        return self.hdu[1].data['FFIINDEX']
    
class Postcard_tesscut(object):
    """TESS FFI data for one postcard across one sector.
    
    TESSCut is a service from MAST to produce TPF cutouts from the TESS FFIs. If
    `eleanor.Source()` is called with `tc=True`, TESSCut is used to produce a large
    postcard-like cutout region rather than downlading a standard eleanor postcard.

    Parameters
    ----------
    filename : str
        Filename of the downloaded postcard.
    location : str, optional
        Filepath to `filename`.
    
    Attributes
    ----------
    dimensions : tuple
        (`x`, `y`, `time`) dimensions of postcard.
    flux, flux_err : numpy.ndarray
        Arrays of shape `postcard.dimensions` containing flux or error on flux 
        for each pixel.
    time : float
        ?
    header : dict
        Stored header information for postcard file.
    center_radec : tuple
        RA & Dec coordinates of the postcard's central pixel.
    center_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of 
        the postcard's central pixel on the FFI.
    origin_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of 
        the postcard's (0,0) pixel on the FFI.
    """
    def __init__(self, cutout, location=None):

        if location is None:
            self.local_path = os.path.join(os.path.expanduser('~'), '.eleanor/tesscut')
        else:
            self.local_path = location

        self.hdu = cutout


    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        """Plots a single frame of a postcard.
        
        Parameters
        ----------
        frame : int, optional
            Index of frame. Default 0.
        
        ax : matplotlib.axes.Axes, optional
            Axes on which to plot. Creates a new object by default.
        
        scale : str
            Scaling for colorbar; acceptable inputs are 'linear' or 'log'.
            Default 'linear'.
        
        **kwargs : passed to matplotlib.pyplot.imshow
        
        Returns
        -------
        ax : matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale == 'log':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dat = np.log10(self.flux[:, :, frame])
                dat[~np.isfinite(dat)] = np.nan
        else:
            dat = self.flux[:, :, frame]

        if ('vmin' not in kwargs) & ('vmax' not in kwargs):
            kwargs['vmin'] = np.nanpercentile(dat, 1)
            kwargs['vmax'] = np.nanpercentile(dat, 99)

        im = ax.imshow(dat, **kwargs)
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        cbar = plt.colorbar(im, ax=ax)
        if scale == 'log':
            cbar.set_label('log$_{10}$ Flux')
        else:
            cbar.set_label('Flux')

        # Reset the x/y ticks to the position in the ACTUAL FFI.
        xticks = ax.get_xticks() + self.center_xy[0]
        yticks = ax.get_yticks() + self.center_xy[1]
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        return ax

    def find_sources(self):
        """Finds the cataloged sources in the postcard and returns a table.

        Returns
        -------
        result : astropy.table.Table
            All the sources in a postcard with TIC IDs or Gaia IDs.
        """
        result = crossmatch_by_position(self.center_radec, 0.5, 'Mast.Tic.Crossmatch').to_pandas()
        result = result[['MatchID', 'MatchRA', 'MatchDEC', 'pmRA', 'pmDEC', 'Tmag']]
        result.columns = ['TessID', 'RA', 'Dec', 'pmRA', 'pmDEC', 'Tmag']
        return result


    @property
    def header(self):
        return self.hdu[1].header

    @property
    def center_radec(self):
        return(self.header['RA_OBJ'], self.header['DEC_OBJ'])

    @property
    def center_xy(self):
        return (self.header['1CRV4P']+16,  self.header['1CRV4P']+16)

    @property
    def origin_xy(self):
        return (self.header['1CRV4P'], self.header['1CRV4P'])

    @property
    def flux(self):
        return self.hdu[1].data['FLUX']

    @property
    def dimensions(self):
        return self.flux.shape

    @property
    def flux_err(self):
        return self.hdu[1].data['FLUX_ERR']

    @property
    def time(self):
        return self.hdu[1].data['TIME']

    @property
    def wcs(self):
        return WCS(self.header)

    @property
    def quality(self):
        sector = self.header['SECTOR']
        eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')
        A = np.loadtxt(eleanorpath + '/metadata/s{0:04d}/quality_s{0:04d}.txt'.format(sector))
        return A

    @property
    def bkg(self):
        sigma_clip = SigmaClip(sigma=3.)
        bkg = MMMBackground(sigma_clip=sigma_clip)
        b = bkg.calc_background(self.flux, axis=(1,2))
        return b
    
    @property 
    def barycorr(self):
        return self.hdu[1].data['TIMECORR']
    
    @property
    def ffiindex(self):
        sector = self.header['SECTOR']
        eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')
        A = np.loadtxt(eleanorpath + '/metadata/s{0:04d}/cadences_s{0:04d}.txt'.format(sector))
        return A

