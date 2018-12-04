import os, sys

from astropy.utils.data import download_file
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
import warnings
import pandas as pd
import copy
from .mast import crossmatch_by_position

__all__ = ['Postcard']
ELLIEURL = 'http://jet.uchicago.edu/tess_postcards/postcards/'


class Postcard(object):
    """
    Attributes
    ----------
    dimensions : tuple
        `x`, `y`, and `time` dimensions of postcard
    header :
        stored header information for postcard file
    center_radec : tuple
        RA & dec coordinates of the postcard's central pixel
    center_xy : tuple
        `x`, `y` coordinates in the FFI of the postcard's central pixel
    flux : numpy ndarray
        array of matrices containing flux in each pixel
    """
    def __init__(self, filename, location=None):
        if location is not None:
            self.filename = '{}{}'.format(location, filename)
            self.local_path = copy.copy(self.filename)
            self.hdu = fits.open(self.local_path)
        else:
            if os.path.isdir('./.ellie/sector_1/postcards')==True:
                self.filename = './.ellie/sector_1/postcards/{}'.format(filename)
                self.local_path = self.filename
                self.hdu = fits.open(self.local_path)
            else:
                self.filename = '{}{}'.format(ELLIEURL, filename)
                local_path = download_file(self.filename, cache=True)
                self.local_path = local_path
                self.hdu = fits.open(self.local_path)

    def __repr__(self):
        return "ellie postcard ({})".format(self.filename)

    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        """ Plot a frame of a tpf """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale is 'log':
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
        """
        Find the sources in a postcard

        Returns
        -------
        result : astropy.table.Table
            All the sources in a postcard with tic ids, gaia ids and others...
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
        raise NotImplementedError('Not yet implemented.')
