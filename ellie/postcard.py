import os, sys

from astropy.utils.data import download_file
from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import warnings

__all__ = ['Postcard']
ELLIEURL = 'http://jet.uchicago.edu/tess_postcards/'


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
    def __init__(self, filename):
        self.filename = '{}{}'.format(ELLIEURL, filename)

    def __repr__(self):
        return "ellie postcard ({})".format(self.filename)

    def download_postcard(self):
        """ Downloads the postcard from the ELLIEURL """
        local_path = download_file(self.filename, cache=True)
        self.local_path = local_path
        self.hdu = fits.open(self.local_path)

    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        ''' Plot a frame of a tpf
        '''

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale is 'log':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dat = np.log10(self.flux[frame])
                dat[~np.isfinite(dat)] = np.nan
        else:
            dat = self.flux[frame]

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

    @property
    def dimensions(self):
        return (self.hdu[0].header['NAXIS1'], self.hdu[0].header['NAXIS2'], self.hdu[0].header['NAXIS3'])

    @property
    def header(self):
        return self.hdu[0].header

    @property
    def center_radec(self):
        return(self.header['CEN_RA'], self.header['CEN_DEC'])

    @property
    def center_xy(self):
        return (self.header['CEN_X'],  self.header['CEN_Y'])

    @property
    def flux(self):
        return self.hdu[0].data
