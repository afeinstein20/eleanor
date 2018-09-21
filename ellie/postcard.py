import os, sys

from astropy.utils.data import download_file
from astropy.io import fits

__all__ = ['Postcard']
ELLIEURL = 'http://jet.uchicago.edu/tess_postcards/'


class Postcard(object):

    def __init__(self, filename):
        self.filename = '{}{}'.format(ELLIEURL, filename)

    def __repr__(self):
        return "I AM A POSTCARD"

    def download_postcard(self):
        """ Downloads the postcard from the ELLIEURL """
        local_path = download_file(self.filename, cache=True)
        self.local_path = local_path
        self.hdu = fits.open(self.local_path)

    def plot(self):
        pass

    @property
    def dimensions(self):
        return (self.hdu[0].header['NAXIS1'], self.hdu[0].header['NAXIS2'], self.hdu[0].header['NAXIS3'])

    @property
    def center_radec(self):
        return(self.header['CEN_RA'], self.header['CEN_DEC'])

    @property
    def center_xy(self):
        return (self.header['CEN_X'],  self.header['CEN_Y'])

    @property
    def flux(self):
        return self.hdu[0].data
