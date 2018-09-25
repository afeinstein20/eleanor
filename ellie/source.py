import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy.io import fits

import urllib
from .mast import *

__all__ = ['Source']

def load_guide():
    """
    Load and return the postcard coordinates guide.
    """
    guide_link = urllib.request.urlopen('http://jet.uchicago.edu/tess_postcards/postcard.txt')
    guide = guide_link.read().decode('utf-8')
    guide = Table.read(guide, format='ascii.basic') # guide to postcard locations
    return guide

class Source(object):
    """
    A single source observed by TESS.

    Parameters
    ----------
    tic : int; optional
        The TIC ID of the source
    gaia : int; optional
        The Gaia DR2 source_id
    coords : (float, float); optional
        The (RA, Dec) coords of the object in degrees

    Additional Attributes
    ---------------------
    tess_mag : float
        The TESS magnitude from the TIC
    sector : int
        Sector in which source was observed by TESS
    camera : int
        TESS camera on which source falls
    chip : int
        TESS chip on which source falls
    position_on_chip : (int, int)
        Predicted (x,y) coords of object center on chip.
    postcard : str
        Name of the best postcard (postcard where source should be located closest to the center).
    position_on_postcard : (int, int)
        Predicted (x, y) coords of object center on the above-named postcard.
        Does NOT take into account additional pointing corrections; these will be applied when making the TPF.
    all_postcards : list of strs
        Names of all postcards where the source appears.
    """
    def __init__(self, tic=None, gaia=None, coords=None):
        self.tic = tic
        self.gaia = gaia
        self.coords = coords
        if self.tic is not None:
            self.coords, self.tess_mag = coords_from_tic(self.tic)
            self.gaia = gaia_from_coords(self.coords)
        elif self.gaia is not None:
            self.coords = coords_from_gaia(self.gaia)
            self.tic, self.tess_mag = tic_from_coords(self.coords)
        elif self.coords is not None:
            self.tic, self.tess_mag = tic_from_coords(self.coords)
            self.gaia = gaia_from_coords(self.coords)
        self.locate_on_tess() # sets sector, camera, chip, chip_position

    def locate_on_tess(self):

        
    def locate_on_chip(self, guide):

        """
        Finds the TESS sector, camera, chip, and position on chip for the source.
        Sets attributes sector, camera, chip, position_on_chip.
        """

        guide_link = urllib.request.urlopen('http://jet.uchicago.edu/tess_postcards/postcard.txt')
        guide = guide_link.read().decode('utf-8')
        guide = Table.read(guide, format='ascii.basic')

        d = {}
        for j,col in enumerate(guide.colnames):
            d[col] = guide[0][j]
        hdr = fits.Header(cards=d) # make WCS info from one postcard header
        xy = WCS(hdr).all_world2pix(self.coords[0], self.coords[1], 1, quiet=True) # position in pixels in FFI dims

        self.sector = None
        for sec in np.unique(guide['SECTOR']):
            for cam in np.unique(guide['CAMERA']):
                for chip in np.unique(guide['CCD']):
                    mask = (guide['SECTOR'] == sec) & (guide['CAMERA'] == cam) 
                                            & (guide['CCD'] == chip)                    
                    if np.sum(mask) == 0:
                        continue # this camera-chip combo is not in the postcard database yet
                    else:
                        random_postcard = guide[mask][0]
                    d = {}
                    for j,col in enumerate(random_postcard.colnames):
                        d[col] = random_postcard[j]
                    hdr = fits.Header(cards=d) # make WCS info from one postcard header
                    xy = WCS(hdr).all_world2pix(self.coords[0], self.coords[1], 1, quiet=True) # position in pixels in FFI dims
                    if (0 <= xy[0] < 2048) & (44 <= xy[1] < 2092):
                        self.sector = sec
                        self.camera = cam
                        self.chip = chip
                        self.position_on_chip = np.ravel(xy)
                        return
        if self.sector is None:
            print("TESS has not (yet) observed your target.")
            return
        
        
    def locate_on_tess(self):
        """
        Finds the best TESS postcard(s) and the position of the source on postcard.
        Sets attributes postcard, position_on_postcard, all_postcards.
        """
        guide = load_guide()
        self.locate_on_chip(guide)


        # Searches through postcards for the given sector, camera, chip
        in_file=[]

        # Searches through rows of the table
        for i in range(len(guide)):

        postcard_inds = np.arange(len(guide))[(guide['SECTOR'] == self.sector) & (guide['CAMERA'] == self.camera) 
                                                & (guide['CCD'] == self.chip)]
        xy = self.position_on_chip
        # Finds postcards containing the source
        for i in postcard_inds: # loop rows

            x_cen, y_cen= guide['POST_CEN_X'][i], guide['POST_CEN_Y'][i]
            l, w = guide['POST_SIZE1'][i]/2., guide['POST_SIZE2'][i]/2.
            print("postcard #{0}: x: {1} - {2}, y: {3} - {4}".format(i,x_cen-l,x_cen+l,y_cen-w,y_cen+w))
            # Checks to see if xy coordinates of source falls within postcard
            if (xy[0] >= x_cen-l) & (xy[0] <= x_cen+l) & (xy[1] >= y_cen-w) & (xy[1] <= y_cen+w):
                in_file.append(i)
        # If more than one postcard is found for a single source, choose the postcard where the
        # source is closer to the center
        if in_file == []:
            print("Sorry! We don't have a postcard for you at the moment.")
            return
        elif len(in_file) > 1:
            dists = np.zeros_like(in_file)
            for i,j in enumerate(in_file):
                dists[i] = np.sqrt( (xy[0]-guide['POST_CENX'][j])**2 + (xy[1]-guide['POST_CENY'][j])**2  )
            best_ind = np.argmin(dists)
        else:
            best_ind = 0

        self.all_postcards = guide['POST_FILE'][in_file]
        self.postcard = guide['POST_FILE'][in_file[best_ind]]

        self.sector = guide['SECTOR'][in_file[best_ind]] # WILL BREAK FOR MULTI-SECTOR TARGETS
        self.camera = guide['CAMERA'][in_file[best_ind]]
        self.chip = guide['CCD'][in_file[best_ind]]

        i = in_file[best_ind]
        postcard_pos_on_ffi = (guide['POST_CENX'][i] - guide['POST_SIZE1'][i]/2.,
                                guide['POST_CENY'][i] - guide['POST_SIZE2'][i]/2.)
        self.position_on_postcard = xy - postcard_pos_on_ffi # as accurate as FFI WCS

        i = in_file[best_ind]
        postcard_pos_on_ffi = (guide['POST_CEN_X'][i] - guide['POST_SIZE1'][i]/2., 
                              guide['POST_CEN_Y'][i] - guide['POST_SIZE2'][i]/2.) # (x,y) on FFI where postcard (x,y) = (0,0)
        self.position_on_postcard = xy - postcard_pos_on_ffi # as accurate as FFI WCS
        

