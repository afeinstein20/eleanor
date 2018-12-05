import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import sys

import urllib
from .mast import *
from .utils import *

__all__ = ['Source', 'multi_sectors']


def multi_sectors(sectors, tic=None, gaia=None, coords=None):
    """Obtain a list of Source objects for a single target, for each of multiple sectors for which the target was observed.
    
    Parameters
    ----------
    tic : int, optional
        The TIC ID of the source.
    gaia : int, optional
        The Gaia DR2 source_id.
    coords : tuple, optional
        The (RA, Dec) coords of the object in degrees.
    sectors : list or str
        The list of sectors for which data should be returned, or `all` to return all sectors 
        for which there are data.
    """
    objs = []
    if sectors == 'all':
<<<<<<< HEAD
        sectors = list(np.arange(1,14,1, dtype=int))
=======
        sectors = list(np.arange(1,14,1))
>>>>>>> 081771d1d2cccd4265c477102654d6a3c61fda72
    if type(sectors) == list:
        for s in sectors:
            star = Source(tic=tic, gaia=gaia, coords=coords, sector=int(s))
            if star.sector is not None:
                objs.append(star)
        return objs
    else:
        raise TypeError("Sectors needs to be either 'all' or a type(list) to work.")
    

def load_postcard_guide():
    """Load and return the postcard coordinates guide."""
    guide_link = urllib.request.urlopen('http://astro.uchicago.edu/~bmontet/TESS_postcards/postcard.guide')
    guide = guide_link.read().decode('utf-8')
    guide = Table.read(guide, format='ascii.basic') # guide to postcard locations
    return guide

class Source(object):
    """A single source observed by TESS.

    Parameters
    ----------
    tic : int, optional
        The TIC ID of the source.
    gaia : int, optional
        The Gaia DR2 source_id.
    coords : tuple or astropy.coordinates.SkyCoord, optional
        The (RA, Dec) coords of the object in degrees or an astropy SkyCoord object.
    fn : str, optional
        Filename of a TPF corresponding to the desired source.
    sector : int or str 
        The sector for which data should be returned, or `recent` to 
        obtain data for the most recent sector which contains this target. 

    Attributes
    ----------
    tess_mag : float
        The TESS magnitude from the TIC.
    sector : int
        Sector in which source was observed by TESS.
    camera : int
        TESS camera on which source falls.
    chip : int
        TESS chip on which source falls.
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
    def __init__(self, tic=None, gaia=None, coords=None, fn=None, sector=None):
        self.tic     = tic
        self.gaia    = gaia
        self.coords  = coords
        self.fn      = fn
        self.premade = False
        self.usr_sec = sector

        if self.fn is not None:
            try:
                hdu = fits.open(self.fn)
            except:
                assert False, "{0} is not a valid filename.".format(fn)
            hdr = hdu[0].header
            self.tic      = hdr['TIC_ID']
            self.tess_mag = hdr['TMAG']
            self.gaia     = hdr['GAIA_ID']
            self.coords   = (hdr['CEN_RA'], hdr['CEN_DEC'])
            self.premade  = True
            self.sector   = hdr['SECTOR']
            self.camera   = hdr['CAMERA']
            self.chip     = hdr['CHIP']
            self.position_on_chip = (hdr['CHIPPOS1'], hdr['CHIPPOS2'])
            self.position_on_postcard = (hdr['POSTPOS1'], hdr['POSTPOS2'])
                    
        else:
            if self.coords is not None:
                assert type(self.coords) is astropy.coordinates.sky_coordinate.SkyCoord, \
                    "Source: coords must be an astropy.coordinates.SkyCoord object."
                if type(self.coords) is astropy.coordinates.sky_coordinate.SkyCoord:
                    self.coords = (c.ra.degree, c.dec.degree)
                elif (len(coords) == 2) & all(isinstance(c, float) for c in self.coords):
                    self.coords  = coords
                else:
                    assert False, ("Source: invalid coords. Valid input types are: "
                                   "(RA, Dec) tuple or astropy.coordinates.SkyCoord object.")

                self.tic, self.tess_mag, sep = tic_from_coords(self.coords)
                self.gaia = gaia_from_coords(self.coords)

            elif self.gaia is not None:
                self.coords = coords_from_gaia(self.gaia)
                self.tic, self.tess_mag = tic_from_coords(self.coords)

            elif self.tic is not None:
                self.coords, self.tess_mag = coords_from_tic(self.tic)
                self.gaia = gaia_from_coords(self.coords)
                
            else:
                assert False, ("Source: one of the following keywords must be given: "
                               "tic, gaia, coords, fn.")

            self.tess_mag = self.tess_mag[0]
            self.locate_on_tess() # sets sector, camera, chip, postcard,
                                  # position_on_chip, position_on_postcard



    def locate_on_chip(self, guide):
        """Finds the TESS sector, camera, chip, and position on chip for the source.
        Sets attributes sector, camera, chip, position_on_chip.
        """
        def cam_chip_loop(sec):
            for cam in np.unique(guide['CAMERA']):
                for chip in np.unique(guide['CCD']):
                    mask = ((guide['SECTOR'] == sec) & (guide['CAMERA'] == cam)
                            & (guide['CCD'] == chip))
                    if np.sum(mask) == 0:
                        continue # this camera-chip combo is not in the postcard database yet
                    else:
                        random_postcard = guide[mask][0]
                    d = {}
                    for j,col in enumerate(random_postcard.colnames):
                        d[col] = random_postcard[j]
                    hdr = fits.Header(cards=d) # make WCS info from one postcard header
                    xy = WCS(hdr).all_world2pix(self.coords[0], self.coords[1], 1, quiet=True) # position in pixels in FFI dims
                    x_zero, y_zero = hdr['POSTPIX1'], hdr['POSTPIX2']
                    xy = np.array([xy[0]+x_zero, xy[1]+y_zero])
                    if (0 <= xy[0] < 2048) & (44 <= xy[1] < 2092):
                        self.sector = sec
                        self.camera = cam
                        self.chip = chip
                        self.position_on_chip = np.ravel(xy)

        guide = load_postcard_guide()
        self.sector=None

        if self.usr_sec is None:
            for sec in np.unique(guide['SECTOR']):
                cam_chip_loop(sec)
            
            if self.sector is None:
                raise SearchError("TESS has not (yet) observed your target.")

        elif self.usr_sec is not None:
            if type(self.usr_sec) == int:
                cam_chip_loop(self.usr_sec)

            # Searches for the most recent sector the object was observed in
            elif self.usr_sec.lower() == 'recent':
                for s in np.arange(15,0,-1):
                    cam_chip_loop(s)
                    if self.sector is not None:
                        break

        return


    def locate_on_tess(self):
        """Finds the best TESS postcard(s) and the position of the source on postcard.
        Sets attributes postcard, position_on_postcard, all_postcards.
        """
        guide = load_postcard_guide()
        self.locate_on_chip(guide)

        if self.sector is None:
            return

        # Searches through postcards for the given sector, camera, chip
        in_file, dists=[], []

        # Searches through rows of the table
        for i in range(len(guide)):
            postcard_inds = np.arange(len(guide))[(guide['SECTOR'] == self.sector) & (guide['CAMERA'] == self.camera)
                                                  & (guide['CCD'] == self.chip)]
        xy = self.position_on_chip

        # Finds postcards containing the source
        for i in postcard_inds: # loop rows

            x_cen, y_cen= guide['CEN_X'][i], guide['CEN_Y'][i]
            l, w = guide['POST_H'][i]/2., guide['POST_W'][i]/2.

            # Checks to see if xy coordinates of source falls within postcard
            if (xy[0] >= x_cen-l) & (xy[0] <= x_cen+l) & (xy[1] >= y_cen-w) & (xy[1] <= y_cen+w):
                in_file.append(i)
                dists.append(np.min([xy[0]-(x_cen-l), (x_cen+l)-xy[0], xy[1]-(y_cen-w), (y_cen+w)-xy[1]]))


        # If more than one postcard is found for a single source, choose the postcard where the
        # source is closer to the center
        if in_file == []:
            raise SearchError("Sorry! We don't have a postcard for you at the moment.")
        elif len(in_file) > 1:
            best_ind = np.argmax(dists)
        else:
            best_ind = 0

        self.all_postcards = guide['POSTNAME'][in_file]
        self.postcard = guide['POSTNAME'][in_file[best_ind]]

        self.sector = guide['SECTOR'][in_file[best_ind]] # WILL BREAK FOR MULTI-SECTOR TARGETS
        self.camera = guide['CAMERA'][in_file[best_ind]]
        self.chip = guide['CCD'][in_file[best_ind]]

        i = in_file[best_ind]
        postcard_pos_on_ffi = (guide['CEN_X'][i] - guide['POST_H'][i]/2.,
                                guide['CEN_Y'][i] - guide['POST_W'][i]/2.)
        self.position_on_postcard = xy - postcard_pos_on_ffi # as accurate as FFI WCS

        i = in_file[best_ind]
        postcard_pos_on_ffi = (guide['CEN_X'][i] - guide['POST_H'][i]/2.,
                              guide['CEN_Y'][i] - guide['POST_W'][i]/2.) # (x,y) on FFI where postcard (x,y) = (0,0)
        self.position_on_postcard = xy - postcard_pos_on_ffi # as accurate as FFI WCS
