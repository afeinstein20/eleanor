import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import os
import sys
from os.path import join, abspath
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px
import warnings
from astroquery.mast import Tesscut

from . import PACKAGEDIR

import urllib
from .mast import *
from .utils import *

__all__ = ['Source', 'multi_sectors']


def multi_sectors(sectors, tic=None, gaia=None, coords=None, tc=False):
    """Obtain a list of Source objects for a single target, for each of multiple sectors for which the target was observed.

    Parameters
    ----------
    sectors : list or str
        The list of sectors for which data should be returned, or `all` to return all sectors
        for which there are data.
    tic : int, optional
        The TIC ID of the source.
    gaia : int, optional
        The Gaia DR2 source_id.
    coords : tuple, optional
        The (RA, Dec) coords of the object in degrees.
    tc : bool, optional
        If True, use a TessCut cutout to produce postcards rather than downloading the eleanor
        postcard data products.
    """
    objs = []

    if sectors == 'all':
        if coords is None:
            if tic is not None:
                coords, _, _ = coords_from_tic(tic)
            elif gaia is not None:
                coords = coords_from_gaia(gaia)

        if coords is not None:
            if type(coords) is SkyCoord:
                coords = (coords.ra.degree, coords.dec.degree)
            result = tess_stars2px(8675309, coords[0], coords[1])
            sector = result[3][result[3] < 12.5]
            sectors = sector.tolist()
        print('Found star in Sector(s) ' +" ".join(str(x) for x in sectors))
    if type(sectors) == list:
        for s in sectors:
            star = Source(tic=tic, gaia=gaia, coords=coords, sector=int(s), tc=tc)
            if star.sector is not None:
                objs.append(star)
        if len(objs) < len(sectors):
            warnings.warn('Only {} targets found instead of desired {}. Your '
                          'target may not have been observed yet in these sectors.'
                          ''.format(len(objs), len(sectors)))
        return objs
    else:
        raise TypeError("Sectors needs to be either 'all' or a type(list) to work.")


def load_postcard_guide(sector):
    """Load and return the postcard coordinates guide."""
    try:
        user_agent = 'eleanor 0.1.6'
        values = {'name': 'eleanor',
                  'language': 'Python' }
        headers = {'User-Agent': user_agent}
        
        data = urllib.parse.urlencode(values)
        data = data.encode('ascii')
        
        guide_link = 'https://users.flatironinstitute.org/dforeman/public_www/tess/postcards_test/s{0:04d}/postcard.guide'.format(sector)
        
        req = urllib.request.Request(guide_link, data, headers)
        with urllib.request.urlopen(req) as response:
            guide = response.read().decode('utf-8')
        
        guide = Table.read(guide, format='ascii.basic') # guide to postcard locations
    except urllib.error.HTTPError:
        return None
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
    tc : bool, optional
        If True, use a TessCut cutout to produce postcards rather than downloading the eleanor
        postcard data products.

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
    def __init__(self, tic=None, gaia=None, coords=None, fn=None, sector=None, fn_dir=None, tc=False):
        self.tic     = tic
        self.gaia    = gaia
        self.coords  = coords
        self.fn      = fn
        self.premade = False
        self.usr_sec = sector
        self.tc      = tc

        if fn_dir is None:
            self.fn_dir = os.path.join(os.path.expanduser('~'), '.eleanor')
        else:
            self.fn_dir  = fn_dir

        if self.fn is not None:
            try:
                hdu = fits.open(self.fn_dir + '/' + self.fn)
            except:
                assert False, "{0} is not a valid filename or not located in directory {1}".format(fn, fn_dir)
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
#            self.position_on_postcard = (hdr['POSTPOS1'], hdr['POSTPOS2'])

        else:
            if self.coords is not None:
                if type(self.coords) is SkyCoord:
                    self.coords = (self.coords.ra.degree, self.coords.dec.degree)
                elif (len(self.coords) == 2) & (all(isinstance(c, float) for c in self.coords) | all(isinstance(c, int) for c in self.coords) ):
                    self.coords  = coords
                else:
                    assert False, ("Source: invalid coords. Valid input types are: "
                                   "(RA [deg], Dec [deg]) tuple or astropy.coordinates.SkyCoord object.")

                self.tic, self.tess_mag, sep, self.tic_version = tic_from_coords(self.coords)
                self.gaia = gaia_from_coords(self.coords)

            elif self.gaia is not None:
                self.coords = coords_from_gaia(self.gaia)
                self.tic, self.tess_mag, sep, self.tic_version = tic_from_coords(self.coords)

            elif self.tic is not None:
                self.coords, self.tess_mag, self.tic_version = coords_from_tic(self.tic)
                self.gaia = gaia_from_coords(self.coords)

            else:
                assert False, ("Source: one of the following keywords must be given: "
                               "tic, gaia, coords, fn.")
                

            self.tess_mag = self.tess_mag[0]            
            if tc == False:
                self.locate_on_tess() # sets sector, camera, chip, postcard,
                                  # position_on_chip, position_on_postcard
            if tc == True:
                self.tesscut_size = 31
                self.locate_with_tesscut() # sets sector, camera, chip, postcard,
                                  # position_on_chip, position_on_postcard
            
                
        self.ELEANORURL = 'https://users.flatironinstitute.org/dforeman/public_www/tess/postcards_test/s{0:04d}/{1}-{2}/'.format(self.sector,
                                                                                                                                 self.camera,
                                                                                                                                 self.chip)


    def locate_on_chip(self):#, guide):
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
#                    xy = np.array([xy[0]+x_zero, xy[1]+y_zero])
                    if (44 <= xy[0] < 2092) & (0 <= xy[1] < 2048):
                        self.sector = sec
                        self.camera = cam
                        self.chip = chip
                        self.position_on_chip = np.ravel(xy)
                        self.position_on_chip[0] += 44

        self.sector=None

        if self.usr_sec is None:
            self.usr_sec = 'recent'

        if self.usr_sec is not None:
            if type(self.usr_sec) == int:
                guide = load_postcard_guide(self.usr_sec)
                if guide is None:
                    raise SearchError("Sorry, this sector isn't available yet. We're working on it!")
                else:
                    cam_chip_loop(self.usr_sec)

            # Searches for the most recent sector the object was observed in
            elif self.usr_sec.lower() == 'recent':
                for s in np.arange(15,0,-1):
                    guide = load_postcard_guide(s)
                    if guide is not None:
                        cam_chip_loop(s)
                        if self.sector is not None:
                            break
            if self.sector is None:
                raise SearchError("TESS has not (yet) observed your target.")

        return guide


    def locate_on_tess(self):
        """Finds the best TESS postcard(s) and the position of the source on postcard.
        Sets attributes postcard, position_on_postcard, all_postcards.
        """
        self.locate_on_chip()
        guide = load_postcard_guide(self.sector)

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
        

    def locate_with_tesscut(self):
        """
        Finds the best TESS postcard(s) and the position of the source on postcard.

        Attributes
        ----------
        postcard : list
        postcard_path : str
        position_on_postcard : list 
        all_postcards : list
        sector : int
        camera : int
        chip : int
        position_on_chip : np.array

        """
        self.postcard = []
        self.position_on_postcard = []
        self.all_postcards = []
        
        self.tc = True
        
        coord = SkyCoord(self.coords[0], self.coords[1], unit="deg")
        
        sector_table = Tesscut.get_sectors(coord)
        self.sector = self.usr_sec

        self.camera = sector_table[sector_table['sector'] == self.sector]['camera'].quantity[0]
        self.chip = sector_table[sector_table['sector'] == self.sector]['ccd'].quantity[0]

        download_dir = self.tesscut_dir()

        fn_exists = self.search_tesscut(download_dir, coord)

        if fn_exists is None:
            manifest = Tesscut.download_cutouts(coord, self.tesscut_size, sector=self.usr_sec, path=download_dir)
            cutout = fits.open(manifest['Local Path'][0])
            self.postcard_path = manifest['Local Path'][0]
        else:
            self.postcard_path = fn_exists
            cutout = fits.open(fn_exists)
        
        self.cutout = cutout
        
        xcoord = cutout[1].header['1CRV4P']
        ycoord = cutout[1].header['2CRV4P']
        
        self.position_on_chip = np.array([xcoord, ycoord])


    def search_tesscut(self, download_dir, coords):
        """Searches to see if the TESSCut cutout has already been downloaded.
        """
        ra =  format(coords.ra.deg, '.6f')
        dec = format(coords.dec.deg, '.6f')

        tesscut_fn = "tess-s{0:04d}-{1}-{2}_{3}_{4}_{5}x{5}_astrocut.fits".format(self.sector,
                                                                                  self.camera,
                                                                                  self.chip,
                                                                                  ra, dec,
                                                                                  self.tesscut_size)
        local_path = os.path.join(download_dir, tesscut_fn)
        if os.path.isfile(local_path):
            return local_path
        else:
            return None

    def tesscut_dir(self):
        """Creates a TESSCut directory in the hidden eleanor directory.
        """
        download_dir = os.path.join(os.path.expanduser('~'), '.eleanor/tesscut')
        if os.path.isdir(download_dir) is False:
            try:
                os.mkdir(download_dir)
            except OSError:
                download_dir = '.'
                warnings.warn('Warning: unable to create {}. '
                              'Downloading TessCut to the current '
                              'working directory instead.'.format(download_dir))
        return download_dir
