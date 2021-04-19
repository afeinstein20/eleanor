import numpy as np
from astropy.wcs import WCS
from astropy.table import Table
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
import re
import os
import sys
from os.path import join, abspath
from tess_stars2px import tess_stars2px_function_entry as tess_stars2px
import warnings
from astroquery.mast import Tesscut
from astroquery.mast import Observations

from . import PACKAGEDIR

import urllib
from .ffi import check_pointing
from .mast import *
from .utils import *
from .maxsector import maxsector
from .update import *

__all__ = ['Source', 'multi_sectors']


def multi_sectors(sectors, tic=None, gaia=None, coords=None, name=None, tc=False, local=False, post_dir=None, pm_dir=None,
                  metadata_path=None, tesscut_size=31):
    """Obtain a list of Source objects for a single target, for each of multiple sectors for which the target was observed.

    Parameters
    ----------
    sectors : list or str
        The list of sectors for which data should be returned, or `'all'` to return all sectors
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
    tesscut_size : int, array-like, astropy.units.Quantity
        The size of the cutout array, when tc is True. Details can be seen in
        astroquery.mast.TesscutClass.download_cutouts
    """
    objs = []

    if sectors == 'all':
        if coords is None:
            if tic is not None:
                coords, _, _, _ = coords_from_tic(tic)
            elif gaia is not None:
                coords = coords_from_gaia(gaia)
            elif name is not None:
                coords = coords_from_name(name)

        if coords is not None:
            if type(coords) is SkyCoord:
                coords = (coords.ra.degree, coords.dec.degree)
            result = tess_stars2px(8675309, coords[0], coords[1])
            sector = result[3][result[3] < maxsector + 0.5]
            sectors = sector.tolist()

        if len(sectors) == 0 or sectors[0] < 0:
            raise SearchError("Your target is not observed by TESS, or maybe you need to run eleanor.Update()")
        else:
            print('Found star in Sector(s) ' +" ".join(str(x) for x in sectors))

    if (type(sectors) == list) or (type(sectors) == np.ndarray):
        for s in sectors:
            star = Source(tic=tic, gaia=gaia, coords=coords, sector=int(s), tc=tc, local=local, post_dir=post_dir, pm_dir=pm_dir,
                          metadata_path=metadata_path, tesscut_size=tesscut_size)
            if star.sector is not None:
                objs.append(star)
        if len(objs) < len(sectors):
            warnings.warn('Only {} targets found instead of desired {}. Your '
                          'target may not have been observed yet in these sectors.'
                          ''.format(len(objs), len(sectors)))
        return objs


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
    name : str, optional
        The name of your target (e.g. "HD#####" or "M31").
    fn : str, optional
        Filename of a TPF corresponding to the desired source.
    sector : int or str
        The sector for which data should be returned, or `recent` to
        obtain data for the most recent sector which contains this target.
    tc : bool, optional
        If True, use a TessCut cutout to produce postcards rather than downloading the eleanor
        postcard data products.
    tesscut_size : int, array-like, astropy.units.Quantity
        The size of the cutout array, when tc is True. Details can be checked in
        astroquery.mast.TesscutClass.download_cutouts

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
    def __init__(self, tic=None, gaia=None, coords=None, name=None, fn=None,
                 sector=None, fn_dir=None, tc=False, local=False, post_dir=None, pm_dir=None,
                 metadata_path=None, tesscut_size=31):
        self.tic       = tic
        self.gaia      = gaia
        self.coords    = coords
        self.name      = name
        self.fn        = fn
        self.premade   = False
        self.usr_sec   = sector
        self.tc        = tc
        self.contratio = None
        self.postcard_path = post_dir
        self.pm_dir = pm_dir
        self.local = local



        if self.pm_dir is None:
            self.pm_dir = self.postcard_path

        if fn_dir is None:
            self.fn_dir = os.path.join(os.path.expanduser('~'), '.eleanor')
            if not os.path.exists(self.fn_dir):
                try:
                    os.mkdir(self.fn_dir)
                except OSError:
                    self.fn_dir = '.'
                    warnings.warn('Warning: unable to create {}. '
                                  'Downloading to the current '
                                  'working directory instead.'.format(self.fn_dir))


        else:
            self.fn_dir  = fn_dir

        self.eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')

        if metadata_path is None:
            self.metadata_path = self.eleanorpath
        else:
            self.metadata_path = metadata_path

        if not os.path.exists(self.eleanorpath):
            try:
                os.mkdir(self.eleanorpath)
            except OSError:
                self.eleanorpath = os.path.dirname(__file__)

        if not os.path.exists(self.metadata_path + '/metadata'):
            os.mkdir(self.metadata_path + '/metadata')

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
            self.chip     = hdr['CCD']
            self.tc       = hdr['TESSCUT']
            self.tic_version = hdr['TIC_V']
            self.postcard = hdr['POSTCARD']

            if self.tc is True:
                post_dir = self.tesscut_dir()
                self.postcard_path = os.path.join(post_dir, self.postcard)
                # workaround to address #137
                # - both self.postcard  and tesscut_dir() contains tesscut, resulting in incorrect path
                self.postcard_path = re.sub(r'tesscut[/\\]tesscut', 'tesscut', self.postcard_path)
                self.cutout = fits.open(self.postcard_path)
#            else:
                #########

            self.position_on_chip = (hdr['CHIPPOS1'], hdr['CHIPPOS2'])
#            self.position_on_postcard = (hdr['POSTPOS1'], hdr['POSTPOS2'])

        else:
            if self.coords is not None:
                if type(self.coords) is SkyCoord:
                    self.coords = (self.coords.ra.degree, self.coords.dec.degree)
                elif (len(self.coords) == 2) & (all(isinstance(c, float) for c in self.coords) | all(isinstance(c, int) for c in self.coords) ):
                    self.coords = coords
                else:
                    assert False, ("Source: invalid coords. Valid input types are: "
                                   "(RA [deg], Dec [deg]) tuple or astropy.coordinates.SkyCoord object.")

                if self.tic is None:
                    self.tic, self.tess_mag, sep, self.tic_version, self.contratio = tic_from_coords(self.coords)
                else:
                    self.tess_mag = [999]
                    self.tic_version = None
                    self.contratio = 0.0
                if self.gaia is None:
                    self.gaia = gaia_from_coords(self.coords)

            elif self.gaia is not None:
                self.coords = coords_from_gaia(self.gaia)
                self.tic, self.tess_mag, sep, self.tic_version, self.contratio = tic_from_coords(self.coords)

            elif self.tic is not None:
                self.coords, self.tess_mag, self.tic_version, self.contratio = coords_from_tic(self.tic)
                self.gaia = gaia_from_coords(self.coords)

            elif self.name is not None:
                self.coords = coords_from_name(self.name)
                self.tic, self.tess_mag, sep, self.tic_version, self.contratio = tic_from_coords(self.coords)
                self.gaia = gaia_from_coords(self.coords)

            else:
                assert False, ("Source: one of the following keywords must be given: "
                               "tic, gaia, coords, fn.")

            if isinstance(self.tess_mag,list):
                self.tess_mag = self.tess_mag[0]

            self.locate_on_tess()
            self.tesscut_size = tesscut_size

            if not os.path.isdir(self.metadata_path + '/metadata/s{:04d}'.format(self.sector)):
                Update(sector=self.sector)

            if tc == False:
                self.locate_postcard(local)
            if tc == True:
                self.locate_with_tesscut() # sets sector, camera, chip, postcard,
                                  # position_on_chip, position_on_postcard


    def locate_on_tess(self):
        """Finds the TESS sector, camera, chip, and position on chip for the source.

        Attributes
        ----------
        sector : int
        camera : int
        chip : int
        position_on_chip : np.array
        """
        sector = None

        if self.usr_sec is None:
            self.usr_sec = 'recent'

        result = tess_stars2px(self.tic, self.coords[0], self.coords[1])
        cameras = result[4][result[3] <= maxsector]
        chips = result[5][result[3] <= maxsector]
        cols = result[6][result[3] <= maxsector]
        rows = result[7][result[3] <= maxsector]
        sectors = result[3][result[3] <= maxsector]

        # tess_stars2px returns array [-1] when star not observed yet
        if len(sectors) < 1 or sectors[0] == np.array([-1]):
            raise SearchError("Tess has not (yet) observed your target.")

        else:
            # Handles cases where users can pass in their sector
            if type(self.usr_sec) == int:
                arg = np.argwhere(sectors == self.usr_sec)[0]
                if len(arg) > 0:
                    self.sector = sectors[arg][0]
                    camera = cameras[arg][0]
                    chip   = chips[arg][0]
                    position_on_chip = np.array([cols[arg][0], rows[arg][0]])

            # Handles cases where the user does not pass in a sector
            elif self.usr_sec.lower() == 'recent':
                self.sector = sectors[-1]
                camera = cameras[-1]
                chip   = chips[-1]
                position_on_chip = np.array([cols[-1], rows[-1]])

        if self.sector is None or type(self.sector) == np.ndarray:
            raise SearchError("TESS has not (yet) observed your target.")
        else:
            self.camera = camera
            self.chip   = chip
            self.position_on_chip = position_on_chip

    def locate_postcard(self, local):
        """ Finds the eleanor postcard, if available, this star falls on.

        Attributes
        ----------
        postcard : str
        postcard_bkg : str
        postcard_path : str
        position_on_postcard : list
        all_postcards : list
        mast_list : astropy.table.Table
        """
        self.mast_list = None

        info_str = "{0:04d}-{1}-{2}-{3}".format(self.sector, self.camera, self.chip, "cal")
        postcard_fmt = "postcard-s{0}-{{0:04d}}-{{1:04d}}"
        postcard_fmt = postcard_fmt.format(info_str)


        eleanorpath = os.path.dirname(__file__)

        guide_url = eleanorpath + '/postcard_centers.txt'
        guide     = Table.read(guide_url, format="ascii")

        col, row = self.position_on_chip[0], self.position_on_chip[1]

        post_args = np.where( (np.abs(guide['x'].data - col) <= 100) &
                              (np.abs(guide['y'].data - row) <= 100) )

        post_cens = guide[post_args]

        # Finds the mostest closest postcard
        closest_x, closest_y = np.argmin(np.abs(post_cens['x'] - col)), np.argmin(np.abs(post_cens['y'] - row))
        self.postcard = postcard_fmt.format(post_cens['x'].data[closest_x],
                                            post_cens['y'].data[closest_y])

        # Keeps track of all postcards that the star falls on
        all_postcards = []
        for i in range(len(post_cens)):
            name = postcard_fmt.format(post_cens['x'].data[i],
                                       post_cens['y'].data[i])
            all_postcards.append(name)
        self.all_postcards = np.array(all_postcards)


        if local == False:

            postcard_obs = Observations.query_criteria(provenance_name="ELEANOR",
                                                       target_name=self.postcard,
                                                       obs_collection="HLSP")

            if len(postcard_obs) > 0:
                product_list = Observations.get_product_list(postcard_obs)
                self.pointing = check_pointing(self.sector, self.camera, self.chip, self.pm_dir)

                if self.pointing is None:
                    extension = ["pc.fits", "bkg.fits", "pm.txt"]
                else:
                    extension = ["pc.fits", "bkg.fits"]

                results = Observations.download_products(product_list, extension=extension,
                                                         download_dir=self.postcard_path)
                postcard_path = results['Local Path'][0]
                self.postcard_path = '/'.join(e for e in postcard_path.split('/')[:-1])
                self.postcard  = results['Local Path'][1].split('/')[-1]
                self.postcard_bkg = results['Local Path'][0].split('/')[-1]
                self.mast_results = results
                self.cutout    = None  # Attribute for TessCut only
                # only downloaded the pointing model if the search for it above failed, so only
                # update it in that case here
                if self.pointing is None:
                    self.pm_dir = self.postcard_path
                    self.pointing = check_pointing(self.sector, self.camera, self.chip, self.pm_dir)


            else:
                print("No eleanor postcard has been made for your target (yet). Using TessCut instead.")
                self.locate_with_tesscut()

        else:
            self.cutout = None #Attribute for TessCut only
            self.postcard_bkg = 'hlsp_eleanor_tess_ffi_' + self.postcard + '_tess_v2_bkg.fits'
            self.postcard = 'hlsp_eleanor_tess_ffi_' + self.postcard + '_tess_v2_pc.fits'

            self.pointing = check_pointing(self.sector, self.camera, self.chip, self.pm_dir)



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
        # Attributes only when using postcards
        self.all_postcards = None

        # Attribute for TessCut
        self.tc = True

        download_dir = self.tesscut_dir()

        coords = SkyCoord(self.coords[0], self.coords[1],
                          unit=(u.deg, u.deg))

        fn_exists = self.search_tesscut(download_dir, coords)

        if fn_exists is None:
            manifest = Tesscut.download_cutouts(coords, self.tesscut_size, sector=self.sector, path=download_dir)
            cutout = fits.open(manifest['Local Path'][0])
            self.postcard_path = manifest['Local Path'][0]
        else:
            self.postcard_path = fn_exists
            cutout = fits.open(fn_exists)

        self.cutout   = cutout
        self.postcard = self.postcard_path.split('/')[-1]

        xcoord = cutout[1].header['1CRV4P']
        ycoord = cutout[1].header['2CRV4P']

        self.position_on_chip = np.array([xcoord, ycoord])


    def search_tesscut(self, download_dir, coords):
        """Searches to see if the TESSCut cutout has already been downloaded.
        """
        ra = format(coords.ra.deg, '.6f')
        dec = format(coords.dec.deg, '.6f')

        if isinstance(self.tesscut_size, int):
            tesscut_fn = "tess-s{0:04d}-{1}-{2}_{3}_{4}_{5}x{6}_astrocut.fits".format(self.sector,
                                                                                  self.camera,
                                                                                  self.chip,
                                                                                  ra, dec,
                                                                                  self.tesscut_size, self.tesscut_size)
        else:
            tesscut_fn = "tess-s{0:04d}-{1}-{2}_{3}_{4}_{5}x{6}_astrocut.fits".format(self.sector,
                                                                                  self.camera,
                                                                                  self.chip,
                                                                                  ra, dec,
                                                                                  self.tesscut_size[1],
                                                                                  self.tesscut_size[0])

        local_path = os.path.join(download_dir, tesscut_fn)
        if os.path.isfile(local_path):
            return local_path
        else:
            return None

    def tesscut_dir(self):
        """Creates a TESSCut directory in the hidden eleanor directory.
        """
        if self.postcard_path is None:
            download_dir = os.path.join(os.path.expanduser('~'), '.eleanor/tesscut')
        else:
            download_dir = self.postcard_path
        if os.path.isdir(download_dir) is False:
            try:
                os.mkdir(download_dir)
            except OSError:
                download_dir = '.'
                warnings.warn('Warning: unable to create {}. '
                              'Downloading TessCut to the current '
                              'working directory instead.'.format(download_dir))
        return download_dir
