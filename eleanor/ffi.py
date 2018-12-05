import os, tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from muchbettermoments import quadratic_2d
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
import requests
from bs4 import BeautifulSoup
import warnings

from .mast import tic_by_contamination

def use_pointing_model(coords, pointing_model):
    """Applies pointing model to correct the position of star(s) on postcard.

    Parameters
    ----------
    coords : tuple
        (`x`, `y`) position of star(s).

    pointing_model : astropy.table.Table
        pointing_model for ONE cadence.

    Returns
    -------
    coords : tuple
        Corrected position of star(s).
    """
    pointing_model = np.reshape(list(pointing_model), (3,3))
    A = np.column_stack([coords[0], coords[1], np.ones_like(coords[0])])
    fhat = np.dot(A, pointing_model)
    return fhat[:,0:2]


class ffi:
    """This class allows the user to download all full-frame images for a given sector,
         camera, and chip. It also allows the user to create their own pointing model
         based on each cadence for a given combination of sector, camera, and chip.

    No individual user should have to download all of the full-frame images because
         stacked postcards will be available for the user to download from MAST.

    Parameters
    ----------
    sector : int, optional
    camera : int, optional
    chip : int, optional
    """
    def __init__(self, sector=None, camera=None, chip=None):
        self.sector = sector
        self.camera = camera
        self.chip   = chip


    def download_ffis(self, download_dir=None):
        """
        Downloads entire sector of data into FFI download directory.

        Parameters
        ----------
        download_dir : str
            Location where the data files will be stored.
            Defaults to "~/.eleanor/sector_{}/ffis" if `None` is passed.
        """

        def findAllFFIs(ca, ch):
            nonlocal year, days, url
            calFiles, urlPaths = [], []
            for d in days:
                path = '/'.join(str(e) for e in [url, year, d, ca])+'-'+str(ch)+'/'
                for fn in BeautifulSoup(requests.get(path).text, "lxml").find_all('a'):
                    if fn.get('href')[-7::] == 'ic.fits':
                        calFiles.append(fn.get('href'))
                        urlPaths.append(path)
            return calFiles, urlPaths

        if self.sector in np.arange(1,14,1):
            year=2019
        else:
            year=2020
        # Current days available for ETE-6
        days = np.arange(129,158,1)

        # This URL applies to ETE-6 simulated data ONLY
        url = 'https://archive.stsci.edu/missions/tess/ete-6/ffi/'
        files, urlPaths = findAllFFIs(self.camera, self.chip)

        if download_dir is None:
            # Creates hidden .eleanor FFI directory
            ffi_dir = self._fetch_ffi_dir()
        else:
            ffi_dir = download_dir

        files_in_dir = os.listdir(ffi_dir)

        local_paths = []
        for i in range(len(files)):
            if files[i] not in files_in_dir:
                os.system('cd {} && curl -O -L {}'.format(ffi_dir, urlPaths[i]+files[i]))
            local_paths.append(ffi_dir+files[i])
        self.local_paths = np.array(local_paths)
        return


    def _fetch_ffi_dir(self):
        """Returns the default path to the directory where FFIs will be downloaded.

        By default, this method will return "~/.eleanor/sector_{}/ffis" and create
        this directory if it does not exist.  If the directory cannot be
        access or created, then it returns the local directory (".").

        Returns
        -------
        download_dir : str
            Path to location of `ffi_dir` where FFIs will be downloaded
        """
        download_dir = os.path.join(os.path.expanduser('~'), '.eleanor',
                                    'sector_{}'.format(self.sector), 'ffis')
        if os.path.isdir(download_dir):
            return download_dir
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(download_dir)
            # downloads locally if OS error occurs
            except OSError:
                warnings.warn('Warning: unable to create {}. '
                              'Downloading FFIs to the current '
                              'working directory instead.'.format(download_dir))
                download_dir = '.'

        return download_dir


    def sort_by_date(self):
        """Sorts FITS files by start date of observation."""
        dates, time = [], []
        for f in self.local_paths:
            hdu = fits.open(f)
            hdr = hdu[1].header
            dates.append(hdr['DATE-OBS'])
        dates, fns = np.sort(np.array([dates, self.local_paths]))
        self.local_paths = fns
        self.dates = dates
        return


    def build_pointing_model(self, pos_predicted, pos_inferred, outlier_removal=False):
        """Builds an affine transformation to correct the positions of stars
           from a possibly incorrect WCS.

        Parameters
        ----------
        pos_predicted : tuple
            Positions taken straight from the WCS; [[x,y],[x,y],...] format.
        pos_inferred : tuple
            Positions taken using any centroiding method; [[x,y],[x,y],...] format.
        outlier_removal : bool, optional
            Whether to clip 1-sigma outlier frames. Default `False`.

        Returns
        -------
        xhat : np.ndarray
            (3, 3) affine transformation matrix between WCS positions 
            and inferred positions.
        """
        A = np.column_stack([pos_predicted[:,0], pos_predicted[:,1], np.ones_like(pos_predicted[:,0])])
        f = np.column_stack([pos_inferred[:,0], pos_inferred[:,1], np.ones_like(pos_inferred[:,0])])
        if outlier_removal == True:
            dist = np.sqrt(np.sum((A - f)**2, axis=1))
            mean, std = np.nanmean(dist), np.nanstd(dist)
            lim = 1.0
            A = A[dist < mean + lim*std]
            f = f[dist < mean + lim*std]
        ATA = np.dot(A.T, A)
        ATAinv = np.linalg.inv(ATA)
        ATf = np.dot(A.T, f)
        xhat = np.dot(ATAinv, ATf)
        fhat = np.dot(A, xhat)

        return xhat


    def pointing_model_per_cadence(self):
        """Step through build_pointing_model for each cadence."""

        def find_isolated(x, y):
            """Finds the most isolated, least contaminated sources for pointing model."""
            isolated = []
            for i in range(len(x)):
                x_list  = np.delete(x, np.where(x==x[i]))
                y_list  = np.delete(y, np.where(y==y[i]))
                closest = np.sqrt( (x[i]-x_list)**2 + (y[i]-y_list)**2 ).argmin()
                dist    = np.sqrt( (x[i]-x_list[closest])**2+ (y[i]-y_list[closest])**2 )
                if dist > 8.0:
                    isolated.append(i)
            return np.array(isolated)


        def isolated_center(x, y, image):
            """Finds the centroid of each isolated source with quadratic_2d.

            Parameters
            ----------
            x : array-like
                Initial guesses of x positions for sources.
            y : array-like
                Initial guesses of y positions for sources.            
            image : np.ndarray
                FFI flux data.
                
            Returns
            -------
            cenx : array-like
                Controid x coordinates for all good input sources.
            ceny : array-like
                Centroid y coordinates for all good input sources.
            good : list
                Indexes into input arrays corresponding to good sources.
            """
            cenx, ceny, good = [], [], []
            print("Finding isolated centers of sources")
            for i in range(len(x)):
                 if x[i] > 0. and y[i] > 0.:
                     tpf = Cutout2D(image, position=(x[i], y[i]), size=(7,7), mode='partial')
                     origin = tpf.origin_original
                     cen = quadratic_2d(tpf.data)
                     cenx.append(cen[0]+origin[0]); ceny.append(cen[1]+origin[1])
                     good.append(i)
            cenx, ceny = np.array(cenx), np.array(ceny)
            return cenx, ceny, good

        def apply_pointing_model(xy, matrix):
            pointing_model = matrix
            centroid_xs, centroid_ys = [], []
            new_coords = use_pointing_model(xy, pointing_model)
            return np.array(new_coords)


        pm_fn = 'pointingModel_{}_{}-{}.txt'.format(self.sector, self.camera, self.chip)
        with open(pm_fn, 'w') as tf:
            tf.write('0 1 2 3 4 5 6 7 8\n')

        hdu = fits.open(self.local_paths[0])
        hdr = hdu[1].header
        pos = [hdr['CRVAL1'], hdr['CRVAL2']]

        r = 6.0#*np.sqrt(1.2)
        contam = [0.0, 5e-3]
        tmag_lim = 12.5

        t  = tic_by_contamination(pos, r, contam, tmag_lim)

        for fn in self.local_paths:
            print(fn)
            hdu = fits.open(fn)
            hdr = hdu[1].header
            xy = WCS(hdr).all_world2pix(t['ra'], t['dec'], 1)
            # Triple checks the sources are on the FFI
            onFrame = np.where( (xy[0]>10) & (xy[0]<2092-10) & (xy[1]>10) & (xy[1]<2048-10) )[0]
            xy  = np.array([xy[0][onFrame], xy[1][onFrame]])
            iso = find_isolated(xy[0], xy[1])
            xy  = np.array([xy[0][iso], xy[1][iso]])
            cenx, ceny, good = isolated_center(xy[0], xy[1], hdu[1].data)

            # Triple checks there are no nans; Nans make people sad
            no_nans = np.where( (np.isnan(cenx)==False) & (np.isnan(ceny)==False))
            pos_inferred = np.array( [cenx[no_nans], ceny[no_nans]] )
            xy = np.array( [xy[0][no_nans], xy[1][no_nans]] )

            solution = self.build_pointing_model(xy.T, pos_inferred.T)

            xy = apply_pointing_model(xy.T, solution)
            matrix = self.build_pointing_model(xy, pos_inferred.T, outlier_removal=True)

            sol    = np.dot(matrix, solution)
            sol    = sol.flatten()

            with open(pm_fn, 'a') as tf:
                tf.write('{}\n'.format(' '.join(str(e) for e in sol) ) )
