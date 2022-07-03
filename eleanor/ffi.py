import os, tqdm
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS, NoConvergence
from astropy.table import Table
from astropy.nddata import Cutout2D
from astropy.utils.data import download_file
import requests
from bs4 import BeautifulSoup
import warnings
import urllib
from .utils import EleanorWarning
from .mast import tic_by_contamination


def check_pointing(sector, camera, chip, path=None):
    """ Checks to see if a pointing model exists locally already.
    """
    # Tries to create a pointing model directory
    if path == None:
        pm_dir = '.'
    else:
        pm_dir = path

    searches = [
        's{0:04d}-{1}-{2}_tess_v2_pm.txt'.format(sector, camera, chip),
        'pointingModel_{0:04d}_{1}-{2}.txt'.format(sector, camera, chip),
        'hlsp_eleanor_tess_ffi_postcard-s{0:04d}-{1}-{2}_tess_v2_pm.txt'.format(sector, camera, chip)
    ]

    # Checks a directory of pointing models, if it exists
    # Returns the pointing model if it's in the pointing model directory
    for search in searches:
        if not os.path.isdir(pm_dir):
            continue
        pm_downloaded = os.listdir(pm_dir)
        pm = [i for i in pm_downloaded if search in i]
        if len(pm) > 0:
            return Table.read(os.path.join(pm_dir, pm[0]), format="ascii.basic")
    warnings.warn("couldn't find pointing model", category=EleanorWarning)


def load_pointing_model(pm_dir, sector, camera, chip):
    """ Loads in pointing model.
    """
    return check_pointing(sector, camera, chip, path=pm_dir)


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
    return fhat


def pm_quality(time, sector, camera, chip, pm=None, pm_dir=None):
        """ Fits a line to the centroid motions using the pointing model.
            A quality flag is set if the centroid is > 2*sigma away from
                the majority of the centroids.
        """

        def outliers(x, y, poly, mask):
            dist = (y - poly[0]*x - poly[1])/np.sqrt(poly[0]**2+1**2)
            std  = np.std(dist)
            ind  = np.where((dist > 2*std) | (dist < -2*std))[0]
            mask[ind] = 1
            return mask

        cen_x, cen_y = 1024, 1024 # Uses a point in the center of the FFI
        cent_x,cent_y = [], []

        if pm is None:
            pm = load_pointing_model(pm_dir, sector, camera, chip)

        # Applies centroids
        for i in range(len(pm)):
            new_coords = use_pointing_model(np.array([cen_x, cen_y]), pm[i])
            cent_x.append(new_coords[0][0])
            cent_y.append(new_coords[0][1])
        cent_x = np.array(cent_x); cent_y = np.array(cent_y)

        # Finds gap in orbits
        t = np.diff(time)
        brk = np.where( t == np.max(t))[0][0]
        brk += 1

        # Initiates lists for each orbit
        x1 = cent_x[0:brk]; y1 = cent_y[0:brk]
        x2 = cent_x[brk:len(cent_x)+1];y2 = cent_y[brk:len(cent_y)+1]

        # Initiates masks
        mask1 = np.zeros(len(x1)); mask2 = np.zeros(len(x2))


        # Loops through and searches for points > 2 sigma away from distribution
        for i in np.arange(0,10,1):
            poly1  = np.polyfit(x1[mask1==0], y1[mask1==0], 1)
            poly2  = np.polyfit(x2[mask2==0], y2[mask2==0], 1)
            mask1  = outliers(x1, y1, poly1, mask1)
            mask2  = outliers(x2, y2, poly2, mask2)

        # Returns a total mask for each orbit
        return np.append(mask1, mask2)


def set_quality_flags(ffi_start, ffi_stop, shortCad_fn, sector, camera, chip,
                      pm=None, pm_dir=None):
    """ Uses the quality flags in a 2-minute target to create quality flags
        in the postcards.
    We create our own quality flag as well, using our pointing model.
    """
    # Obtains information for 2-minute target
    twoMin     = fits.open(shortCad_fn)
    twoMinTime = twoMin[1].data['TIME']-twoMin[1].data['TIMECORR']
    finite     = np.isfinite(twoMinTime)
    twoMinQual = twoMin[1].data['QUALITY']

    twoMinTime = twoMinTime[finite]
    twoMinQual = twoMinQual[finite]

    perFFIcad = []
    for i in range(len(ffi_start)):
        where = np.where( (twoMinTime > ffi_start[i]) &
                          (twoMinTime < ffi_stop[i]) )[0]
        perFFIcad.append(where)

    perFFIcad = np.array(perFFIcad)
    # Binary string for values which apply to the FFIs
    ffi_apply = int('100010101111', 2)

    convolve_ffi = []
    for cadences in perFFIcad:
        v = np.bitwise_or.reduce(twoMinQual[cadences])
        convolve_ffi.append(v)
    convolve_ffi = np.array(convolve_ffi)

    flags    = np.bitwise_and(convolve_ffi, ffi_apply)
    pm_flags = pm_quality(ffi_stop, sector, camera,
                          chip, pm=pm, pm_dir=pm_dir) * 131072

    pm_flags[ ((ffi_stop>1420.) & (ffi_stop < 1424.)) ] = 131072

    return np.bitwise_or(flags, pm_flags)

def centroid_quadratic(data, mask=None):
    """Computes the quadratic estimate of the centroid in a 2d-array.

    This method will fit a simple 2D second-order polynomial
    $P(x, y) = a + bx + cy + dx^2 + exy + fy^2$
    to the 3x3 patch of pixels centered on the brightest pixel within
    the image.  This function approximates the core of the Point
    Spread Function (PSF) using a bivariate quadratic function, and returns
    the maximum (x, y) coordinate of the function using linear algebra.

    For the motivation and the details around this technique, please refer
    to Vakili, M., & Hogg, D. W. 2016, ArXiv, 1610.05873.

    Caveat: if the brightest pixel falls on the edge of the data array, the fit
    will tend to fail or be inaccurate.

    As used in the Lightkurve package of Barentsen et al.

    Parameters
    ----------
    data : 2D array
        The 2D input array representing the pixel values of the image.
    mask : array_like (bool), optional
        A boolean mask, with the same shape as `data`, where a **True** value
        indicates the corresponding element of data is masked.

    Returns
    -------
    column, row : tuple
        The coordinates of the centroid in column and row.  If the fit failed,
        then (NaN, NaN) will be returned.
    """
    # Step 1: identify the patch of 3x3 pixels (z_)
    # that is centered on the brightest pixel (xx, yy)
    if mask is not None:
        data = data * mask
    arg_data_max = np.nanargmax(data)
    yy, xx = np.unravel_index(arg_data_max, data.shape)
    # Make sure the 3x3 patch does not leave the TPF bounds
    if yy < 1:
        yy = 1
    if xx < 1:
        xx = 1
    if yy > (data.shape[0] - 2):
        yy = data.shape[0] - 2
    if xx > (data.shape[1] - 2):
        xx = data.shape[1] - 2

    z_ = data[yy-1:yy+2, xx-1:xx+2]

    # Next, we will fit the coefficients of the bivariate quadratic with the
    # help of a design matrix (A) as defined by Eqn 20 in Vakili & Hogg
    # (arxiv:1610.05873). The design matrix contains a
    # column of ones followed by pixel coordinates: x, y, x**2, xy, y**2.
    A = np.array([[1, -1, -1, 1,  1, 1],
                  [1,  0, -1, 0,  0, 1],
                  [1,  1, -1, 1, -1, 1],
                  [1, -1,  0, 1,  0, 0],
                  [1,  0,  0, 0,  0, 0],
                  [1,  1,  0, 1,  0, 0],
                  [1, -1,  1, 1, -1, 1],
                  [1,  0,  1, 0,  0, 1],
                  [1,  1,  1, 1,  1, 1]])
    # We also pre-compute $(A^t A)^-1 A^t$, cf. Eqn 21 in Vakili & Hogg.
    At = A.transpose()
    # In Python 3 this can become `Aprime = np.linalg.inv(At @ A) @ At`
    Aprime = np.matmul(np.linalg.inv(np.matmul(At, A)), At)

    # Step 2: fit the polynomial $P = a + bx + cy + dx^2 + exy + fy^2$
    # following Equation 21 in Vakili & Hogg.
    # In Python 3 this can become `Aprime @ z_.flatten()`
    a, b, c, d, e, f = np.matmul(Aprime, z_.flatten())

    # Step 3: analytically find the function maximum,
    # following https://en.wikipedia.org/wiki/Quadratic_function
    det = 4 * d * f - e ** 2
    if abs(det) < 1e-6:
        return np.nan, np.nan  # No solution
    xm = - (2 * f * b - c * e) / det
    ym = - (2 * d * c - b * e) / det
    return xx + xm, yy + ym


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
        self.ffiindex = None

    def download_ffis(self, download_dir=None):
        """
        Downloads entire sector of data into FFI download directory.

        Parameters
        ----------
        download_dir : str
            Location where the data files will be stored.
            Defaults to "~/.eleanor/sector_{}/ffis" if `None` is passed.
        """

        def findAllFFIs():
            nonlocal url
            sub_paths = []
            subsub_paths = []
            calFiles, urlPaths = [], []

            paths = BeautifulSoup(requests.get(url).text, "lxml").find_all('a')

            for direct in paths:
                subdirect = direct.get('href')
                if ('2018' in subdirect) or ('2019' in subdirect):
                    sub_paths.append(os.path.join(url, subdirect))

            for sp in sub_paths:
                for fn in BeautifulSoup(requests.get(sp).text, "lxml").find_all('a'):
                    subsub = fn.get('href')
                    if (subsub[0] != '?') and (subsub[0] != '/'):
                        subsub_paths.append(os.path.join(sp, subsub))

            subsub_paths = [os.path.join(i, '{}-{}/'.format(self.camera, self.chip)) for i in subsub_paths]

            for sbp in subsub_paths:
                for fn in BeautifulSoup(requests.get(sbp).text, "lxml").find_all('a'):
                    if 'ffic.fits' in fn.get('href'):
                        calFiles.append(fn.get('href'))
                        urlPaths.append(sbp)

            return np.array(calFiles), np.array(urlPaths)

        # This URL applies to ETE-6 simulated data ONLY
        url = 'https://archive.stsci.edu/missions/tess/ffi/'

        url = os.path.join(url, "s{0:04d}".format(self.sector))
        files, urlPaths = findAllFFIs()

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
                os.makedirs(download_dir)
            # downloads locally if OS error occurs
            except OSError:
                warnings.warn('Warning: unable to create {}. '
                              'Downloading FFIs to the current '
                              'working directory instead.'.format(download_dir))
                download_dir = '.'

        return download_dir

    def sort_by_date(self):
        """Sorts FITS files by start date of observation."""
        dates, time, index = [], [], []
        for f in self.local_paths:
            hdu = fits.open(f)
            hdr = hdu[1].header
            dates.append(hdr['DATE-OBS'])
            if 'ffiindex' in hdu[0].header:
                index.append(hdu[0].header['ffiindex'])
        if len(index) == len(dates):
            dates, index, fns = np.sort(np.array([dates, index, self.local_paths]))
            self.ffiindex = index.astype(int)
        else:
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


    def pointing_model_per_cadence(self, out_dir=None, n_sources=350):
        """Step through build_pointing_model for each cadence."""

        def find_isolated(x, y):
            """Finds the most isolated, least contaminated sources for pointing model."""
            init_d   = 8.0
            counter  = 0.0
            isolated = []

            while ((init_d - counter) > 0) and (counter < 3.5):
                for i in range(len(x)):
                    x_list  = np.delete(x, np.where(x==x[i]))
                    y_list  = np.delete(y, np.where(y==y[i]))
                    closest = np.sqrt( (x[i]-x_list)**2 + (y[i]-y_list)**2 ).argmin()
                    dist    = np.sqrt( (x[i]-x_list[closest])**2 + (y[i]-y_list[closest])**2 )
                    if (dist > (init_d - counter)) and (i not in isolated):
                        isolated.append(i)
                    if len(isolated) > n_sources:
                        break
                counter += 0.1

            return np.array(isolated)


        def isolated_center(x, y, image):
            """Finds the centroid of each isolated source with centroid_quadratic.

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

            for i in range(len(x)):
                 if x[i] > 0. and y[i] > 0.:
                     tpf = Cutout2D(image, position=(x[i], y[i]), size=(7,7), mode='partial')
                     origin = tpf.origin_original
                     cen = centroid_quadratic(tpf.data - np.nanmedian(tpf.data))
                     cenx.append(cen[0]+origin[0]); ceny.append(cen[1]+origin[1])
                     good.append(i)
            cenx, ceny = np.array(cenx), np.array(ceny)
            return cenx, ceny, good

        def apply_pointing_model(xy, matrix):
            pointing_model = matrix
            centroid_xs, centroid_ys = [], []
            new_coords = use_pointing_model(np.array(xy).T, pointing_model)
            return np.array(new_coords)

        pm_fn = 's{0:04d}-{1}-{2}_tess_v2_pm.txt'.format(self.sector, self.camera, self.chip)

        eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')

        qf = np.loadtxt(eleanorpath + '/metadata/s{0:04d}/quality_s{0:04d}.txt'.format(self.sector,
                                                                                                                        self.sector))

        if out_dir is not None:
            pm_fn = out_dir+ '/' + pm_fn

        with open(pm_fn, 'w') as tf:
            tf.write('0 1 2 3 4 5 6 7 8\n')

        pos = None
        for fn in self.local_paths:
            with fits.open(fn) as hdu:
                hdr = hdu[1].header
            if 'CRVAL1' not in hdr or 'CRVAL2' not in hdr:
                continue
            pos = [hdr['CRVAL1'], hdr['CRVAL2']]
        if pos is None:
            raise ValueError("no WCS")

        r = 6.0*np.sqrt(1.2)
        contam = [0.0, 0.5]
        tmag_lim = [7.5, 12.5]

        t  = tic_by_contamination(pos, r, contam, tmag_lim).group_by('contratio')

        for i, fn in enumerate(self.local_paths):
            with fits.open(fn) as hdu:
                hdr = hdu[1].header
                data = hdu[1].data

            try:
                xy = WCS(hdr).all_world2pix(t['ra'], t['dec'], 1)

                # Triple checks the sources are on the FFI
                onFrame = np.where( (xy[0]>10) & (xy[0]<2092-10) & (xy[1]>10) & (xy[1]<2048-10) )[0]
                xy  = np.array([xy[0][onFrame], xy[1][onFrame]])
                iso = find_isolated(xy[0], xy[1])
                if len(iso) > 0:
                    xy  = np.array([xy[0][iso], xy[1][iso]])
                    cenx, ceny, good = isolated_center(xy[0], xy[1], data)

                    # Triple checks there are no nans; Nans make people sad
                    no_nans = np.where( (np.isnan(cenx)==False) & (np.isnan(ceny)==False))
                    pos_inferred = np.array( [cenx[no_nans], ceny[no_nans]] )
                    xy = np.array( [xy[0][no_nans], xy[1][no_nans]] )

                    solution = self.build_pointing_model(xy.T, pos_inferred.T)

                    xy = apply_pointing_model(xy.T, solution)
                    matrix = self.build_pointing_model(xy, pos_inferred.T, outlier_removal=True)

                    sol    = np.dot(matrix, solution)
                    sol    = sol.flatten()
                else:
                    sol = np.full((9,), 1e5)

            except NoConvergence:
                if qf[i] != 0:
                    sol = np.full((9,), 1e5)
                else:
                    a   = np.zeros((3, 3), int)
                    np.fill_diagonal(a, 1)
                    sol = np.reshape(a, (9,))

            with open(pm_fn, 'a') as tf:
                tf.write('{}\n'.format(' '.join(str(e) for e in sol) ) )

        with open(pm_fn, "r") as tf:
            return Table.read(tf.read(), format='ascii.basic')
