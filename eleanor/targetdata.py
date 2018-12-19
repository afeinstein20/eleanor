import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from photutils import MMMBackground
from lightkurve import SFFCorrector, lightcurve
from scipy.optimize import minimize
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from time import strftime
from astropy.io import fits
from muchbettermoments import quadratic_2d
from urllib.request import urlopen
import os
import os.path
import warnings
import pickle

from .ffi import use_pointing_model, load_pointing_model
from .postcard import Postcard

__all__  = ['TargetData']

class TargetData(object):
    """
    Object containing the light curve, target pixel file, and related information
    for any given source.

    Parameters
    ----------
    source : ellie.Source
        The source object to use.
    height : int, optional
        Height in pixels of TPF to retrieve. Default value is 9 pixels. Must be an odd number,
        or else will return an aperture one pixel taller than requested so target
        falls on central pixel.
    width : int, optional
        Width in pixels of TPF to retrieve. Default value is 9 pixels. Must be an odd number,
        or else will return an aperture one pixel wider than requested so target
        falls on central pixel.


    Attributes
    ----------
    header : dict
        FITS header for saving/loading data.
    source_info : eleanor.Source
        Pointer to input source.
    aperture :
        Aperture to use if overriding default. To use default, set to `None`.
    tpf : np.ndarray
        Target pixel file of fluxes; array with shape `dimensions`.
    time : np.ndarray
        Time series.
    post_obj : eleanor.Postcard
        Pointer to Postcard objects containing this TPF.
    pointing_model : astropy.table.Table
        Table of matrices describing the transformation matrix from FFI default
        WCS and eleanor's corrected pointing.
    tpf_err : np.ndarray
        Errors on fluxes in `tpf`.
    centroid_xs : np.ndarray
        Position of the source in `x` inferred from pointing model; has same length as `time`.
        Position is relative to the pixel coordinate system of the postcard.
    centroid_ys : np.ndarray
        Position of the source in `y` inferred from pointing model; has same length as `time`.
        Position is relative to the pixel coordinate system of the postcard.
    cen_x : int
        Median `x` position of the source.
        Position is relative to the pixel coordinate system of the postcard.
    cen_y : int
        Median `y` position of the source.
        Position is relative to the pixel coordinate system of the postcard.
    dimensions : tuple
        Shape of `tpf`. Should be (`time`, `height`, `width`).
    all_apertures : list
        List of aperture objects.
    aperture : array-like
        Chosen aperture for producing `raw_flux` lightcurve. Format is array
        with shape (`height`, `width`). All entries are floats in range [0,1].
    all_lc_err : np.ndarray
        Estimated uncertainties on `all_raw_lc`.
    all_raw_lc : np.ndarray
        All lightcurves extracted using `all_apertures`.
        Has shape (N_apertures, N_time).
    all_corr_lc : np.ndarray
        All systematics-corrected lightcurves. See `all_raw_lc`.
    best_ind : int
        Index into `all_apertures` producing the best (least noisy) lightcurve.
    corr_flux : np.ndarray
        Systematics-corrected version of `raw_flux`.
    flux_err : np.ndarray
        Estimated uncertainty on `raw_flux`.
    raw_flux : np.ndarray
        Un-systematics-corrected lightcurve derived using `aperture` and `tpf`.
    x_com : np.ndarray
        Position of the source in `x` inferred from TPF; has same length as `time`.
        Position is relative to the pixel coordinate system of the TPF.
    y_com : np.ndarray
        Position of the source in `y` inferred from TPF; has same length as `time`.
        Position is relative to the pixel coordinate system of the TPF.
    quality : int
        Quality flag.

    Notes
    -----
    `save()` and `load()` methods write/read these data to a FITS file with format:

    Extension[0] = header

    Extension[1] = (N_time, height, width) TPF, where n is the number of cadences in an observing run

    Extension[2] = (3, N_time) time, raw flux, systematics corrected flux
    """

    def __init__(self, source, height=9, width=9, save_postcard=True, do_pca=True):
        self.source_info = source

        if source.premade:
            self.load()

        else:
            self.aperture = None
            self.post_obj = Postcard(source.postcard, source.ELEANORURL)
            self.flux_bkg = self.post_obj.bkg
            self.time  = self.post_obj.time
            self.pointing_model = load_pointing_model(source.sector, source.camera, source.chip)
            self.get_tpf_from_postcard(source.coords, source.postcard, height, width, save_postcard)
            self.set_quality()
            self.create_apertures(height, width)
            self.get_lightcurve()
            if do_pca == True:
                self.pca()
            else:
                self.modes = None
                self.pca_flux = None
            self.center_of_mass()


    def get_tpf_from_postcard(self, pos, postcard, height, width, save_postcard):
        """Gets TPF from postcard."""

        self.tpf         = None
        self.centroid_xs = None
        self.centroid_ys = None

        xy = WCS(self.post_obj.header).all_world2pix(pos[0], pos[1], 1)

        # Apply the pointing model to each cadence to find the centroids
        centroid_xs, centroid_ys = [], []
        for i in range(len(self.pointing_model)):
            new_coords = use_pointing_model(np.array(xy), self.pointing_model[i])
            centroid_xs.append(new_coords[0][0])
            centroid_ys.append(new_coords[0][1])
        self.centroid_xs = np.array(centroid_xs)
        self.centroid_ys = np.array(centroid_ys)

        # Define tpf as region of postcard around target
        med_x, med_y = np.nanmedian(self.centroid_xs), np.nanmedian(self.centroid_ys)

        med_x, med_y = int(np.round(med_x,0)), int(np.round(med_y,0))

        post_flux = np.transpose(self.post_obj.flux, (2,0,1))
        post_err  = np.transpose(self.post_obj.flux_err, (2,0,1))

        self.cen_x, self.cen_y = med_x, med_y

        y_length, x_length = int(np.floor(height/2.)), int(np.floor(width/2.))

        y_low_lim = med_y-y_length
        y_upp_lim = med_y+y_length+1
        x_low_lim = med_x-x_length
        x_upp_lim = med_x+x_length+1

        if height % 2 == 0 or width % 2 == 0:
            warnings.warn('We force our TPFs to have an odd height and width so we can properly center our apertures.')

        post_y_upp, post_x_upp = self.post_obj.dimensions[0], self.post_obj.dimensions[1]

        # Fixes the postage stamp if the user requests a size that is too big for the postcard
        if y_low_lim <= 0:
            y_low_lim = 0
        if x_low_lim <= 0:
            x_low_lim = 0
        if y_upp_lim  > post_y_upp:
            y_upp_lim = post_y_upp+1
        if x_upp_lim >  post_x_upp:
            x_upp_lim = post_x_upp+1

        if (x_low_lim==0) or (y_low_lim==0) or (x_upp_lim==post_x_upp) or (y_upp_lim==post_y_upp):
            warnings.warn("The size postage stamp you are requesting falls off the edge of the postcard.")
            warnings.warn("WARNING: Your postage stamp may not be centered.")

        self.tpf     = post_flux[:, y_low_lim:y_upp_lim, x_low_lim:x_upp_lim]
        self.tpf_err = post_err[: , y_low_lim:y_upp_lim, x_low_lim:x_upp_lim]
        self.dimensions = np.shape(self.tpf)

        self.bkg_subtraction()

        self.tpf = self.tpf
        if save_postcard == False:
            try:
                os.remove(source.postcard.filename)
            except OSError:
                pass
        return


    def create_apertures(self, height, width):
        """Creates a range of sizes and shapes of apertures to test."""
        import eleanor
        self.all_apertures = None

        # Saves some time by pre-loading apertures for 9x9 TPFs
        ap_path = eleanor.__path__[0]+'/default_apertures.pickle'
        pickle_in = open(ap_path, "rb")
        pickle_dict = pickle.load(pickle_in)
        self.aperture_names = np.array(list(pickle_dict.keys()))
        all_apertures  = np.array(list(pickle_dict.values()))

        default = 9

        # Creates aperture based on the requested size of TPF
        if (height, width) == (default, default):
            self.all_apertures = all_apertures

        else:
            new_aps = []

            h_diff = height-default; w_diff = width-default
            half_h = int(np.abs(h_diff/2)) ; half_w = int(np.abs(w_diff/2))

            # HEIGHT PADDING
            if h_diff > 0:
                h_pad = (half_h, half_h)
            elif h_diff < 0:
                warnings.warn('WARNING: Making a TPF smaller than (9,9) may provide inadequate results.')
                h_pad = (0,0)
                all_apertures = all_apertures[:, half_h:int(len(all_apertures[0][:,0])-half_h), :]
            else:
                h_pad = (0,0)

            # WIDTH PADDING
            if w_diff > 0:
                w_pad = (half_w, half_w)
            elif w_diff < 0:
                warnings.warn('WARNING: Making a TPF smaller than (9,9) may provide inadequate results.')
                w_pad = (0,0)
                all_apertures = all_apertures[:, :, half_w:int(len(all_apertures[0][0])-half_w)]
            else:
                w_pad=(0,0)

            for a in range(len(all_apertures)):
                new_aps.append(np.pad(all_apertures[a], (h_pad, w_pad), 'constant', constant_values=(0)))
            self.all_apertures = np.array(new_aps)


    def bkg_subtraction(self, scope="tpf", sigma=2.5):
        """Subtracts background flux from target pixel file.

        Parameters
        ----------
        scope : string, "tpf" or "postcard"
            If `tpf`, will use data from the target pixel file only to estimate and remove the background.
            If `postcard`, will use data from the entire postcard region to estimate and remove the background.
        sigma : float
            The standard deviation cut used to determine which pixels are representative of the background in each cadence.
        """
        time = self.time
        flux = self.tpf

        self.tpf_flux_bkg = []

        sigma_clip = SigmaClip(sigma=sigma)
        bkg = MMMBackground(sigma_clip=sigma_clip)

        for i in range(len(time)):
            bkg_value = bkg.calc_background(flux[i])
            self.tpf_flux_bkg.append(bkg_value)

        self.tpf_flux_bkg = np.array(self.tpf_flux_bkg)
        


    def get_lightcurve(self, aperture=False):
        """Extracts a light curve using the given aperture and TPF.
        Can pass a user-defined aperture mask, otherwise determines which of a set of pre-determined apertures
        provides the lowest scatter in the light curve.
        Produces a mask, a numpy.ndarray object of the same shape as the target pixel file, which every pixel assigned
        a weight in the range [0, 1].

        Parameters
        ----------
        aperture : numpy.ndarray
            (`height`, `width`) array of floats in the range [0,1] with desired weights for each pixel to
            create a light curve. If not set, ideal aperture is inferred automatically. If set, uses this
            aperture at the expense of all other set apertures.
        """

        def apply_mask(mask):
            lc     = np.zeros(len(self.tpf))
            lc_err = np.zeros(len(self.tpf))
            for cad in range(len(self.tpf)):
                lc[cad]     = np.sum( self.tpf[cad] * mask)
                lc_err[cad] = np.sqrt( np.sum( self.tpf_err[cad]**2 * mask))
            self.raw_flux   = np.array(lc)
            self.corr_flux  = self.k2_correction(flux=lc)
            self.flux_err   = np.array(lc_err)
            return

        self.flux_err = None

        if (self.aperture is None):

            self.all_lc_err  = None

            all_raw_lc  = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_lc_err  = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_corr_lc = np.copy(all_raw_lc)
            all_raw_tpf = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_corr_tpf= np.copy(all_raw_tpf)

            stds, tpf_stds = [], []
            for a in range(len(self.all_apertures)):
                for cad in range(len(self.tpf)):
                    try:
                        all_lc_err[a, cad]   = np.sqrt( np.sum( self.tpf_err[cad]**2 * self.all_apertures[a] ))
                        all_raw_lc[a, cad]   = np.sum( (self.tpf[cad]) * self.all_apertures[a] )
                        all_raw_tpf[a, cad]  = np.sum( (self.tpf[cad]-self.tpf_flux_bkg[cad]) * self.all_apertures[a] )
                    except ValueError:
                        continue

                ## Remove something from all_raw_lc before passing into jitter_corr ##
                try:
                    all_corr_lc[a] = self.k2_correction(flux=all_raw_lc[a]/np.nanmedian(all_raw_lc[a]))
                    all_corr_tpf[a]= self.k2_correction(flux=all_raw_tpf[a]/np.nanmedian(all_raw_tpf[a]))
                except IndexError:
                    continue

                q = self.quality == 0

                lc_obj = lightcurve.LightCurve(time = self.time[q][0:500],
                                       flux = all_corr_lc[a][q][0:500])
                flat_lc = lc_obj_tpf.flatten(polyorder=2, window_length=51)
                stds.append( np.std(flat_lc.flux))

                lc_obj_tpf = lightcurve.LightCurve(time = self.time[q][0:500],
                                                   flux = all_corr_tpf[a][q][0:500])
                flat_lc_tpf = lc_obj_tpf.flatten(polyorder=2, window_length=51)
                tpf_stds.append( np.std(flat_lc_tpf.flux))

                all_corr_lc[a]  = all_corr_lc[a]  * np.nanmedian(all_raw_lc[a])
                all_corr_tpf[a] = all_corr_tpf[a] * np.nanmedian(all_raw_tpf[a])

            self.all_raw_lc  = np.array(all_raw_lc)
            self.all_lc_err  = np.array(all_lc_err)
            self.all_corr_lc = np.array(all_corr_lc)

            best_ind_tpf = np.where(tpf_stds == np.min(tpf_stds))[0][0]
            best_ind     = np.where(stds == np.min(stds))[0][0]

            ## Checks if postcard or tpf level bkg subtraction is better ##
            ## Prints bkg_type to TPF header ##
            if stds[best_ind] <= post_stds[best_ind_tpf]:
                best_ind = best_ind
                self.bkg_type = 'PC_LEVEL'
            else:
                best_ind = best_ind_tpf
                self.bkg_type = 'TPF_LEVEL'
                all_corr_lc   = all_corr_tpf
                all_raw_lc    = all_raw_tpf

            self.corr_flux= self.all_corr_lc[best_ind]
            self.raw_flux = self.all_raw_lc[best_ind]
            self.aperture = self.all_apertures[best_ind]
            self.flux_err = self.all_lc_err[best_ind]
            self.aperture_size = np.sum(self.aperture)
            self.best_ind = best_ind
        else:
            if np.shape(aperture) == np.shape(self.tpf[0]):
                self.aperture = aperture
                apply_mask(self.aperture)
            elif self.aperture is not None:
                apply_mask(self.aperture)
            else:
                raise Exception("We could not find a custom aperture. Please either create a 2D array that is the same shape as the TPF. "
                                "Or, create a custom aperture using the function TargetData.custom_aperture(). See documentation for inputs.")

        return


    def pca(self, matrix_fn = 'a_matrix.txt', flux=None, modes=8):
        """ Applies cotrending basis vectors, found through principal component analysis, to light curve to
        remove systematics shared by nearby stars.

        Parameters
        ----------
        flux : numpy.ndarray
            Flux array to which cotrending basis vectors are applied. Default is `self.corr_flux`.
        modes : int
            Number of cotrending basis vectors to apply. Default is 8.
        """
        if flux is None:
            flux = self.corr_flux

        matrix_file = urlopen('https://archipelago.uchicago.edu/tess_postcards/{}'.format(matrix_fn))
        A = [float(x) for x in matrix_file.read().decode('utf-8').split()]
        for i in range(0, len(A), 16):
            yield A[i:i+16]

        def matrix(f):
            nonlocal A
            ATA     = np.dot(A.T, A)
            invATA  = np.linalg.inv(ATA)
            A_coeff = np.dot(invATA, A.T)
            return np.dot(A_coeff, f)

        self.modes    = modes
        self.pca_flux = flux - np.dot(A[:,0:modes], matrix(flux)[0:modes])
        return


    def center_of_mass(self):
        """
        Calculates the position of the source across all cadences using `muchbettermoments` and `self.best_aperture`.

        Finds the brightest pixel in a (`height`, `width`) region summed up over all cadence.
        Searches a smaller (3x3) region around this pixel at each cadence and uses `muchbettermoments` to find the maximum.
        """

        self.x_com = []
        self.y_com = []

        summed_pixels = np.sum(self.aperture * self.tpf, axis=0)
        brightest = np.where(summed_pixels == np.max(summed_pixels))
        cen = [brightest[0][0], brightest[1][0]]

        if cen[0] < 3.0:
            cen[0] = 3
        if cen[1] < 3.0:
            cen[1] = 3
        if cen[0]+3 > np.shape(self.tpf[0])[0]:
            cen[0] = np.shape(self.tpf[0])[0]-3
        if cen[1]+3 > np.shape(self.tpf[0])[1]:
            cen[1] = np.shape(self.tpf[0])[1]-3

        for a in range(len(self.tpf)):
            data = self.tpf[a, cen[0]-3:cen[0]+2, cen[1]-3:cen[1]+2]
            c_0  = quadratic_2d(data)
            c_frame = [cen[0]+c_0[0], cen[1]+c_0[1]]
            self.x_com.append(c_frame[0])
            self.y_com.append(c_frame[1])
        return



    def set_quality(self):
        """ Reads in quality flags set in the postcard
        """
        file = urlopen('https://archipelago.uchicago.edu/tess_postcards/quality_flags.txt')
        tess_quality = Table.read(file.read().decode('utf-8'), format='ascii.basic')
        tess_quality_flags = [q[0] for q in tess_quality]
        tess_quality_flags.append(0.0)
        lim = 2.5
        bad = np.where( (self.centroid_xs > np.mean(self.centroid_xs)+lim*np.std(self.centroid_xs)) | (self.centroid_ys > np.mean(self.centroid_ys)+lim*np.std(self.centroid_ys)))

        quality = np.zeros(np.shape(self.time))
        quality[bad] = 1
        self.quality = quality+tess_quality_flags


    def psf_lightcurve(self, nstars=1, model='gaussian', xc=[4.5], yc=[4.5]):
        """
        Performs PSF photometry for a selection of stars on a TPF.

        Parameters
        ----------
        nstars: int, optional
            Number of stars to be modeled on the TPF.
        model: string, optional
            PSF model to be applied. Presently must be `gaussian`, which models a single Gaussian.
            Will be extended in the future once TESS PRF models are made publicly available.
        xc: list, optional
            The x-coordinates of stars in the zeroth cadence. Must have length `nstars`.
            While the positions of stars will be fit in all cadences, the relative positions of
            stars will be fixed following the delta values from this list.
        yc: list, optional
            The y-coordinates of stars in the zeroth cadence. Must have length `nstars`.
            While the positions of stars will be fit in all cadences, the relative positions of
            stars will be fixed following the delta values from this list.
        """
        import tensorflow as tf
        from vaneska.models import Gaussian
        from tqdm import tqdm

        if len(xc) != nstars:
            raise ValueError('xc must have length nstars')
        if len(yc) != nstars:
            raise ValueError('yc must have length nstars')


        flux = tf.Variable(np.ones(nstars)*1000, dtype=tf.float64)
        bkg = tf.Variable(np.nanmedian(self.tpf[0]), dtype=tf.float64)
        xshift = tf.Variable(0.0, dtype=tf.float64)
        yshift = tf.Variable(0.0, dtype=tf.float64)

        if model == 'gaussian':

            gaussian = Gaussian(shape=self.tpf.shape[1:], col_ref=0, row_ref=0)

            a = tf.Variable(initial_value=1., dtype=tf.float64)
            b = tf.Variable(initial_value=0., dtype=tf.float64)
            c = tf.Variable(initial_value=1., dtype=tf.float64)

            if nstars == 1:
                mean = gaussian(flux, xc[0]+xshift, yc[0]+yshift, a, b, c)
            else:
                mean = [gaussian(flux[j], xc[j]+xshift, yc[j]+yshift, a, b, c) for j in range(nstars)]
        else:
            raise ValueError('This model is not incorporated yet!') # we probably want this to be a warning actually,
                                                                    # and a gentle return

        mean += bkg

        data = tf.placeholder(dtype=tf.float64, shape=self.tpf[0].shape)
        nll = tf.reduce_sum(tf.squared_difference(mean, data))

        var_list = [flux, xshift, yshift, a, b, c, bkg]
        grad = tf.gradients(nll, var_list)

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(nll, var_list, method='TNC', tol=1e-4)

        fout = np.zeros((len(self.tpf), nstars))
        xout = np.zeros(len(self.tpf))
        yout = np.zeros(len(self.tpf))

        for i in tqdm(range(len(self.tpf))):
            optimizer.minimize(session=sess, feed_dict={data:self.tpf[i]}) # we could also pass a pointing model here
                                                                           # and just fit a single offset in all frames

            fout[i] = sess.run(flux)
            xout[i] = sess.run(xshift)
            yout[i] = sess.run(yshift)

        sess.close()

        self.psf_flux = fout[:,0]
        return


    def custom_aperture(self, shape=None, r=0.0, l=0.0, w=0.0, theta=0.0, pos=None, method='exact'):
        """
        Creates a custom circular or rectangular aperture of arbitrary size.

        Parameters
        ----------
        shape: str, optional
            The shape of the aperture to be used. Must be either `circle` or `rectangle.`
        r: float, optional
            If shape is `circle` the radius of the circular aperture to be used.
        l: float, optional
            If shape is `rectangle` the length of the rectangular aperture to be used.
        w: float, optional
            If shape is `rectangle` the width of the rectangular aperture to be used.
        theta: float, optional
            If shape is `rectangle` the rotation of the rectangle relative to detector coordinate.
            Uses units of radians.
        pos: tuple, optional
            The center of the aperture, in TPF coordinates. If not set, defaults to the center of the TPF.
        method: str, optional
            The method of producing a light curve to be used, either `exact`, `center`, or `subpixel`.
            Passed through to photutils and used as intended by that package.
        """
        if shape is None:
            raise Exception("Please select a shape: circle or rectangle")

        shape = shape.lower()
        if pos is None:
            pos = (self.tpf.shape[1]/2, self.tpf.shape[2]/2)
        else:
            pos = pos

        if shape == 'circle':
            if r == 0.0:
                raise Exception("Please set a radius (in pixels) for your aperture")
            else:
                aperture = CircularAperture(pos, r=r)
                self.aperture = aperture.to_mask(method=method)[0].to_image(shape=((
                            np.shape(self.tpf[0]))))

        elif shape == 'rectangle':
            if l==0.0 or w==0.0:
                raise Exception("For a rectangular aperture, please set both length and width: custom_aperture(shape='rectangle', l=#, w=#)")
            else:
                aperture = RectangularAperture(pos, l=l, w=w, t=theta)
                self.aperture = aperture.to_mask(method=method)[0].to_image(shape=((
                            np.shape(self.tpf[0]))))
        else:
            raise ValueError("Aperture shape not recognized. Please set shape == 'circle' or 'rectangle'")


    def find_break(self):
        t   = np.diff(self.time)
        ind = np.where( t > np.mean(t)+2*np.std(t))[0][0]
        return ind + 1


    def k2_correction(self, flux):
        """Remove any systematics that are correlated with spacecraft pointing, as inferred through telescope
        pointing model.

        Parameters
        ----------
        flux : numpy.ndarray
            Flux array to which detrending applied.
        """
        brk = self.find_break()

        r1 = np.arange(0, brk, 1)
        r2 = np.arange(brk,len(self.time))

        t1 = self.time[r1]; f1 = flux[r1]
        t2 = self.time[r2]; f2 = flux[r2]

        sff = SFFCorrector()
        corr_lc_obj_1 = sff.correct(time=t1, flux=f1,
                                    centroid_col=self.centroid_xs[r1],
                                    centroid_row=self.centroid_ys[r1],
                                    windows=1, polyorder=2, niters=3, sigma_1=3, sigma_2=5,
                                    restore_trend=True, bins=15)
        corr_lc_obj_2 = sff.correct(time=t2, flux=f2,
                                    centroid_col=self.centroid_xs[r2],
                                    centroid_row=self.centroid_ys[r2],
                                    windows=1, polyorder=2, niters=3, sigma_1=3, sigma_2=5,
                                    restore_trend=True, bins=15)
        return np.append(corr_lc_obj_1.flux, corr_lc_obj_2.flux)


    def jitter_corr(self, flux, quality, cen=0.0):
        """
        Corrects for jitter in the light curve by quadratically regressing with centroid position.
        Following Equation 1 of Knutson et al. 2008, ApJ, 673, 526.

        Parameters
        ----------
        flux: numpy.ndarray
            Time series of raw flux observations to be corrected.
        cen: float, optional
            Center of the 2-d paraboloid for which the correction is performed.
        """
        flux = np.array(flux)

        # Inputs: light curve & quality flag
        def norm(l, q):
            l = l[q]
            l /= np.nanmedian(l)
            l -= 1
            return l

        def fhat(xhat, data):
            return np.dot(data, xhat)

        def xhat(mat, lc):
            ATA = np.dot(mat.T, mat)
            ATAinv = np.linalg.inv(ATA)
            ATf = np.dot(mat.T, lc)
            xhat = np.dot(ATAinv, ATf)
            return xhat

        q = quality == 0
        norm_l = norm(flux, q)
        cm     = np.column_stack( (self.centroid_xs[q]   , self.centroid_ys[q],
                                   self.centroid_xs[q]**2, self.centroid_ys[q]**2))
        x = xhat(cm, norm_l)

        cm  = np.column_stack( (self.centroid_xs   , self.centroid_ys,
                                self.centroid_xs**2, self.centroid_ys**2))

        f_mod = fhat(x, cm)
        lc_reg = flux/f_mod
        # Breaks the light curve into two sections
        brk = self.find_break()
        f   = np.arange(0, brk, 1); s = np.arange(brk, len(self.time), 1)

        poly_fit1 = np.polyval( np.polyfit(self.time[f], flux[f], 1), self.time[f])
        poly_fit2 = np.polyval( np.polyfit(self.time[s], flux[s], 1), self.time[s])
        return np.append( lc_reg[f], lc_reg[s])




    def set_header(self):
        """Defines the header for the TPF."""
        self.header = self.post_obj.header
        self.header.update({'CREATED':strftime('%Y-%m-%d')})

        # Removes postcard specific header information
        for keyword in ['POST_H', 'POST_W', 'CEN_X', 'CEN_Y', 'CEN_RA', 'CEN_DEC', 'POSTPIX1', 'POSTPIX2']:
            self.header.remove(keyword)

        # Adds TPF specific header information
        self.header.append(fits.Card(keyword='TIC_ID', value=self.source_info.tic,
                                     comment='TESS Input Catalog ID'))
        self.header.append(fits.Card(keyword='TMAG', value=self.source_info.tess_mag,
                                     comment='TESS mag'))
        self.header.append(fits.Card(keyword='GAIA_ID', value=self.source_info.gaia,
                                     comment='Associated Gaia ID'))
        self.header.append(fits.Card(keyword='SECTOR', value=self.source_info.sector,
                                     comment='Sector'))
        self.header.append(fits.Card(keyword='CAMERA', value=self.source_info.camera,
                                     comment='Camera'))
        self.header.append(fits.Card(keyword='CHIP', value=self.source_info.chip,
                                     comment='Chip'))
        self.header.append(fits.Card(keyword='CHIPPOS1', value=self.source_info.position_on_chip[0],
                                     comment='central x pixel of TPF in FFI chip'))
        self.header.append(fits.Card(keyword='CHIPPOS2', value=self.source_info.position_on_chip[1],
                                     comment='central y pixel of TPF in FFI'))
        self.header.append(fits.Card(keyword='POSTCARD', value=self.source_info.postcard,
                                     comment='Postcard'))
        self.header.append(fits.Card(keyword='POSTPOS1', value= self.source_info.position_on_postcard[0],
                                     comment='predicted x pixel of source on postcard'))
        self.header.append(fits.Card(keyword='POSTPOS2', value= self.source_info.position_on_postcard[1],
                                     comment='predicted y pixel of source on postcard'))
        self.header.append(fits.Card(keyword='CEN_RA', value = self.source_info.coords[0],
                                     comment='RA of TPF source'))
        self.header.append(fits.Card(keyword='CEN_DEC', value=self.source_info.coords[1],
                                     comment='DEC of TPF source'))
        self.header.append(fits.Card(keyword='TPF_H', value=np.shape(self.tpf[0])[0],
                                     comment='Height of the TPF in pixels'))
        self.header.append(fits.Card(keyword='TPF_W', value=np.shape(self.tpf[0])[1],
                                           comment='Width of the TPF in pixels'))
#        self.header.append(fits.Card(keyword='BKG_LVL', value=self.bkg_type,
#                                     comment='Stage at which background is subtracted'))

        if self.modes is not None:
            self.header.append(fits.Card(keyword='MODES', value=self.modes,
                                         comment='Number of modes used in PCA analysis'))


    def save(self, output_fn=None, directory=None):
        """Saves a created TPF object to a FITS file.

        Parameters
        ----------
        output_fn : str, optional
            Filename to save output as. Overrides default naming.
        directory : str, optional
            Directory to save file into.
        """

        self.set_header()

        # if the user did not specify a directory, set it to default
        if directory is None:
            directory = self.fetch_dir()

        raw       = [e+'_raw'  for e in self.aperture_names]
        errors    = [e+'_err'  for e in self.aperture_names]
        corrected = [e+'_corr' for e in self.aperture_names]

        # Creates table for first extension (tpf, tpf_err, best lc, time, centroids)
        ext1 = Table()
        ext1['TIME']       = self.time
        ext1['TPF']        = self.tpf
        ext1['TPF_ERR']    = self.tpf_err
        ext1['RAW_FLUX']   = self.raw_flux
        ext1['CORR_FLUX']  = self.corr_flux
        ext1['FLUX_ERR']   = self.flux_err
        ext1['QUALITY']    = self.quality
        ext1['X_CENTROID'] = self.centroid_xs
        ext1['Y_CENTROID'] = self.centroid_ys
        ext1['X_COM']      = self.x_com
        ext1['Y_COM']      = self.y_com


        if self.bkg_type == "PC_LEVEL":
            ext1['FLUX_BKG'] = self.flux_bkg
        else:
            ext1['FLUX_BKG'] = self.flux_bkg + self.tpf_flux_bkg 
        

        if self.pca_flux is not None:
            ext1['PCA_FLUX'] = self.pca_flux

        # Creates table for second extension (all apertures)
        ext2 = Table()
        for i in range(len(self.all_apertures)):
            ext2[self.aperture_names[i]] = self.all_apertures[i]

        # Creates table for third extention (all raw & corrected fluxes and errors)
        ext3 = Table()
        for i in range(len(raw)):
            ext3[raw[i]]       = self.all_raw_lc[i]
            ext3[corrected[i]] = self.all_corr_lc[i]
            ext3[errors[i]]    = self.all_lc_err[i]

        # Appends aperture to header
        self.header.append(fits.Card(keyword='APERTURE', value=self.aperture_names[self.best_ind],
                                     comment='Best aperture used for lightcurves in extension 1'))

        # Writes TPF to FITS file
        primary_hdu = fits.PrimaryHDU(header=self.header)
        data_list = [primary_hdu, fits.BinTableHDU(ext1), fits.BinTableHDU(ext2), fits.BinTableHDU(ext3)]
        hdu = fits.HDUList(data_list)

        hdu.writeto(os.path.join(directory,
                                 'hlsp_eleanor_tess_ffi_lc_TIC{}_s{}_v0.1.fits'.format(
                    self.source_info.tic, self.source_info.sector)),
                    overwrite=True)



    def load(self, directory=None):
        """
        Loads in and sets all the attributes for a pre-created TPF file.

        Parameters
        ----------
        directory : str, optional
            Directory to load file from.
        """

        if directory is None:
            directory = self.fetch_dir()

        hdu = fits.open(os.path.join(directory, self.source_info.fn))
        hdr = hdu[0].header
        self.header = hdr
        # Loads in everything from the first extension
        cols  = hdu[1].columns.names
        table = hdu[1].data
        self.time        = table[cols[0]]
        self.tpf         = table[cols[1]]
        self.tpf_err     = table[cols[2]]
        self.raw_flux    = table[cols[3]]
        self.corr_flux   = table[cols[4]]
        self.flux_err    = table[cols[5]]
        self.quality     = table[cols[6]]
        self.centroid_xs = table[cols[7]]
        self.centroid_ys = table[cols[8]]

        # Loads in apertures from second extension
        self.all_apertures = []
        cols  = hdu[2].columns.names
        table = hdu[2].data
        for i in cols:
            if i == 'custom':
                self.custom_aperture = table[i]
            elif i == hdr['aperture']:
                self.aperture = table[i]
            else:
                self.all_apertures.append(table[i])

        # Loads in remaining light curves from third extension
        cols  = hdu[3].columns.names
        table = hdu[3].data
        self.all_raw_lc  = []
        self.all_corr_lc = []
        self.all_lc_err  = []
        for i in cols:
            if i[-4::] == 'corr':
                self.all_corr_lc.append(table[i])
            elif i[-3::] == 'err':
                self.all_lc_err.append(table[i])
            else:
                self.all_raw_lc.append(table[i])
        return

    def fetch_dir(self):
        """
        Returns the default path to the directory where files will be saved
        or loaded.

        By default, this method will return "~/.eleanor" and create
        this directory if it does not exist.  If the directory cannot be
        access or created, then it returns the local directory (".").

        Returns
        -------
        download_dir : str
            Path to location of `ffi_dir` where FFIs will be downloaded
        """
        download_dir = os.path.join(os.path.expanduser('~'), '.eleanor')
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
