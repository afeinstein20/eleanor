import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import SFFCorrector
from scipy.optimize import minimize
from astropy.table import Table, Column
from astropy.wcs import WCS
from time import strftime
from astropy.io import fits
from muchbettermoments import quadratic_2d
import urllib
import os

from .ffi import use_pointing_model
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
        Height in pixels of TPF to retrieve. Default 9.
    width : int, optional
        Width in pixels of TPF to retrieve. Default 9.


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

    def __init__(self, source, height=9, width=9, save_postcard=True):
        self.source_info = source

        if source.premade:
            self.load()

        else:
            self.aperture = None
            self.post_obj = Postcard(source.postcard)
            self.time  = self.post_obj.time
            self.load_pointing_model(source.sector, source.camera, source.chip)
            self.get_tpf_from_postcard(source.coords, source.postcard, height, width, save_postcard)
            self.create_apertures(height, width)
            self.get_lightcurve()
            self.center_of_mass()
            self.set_quality()
            self.set_header()


    def load_pointing_model(self, sector, camera, chip):

        pointing_link = urllib.request.urlopen('http://jet.uchicago.edu/tess_postcards/pointingModel_{}_{}-{}.txt'.format(sector,
                                                                                                                          camera,
                                                                                                                          chip))
        pointing = pointing_link.read().decode('utf-8')
        pointing = Table.read(pointing, format='ascii.basic') # guide to postcard locations
        self.pointing_model = pointing
        return


    def get_tpf_from_postcard(self, pos, postcard, height, width, save_postcard):
        """Gets TPF from postcard."""

        self.tpf = None
        self.centroid_xs = None
        self.centroid_ys = None

        xy = WCS(self.post_obj.header).all_world2pix(pos[0], pos[1], 1)
        
        # Apply the pointing model to each cadence to find the centroids
        centroid_xs, centroid_ys = [], []
        for i in range(len(self.pointing_model)):
            new_coords = use_pointing_model(xy, self.pointing_model[i])
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
            print("The size postage stamp you are requesting falls off the edge of the postcard.")
            print("WARNING: Your postage stamp may not be centered.")

        self.tpf     = post_flux[:, y_low_lim:y_upp_lim, x_low_lim:x_upp_lim]
        self.tpf_err = post_err[: , y_low_lim:y_upp_lim, x_low_lim:x_upp_lim]
        self.dimensions = np.shape(self.tpf)

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
        if (height,width) == (9,9):
            ap_path = eleanor.__path__[0]+'/default_apertures.npy'
            self.all_apertures = np.load(ap_path)

        # Creates aperture based on the requested size of TPF
        else:
            # Creates a circular and rectangular aperture
            def circle(pos,r):
                return CircularAperture(pos,r)
            def rectangle(pos, l, w, t):
                return RectangularAperture(pos, l, w, t)

            # Completes either binary or weighted aperture photometry
            def binary(data, apertures, err):
                return aperture_photometry(data, apertures, error=err, method='center')
            def weighted(data, apertures, err):
                return aperture_photometry(data, apertures, error=err, method='exact')

            r_list = np.arange(1.5,4,0.5)

            # Center gives binary mask; exact gives weighted mask
            circles, rectangles, self.all_apertures = [], [], []

            center = (width/2, height/2)

            for r in r_list:
                ap_circ = circle( center, r )
                ap_rect = rectangle( center, r, r, 0.0)
                circles.append(ap_circ); rectangles.append(ap_rect)
                for method in ['center', 'exact']:
                    circ_mask = ap_circ.to_mask(method=method)[0].to_image(shape=((
                                np.shape( self.tpf[0]))))
                    rect_mask = ap_rect.to_mask(method=method)[0].to_image(shape=((
                                np.shape( self.tpf[0]))))
                    self.all_apertures.append(circ_mask)
                    self.all_apertures.append(rect_mask)
            self.all_apertures = np.array(self.all_apertures)


    def get_lightcurve(self, aperture=False):
        """Extracts a light curve using the given aperture and TPF.

        Allows the user to pass in a mask to use, otherwise sets best lightcurve and aperture (min std).
        Mask is a 2D array of the same shape as TPF (9x9).

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
            self.corr_flux  = self.jitter_corr(flux=lc)
            self.flux_err   = np.array(lc_err)
            return


        self.flux_err   = None

        if (self.aperture is None):

            self.all_lc_err  = None

            all_raw_lc     = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_lc_err = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_corr_lc = np.copy(all_raw_lc)

            stds = []
            for a in range(len(self.all_apertures)):
                for cad in range(len(self.tpf)):
                    all_lc_err[a][cad] = np.sqrt( np.sum( self.tpf_err[cad]**2 * self.all_apertures[a] ))
                    all_raw_lc[a][cad] = np.sum( self.tpf[cad] * self.all_apertures[a] )

                all_corr_lc[a] = self.jitter_corr(flux=all_raw_lc[a]/np.nanmedian(all_raw_lc[a]))*np.nanmedian(all_raw_lc[a])
                stds.append( np.std(all_corr_lc[a]) )

            self.all_raw_lc  = np.array(all_raw_lc)
            self.all_lc_err  = np.array(all_lc_err)
            self.all_corr_lc = np.array(all_corr_lc)

            best_ind = np.where(stds == np.min(stds))[0][0]
            self.best_ind = best_ind
            self.corr_flux= self.all_corr_lc[best_ind]
            self.raw_flux = self.all_raw_lc[best_ind]
            self.aperture = self.all_apertures[best_ind]
            self.flux_err = self.all_lc_err[best_ind]

        else:
            if np.shape(aperture) == np.shape(self.tpf[0]):
                self.aperture = aperture
                apply_mask(self.aperture)
            elif self.aperture is not None:
                apply_mask(self.aperture)
            else:
                print("We could not find a custom aperture. Please either create a 2D array that is the same shape as the TPF.")
                print("Or, create a custom aperture using the function TargetData.custom_aperture(). See documentation for inputs.")

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
        cen = (brightest[0][0], brightest[1][0])
        for a in range(len(self.tpf)):
            data = self.tpf[a, cen[0]-3:cen[0]+2, cen[1]-3:cen[1]+2]
            c_0  = quadratic_2d(data)
            c_frame = [cen[0]+c_0[0], cen[1]+c_0[1]]
            self.x_com.append(c_frame[0])
            self.y_com.append(c_frame[1])
        return



    def set_quality(self):
        """Currently (10/13/2018), this function sets a flag for when the centroid is
        3 sigma away from the mean either in the x or y direction.
        Hopefully in the future, MAST will put in some quality flags for us.
        Our flags and their flags will be combnied, if they create flags.
        """
        bad = np.where( (self.centroid_xs > np.mean(self.centroid_xs)+3*np.std(self.centroid_xs)) | (self.centroid_ys > np.mean(self.centroid_ys)+3*np.std(self.centroid_ys)))

        quality = np.zeros(np.shape(self.time))
        quality[bad] = 1
        self.quality = quality



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
            print("Please select a shape: circle or rectangle")

        shape = shape.lower()
        if pos is None:
            pos = (self.tpf.shape[1]/2, self.tpf.shape[2]/2)
        else:
            pos = pos

        if shape == 'circle':
            if r == 0.0:
                print ("Please set a radius (in pixels) for your aperture")
            else:
                aperture = CircularAperture(pos, r=r)
                self.aperture = aperture.to_mask(method=method)[0].to_image(shape=((
                            np.shape(self.tpf[0]))))

        elif shape == 'rectangle':
            if l==0.0 or w==0.0:
                print("For a rectangular aperture, please set both length and width: custom_aperture(shape='rectangle', l=#, w=#)")
            else:
                aperture = RectangularAperture(pos, l=l, w=w, t=theta)
                self.aperture = aperture.to_mask(method=method)[0].to_image(shape=((
                            np.shape(self.tpf[0]))))
        else:
            print("Aperture shape not recognized. Please set shape == 'circle' or 'rectangle'")




    def jitter_corr(self, flux, cen=0.0):
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
        def parabola(params, x, y, f_obs, y_err):
            nonlocal cen
            c1, c2, c3, c4, c5 = params
            f_corr = f_obs * (c1 + c2*(x-cen) + c3*(x-cen)**2 + c4*(y-cen) + c5*(y-cen)**2)
            return np.sum( ((1-f_corr)/y_err)**2)


        # Masks out anything >= 2.5 sigma above the mean
        mask = np.ones(len(flux), dtype=bool)
        for i in range(5):
            lc_new = []
            std_mask = np.std(flux[mask])

            inds = np.where(flux <= np.mean(flux)-2.5*std_mask)
            y_err = np.ones(len(flux))**np.std(flux)
            for j in inds:
                y_err[j] = np.inf
                mask[j]  = False

            if i == 0:
                initGuess = [3, 3, 3, 3, 3]
            else:
                initGuess = test.x

            bnds = ((-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.))
            centroid_xs, centroid_ys = self.centroid_xs-np.median(self.centroid_xs), self.centroid_ys-np.median(self.centroid_ys)
            test = minimize(parabola, initGuess, args=(centroid_xs, centroid_ys, flux, y_err), bounds=bnds)
            c1, c2, c3, c4, c5 = test.x
        lc_new = flux * (c1 + c2*(centroid_xs-cen) + c3*(centroid_xs-cen)**2 + c4*(centroid_ys-cen) + c5*(centroid_ys-cen)**2)
        return np.copy(lc_new)


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



    def save(self, output_fn=None, directory=None):
        """Saves a created TPF object to a FITS file.

        Parameters
        ----------
        output_fn : str, optional
            Filename to save output as. Overrides default naming.
        directory : str, optional
            Directory to save file into.
        """

        # if the user did not specify a directory, set it to default
        if directory is None:
            directory = self.fetch_dir()

        # Creates column names for FITS tables
        r = np.arange(1.5,4,0.5)
        colnames=[]
        for i in r:
            colnames.append('_'.join(str(e) for e in ['circle', 'binary', i]))
            colnames.append('_'.join(str(e) for e in ['rectangle', 'binary', i]))
            colnames.append('_'.join(str(e) for e in ['circle', 'weighted', i]))
            colnames.append('_'.join(str(e) for e in ['rectangle', 'weighted', i]))
        raw       = [e+'_raw' for e in colnames]
        errors    = [e+'_err' for e in colnames]
        corrected = [e+'_corr' for e in colnames]

        # Creates table for first extension (tpf, tpf_err, best lc, time, centroids)
        ext1 = Table()
        ext1['TIME']       = self.time
        ext1['TPF']        = self.tpf
#        ext1['FLUX']       = self.tpf
        ext1['TPF_ERR']    = self.tpf_err
        ext1['RAW_FLUX']   = self.raw_flux
        ext1['CORR_FLUX']  = self.corr_flux
        ext1['FLUX_ERR']   = self.flux_err
        ext1['QUALITY']    = self.quality
        ext1['X_CENTROID'] = self.centroid_xs
        ext1['Y_CENTROID'] = self.centroid_ys
        ext1['X_COM']      = self.x_com
        ext1['Y_COM']      = self.y_com

        # Creates table for second extension (all apertures)
        ext2 = Table()
        for i in range(len(self.all_apertures)):
            ext2[colnames[i]] = self.all_apertures[i]

        # Appends custom aperture to the end, if there is one
        if self.custom_aperture is not None:
            ext2['custom'] = self.custom_aperture

        # Creates table for third extention (all raw & corrected fluxes and errors)
        ext3 = Table()
        for i in range(len(raw)):
            ext3[raw[i]]       = self.all_raw_lc[i]
            ext3[corrected[i]] = self.all_corr_lc[i]
            ext3[errors[i]]    = self.all_lc_err[i]

        # Appends aperture to header
        self.header.append(fits.Card(keyword='APERTURE', value=colnames[self.best_ind],
                                     comment='Best aperture used for lightcurves in extension 1'))

        # Writes TPF to FITS file
        primary_hdu = fits.PrimaryHDU(header=self.header)
        data_list = [primary_hdu, fits.BinTableHDU(ext1), fits.BinTableHDU(ext2), fits.BinTableHDU(ext3)]
        hdu = fits.HDUList(data_list)

        if output_fn==None:
            hdu.writeto(os.path.join(directory,
                        'hlsp_eleanor_tess_ffi_lc_TIC{}_s{}_v0.1.fits'.format(
                        self.source_info.tic, self.source_info.sector),
                        overwrite=True))
        else:
            hdu.writeto(output_fn)



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
