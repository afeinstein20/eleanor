import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import SFFCorrector
from scipy.optimize import minimize

from .ffi import use_pointing_model
from .postcard import Postcard

__all__  = ['TargetData']

class TargetData(object):
    """
    Object containing the light curve, target pixel file, and related information
    for any given source.

    Parameters
    ----------
    source : an ellie.Source object

    Attributes
    ----------
    tpf : [lightkurve TargetPixelFile object](https://lightkurve.keplerscience.org/api/targetpixelfile.html)
        target pixel file
    best_lightcurve : [lightkurve LightCurve object](https://lightkurve.keplerscience.org/api/lightcurve.html)
        extracted light curve
    centroid_trace : (2, N_time) np.ndarray
        (xs, ys) in TPF pixel coordinates as a function of time
    best_aperture :

    all_lightcurves
    all_apertures
    """
    def __init__(self, source, height=9, width=9):
        self.source_info = source
        self.custom_aperture = None

        if source.premade is not None:
            self.load()

        else:
            self.post_obj = Postcard(source.postcard)
            self.time  = self.post_obj.time
            self.load_pointing_model(source.sector, source.camera, source.chip)
            self.get_tpf_from_postcard(source.coords, source.postcard, height, width)
            self.create_apertures(height, width)
            self.get_lightcurve()
            self.center_of_mass()
            self.set_quality()
            self.set_header()


    def load_pointing_model(self, sector, camera, chip):
        from astropy.table import Table
        import urllib
        pointing_link = urllib.request.urlopen('http://jet.uchicago.edu/tess_postcards/pointingModel_{}_{}-{}.txt'.format(sector,
                                                                                                                          camera,
                                                                                                                          chip))
        pointing = pointing_link.read().decode('utf-8')
        pointing = Table.read(pointing, format='ascii.basic') # guide to postcard locations
        self.pointing_model = pointing
        return


    def get_tpf_from_postcard(self, pos, postcard, height, width):
        """
        Creates a FITS file for a given source that includes:
            Extension[0] = header
            Extension[1] = (9x9xn) TPF, where n is the number of cadences in an observing run
            Extension[2] = (3 x n) time, raw flux, systematics corrected flux
        Defines
            self.tpf     = flux cutout from postcard
            self.tpf_err = flux error cutout from postcard
            self.centroid_xs = pointing model corrected x pixel positions
            self.centroid_ys = pointing model corrected y pixel positions
        """
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D

        self.tpf = None
        self.centroid_xs = None
        self.centroid_ys = None

        def apply_pointing_model(xy):
            centroid_xs, centroid_ys = [], []
            for i in range(len(self.pointing_model)):
                new_coords = use_pointing_model(xy, self.pointing_model[i])
                centroid_xs.append(new_coords[0][0])
                centroid_ys.append(new_coords[0][1])
            self.centroid_xs = np.array(centroid_xs)
            self.centroid_ys = np.array(centroid_ys)
            return

        xy = WCS(self.post_obj.header).all_world2pix(pos[0], pos[1], 1)

        apply_pointing_model(xy)

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
        return


    def create_apertures(self, height, width):
        """
        Finds the "best" aperture (i.e. the one that produces the smallest std light curve) for a range of
        sizes and shapes.
        Defines
            self.all_aperture = an array of masks for all apertures tested
        """
        from photutils import CircularAperture, RectangularAperture

        self.all_apertures = None

        # Saves some time by pre-loading apertures for 9x9 TPFs
        if (height,width) == (9,9):
            self.all_apertures = np.load('default_apertures.npy')

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

        return



    def get_lightcurve(self, custom_mask=False):
        """
        Extracts a light curve using the given aperture and TPF.
        Allows the user to pass in a mask to use, otherwise sets
            "best" lightcurve and aperture (min std)
        Mask is a 2D array of the same shape as TPF (9x9)
        Defines:
            self.flux
            self.flux_err
            self.aperture
            self.all_lc     (if mask=None)
            self.all_lc_err (if mask=None)
        """

        self.flux       = None
        self.flux_err   = None
        self.aperture   = None

        if custom_mask is False:

            self.all_lc      = None
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
            if self.custom_aperture is None:
                print("You have not defined a custom aperture. You can do this by calling the function .custom_aperture")
            else:
                lc = np.zeros(len(self.tpf))
                for cad in range(len(self.tpf)):
                    lc[cad]     = np.sum( self.tpf[cad]     * self.custom_aperture )
                    lc_err[cad] = np.sqrt( np.sum( self.tpf_err[cad]**2 * self.custom_aperture ))
                self.raw_flux   = np.array(lc)
                self.corr_flux  = self.jitter_corr(flux=lc)
                self.flux_err   = np.array(lc_err)

        return


    def center_of_mass(self):
        """
        Calculates the center of mass of the source across all cadences using muchbettermoments and self.best_aperture 
        Finds the brightest pixel in a (9x9) region summed up over all cadence
        Searches a smaller (3x3) region around this pixel at each cadence and uses muchbettermoments to find the maximum
            
        Sets:
            self.x_com
            self.y_com
        """
        from muchbettermoments import quadratic_2d
        from astropy.nddata.utils import Cutout2D

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
            self.y_com.append(c_frame[0])
        return



    def set_quality(self):
        """
        Currently (10/13/2018), this function sets a flag for when the centroid is
            3 sigma away from the mean either in the x or y direction
        Hopefully in the future, MAST will put in some quality flags for us
        Our flags and their flags will be combnied, if they create flags
        """
        bad = np.where( (self.centroid_xs > np.mean(self.centroid_xs)+3*np.std(self.centroid_xs)) | (self.centroid_ys > np.mean(self.centroid_ys)+3*np.std(self.centroid_ys)))

        quality = np.zeros(np.shape(self.time))
        quality[bad] = 1
        self.quality = quality



    def psf_lightcurve(self, nstars=1, model='gaussian', xc=[4.5], yc=[4.5]):
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


    def custom_aperture(self, height, width, shape=None, r=0.0, l=0.0, w=0.0, theta=0.0, pos=None, method='exact'):

        """
        Allows the user to input their own aperture of a given shape (either 'circle' or
            'rectangle' are accepted) of a given size {radius of circle: r, length of rectangle: l,
            width of rectangle: w, rotation of rectangle: t}
        Pos is the position given in pixel space
        Method defaults to 'exact'
        Defines
            self.custom_aperture: 2D array of shape (9x9)
        """
        from photutils import CircularAperture, RectangularAperture

        self.custom_aperture = None

        shape = shape.lower()
        if pos is None:
            pos = (width/2, height/2)
        else:
            pos = pos

        if shape == 'circle':
            if r == 0.0:
                print ("Please set a radius (in pixels) for your aperture")
            else:
                aperture = CircularAperture(pos, r=r)
                self.custom_aperture = aperture.to_mask(method=method)[0].to_image(shape=((
                            np.shape(self.tpf[0]))))

        elif shape == 'rectangle':
            if l==0.0 or w==0.0:
                print("For a rectangular aperture, please set both length and width: custom_aperture(shape='rectangle', l=#, w=#)")
            else:
                aperture = RectangularAperture(pos, l=l, w=w, t=theta)
                self.custom_aperture = aperture.to_mask(method=method)[0].to_image(shape=((
                            np.shape(self.tpf[0]))))
        else:
            print("Aperture shape not recognized. Please set shape == 'circle' or 'rectangle'")




    def jitter_corr(self, cen=0.0, flux=None):
        """
        Corrects for jitter in the light curve by quadratically regressing with centroid position.
        """

        def parabola(params, x, y, f_obs, y_err):
            nonlocal cen
            c1, c2, c3, c4, c5 = params
            f_corr = f_obs * (c1 + c2*(x-cen) + c3*(x-cen)**2 + c4*(y-cen) + c5*(y-cen)**2)
            return np.sum( ((1-f_corr)/y_err)**2)

        flux=np.array(flux)

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


    def rotation_corr(self):
        """ Corrects for spacecraft roll using Lightkurve """
        sff = SFFCorrector()
        lc_new = sff.correct(self.time, self.flux, self.centroid_xs, self.centroid_ys, niters=1,
                                   windows=1, polyorder=5)
        self.flux = np.copy(lc_new.flux)


    def set_header(self):
        """
        Defines the header for the TPF
        Sets:
            self.header
        """
        from time import strftime
        from astropy.io import fits

        self.header = self.post_obj.header
        self.header.update({'CREATED':strftime('%Y-%m-%d')})

        # Removes postcard specific header information
        for keyword in ['POST_HEIGHT', 'POST_WIDTH', 'CEN_X', 'CEN_Y', 'CEN_RA', 'CEN_DEC', 'POSTPIX1', 'POSTPIX2']:
            self.header.remove(keyword)

        # Adds TPF specific header information
        self.header.append(fits.Card(keyword='TIC_ID', value=self.source_info.tic,
                                     comment='TESS Input Catalog ID'))
        self.header.append(fits.Card(keyword='TMAG', value=self.source_info.tess_mag,
                                     comment='TESS mag'))
        self.header.append(fits.Card(keyword='GAIA_ID', value=self.source_info.gaia,
                                     comment='Associated Gaia ID'))
#        self.header.append(fits.Card(keyword='CRPIX1', value=self.cen_x + self.post_obj.origin_xy[0],
#                                     comment='central pixel of TPF in FFI'))
#        self.header.append(fits.Card(keyword='CRPIX2', value=self.cen_y + self.post_obj.origin_xy[1],
#                                     comment='central pixel of TPF in FFI'))
        self.header.append(fits.Card(keyword='POSTPIX1', value= self.source_info.coords[0]-self.dimensions[1],
                                     comment='origin of postcard axis 1'))
        self.header.append(fits.Card(keyword='POSTPIX2', value= self.source_info.coords[1]-self.dimensions[2],
                                     comment='origin of postcard axis 2'))
        self.header.append(fits.Card(keyword='CEN_RA', value = self.source_info.coords[0],
                                     comment='RA of TPF source'))
        self.header.append(fits.Card(keyword='CEN_DEC', value=self.source_info.coords[1],
                                     comment='DEC of TPF source'))
        self.header.append(fits.Card(keyword='TPF_H', value=np.shape(self.tpf[0])[0],
                                     comment='Height of the TPF in pixels'))
        self.header.append(fits.Card(keyword='TPF_W', value=np.shape(self.tpf[0])[1],
                                           comment='Width of the TPF in pixels'))



    def save(self, output_fn=None):
        """
        Saves a created TPF object to a FITS file
        """
        from astropy.io import fits
        from astropy.table import Table, Column


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
        ext1['time']       = self.time
        ext1['tpf']        = self.tpf
        ext1['tpf_err']    = self.tpf_err
        ext1['raw_flux']   = self.raw_flux
        ext1['corr_flux']  = self.corr_flux
        ext1['flux_err']   = self.flux_err
        ext1['quality']    = self.quality
        ext1['x_centroid'] = self.centroid_xs
        ext1['y_centroid'] = self.centroid_ys

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
            hdu.writeto('hlsp_eleanor_tess_ffi_lc_TIC{}_s{}_v0.1.fits'.format(self.source_info.tic, self.source_info.sector), overwrite=True)
        else:
            hdu.writeto(output_fn)



    def load(self):
        """
        Loads in and sets all the attributes for a pre-created TPF file
        """
        from astropy.io import fits
        from astropy.table import Table

        hdu = fits.open(self.source_info.fn)
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
