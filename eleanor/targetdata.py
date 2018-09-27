import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
from lightkurve import SFFCorrector
from scipy import ndimage
from scipy.optimize import minimize

from ffi import use_pointing_model
from postcard import Postcard

__all__ = ['TargetData']

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
    def __init__(self, source):
        self.post_obj = Postcard(source.postcard)
        self.load_pointing_model(source.sector, source.camera, source.chip)
        self.get_tpf_from_postcard(source.coords, source.postcard)
        self.fit_apertures()
#        self.get_lightcurve()


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
                                                                                                                          
        
    def get_tpf_from_postcard(self, pos, postcard):
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
        from muchbettermoments import quadratic_2d
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
            self.centroid_xs = centroid_xs
            self.centroid_ys = centroid_ys
            return

        xy = WCS(self.post_obj.header).all_world2pix(pos[0], pos[1], 1)
        apply_pointing_model(xy)

        # Define tpf as region of postcard around target
        med_x, med_y = np.nanmedian(self.centroid_xs), np.nanmedian(self.centroid_ys)
        med_x, med_y = int(np.floor(med_x)), int(np.floor(med_y))

        post_flux = np.transpose(self.post_obj.flux, (2,0,1))        
        post_err  = np.transpose(self.post_obj.flux_err, (2,0,1))
        self.tpf  = post_flux[:, med_y-4:med_y+5, med_x-4:med_x+5]
        self.tpf_err = post_err[:, med_y-4:med_y+5, med_x-4:med_x+5]
        return


    def fit_apertures(self):
        """
        Finds the "best" aperture (i.e. the one that produces the smallest std light curve) for a range of
        sizes and shapes.
        Defines
            self.lc = the resulting light curve from the "best" aperture
            self.aperture = the "best" aperture
            self.all_lc = an array of light curves from all apertures tested
            self.all_aperture = an array of masks for all apertures tested
        """
        from photutils import CircularAperture, RectangularAperture, aperture_photometry, ApertureMask
        from astropy.table import Table

        self.aperture     = None
        self.lc           = None
        self.all_lc       = None
        self.all_aperture = None

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

        r_list = np.arange(1,4,0.5)
        circles, rectangles = [], []
        plt.imshow(self.tpf[0], origin='lower')
        plt.show()
        for r in r_list:
            circles.append(circle((4,4),r))
            rectangles.append(rectangle((4,4),r,r,0.0))
            
        for i in range(len(self.tpf)):
            bc = binary(self.tpf[i], circles, self.tpf_err[i])
            br = binary(self.tpf[i], rectangles, self.tpf_err[i])
            wc = weighted(self.tpf[i], circles, self.tpf_err[i])
            if i == 0:
                binary_circ = Table(names=bc.colnames)
                binary_rect = Table(names=br.colnames)
                weight_circ = Table(names=wc.colnames)
            binary_circ.add_row(bc[0])
        print(len(binary_circ['aperture_sum_3']))



    def custom_aperture(self, shape=None, r=0.0, l=0.0, w=0.0, t=0.0, pointing=True,
                        jitter=True, roll=True, input_fn=None, pos=[]):
        # NOTE: this should probably return a pixel mask rather than a light curve
        """
        Allows the user to input their own aperture of a given shape (either 'circle' or
            'rectangle' are accepted) of a given size {radius of circle: r, length of rectangle: l,
            width of rectangle: w, rotation of rectangle: t}
        The user can have the aperture not follow the pointing model by setting pointing=False
        The user can determine which kinds of corrections would like to be applied to their light curve
            jitter & roll are automatically set to True
        Pos is the position given in pixel space
        """
        def create_ap(pos):
            """ Defines the custom aperture, as inputted by the user """
            nonlocal shape, r, l, w, t
            if shape=='circle':
                return CircularAperture(pos, r)
            elif shape=='rectangle':
                return RectangularAperture(pos, l, w, t)
            else:
                print("Shape of aperture not recognized. Please input: circle or rectangle")
            return

        # Checks for all of the correct inputs to create an aperture
        if shape=='circle' and r==0.0:
            print("Please input a radius of your aperture when calling custom_aperture(shape='circle', r=#)")
            return
        if shape=='rectangle' and (l==0.0 or w==0.0):
            print("You are missing a dimension of your rectangular aperture. Please set custom_aperture(shape='rectangle', l=#, w=#)")
            return

        if self.tic != None:
            hdu = fits.open(self.tic_fn)
        elif self.gaia != None:
            hdu = fits.open(self.gaia_fn)
        else:
            hdu = fits.open(input_fn)

        tpf = hdu[0].data

        if len(pos) < 2:
            center = [4,4]
        else:
            center = pos
        # Grabs the pointing model if the user has set pointing==True
        if pointing==True:
            pm  = self.get_pointing(header = hdu[0].header)
            x = center[0]*np.cos(pm['medT'].data) - center[1]*np.sin(pm['medT'].data) + pm['medX'].data
            y = center[0]*np.sin(pm['medT'].data) + center[1]*np.cos(pm['medT'].data) + pm['medY'].data

        x, y = x-np.median(x)+center[0], y-np.median(y)+center[1]
        lc = cust_lc(center, x, y)

        if pointing==False and roll==True:
            print("Sorry, our roll correction, lightkurve.SFFCorrector, only works when you use the pointing model.\nFor now, we're turning roll corrections off.")
            roll=False

        lc = self.system_corr(lc, x, y, jitter=jitter, roll=roll)
        return lc

#    def get_lightcurve(self):
#        """
#        Extracts a light curve using the given aperture and TPF.
#        """
#        lc = aperture_photometry(self.tpf, self.aperture)['aperture_sum'].data[0]
#        self.lc = lc


    def jitter_corr(self):
        """
        Corrects for jitter in the light curve by quadratically regressing with centroid position.
        """
        x_pos, y_pos = self.centroid_x, self.centroid_y
        lc = self.lc

        def parabola(params, x, y, f_obs, y_err):
            c1, c2, c3, c4, c5 = params
            f_corr = f_obs * (c1 + c2*(x-2.5) + c3*(x-2.5)**2 + c4*(y-2.5) + c5*(y-2.5)**2)
            return np.sum( ((1-f_corr)/y_err)**2)

        # Masks out anything >= 2.5 sigma above the mean
        mask = np.ones(len(lc), dtype=bool)
        for i in range(5):
            lc_new = []
            std_mask = np.std(lc[mask])

            inds = np.where(lc <= np.mean(lc)-2.5*std_mask)
            y_err = np.ones(len(lc))**np.std(lc)
            for j in inds:
                y_err[j] = np.inf
                mask[j]  = False

            if i == 0:
                initGuess = [3, 3, 3, 3, 3]
            else:
                initGuess = test.x

            bnds = ((-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.), (-15.,15.))
            test = minimize(parabola, initGuess, args=(x_pos, y_pos, lc, y_err), bounds=bnds)
            c1, c2, c3, c4, c5 = test.x
            lc_new = lc * (c1 + c2*(x_pos-2.5) + c3*(x_pos-2.5)**2 + c4*(y_pos-2.5) + c5*(y_pos-2.5)**2)

        self.lc = np.copy(lc_new)

    def rotation_corr(self):
        """ Corrects for spacecraft roll using Lightkurve """
        time = np.arange(0, len(self.lc), 1)
        sff = SFFCorrector()
        x_pos, y_pos = self.centroid_x, self.centroid_y
        lc_new = sff.correct(time, self.lc, x_pos, y_pos, niters=1,
                                   windows=1, polyorder=5)
        self.lc = np.copy(lc_new)


    def system_corr(self, jitter=False, roll=False):
        """
        Allows for systematics correction of a given light curve
        """
        if jitter==True:
            self.jitter_corr()
        if roll==True:
            self.rotation_corr()
