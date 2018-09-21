import numpy as np
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
from lightkurve import SFFCorrector
from scipy import ndimage
from scipy.optimize import minimize


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
        self.get_tpf_from_postcard(source.pos, source.postcard)
        self.choose_aperture()
        self.get_lightcurve()

    def get_tpf_from_postcard(self, pos, postcard):
        """
        Creates a FITS file for a given source that includes:
            Extension[0] = header
            Extension[1] = (9x9xn) TPF, where n is the number of cadences in an observing run
            Extension[2] = (3 x n) time, raw flux, systematics corrected flux
        """
        self.tpf = None
        self.centroid_xs = None
        self.centroid_ys = None

        def init_shift(xy, postcard):
            """ Offsets (x,y) coords of source by pointing model """
            theta, delX, delY = postcard.pointing[0]['medT'], postcard.pointing[0]['medX'], postcard.pointing[0]['medY']

            x = xy[0]*np.cos(theta) - xy[1]*np.sin(theta) + delX
            y = xy[0]*np.sin(theta) + xy[1]*np.cos(theta) + delY

            return np.array([x,y])

        def centering_shift(tpf):
            """ Creates an additional shift to put source at (4,4) of TPF file """
            """                  Returns: required pixel shift                 """
            xdim, ydim = len(tpf[0][0]), len(tpf[0][1])
            center = quadratic_2d(tpf[0])
            shift = [int(round(xdim/2-center[1])), int(round(xdim/2-center[0]))]

            return shift

        # Calculate pixel distance between center of postcard and target in FFI pixel coords
        delta_pix = np.array([xy_shift[0] - postcard.center_pos[0], xy_shift[1] - postcard.center_pos[1]])
        delta_init = np.array([xy[0] - postcard.center_pos[0], xy[1] - postcard.center_pos[1]])
        # Apply shift to coordinates for center of tpf
        newX = int(np.ceil(postcard.dims[1]/2. + delta_pix[1]))
        newY = int(np.ceil(postcard.dims[0]/2. + delta_pix[0]))

        init_x = int(np.ceil(postcard.dims[0]/2. + delta_init[1]))
        init_y = int(np.ceil(postcard.dims[1]/2. + delta_init[0]))
        # Define tpf as region of postcard around target
        post_fits = postcard.data  # ??
        tpf = post_fits[:, newX-6:newX+7, newY-6:newY+7]
        init_tpf = post_fits[:, init_x-6:init_x+7, init_y-6:init_y+7]

        self.tpf = tpf


    def choose_aperture(self):
        """
        Finds the "best" aperture (i.e. the one that produces the smallest std light curve) for a range of
        sizes and shapes.
        """
        self.aperture = None

        r_list = np.arange(1.5, 3.5, 0.5) # aperture radii to try
        matrix = np.zeros( (len(r_list), 2) )
        system = np.zeros( (len(r_list), 2) )
        sigma  = np.zeros( (len(r_list), 2) )

        for i in range(len(r_list)):
            pos = (x_point, y_point)
            circ, rect = aperture(r_list[i], pos)
            # Completes aperture sums for each tpf.flux and each aperture shape
            matrix[i][0] = aperture_photometry(self.tpf, circ)['aperture_sum'].data[0]
            matrix[i][1] = aperture_photometry(self.tpf, rect)['aperture_sum'].data[0]
            matrix[i][0] = matrix[i][0] / np.nanmedian(matrix[i][0])
            matrix[i][1] = matrix[i][1] / np.nanmedian(matrix[i][1])

        # Creates a complete, systematics corrected light curve for each aperture
        for i in range(len(r_list)):
            lc_circ = self.system_corr(matrix[i][0], x, y, jitter=True, roll=True)
            system[i][0] = lc_circ
            sigma[i][0] = np.std(lc_circ)
            lc_rect = self.system_corr(matrix[i][1], x, y, jitter=True, roll=True)
            system[i][1] = lc_rect
            sigma[i][1] = np.std(lc_rect)

        best = np.where(sigma==np.min(sigma))
        r_ind, s_ind = best[0][0], best[1][0]

        def centroidOffset(tpf, file_cen):
            """ Finds offset between center of TPF and centroid of first cadence """
            tpf_com = Cutout2D(tpf[0], position=(len(tpf[0])/2, len(tpf[0])/2), size=(4,4))
            com = ndimage.measurements.center_of_mass(tpf_com.data.T - np.median(tpf_com.data))
            return len(tpf_com.data)/2-com[0], len(tpf_com.data)/2-com[1]

        def aperture(r, pos):
            """ Creates circular & rectangular apertures of given size """
            circ = CircularAperture(pos, r)
            rect = RectangularAperture(pos, r, r, 0.0)
            return circ, rect

        file_cen = len(self.tpf[0])/2.
        initParams = self.pointing[0]
        theta, delX, delY = self.pointing['medT'].data, self.pointing['medX'].data, self.pointing['medY'].data

        startX, startY = centroidOffset(tpf, file_cen)

        x, y = [], []
        for i in range(len(theta)):
            x.append( startX*np.cos(theta[i]) - startY*np.sin(theta[i]) + delX[i] )
            y.append( startX*np.sin(theta[i]) + startY*np.cos(theta[i]) + delY[i] )
        x, y = np.array(x), np.array(y)

        print("*************")
        print("We're doing our best to find the ideal aperture shape & size for your source.")
        print("*************")

        radius, shape, lc, uncorr = findLC(x, y)

        self.lc = lc
        self.aperture = (radius, shape) # ?
        self.centroid_xs = x
        self.centroid_ys = y


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

    def get_lightcurve(self):
        """
        Extracts a light curve using the given aperture and TPF.
        """
        lc = aperture_photometry(self.tpf, self.aperture)['aperture_sum'].data[0])
        self.lc = lc


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
