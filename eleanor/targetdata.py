import numpy as np
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from photutils import MMMBackground
from lightkurve import SFFCorrector, lightcurve
from scipy.optimize import minimize
from astropy import time, coordinates as coord, units as u
from astropy.coordinates import SkyCoord, Angle
from astropy.table import Table, Column
from astropy.wcs import WCS
from astropy.stats import SigmaClip
from time import strftime
from astropy.io import fits
from muchbettermoments import quadratic_2d
from scipy.stats import mode
from urllib.request import urlopen
import os, sys, copy
import os.path
import warnings
import pickle
import eleanor

from .ffi import use_pointing_model, load_pointing_model
from .postcard import Postcard, Postcard_tesscut


__all__  = ['TargetData']

class TargetData(object):
    """
    Object containing the light curve, target pixel file, and related information
    for any given source.

    Parameters
    ----------
    source : eleanor.Source
        The source object to use.
    height : int, optional
        Height in pixels of TPF to retrieve. Default value is 13 pixels. Must be an odd number,
        or else will return an aperture one pixel taller than requested so target
        falls on central pixel.
    width : int, optional
        Width in pixels of TPF to retrieve. Default value is 13 pixels. Must be an odd number,
        or else will return an aperture one pixel wider than requested so target
        falls on central pixel.
    bkg_size : int, optional
        Size of box to use for background estimation. If not set, will default to the width of the 
        target pixel file.
    crowded_field : bool, optional
        If true, will return a light curve built using a small aperture (not more than 8 pixels in size).
    do_pca : bool, optional
        If true, will return a PCA-corrected light curve.
    do_psf : bool, optional
        If true, will return a light curve made with a simple PSF model.
    cal_cadences : tuple, optional
        Start and end cadence numbers to use for optimal aperture selection.
    try_load: bool, optional
        If true, will search hidden ~/.eleanor directory to see if TPF has already
        been created. 

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

    def __init__(self, source, height=13, width=13, save_postcard=True, do_pca=False, do_psf=False, 
                 bkg_size=None, crowded_field=False, cal_cadences=None, try_load=True, language='English'):

        self.source_info = source 
        self.language = language
        self.pca_flux = None
        self.psf_flux = None

        if self.source_info.premade is True:
            self.load(directory=self.source_info.fn_dir)

        else:            
            fnf = True
            # Checks to see if file exists already
            if try_load==True:
                try:
                    default_fn = 'hlsp_eleanor_tess_ffi_tic{0}_s{1:02d}_tess_v{2}_lc.fits'.format(self.source_info.tic,
                                                                                                  self.source_info.sector,
                                                                                                  eleanor.__version__)
                    self.load(fn=default_fn)
                    fnf = False
                except:
                    pass

            if fnf is True:
                self.aperture = None
                
                if source.tc == False:
                    self.post_obj = Postcard(source.postcard, source.ELEANORURL)
                else:
                    self.post_obj = Postcard_tesscut(source.cutout)
                
                self.ffiindex = self.post_obj.ffiindex
                self.flux_bkg = self.post_obj.bkg 
                self.get_time(source.coords)

                if bkg_size is None:
                    bkg_size = width

                # Uses the contamination ratio for crowded field if available and 
                # not already set by the user   
                if crowded_field is True:
                    self.crowded_field = 1
                else:
                    if self.source_info.contratio is not None:
                        self.crowded_field = self.source_info.contratio
                    else:
                        self.crowded_field = 0


                if cal_cadences is None:
                    self.cal_cadences = (0, len(self.post_obj.time))
                else:
                    self.cal_cadences = cal_cadences
            
                try:
                    self.pointing_model = load_pointing_model(source.sector, source.camera, source.chip)
                except:
                    self.pointing_model = None
                    
                self.get_tpf_from_postcard(source.coords, source.postcard, height, width, bkg_size, save_postcard, source)
                self.set_quality()
                self.get_cbvs()
            

                self.create_apertures(height, width)
            
                self.get_lightcurve()

                if do_pca == True:
                    self.corrected_flux(pca=True)
                else:
                    self.modes = None
                    self.pca_flux = None

                if do_psf == True:
                    self.psf_lightcurve()
                else:
                    self.psf_flux = None

                self.center_of_mass()
            
                self.set_header()


    def get_time(self, coords):
        """Gets time, including light travel time correction to solar system barycenter for object given location"""
        t0 = self.post_obj.time - self.post_obj.barycorr
        
        ra = Angle(coords[0], u.deg)
        dec = Angle(coords[1], u.deg)
        
        greenwich = coord.EarthLocation.of_site('greenwich')
        times = time.Time(t0+2457000, format='jd',
                   scale='utc', location=greenwich)
        ltt_bary = times.light_travel_time(SkyCoord(ra, dec)).value
        
        self.time = t0 + ltt_bary
        self.barycorr = ltt_bary

    def get_tpf_from_postcard(self, pos, postcard, height, width, bkg_size, save_postcard, source):
        """Gets TPF from postcard."""

        self.tpf         = None
        self.centroid_xs = None
        self.centroid_ys = None

        xy = WCS(self.post_obj.header).all_world2pix(pos[0], pos[1], 1)
        
        # Apply the pointing model to each cadence to find the centroids
        
        if self.pointing_model is None:
            self.centroid_xs = np.zeros_like(self.post_obj.time)
            self.centroid_ys = np.zeros_like(self.post_obj.time)
        else:
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
        
        if source.tc == False:

            post_flux = np.transpose(self.post_obj.flux, (2,0,1))
            post_err  = np.transpose(self.post_obj.flux_err, (2,0,1))
        else:
            post_flux = self.post_obj.flux + 0.0
            post_err = self.post_obj.flux_err + 0.0


        self.cen_x, self.cen_y = med_x, med_y
        
        y_length, x_length = int(np.floor(height/2.)), int(np.floor(width/2.))
        y_bkg_len, x_bkg_len = int(np.floor(bkg_size/2.)), int(np.floor(bkg_size/2.))

        y_low_lim = med_y-y_length
        y_upp_lim = med_y+y_length+1
        x_low_lim = med_x-x_length
        x_upp_lim = med_x+x_length+1
        
        y_low_bkg = med_y-y_bkg_len
        y_upp_bkg = med_y+y_bkg_len + 1
        x_low_bkg = med_x-x_bkg_len
        x_upp_bkg = med_x+x_bkg_len + 1
    

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
            
        if y_low_bkg <= 0:
            y_low_bkg = 0
        if x_low_bkg <= 0:
            x_low_bkg = 0
        if y_upp_bkg  > post_y_upp:
            y_upp_bkg = post_y_upp+1
        if x_upp_bkg >  post_x_upp:
            x_upp_bkg = post_x_upp+1

            
        if source.tc == False:
            
            if (x_low_lim==0) or (y_low_lim==0) or (x_upp_lim==post_x_upp) or (y_upp_lim==post_y_upp):
                warnings.warn("The size postage stamp you are requesting falls off the edge of the postcard.")
                warnings.warn("WARNING: Your postage stamp may not be centered.")

            self.tpf     = post_flux[:, y_low_lim:y_upp_lim, x_low_lim:x_upp_lim]
            self.bkg_tpf = post_flux[:, y_low_bkg:y_upp_bkg, x_low_bkg:x_upp_bkg]
            self.tpf_err = post_err[: , y_low_lim:y_upp_lim, x_low_lim:x_upp_lim]
            
        else:            
            if (height > 31) or (width > 31):
                raise ValueError("Maximum allowed TPF size is 31 x 31 pixels.")

            self.tpf     = post_flux[:, 15-y_length:15+y_length+1, 15-x_length:15+x_length+1]
            self.bkg_tpf = post_flux
            self.tpf_err = post_err[:, 15-y_length:15+y_length+1, 15-x_length:15+x_length+1]           
            
        self.dimensions = np.shape(self.tpf)
        
        
        summed_tpf = np.nansum(self.tpf, axis=0)
        mpix = np.unravel_index(summed_tpf.argmax(), summed_tpf.shape)
        if np.abs(mpix[0] - x_length) > 1:
            self.crowded_field = 1
        if np.abs(mpix[1] - y_length) > 1:
            self.crowded_field = 1
            

        self.bkg_subtraction()

        self.tpf = self.tpf

        if save_postcard == False:
            try:
                if self.source_info.tc == False:
                    os.remove(self.post_obj.local_path)
                else:
                    os.remove(self.source_info.postcard_path)
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
        

        default = 13

        # Creates aperture based on the requested size of TPF
        if (height, width) == (default, default):
            self.all_apertures = all_apertures

        else:
            new_aps = []

            h_diff = height-default; w_diff = width-default
            half_h = int(np.ceil(np.abs(h_diff/2))) ; half_w = int(np.ceil(np.abs(w_diff/2)))

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
        flux = self.bkg_tpf

        self.tpf_flux_bkg = []

        sigma_clip = SigmaClip(sigma=sigma)
        bkg = MMMBackground(sigma_clip=sigma_clip)

        for i in range(len(time)):
            bkg_value = bkg.calc_background(flux[i])
            self.tpf_flux_bkg.append(bkg_value)

        self.tpf_flux_bkg = np.array(self.tpf_flux_bkg)
        


    def get_lightcurve(self, aperture=None):
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
                lc[cad]     = np.nansum( self.tpf[cad] * mask)
                lc_err[cad] = np.sqrt( np.nansum( self.tpf_err[cad]**2 * mask))
            self.raw_flux   = np.array(lc) 
            self.corr_flux  = self.corrected_flux(flux=lc, skip=50)
            self.flux_err   = np.array(lc_err)
            return

        self.flux_err = None

        if self.language == 'Australian':
            print("G'day Mate! ʕ •ᴥ•ʔ Your light curves are being translated ...")

        if (self.aperture is None):

            self.all_lc_err  = None

            all_raw_lc_pc_sub  = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_lc_err  = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_corr_lc_pc_sub = np.copy(all_raw_lc_pc_sub)
            all_raw_lc_tpf_sub = np.zeros((len(self.all_apertures), len(self.tpf)))
            all_corr_lc_tpf_sub = np.copy(all_raw_lc_tpf_sub)
            
            for epoch in range(len(self.time)):
                self.tpf[epoch] -= self.tpf_flux_bkg[epoch]

            pc_stds = np.ones(len(self.all_apertures)) 
            tpf_stds = np.ones(len(self.all_apertures)) 
            
            ap_size = np.nansum(self.all_apertures, axis=(1,2))
            

            for a in range(len(self.all_apertures)):       
                for cad in range(len(self.tpf)):
                    try:
                        all_lc_err[a, cad]   = np.sqrt( np.nansum( self.tpf_err[cad]**2 * self.all_apertures[a] ))
                        all_raw_lc_tpf_sub[a, cad]   = np.nansum( (self.tpf[cad]) * self.all_apertures[a] )
                        all_raw_lc_pc_sub[a, cad]  = np.nansum( (self.tpf[cad] + self.tpf_flux_bkg[cad]) * self.all_apertures[a] )
                        
                    except ValueError:
                        continue

                ## Remove something from all_raw_lc before passing into jitter_corr ##
                try:
                    all_corr_lc_pc_sub[a] = self.corrected_flux(flux=all_raw_lc_pc_sub[a]/np.nanmedian(all_raw_lc_pc_sub[a]))
                    all_corr_lc_tpf_sub[a]= self.corrected_flux(flux=all_raw_lc_tpf_sub[a]/np.nanmedian(all_raw_lc_tpf_sub[a]))
                except IndexError:
                    continue

                q = self.quality == 0

                lc_obj_tpf = lightcurve.LightCurve(time = self.time[q][self.cal_cadences[0]:self.cal_cadences[1]],
                                       flux = all_corr_lc_tpf_sub[a][q][self.cal_cadences[0]:self.cal_cadences[1]])
                flat_lc_tpf = lc_obj_tpf.flatten(polyorder=2, window_length=51)
                
                tpf_stds[a] =  np.std(flat_lc_tpf.flux)

                lc_obj_pc = lightcurve.LightCurve(time = self.time[q][self.cal_cadences[0]:self.cal_cadences[1]],
                                                   flux = all_corr_lc_pc_sub[a][q][self.cal_cadences[0]:self.cal_cadences[1]])
                flat_lc_pc = lc_obj_pc.flatten(polyorder=2, window_length=51)
                pc_stds[a] = np.std(flat_lc_pc.flux)

                all_corr_lc_pc_sub[a]  = all_corr_lc_pc_sub[a]  * np.nanmedian(all_raw_lc_pc_sub[a])
                all_corr_lc_tpf_sub[a] = all_corr_lc_tpf_sub[a] * np.nanmedian(all_raw_lc_tpf_sub[a])
            self.all_raw_lc  = np.array(all_raw_lc_pc_sub)
            self.all_lc_err  = np.array(all_lc_err)
            self.all_corr_lc = np.array(all_corr_lc_pc_sub)
            
            if self.language == 'Australian':
                for i in range(len(self.all_raw_lc)):
                    med = np.nanmedian(self.all_raw_lc[i])
                    self.all_raw_lc[i] = (med-self.all_raw_lc[i]) + med

                    med = np.nanmedian(self.all_corr_lc[i])
                    self.all_corr_lc[i] = (med-self.all_corr_lc[i]) + med


            if self.crowded_field > 0.15:
                tpf_stds[ap_size > 8] = 1.0
                pc_stds[ap_size > 8] = 1.0

            best_ind_tpf = np.where(tpf_stds == np.min(tpf_stds))[0][0]
            best_ind_pc  = np.where(pc_stds == np.min(pc_stds))[0][0]
            

            ## Checks if postcard or tpf level bkg subtraction is better ##
            ## Prints bkg_type to TPF header ##
            if pc_stds[best_ind_pc] <= tpf_stds[best_ind_tpf]:
                best_ind = best_ind_pc
                self.bkg_type = 'PC_LEVEL'
                for epoch in range(len(self.time)):
                    self.tpf[epoch] += self.tpf_flux_bkg[epoch]
            else:
                best_ind = best_ind_tpf
                self.bkg_type = 'TPF_LEVEL'
                self.all_raw_lc  = np.array(all_raw_lc_tpf_sub)
                self.all_corr_lc = np.array(all_corr_lc_tpf_sub)

            self.corr_flux= self.all_corr_lc[best_ind]
            self.raw_flux = self.all_raw_lc[best_ind]
            self.aperture = self.all_apertures[best_ind]
            self.flux_err = self.all_lc_err[best_ind]
            self.aperture_size = np.nansum(self.aperture)
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
    

    def get_cbvs(self):
        """ Obtains the cotrending basis vectors (CBVs) as convolved down from the short-cadence targets.
        Parameters
        ----------
        """
        
        try:
            matrix_file = urlopen('https://archipelago.uchicago.edu/tess_postcards/metadata/s{0:04d}/cbv_components_s{0:04d}_{1:04d}_{2:04d}.txt'.format(self.source_info.sector,
                                                                                                                                                         self.source_info.camera,
                                                                                                                                                         self.source_info.chip))
            A = [float(x) for x in matrix_file.read().decode('utf-8').split()]
            cbvs = np.asarray(A)
            self.cbvs = np.reshape(cbvs, (len(self.time), 16))
            
        except:
            self.cbvs = np.zeros((len(self.time), 16))
        return


    def center_of_mass(self):
        """
        Calculates the position of the source across all cadences using `muchbettermoments` and `self.best_aperture`.

        Finds the brightest pixel in a (`height`, `width`) region summed up over all cadence.
        Searches a smaller (3x3) region around this pixel at each cadence and uses `muchbettermoments` to find the maximum.
        """

        self.x_com = []
        self.y_com = []

        summed_pixels = np.nansum(self.aperture * self.tpf, axis=0)
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
            
        self.x_com = np.array(self.x_com)
        self.y_com = np.array(self.y_com)

        return



    def set_quality(self):
        """ Reads in quality flags set in the postcard
        """
        self.quality = np.array(self.post_obj.quality)
        return


    def psf_lightcurve(self, data_arr = None, err_arr = None, bkg_arr = None, nstars=1, model='gaussian', likelihood='gaussian',
                       xc=None, yc=None, verbose=False,
                       err_method=True, ignore_pixels=None):
        """
        Performs PSF photometry for a selection of stars on a TPF.

        Parameters
        ----------
        data_arr: numpy.ndarray, optional
            Data array to fit with the PSF model. If None, will default to `TargetData.tpf`.
        err_arr: numpy.ndarray, optional
            Uncertainty array to fit with the PSF model. If None, will default to `TargetData.tpf_flux_err`.
        bkg_arr: numpy.ndarray, optional
            List of background values to include as initial guesses for the background model. If None,
            will default to `TargetData.flux_bkg`.
        nstars: int, optional
            Number of stars to be modeled on the TPF.
        model: string, optional
            PSF model to be applied. Presently must be `gaussian`, which models a single Gaussian.
            Will be extended in the future once TESS PRF models are made publicly available.
        likelihood: string, optinal
            The data statistics given the parameters. Options are: 'gaussian' and 'poisson'.
        xc: list, optional
            The x-coordinates of stars in the zeroth cadence. Must have length `nstars`.
            While the positions of stars will be fit in all cadences, the relative positions of
            stars will be fixed following the delta values from this list.
        yc: list, optional
            The y-coordinates of stars in the zeroth cadence. Must have length `nstars`.
            While the positions of stars will be fit in all cadences, the relative positions of
            stars will be fixed following the delta values from this list.
        verbose: bool, optional
            If True, return information about the shape of the PSF at every cadence as well as the
            PSF-inferred centroid shape.
        err_method: bool, optional
            If True, use the photometric uncertainties for each pixel in the TPF as delivered by the
            TESS team. Otherwise, each pixel takes an equal uncertainty. If `err_arr` is passed
            through instead, this setting is ignored.
        ignore_pixels: int, optional
            If not None, ignore a certain percentage of the brightest pixels away from the source
            target, effectively masking other nearby, bright stars. This strategy appears to do a
            reasonable job estimating the background more accurately in relatively crowded regions.
        """
        import tensorflow as tf
        from .models import Gaussian, Moffat
        from tqdm import tqdm
        
        tf.logging.set_verbosity(tf.logging.ERROR)

        if data_arr is None:
            data_arr = self.tpf + 0.0
        if err_arr is None:
            if err_method == True:
                err_arr = (self.tpf_err + 0.0) ** 2
            else:
                err_arr = np.ones_like(data_arr)
        if bkg_arr is None:
            bkg_arr = self.flux_bkg + 0.0

        if yc is None:
            yc = 0.5 * np.ones(nstars) * np.shape(data_arr[0])[1]
        if xc is None:
            xc = 0.5 * np.ones(nstars) * np.shape(data_arr[0])[0]

        dsum = np.nansum(data_arr, axis=(0))
        modepix = np.where(dsum == mode(dsum, axis=None)[0][0])
        if len(modepix[0]) > 2.5:
            for i in range(len(bkg_arr)):
                err_arr[i][modepix] = np.inf

        if ignore_pixels is not None:
            tpfsum = np.nansum(data_arr, axis=(0))
            percentile = 100-ignore_pixels
            tpfsum[int(xc[0]-1.5):int(xc[0]+2.5),int(yc[0]-1.5):int(yc[0]+2.5)] = 0.0
            err_arr[:, tpfsum > np.percentile(dsum, percentile)] = np.inf

        if len(xc) != nstars:
            raise ValueError('xc must have length nstars')
        if len(yc) != nstars:
            raise ValueError('yc must have length nstars')


        flux = tf.Variable(np.ones(nstars)*np.max(data_arr[0]), dtype=tf.float64)
        bkg = tf.Variable(bkg_arr[0], dtype=tf.float64)
        xshift = tf.Variable(0.0, dtype=tf.float64)
        yshift = tf.Variable(0.0, dtype=tf.float64)

        if (model == 'gaussian'):

            gaussian = Gaussian(shape=data_arr.shape[1:], col_ref=0, row_ref=0)

            a = tf.Variable(initial_value=1., dtype=tf.float64)
            b = tf.Variable(initial_value=0., dtype=tf.float64)
            c = tf.Variable(initial_value=1., dtype=tf.float64)
        

            if nstars == 1:
                mean = gaussian(flux, xc[0]+xshift, yc[0]+yshift, a, b, c)
            else:
                mean = [gaussian(flux[j], xc[j]+xshift, yc[j]+yshift, a, b, c) for j in range(nstars)]
                mean = np.sum(mean, axis=0)
                
            var_list = [flux, xshift, yshift, a, b, c, bkg]
            
            var_to_bounds = {flux: (0, np.infty), 
                             xshift: (-1.0, 1.0),
                             yshift: (-1.0, 1.0),
                             a: (0, np.infty),
                             b: (-0.5, 0.5),
                             c: (0, np.infty)
                            }
            
        elif model == 'moffat':

            moffat = Moffat(shape=data_arr.shape[1:], col_ref=0, row_ref=0)

            a = tf.Variable(initial_value=1., dtype=tf.float64)
            b = tf.Variable(initial_value=0., dtype=tf.float64)
            c = tf.Variable(initial_value=1., dtype=tf.float64)
            beta = tf.Variable(initial_value=1, dtype=tf.float64)
        

            if nstars == 1:
                mean = moffat(flux, xc[0]+xshift, yc[0]+yshift, a, b, c, beta)
            else:
                mean = [moffat(flux[j], xc[j]+xshift, yc[j]+yshift, a, b, c, beta) for j in range(nstars)]
                mean = np.sum(mean, axis=0)
                
            var_list = [flux, xshift, yshift, a, b, c, beta, bkg]
            
            var_to_bounds = {flux: (0, np.infty), 
                             xshift: (-2.0, 2.0),
                             yshift: (-2.0, 2.0),
                             a: (0, 3.0),
                             b: (-0.5, 0.5),
                             c: (0, 3.0),
                             beta: (0, 10)
                            }
            

            betaout = np.zeros(len(data_arr))
            

        else:
            raise ValueError('This model is not incorporated yet!') # we probably want this to be a warning actually,
                                                                    # and a gentle return
                
        aout = np.zeros(len(data_arr))
        bout = np.zeros(len(data_arr))
        cout = np.zeros(len(data_arr))
        xout = np.zeros(len(data_arr))
        yout = np.zeros(len(data_arr))

        mean += bkg

        data = tf.placeholder(dtype=tf.float64, shape=data_arr[0].shape)
        derr = tf.placeholder(dtype=tf.float64, shape=data_arr[0].shape)
        bkgval = tf.placeholder(dtype=tf.float64)

        if likelihood == 'gaussian':
            nll = tf.reduce_sum(tf.truediv(tf.squared_difference(mean, data), derr))
        elif likelihood == 'poisson':
            nll = tf.reduce_sum(tf.subtract(mean+bkgval, tf.multiply(data+bkgval, tf.log(mean+bkgval))))
        else:
            raise ValueError("likelihood argument {0} not supported".format(likelihood))

        grad = tf.gradients(nll, var_list)

        sess = tf.Session(config=tf.ConfigProto(device_count={'GPU': 0}))
        sess.run(tf.global_variables_initializer())
        


        
        optimizer = tf.contrib.opt.ScipyOptimizerInterface(nll, var_list, method='TNC', tol=1e-4, var_to_bounds=var_to_bounds)

        fout = np.zeros((len(data_arr), nstars))
        bkgout = np.zeros(len(data_arr))

        llout = np.zeros(len(data_arr))

        for i in tqdm(range(len(data_arr))):
            optim = optimizer.minimize(session=sess, feed_dict={data:data_arr[i], derr:err_arr[i], bkgval:bkg_arr[i]}) # we could also pass a pointing model here
                                                                           # and just fit a single offset in all frames

            fout[i] = sess.run(flux)
            bkgout[i] = sess.run(bkg)
            
            if model == 'gaussian':
                aout[i] = sess.run(a)
                bout[i] = sess.run(b)
                cout[i] = sess.run(c)
                xout[i] = sess.run(xshift)
                yout[i] = sess.run(yshift)
                llout[i] = sess.run(nll, feed_dict={data:data_arr[i], derr:err_arr[i], bkgval:bkg_arr[i]})
                
            if model == 'moffat':
                aout[i] = sess.run(a)
                bout[i] = sess.run(b)
                cout[i] = sess.run(c)
                xout[i] = sess.run(xshift)
                yout[i] = sess.run(yshift)
                llout[i] = sess.run(nll, feed_dict={data:data_arr[i], derr:err_arr[i], bkgval:bkg_arr[i]})
                betaout[i] = sess.run(beta)
            

        sess.close()

        self.psf_flux = fout[:,0]

        if self.language == 'Australian':
            self.psf_flux = (np.nanmedian(self.psf_flux) - self.psf_flux) + np.nanmedian(self.psf_flux)

        self.psf_bkg = bkgout
        
        if verbose:
            if model == 'gaussian':
                self.psf_a = aout
                self.psf_b = bout
                self.psf_c = cout
                self.psf_x = xout
                self.psf_y = yout
                self.psf_ll = llout
            if model == 'moffat':
                self.psf_a = aout
                self.psf_b = bout
                self.psf_c = cout
                self.psf_x = xout
                self.psf_y = yout
                self.psf_ll = llout
                self.psf_beta = betaout
            if nstars > 1:
                self.all_psf = fout
        return



    def custom_aperture(self, shape=None, r=0.0, h=0.0, w=0.0, theta=0.0, pos=None, method='exact'):
        """
        Creates a custom circular or rectangular aperture of arbitrary size.

        Parameters
        ----------
        shape: str, optional
            The shape of the aperture to be used. Must be either `circle` or `rectangle.`
        r: float, optional
            If shape is `circle` the radius of the circular aperture to be used.
        h: float, optional
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
                self.aperture = aperture.to_mask(method=method).to_image(shape=((
                            np.shape(self.tpf[0]))))

        elif shape == 'rectangle':
            if h==0.0 or w==0.0:
                raise Exception("For a rectangular aperture, please set both length and width: custom_aperture(shape='rectangle', h=#, w=#, theta=#)")
            else:
                aperture = RectangularAperture(pos, h=h, w=w, theta=theta)
                self.aperture = aperture.to_mask(method=method).to_image(shape=((
                            np.shape(self.tpf[0]))))

        else:
            raise ValueError("Aperture shape not recognized. Please set shape == 'circle' or 'rectangle'")


    def find_break(self):
        t   = np.diff(self.time)
        ind = np.where( t == np.max(t))[0][0]
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


    def corrected_flux(self, flux=None, skip=30, modes=3, pca=False, bkg=None):
        """
        Corrects for jitter in the light curve by quadratically regressing with centroid position.
        Parameters
        ----------
        skip: int
            The number of cadences at the start of each orbit to skip in determining optimal model weights.
        """
        self.modes = modes

        if flux is None:
            flux = self.raw_flux
            
        if bkg is None:
            bkg = self.flux_bkg
            
        if pca == True:
            flux = self.raw_flux - bkg*np.sum(self.aperture)

        flux = np.array(flux)
        
        med = np.nanmedian(flux)

        quality = self.quality

        cx = self.centroid_xs 
        cy = self.centroid_ys
        t  = self.time-self.time[0]

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

        def rotate_centroids(centroid_col, centroid_row):
            centroids = np.array([centroid_col, centroid_row])
            _, eig_vecs = np.linalg.eigh(np.cov(centroids))
            return np.dot(eig_vecs, centroids)

        def calc_corr(mask, cx, cy, skip=50):
            nonlocal quality, flux, bkg

            qm = quality[mask] == 0

            medval = np.nanmedian(flux[mask][qm])
            norm_l = norm(flux[mask], qm)

            #cx, cy = rotate_centroids(cx[mask], cy[mask])
            cx = cx[mask]
            cy = cy[mask]
            cx -= np.median(cx)
            cy -= np.median(cy)
            

            bkg_use = bkg[mask]
            bkg_use -= np.min(bkg_use)
            
            vv = self.cbvs[mask][:,0:modes]
                
        
                   
            if pca == False:
                cm = np.column_stack((t[mask][qm][skip:], np.ones_like(t[mask][qm][skip:])))
                cm_full = np.column_stack((t[mask], np.ones_like(t[mask])))
                
                if np.std(vv) > 1e-10:
                    cm = np.column_stack((cm, vv[qm][skip:]))
                    cm_full = np.column_stack((cm_full, vv))


                if np.std(bkg) > 1e-10:
                    cm = np.column_stack((cm, bkg_use[qm][skip:]))
                    cm_full = np.column_stack((cm_full, bkg_use))

                if np.std(cx) > 1e-10:
                    cm = np.column_stack((cm, cx[qm][skip:], cy[qm][skip:], cx[qm][skip:]**2, cy[qm][skip:]**2))
                    cm_full = np.column_stack((cm_full, cx, cy, cx**2, cy**2))
                
            else:
                
                cm = np.column_stack((vv[qm][skip:], np.ones_like(t[mask][qm][skip:])))
                cm_full = np.column_stack((vv, np.ones_like(t[mask])))


            x = xhat(cm, norm_l[skip:])
            fmod = fhat(x, cm_full)
            lc_pred = (fmod+1)
            return lc_pred*medval


        brk = self.find_break()
        f   = np.arange(0, brk, 1); s = np.arange(brk, len(self.time), 1)

        lc_pred = calc_corr(f, cx, cy, skip)
        corr_f = flux[f]/lc_pred * med

        lc_pred = calc_corr(s, cx, cy, skip)
        corr_s = flux[s]/lc_pred * med

        if pca==True:
            self.pca_flux = np.append(corr_f, corr_s)
        else:
            return np.append(corr_f, corr_s)



    def set_header(self):
        """Defines the header for the TPF."""
        self.header = copy.deepcopy(self.post_obj.header)
        self.header.update({'CREATED':strftime('%Y-%m-%d')})

        # Removes postcard specific header information
#        for keyword in ['POST_H', 'POST_W', 'CEN_X', 'CEN_Y', 'CEN_RA', 'CEN_DEC', 'POSTPIX1', 'POSTPIX2', 'SECTOR', 'VERSION']:
#            self.header.remove(keyword)

        # Adds TPF specific header information
        self.header.append(fits.Card(keyword='FILTER', value='TESS',
                                     comment='Filter keyword'))
        self.header.append(fits.Card(keyword='VERSION', value=eleanor.__version__,
                                     comment='eleanor version used for light curve production'))
        self.header.append(fits.Card(keyword='TIC_ID', value=self.source_info.tic,
                                     comment='TESS Input Catalog ID'))
        self.header.append(fits.Card(keyword='TMAG', value=self.source_info.tess_mag,
                                     comment='TESS mag'))
        self.header.append(fits.Card(keyword='TIC_V', value = self.source_info.tic_version,
                                     comment='TIC Version'))
        self.header.append(fits.Card(keyword='GAIA_ID', value=self.source_info.gaia,
                                     comment='Associated Gaia ID'))
        self.header.append(fits.Card(keyword='SECTOR', value=self.source_info.sector,
                                     comment='Sector'))
        self.header.append(fits.Card(keyword='CAMERA', value=self.source_info.camera,
                                     comment='Camera'))
        self.header.append(fits.Card(keyword='CCD', value=self.source_info.chip,
                                     comment='CCD'))
        self.header.append(fits.Card(keyword='CHIPPOS1', value=self.source_info.position_on_chip[0],
                                     comment='central x pixel of TPF in FFI chip'))
        self.header.append(fits.Card(keyword='CHIPPOS2', value=self.source_info.position_on_chip[1],
                                     comment='central y pixel of TPF in FFI'))
        self.header.append(fits.Card(keyword='POSTCARD', value=self.source_info.postcard,
                                     comment=''))
#        self.header.append(fits.Card(keyword='POSTPOS1', value= self.source_info.position_on_postcard[0],
#                                     comment='predicted x pixel of source on postcard'))
#        self.header.append(fits.Card(keyword='POSTPOS2', value= self.source_info.position_on_postcard[1],
#                                     comment='predicted y pixel of source on postcard'))
        self.header.append(fits.Card(keyword='CEN_RA', value = self.source_info.coords[0],
                                     comment='RA of TPF source'))
        self.header.append(fits.Card(keyword='CEN_DEC', value=self.source_info.coords[1],
                                     comment='DEC of TPF source'))
        self.header.append(fits.Card(keyword='TPF_H', value=np.shape(self.tpf[0])[0],
                                     comment='Height of the TPF in pixels'))
        self.header.append(fits.Card(keyword='TPF_W', value=np.shape(self.tpf[0])[1],
                                     comment='Width of the TPF in pixels'))
        self.header.append(fits.Card(keyword='TESSCUT', value=self.source_info.tc,
                                     comment='If TessCut was used to make this TPF'))

        if self.source_info.tc == False:
            self.header.append(fits.Card(keyword='BKG_SIZE', value=np.shape(self.bkg_tpf[0])[1],
                                         comment='Size of region used for background subtraction'))

        self.header.append(fits.Card(keyword='BKG_LVL', value=self.bkg_type,
                                     comment='Stage at which background is subtracted'))
        self.header.append(fits.Card(keyword='URL', value=self.source_info.ELEANORURL,
                                     comment='URL eleanor files are located at'))

        #if self.modes is not None:
        #    self.header.append(fits.Card(keyword='MODES', value=self.modes,
        #                                 comment='Number of modes used in PCA analysis'))


    def save(self, output_fn=None, directory=None):
        """Saves a created TPF object to a FITS file.

        Parameters
        ----------
        output_fn : str, optional
            Filename to save output as. Overrides default naming.
        directory : str, optional
            Directory to save file into.
        """
        if self.language == 'Australian':
            raise ValueError("These light curves are upside down. Please don't save them ...")
        
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
        ext1['BARYCORR']   = self.barycorr
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
        ext1['FLUX_BKG']   = self.flux_bkg
        ext1['FFIINDEX']   = self.ffiindex

        if self.bkg_type == "PC_LEVEL":
            ext1['FLUX_BKG'] = self.flux_bkg
        else:
            ext1['FLUX_BKG'] = self.flux_bkg + self.tpf_flux_bkg 
        

        if self.pca_flux is not None:
            ext1['PCA_FLUX'] = self.pca_flux
        if self.psf_flux is not None:
            ext1['PSF_FLUX'] = self.psf_flux

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

        if output_fn == None:
            path = os.path.join(directory, 'hlsp_eleanor_tess_ffi_tic{0}_s{1:02d}_tess_v{2}_lc.fits'.format(
                    self.source_info.tic, self.source_info.sector, eleanor.__version__))
        else:
            path = os.path.join(directory, output_fn)

        hdu.writeto(path, overwrite=True)




    def load(self, directory=None, fn=None):
        """
        Loads in and sets all the attributes for a pre-created TPF file.

        Parameters
        ----------
        directory : str, optional
            Directory to load file from.
        """

        if directory is None:
            directory = self.fetch_dir()
        if fn is None:
            fn = self.source_info.fn
            
        hdu = fits.open(os.path.join(directory, fn))
        hdr = hdu[0].header
        self.header = hdr
        self.bkg_type =hdr['BKG_LVL']
        # Loads in everything from the first extension
        cols  = hdu[1].columns.names
        table = hdu[1].data
        self.time        = table['TIME']
        self.tpf         = table['TPF']
        self.tpf_err     = table['TPF_ERR']
        self.raw_flux    = table['RAW_FLUX']
        self.corr_flux   = table['CORR_FLUX']
        self.flux_err    = table['FLUX_ERR']
        self.quality     = table['QUALITY']
        self.centroid_xs = table['X_CENTROID']
        self.centroid_ys = table['Y_CENTROID']
        self.x_com       = table['X_COM']
        self.y_com       = table['Y_COM']
        self.flux_bkg    = table['FLUX_BKG']
        self.ffiindex    = table['FFIINDEX']
        self.barycorr    = table['BARYCORR']

        if 'PSF_FLUX' in cols:
            self.psf_flux = table['PSF_FLUX']
        if 'PCA_FLUX' in cols:
            self.pca_flux = table['PCA_FLUX']

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
        
        names = []
        for i in cols:
            name = ('_').join(i.split('_')[0:-1])
            names.append(name)

            if i[-4::] == 'corr':
                self.all_corr_lc.append(table[i])
            elif i[-3::] == 'err':
                self.all_lc_err.append(table[i])
            else:
                self.all_raw_lc.append(table[i])

        self.aperture_names = np.unique(names)
        self.best_ind = np.where(self.aperture_names == hdr['aperture'])[0][0]


        if self.source_info.tc == False:

            if os.path.isfile(self.source_info.postcard_path) == True:
                post_fn = self.source_info.postcard_path.split('/')[-1]
                post_path = '/'.join(self.source_info.postcard_path.split('/')[0:-1])
                self.post_obj = Postcard(filename=post_fn, ELEANORURL=self.source_info.ELEANORURL,
                                         location=post_path)
        else:
            self.post_obj =Postcard_tesscut(self.source_info.cutout,
                                            location=self.source_info.postcard_path)

                
        self.get_cbvs()

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
        download_dir    = os.path.join(os.path.expanduser('~'), '.eleanor')
        if os.path.isdir(download_dir):
            return download_dir
        else:
            # if it doesn't exist, make a new cache directory
            try:
                os.mkdir(download_dir)
            # downloads locally if OS error occurs
            except OSError:
                download_dir = '.'
                warnings.warn('Warning: unable to create {}. '
                              'Downloading TPFs to the current '
                              'working directory instead.'.format(download_dir))

        return download_dir
