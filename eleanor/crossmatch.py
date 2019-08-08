import numpy as np
import urllib
import os
import socket
import pandas as pd
from astroquery.mast import Observations
from astropy.io import fits

from .utils import *

__all__ = ['Crossmatch']

class Crossmatch(object):
    """
    This class can be used to find the same light curve from different
    pipelines. The current available light curves are from the 
    TESS Asteroseismic Science Consortium (TASC) and Oelkers & Stassun (2019).
    Oelkers & Stassun light curves are only available through Sector 5.

    Parameters
    ----------
    obj :
        Object must be an eleanor.TargetData object.
    """

    def __init__(self, object):
        self.sector = object.source_info.sector
        self.camera = object.source_info.camera
        self.chip   = object.source_info.chip
        self.tic    = object.source_info.tic
        self.download_dir = os.path.join(os.path.expanduser('~'), '.eleanor')


    def tasoc_lc(self):
        """
        Grabs the T'DA available light curves for your target.
        For more information, see the TASOC light curve documentation: https://tasoc.dk/code/.

        Parameters
        ----------

        Attributes
        ----------
        tasoc_header : 
        tasoc_tpf : np.2darray
        tasoc_aperture : np.2darray
        tasoc_time : np.array
        tasoc_quality : np.array
        tasoc_timecorr : np.array
        tasoc_cadenceno : np.array
        tasoc_flux_raw : np.array
        tasoc_flux_raw_err : np.array
        tasoc_flux_corr : np.array
        tasoc_flux_corr_err : np.array
        tasoc_flux_bkg : np.array
        tasoc_pixel_quality : np.array
             Quality flags for the data; use these not `tasoc_quality`.
        tasoc_pos_corr1 : np.array
        tasoc_pos_corr2 : np.array
        tasoc_mom_centr1 : np.array
        tasoc_mom_centr2 : np.array
        """
        products = Observations.query_object(objectname="TIC"+str(self.tic))

        column = np.where( (products['provenance_name'] == 'TASOC') &
                           (products['target_name'] == str(self.tic)) & 
                           (products['sequence_number'] == self.sector) )[0]

        if len(column) > 0:
            download = Observations.get_product_list(products[column])
            manifest = Observations.download_products(download, download_dir=self.download_dir)
            self.tasoc_path = manifest["Local Path"].data[0]

            hdu = fits.open(self.tasoc_path)

            self.tasoc_header   = hdu[0].header
            self.tasoc_tpf      = hdu[2].data
            self.tasoc_aperture = hdu[3].data
            self.tasoc_time      = hdu[1].data['TIME']
            self.tasoc_quality   = hdu[1].data['QUALITY']
            self.tasoc_timecorr  = hdu[1].data['TIMECORR']
            self.tasoc_cadenceno = hdu[1].data['CADENCENO']
            self.tasoc_flux_raw  = hdu[1].data['FLUX_RAW']
            self.tasoc_flux_bkg  = hdu[1].data['FLUX_BKG']
            self.tasoc_flux_corr = hdu[1].data['FLUX_CORR']
            self.tasoc_pos_corr1 = hdu[1].data['POS_CORR1']
            self.tasoc_pos_corr2 = hdu[1].data['POS_CORR2']
            self.tasoc_mom_centr1 = hdu[1].data['MOM_CENTR1']
            self.tasoc_mom_centr2 = hdu[1].data['MOM_CENTR2']
            self.tasoc_pixel_quality = hdu[1].data['PIXEL_QUALITY']
            self.tasoc_flux_raw_err  = hdu[1].data['FLUX_RAW_ERR']
            self.tasoc_flux_corr_err = hdu[1].data['FLUX_CORR_ERR']

        else:
            raise SearchError("No TASOC light curve found.")



    def oelkers_lc(self):
        """
        Grabs the Oelkers & Stassun (2019) associated light curve.

        Parameters
        ----------
        
        Attributes
        ----------
        time : np.array
        mag  : np.array
        mag_err : np.array

        """
        fn = "{1}_sector{0:02d}_{2}_{3}.lc".format(self.sector, self.tic,
                                                   self.camera, self.chip)
        oelkers_url = "http://astro.phy.vanderbilt.edu/~oelkerrj/tess_ffi/sector{0:02d}/clean/{1}".format(self.sector,
                                                                                                          fn)
        
        dne = True # Does Not Exist
        try:
            urllib.request.urlopen(oelkers_url, timeout=3)
            dne = False
        except socket.timeout:
            print("There is no Oelkers & Stassun light curve found.")
            return

        if dne is False:
            tab = pd.read_csv(oelkers_url, delimiter=' ', header=None)
            self.os_time    = tab[0].values
            self.os_mag     = tab[1].values
            self.os_mag_err = tab[2].values

