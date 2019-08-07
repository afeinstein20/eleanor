import numpy as np
import urllib, os
import pandas as pd

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


    def tasoc_lc(self, sectors=None):
        """
        Grabs the T'DA available light curves for your target.
        
        Parameters
        ----------
        sectors : list, optional
            Input the sectors you want to grab for a given light curve.
            Default is the assigned sector in your eleanor.TargetData object.
        """
        return


    def oelkers_ls(self, sectors=None):
        """
        Grabs the Oelkers & Stassun (2019) associated light curve.

        Parameters
        ----------

        """
        fn = "{1}_sector{0:02d}_{2}_{3}.lc".format(self.sector, self.tic,
                                                   self.camera, self.chip)
        oelkers_url = "http://astro.phy.vanderbilt.edu/~oelkerrj/tess_ffi/sector{0:02d}/clean/{1}".format(self.sector,
                                                                                                          fn)
        print(fn)
        print(oelkers_url)
        try:
            tab = pd.read_csv(oelkers_url, delimiter=' ', header=None)
            self.os_time    = tab[0].values
            self.os_mag     = tab[1].values
            self.os_mag_err = tab[2].values
        except:
            raise ValueError("There is no Oelkers & Stassun light curve for this target/sector.")
