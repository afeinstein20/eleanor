import numpy as np
from astropy.io import fits

class ffi:

    def __init__(self, sector=None, camera=None, chip=None):
        self.sector = sector
        self.camera = camera
        self.chip   = chip


    def download_ffis(self):
        """ Downloads entire sector of data into .ellie/ffis/sector directory """
        from astropy.utils.data import download_file
        import requests
        from bs4 import BeautifulSoup

        def findAllFFIs(ca, ch):
            nonlocal year, days, url
            calFiles = []
            for d in days:
                path = '/'.join(str(e) for e in [url, year, d, ca])+'-'+str(ch)+'/'
                for fn in BeautifulSoup(requests.get(path).text, "lxml").find_all('a'):
                    if fn.get('href')[-7::] == 'ic.fits':
                        calFiles.append(path+fn.get('href'))
            return calFiles

        if self.sector in np.arange(1,14,1):
            year=2019
        else:
            year=2020
        # Current days available for ETE-6
        days = np.arange(129,130,1)

        # This URL applies to ETE-6 simulated data ONLY
        url = 'https://archive.stsci.edu/missions/tess/ete-6/ffi/'
        files = findAllFFIs(self.camera, self.chip)

        local_paths = []
        for f in files:
            path = download_file(f, cache=True)
            local_paths.append(path)
        self.local_paths = np.array(local_paths)
        return


    def sort_by_date(self):
        """ Sorts FITS files by start date of observation """
        dates, time = [], []
        for f in self.local_paths:
            hdu = fits.open(f)
            hdr = hdu[1].header
            dates.append(hdr['DATE-OBS'])
        dates, fns = np.sort(np.array([dates, self.local_paths]))
        self.local_paths = fns
        self.dates = dates
        return
    

    def build_pointing_model(pos_predicted, pos_inferred, outlier_removal = False):
        """Builds an affine transformation to correct the positions of stars from a possibly incorrect WCS"""
        A = np.column_stack([pos_predicted[:,0], pos_predicted[:,1], np.ones_like(pos_predicted[:,0])])
        f = np.column_stack([pos_inferred[:,0], pos_inferred[:,1], np.ones_like(pos_inferred[:,0])])
        
        if outlier_removal == True:
            dist = np.sqrt(np.sum((A - f)**2, axis=1))
            mean, std = np.mean(dist), np.std(dist)
            A = A[dist_orig < mean + 3*std]
            f = f[dist_orig < mean + 3*std]
        
        ATA = np.dot(A.T, A)
        ATAinv = np.linalg.inv(ATA)
        ATf = np.dot(A.T, f)
        xhat = np.dot(ATAinv, ATf)
        fhat = np.dot(A, xhat)
        
        return xhat

    
    def use_pointing_model(coords, pointing_model):
        """Calculates the true position of a star/many stars given the predicted pixel location and pointing model"""
        A = np.column_stack([coords[:,0], coords[:,1], np.ones_like(coords[:,0])])
        fhat = np.dot(A, pointing_model)
        return fhat[:,0:2]

        

