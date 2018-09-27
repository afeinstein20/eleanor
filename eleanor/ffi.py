import numpy as np
from astropy.io import fits


def use_pointing_model(coords, pointing_model):
    """ Calculates the true position of a star/many stars given the predicted pixel location and pointing model"""
    """ pointing_model is passed in as an astropy.table.Table row and reshaped into a (3x3) matrix """
    """ pointing_model is for ONE cadence at a time """
    """ Coords given in (x,y) """
    pointing_model = np.reshape(list(pointing_model), (3,3))
    A = np.column_stack([coords[0], coords[1], np.ones_like(coords[0])])
    fhat = np.dot(A, pointing_model)
    return fhat[:,0:2]


class ffi:
    """
    This class allows the user to download all full-frame images for a given sector,
         camera, and chip. It also allows the user to create their own pointing model
         based on each cadence for a given combination of sector, camera, and chip.
    No individual user should have to download all of the full-frame images because 
         stacked postcards will be available for the user to download from MAST.
    """
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


    def build_pointing_model(self, pos_predicted, pos_inferred, outlier_removal=False):
        """Builds an affine transformation to correct the positions of stars from a possibly incorrect WCS"""
        """ pos_predicted are positions taken straight from the WCS """
        """ pos_inferred are positions taken using any centroiding method """
        """ [[x,y],[x,y],...] format """
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


    def pointing_model_per_cadence(self):
        """ Step through build_pointing_model for each cadence """
        from muchbettermoments import quadratic_2d
        from mast import tic_by_contamination
        from astropy.wcs import WCS
        from astropy.nddata import Cutout2D

        def find_isolated(x, y):
            """ Finds the most isolated, least contaminated sources for pointing model """
            isolated = []
            for i in range(len(x)):
                x_list  = np.delete(x, np.where(x==x[i]))
                y_list  = np.delete(y, np.where(y==y[i]))
                closest = np.sqrt( (x[i]-x_list)**2 + (y[i]-y_list)**2 ).argmin()
                dist    = np.sqrt( (x[i]-x_list[closest])**2+ (y[i]-y_list[closest])**2 )
                if dist > 8.0:
                    isolated.append(i)
            return np.array(isolated)


        def isolated_center(x, y, image):
            """ Finds the center of each isolated TPF with quadratic_2d """
            cenx, ceny, good = [], [], []
            for i in range(len(x)):
                 if x[i] > 0. and y[i] > 0.:
                     tpf = Cutout2D(image, position=(x[i], y[i]), size=(7,7), mode='partial')
                     cen = quadratic_2d(tpf.data)
                     cenx.append(cen[0]); ceny.append(cen[1])
                     good.append(i)
            cenx, ceny = np.array(cenx), np.array(ceny)
            return cenx, ceny, good

        pm_fn = 'pointingModel_{}_{}-{}.txt'.format(self.sector, self.camera, self.chip)
        with open(pm_fn, 'w') as tf:
            tf.write('')

        hdu = fits.open(self.local_paths[0])
        hdr = hdu[1].header
        pos = [hdr['CRVAL1'], hdr['CRVAL2']]

        r = 6.0*np.sqrt(1.2)
        contam = [0.0, 5e-3]
        tmag_lim = 12.5

        t  = tic_by_contamination(pos, r, contam, tmag_lim)

        for fn in self.local_paths:
            hdu = fits.open(fn)
            hdr = hdu[1].header
            xy = WCS(hdr).all_world2pix(t['ra'], t['dec'], 1)
            # Triple checks the sources are on the FFI
            onFrame = np.where( (xy[0]>0) & (xy[0]<2092) & (xy[1]>0) & (xy[1]<2048) )[0]
            xy  = np.array([xy[0][onFrame], xy[1][onFrame]])
            iso = find_isolated(xy[0], xy[1])
            xy  = np.array([xy[0][iso], xy[1][iso]])
            pos_predicted = np.reshape(xy, (len(xy[0]),2) )

            cenx, ceny, good = isolated_center(xy[0], xy[1], hdu[1].data)

            pos_inferred = np.empty( (len(xy[0]),2) )
            for i in range(len(pos_inferred)):
                if cenx[i] > 3.5:
                    newX = pos_predicted[i][0]+cenx[i]
                else:
                    newX = pos_predicted[i][0]-cenx[i]
                if ceny[i] > 3.5:
                    newY = pos_predicted[i][1]+ceny[i]
                else:
                    newY = pos_predicted[i][1]-ceny[i]
                pos_inferred[i][0] = newX
                pos_inferred[i][1] = newY
            solution = self.build_pointing_model(pos_predicted, pos_inferred)
            solution = np.reshape(solution, (9,) )
            with open(pm_fn, 'a') as tf:
                tf.write('{}\n'.format(' '.join(str(e) for e in solution) ) )
        return
    
    
