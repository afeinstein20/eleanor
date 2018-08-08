import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import sys, os, math
from gaiaTIC import ticSearchByContam as tsbc
from photutils import Background2D, MedianBackground
from pixelCorrection import sortByDate
from astropy.stats import SigmaClip
from scipy.interpolate import BSpline, splrep, splev
from scipy.optimize import minimize
from photutils import CircularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
from scipy import signal


def parabola(params, x, y, f_obs):
    c1, c2, c3, c4, c5 = params
    f_corr = f_obs * (c1 + c2*(x-2.5) + c3*(x-2.5)**2 + c4*(y-2.5) + c5*(y-2.5)**2)
    print(np.sum((f_obs-f_corr)**2))
    return np.sum((f_obs-f_corr)**2)


def lightcurve(camera, chip):
    file = './198593129_tpf.fits'
    
    theta, delx, dely = np.loadtxt('pointingModel_{}-{}.txt'.format(camera, chip), skiprows=1, 
                                   usecols=(1,2,3), unpack=True)
    tpf  = ktpf.from_fits(file)
    hdu = fits.open(file)

    cen_x, cen_y = len(tpf.flux[0])/2., len(tpf.flux[0])/2.

    # Creates estimated center location taking the pointing model into account
    x_point = cen_x * np.cos(np.radians(theta)) - cen_y * np.sin(np.radians(theta)) - delx
    y_point = cen_x * np.sin(np.radians(theta)) + cen_y * np.cos(np.radians(theta)) - dely

    # Creates the light curve given different center locations per cadence
    lc = []
    for f in range(len(tpf.flux)):
        pos = [x_point[f], y_point[f]]
        ap = CircularAperture(pos, 1.5)
        lc.append(aperture_photometry(tpf.flux[f], ap)['aperture_sum'].data[0])
    lc = np.array(lc / np.nanmedian(lc))
    plt.plot(np.arange(0,len(lc),1),lc,'k')

    c1, c2, c3, c4, c5 = 1.0, 3.4, 1.3, 2.6, 1.7
    lc = lc / ( c1 + c2*(x_point-2.5) + c3*(x_point-2.5)**2 +
                c4*(y_point-2.5) + c5*(y_point-2.5)**2 )
    plt.plot(np.arange(0,len(lc),1),lc,'r')
    plt.show()

    # Masks out anything >= 3 sigma above the mean
    for i in range(5):
        lc_new = []
        inds = np.where(lc <= np.mean(lc)-3*np.std(lc))
        for j in inds:
            lc[j] = 1.0
        
        initGuess = [10, 10, 10, 10, 10]
        test = minimize(parabola, initGuess, args=(x_point, y_point, lc))
        print(test.x, test.success)
        c1, c2, c3, c4, c5 = test.x
        lc_new = lc * (c1 + c2*(x_point-2.5) + c3*(x_point-2.5)**2 + c4*(y_point-2.5) + c5*(y_point-2.5)**2)
        
#        plt.plot(np.arange(0,len(lc),1),lc, 'k', linewidth=3)
#        plt.plot(np.arange(0,len(lc_new),1), lc_new, 'r-')
#        plt.show()
#        plt.close()


    

lightcurve(3, 3)
