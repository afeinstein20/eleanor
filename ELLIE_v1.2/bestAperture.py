import os, sys
import numpy as np
import matplotlib.pyplot as plt
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
from removeSystematics import jitter_correction, roll_correction
from scipy import ndimage

""" Creates the Apertures """
def aperture(r, pos):
    circ = CircularAperture(pos, r)
    rect = RectangularAperture(pos, r, r, 0.0)
    return circ, rect


""" Finds the lightcurve with the least noise """
def findLC(tpf, x, y):

    r_list = np.arange(1.5, 3.5, 0.5)
    matrix = np.zeros( (len(r_list), 2, len(tpf.flux)) )
    
    for i in range(len(r_list)):
        for j in range(len(tpf.flux)):
            pos = (x[j], y[j])
            circ, rect = aperture(r_list[i], pos)
            # Completes aperture sums for each tpf.flux and each aperture shape
            matrix[i][0][j] = aperture_photometry(tpf.flux[j], circ)['aperture_sum'].data[0]
            matrix[i][1][j] = aperture_photometry(tpf.flux[j], rect)['aperture_sum'].data[0]
        matrix[i][0] = matrix[i][0] / np.nanmedian(matrix[i][0])
        matrix[i][1] = matrix[i][1] / np.nanmedian(matrix[i][1])

    time  = np.arange(0,len(x), 1)
    sigma = np.zeros( (len(r_list),2) )
    for i in range(len(r_list)):
        # Stitches together lightcurve for circular aperture
        lc_circ = jitter_correction(matrix[i][0], x, y)
        lc_circ = roll_correction(lc_circ, x, y)
        plt.plot(time, lc_circ, label='Circ of {}'.format(r_list[i]))
        # Stitches together lightcurve for rectangular aperture
        lc_rect = jitter_correction(matrix[i][1], x, y)
        lc_rect = roll_correction(lc_rect, x, y)
        plt.plot(time, lc_rect, label='Rect of {}'.format(r_list[i]))

        # Finds std of each lightcurve
        sigma[i][0] = np.std(lc_circ)
        sigma[i][1] = np.std(lc_rect)
        
#    plt.legend()
#    plt.show()

    best = np.where(sigma==np.min(sigma))
    r_ind, s_ind = best[0][0], best[1][0]

    return r_list[r_ind], s_ind


""" Finds offset between center of TPF and centroid of first cadence """
def centroidOffset(tpf, theta, delx, dely):
    file_cen = len(tpf.flux[0])/2.
    # Finds centroid of 2x2 region around the center of FITS file (which is where the star *should* be)
    tpf_init = tpf.flux[0]
    tpf_com  = tpf_init[2:6, 2:6]
    com = ndimage.measurements.center_of_mass(tpf_com.T - np.median(tpf_com))
    return com[0]-len(tpf_com)/2., com[1]-len(tpf_com[0])/2.


""" Temporary Main Function """
def main(id, camera, chip):
    file = './figures/{}_tpf.fits'.format(id)
    tpf  = ktpf.from_fits(file)
    file_cen = len(tpf.flux[0])/2.

    pointing = 'pointingModel_{}-{}.txt'.format(camera, chip)
    theta, delX, delY = np.loadtxt(pointing, skiprows=2, usecols=(1,2,3), unpack=True)
    initParams = np.loadtxt(pointing, skiprows=1, usecols=(1,2,3))[0]

    startX, startY = centroidOffset(tpf, initParams[0], initParams[1], initParams[2])
    startX, startY = file_cen+startX, file_cen+startY

    plt.imshow(tpf.flux[0], origin='lower')#, vmin=40, vmax=200)
    plt.plot(startX, startY, 'ko')
    plt.show()
    plt.close()

    x, y = [startX], [startY]
    for i in range(len(theta)):
        if i == 0:
            x.append( startX*np.cos(theta[i]) - startY*np.sin(theta[i]) - delX[i] )
            y.append( startX*np.sin(theta[i]) + startY*np.cos(theta[i]) - delY[i] )
        else:
            x.append( x[i-1]*np.cos(theta[i]) - y[i-1]*np.sin(theta[i]) - delX[i] )
            y.append( x[i-1]*np.sin(theta[i]) + y[i-1]*np.cos(theta[i]) - delY[i] )

    radius, shape = findLC(tpf, np.array(x), np.array(y))
"""
    lc = []
    for i in range(len(x)):
        pos = (x[i], y[i])
        circ, rect = aperture(radius, (startX, startY))
        if shape == 0:
            ap = circ
        else:
            ap = rect

        flux = aperture_photometry(tpf.flux[i], ap)['aperture_sum'].data[0]
        lc.append(flux)
    lc = lc / np.nanmedian(lc)
    plt.plot(np.arange(0,len(x),1), lc)
    plt.show()
"""
main(198593129, 3, 3)
