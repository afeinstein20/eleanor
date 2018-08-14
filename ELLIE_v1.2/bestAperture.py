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


""" Finds the lightcurve with the lease noise """
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

    sigma = np.zeros( (len(r_list), 2) )
    for i in range(len(r_list)):
        # Stitches together lightcurve for circular aperture
        lc_circ = matrix[i][0] / np.nanmedian(matrix[i][0])
        print(lc_circ)
        lc_circ = jitter_correction(lc_circ, x, y)
        lc_circ = roll_correction(lc_circ, x, y)
                                              
        # Stitches together lightcurve for rectangular aperture
        lc_rect = matrix[i][1] / np.nanmedian(matrix[i][1])
        lc_rect = jitter_correction(lc_rect, x, y)
        lc_rect = roll_correction(lc_rect, x, y)

        # Finds std of each lightcurve
        sigma[i][0] = np.std(lc_circ)
        sigma[i][1] = np.std(lc_rect)
    
    # Finds lowest sigma -- post systematics correction
    best = np.where(sigma==np.min(sigma))
    r_ind, shape_ind = best[0][0], best[1][0]
    return matrix[r_ind,shape_ind] / np.nanmedian(matrix[r_ind,shape_ind])



""" Finds offset between center of TPF and centroid of first cadence """
def centroidOffset(tpf, theta, delx, dely):
    file_cen = len(tpf.flux[0])/2.
    x_init = int(np.round(file_cen*np.cos(theta) - file_cen*np.sin(theta) - delx, 0))
    y_init = int(np.round(file_cen*np.sin(theta) + file_cen*np.cos(theta) - dely, 0))

    # Finds centroid of 2x2 region around the center of FITS file (which is where the star *should* be)
    tpf_init = tpf.flux[0]
    tpf_com  = tpf_init[x_init-2:x_init+3, y_init-2:y_init+3]
    com = ndimage.measurements.center_of_mass(tpf_com.T - np.median(tpf_com))

    startX = file_cen - com[0]
    startY = file_cen - com[1]
    return startX, startY


""" Temporary Main Function """
def main(id, camera, chip):
    file = './figures/{}_tpf.fits'.format(id)
    tpf  = ktpf.from_fits(file)

    pointing = 'pointingModel_{}-{}.txt'.format(camera, chip)
    theta, delX, delY = np.loadtxt(pointing, skiprows=1, usecols=(1,2,3), unpack=True)
    
    startX, startY = centroidOffset(tpf, theta[0], delX[0], delY[0])
    x, y = [startX], [startY]

    for i in range(len(theta)-1):
        if i == 0:
            x.append( startX*np.cos(theta[i+1]) - startY*np.sin(theta[i+1]) + delX[i+1] )
            y.append( startX*np.sin(theta[i+1]) + startY*np.cos(theta[i+1]) + delY[i+1] )
        else:
            x.append( x[i-1]*np.cos(theta[i+1]) - y[i-1]*np.sin(theta[i+1]) + delX[i+1] )
            y.append( x[i-1]*np.sin(theta[i+1]) + y[i-1]*np.cos(theta[i+1]) + delY[i+1] )

    ideal = findLC(tpf, x, y)
    plt.plot(np.arange(0,len(tpf.flux),1), ideal)
    plt.show()


main(198593129, 3, 3)
