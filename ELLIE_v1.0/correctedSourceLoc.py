import numpy as np
import matplotlib.pyplot as plt
#from locateSources import overplotSources
from world2pix import openFITS, getPixelCoords
from astropy.convolution import Gaussian2DKernel
from photutils import Background2D, MedianBackground, detect_threshold, source_properties, detect_sources
from astropy.stats import gaussian_fwhm_to_sigma
from astropy import units as u
from getFiles import passInfo
"""
# --------------------------
# Creates segmentation map of sources
# --------------------------
def segMap(fitsData):
    # Identifies the background of the image
    bkg_est = MedianBackground()
    bkg = Background2D(fitsData, (50,50), filter_size = (3,3), bkg_estimator = bkg_est)

    # Identifies sources that are 2 sigma above the background
    threshold = bkg.background + (2. * bkg.background_rms)
    sigma = 2. * gaussian_fwhm_to_sigma

    # Creates segmentation map of sources with >= 2 pixel size at >= 2 sigma above background
    segm = detect_sources(fitsData, threshold, npixels = 4)
    print("Finished segmentation map")
    return segm
"""
# --------------------------
#   Plots segmentation map
# --------------------------
def plotSegMap(segm, x, y, cat_table):
    fig, ax = plt.subplots()
    ax.imshow(segm.data, origin = 'lower', interpolation = 'nearest', cmap = 'gist_heat_r')
    ax.plot(x, y, 'go', ms = 3)
    ax.plot(cat_table['xcentroid'], cat_table['ycentroid'], 'kx', ms = 3)
    return ax
"""
# --------------------------
#   Finds nearest source
# --------------------------
def nearest(x, x_list, y, y_list):
    # Removes itself from the list
    x_list = np.delete(x_list, np.where(x_list == x))
    y_list = np.delete(y_list, np.where(y_list == y))
    # Calculates distance w/ everything in list
    dist = np.sqrt( (x-x_list)**2 + (y-y_list)**2 )
    # Gets index of closest point
    closest = dist.argmin()

    # Returns distance to closest point and index of closest point
    return np.sqrt( (x-x_list[closest])**2 + (y-y_list[closest])**2 ), closest

# --------------------------
# Creates list of isolate sources & closest segment
# --------------------------
def isolated(x, y, cenX, cenY):
    isoSource, isoSeg = [], []
    for i in range(len(x)):
        cdist, cdist_arg = nearest(x[i], x, y[i], y)
        # Checks to see if the nearest source is >= 4.0 pixels away
        if cdist >= 4.0:
            cseg, cseg_arg = nearest(x[i], cenX, y[i], cenY)
            # Checks to see if the nearest centroid <= 2.0 pixels away
            if cseg <= 2.0:
                isoSource.append(i)
                isoSeg.append(cseg_arg)

    return isoSource, isoSeg

# -------------------------- 
# Writes mean, median, & weighted average
# correction factors to file
# -------------------------- 
def corrFile(deltaX, deltaY, year, day, cadence, camera, chip):
    meanX, meanY     = np.mean(deltaX)   , np.mean(deltaY)
    medX , medY      = np.median(deltaX) , np.median(deltaY)
    weightX, weightY = np.average(deltaX), np.average(deltaY)

    rowList = [cadence, camera, chip, meanX, meanY, medX, medY, weightX, weightY]

    file = 'corrFactors_{}_{}-{}-{}.txt'.format(year, day, camera, chip)
    with open(file, 'a') as tf:
        tf.write('{}\n'.format(' '.join(str(e) for e in rowList)))
    return
"""
# --------------------------
# Corrects source position using segmentation map
# and associated centroids
# --------------------------
def correctXY():
#    year, daynum, filenames, dir, camera, chip = passInfo()
#    filenames = np.sort(filenames)

    deltaX, deltaY = np.array([]), np.array([])
    cad, oldCad = 0, 0
    
    for fn in filenames:
        print(fn)
        cad = fn[11:17]
        cenX, cenY, sourceIndex, segIndex = [],[],[],[]

        id, ra, dec, x, y, gmag, mast, mheader = getPixelCoords(fn, dir)

        print("Making segmentation map")

        segm = segMap(mast)
        cat = source_properties(mast, segm).to_table()
        print("Obtaining centroid positions")
        cenX, cenY = cat['xcentroid'].value, cat['ycentroid'].value # Removes units
        print("Identifying isolated sources for pixel location correction")
        sourceIndex, segIndex = isolated(x, y, cenX, cenY)
        deltaX = np.append(deltaX, np.abs(x[sourceIndex]-cenX[segIndex]))
        deltaY = np.append(deltaY, np.abs(y[sourceIndex]-cenY[segIndex]))

        corrFile(np.array(deltaX), np.array(deltaY), year, daynum, cad, camera, chip)
        deltaX, deltaY = np.array([]), np.array([])
