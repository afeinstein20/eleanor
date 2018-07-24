import sys, os
import numpy as np
import matplotlib.pyplot as plt
from filesAndConvert import openFITS, radec2pixel, download, passInfo
from astropy.convolution import Gaussian2DKernel
from photutils import Background2D, MedianBackground, detect_threshold, source_properties, detect_sources
from astropy.stats import gaussian_fwhm_to_sigma
from lightkurve import KeplerTargetPixelFile as ktpf
from scipy import ndimage as ni
import collections

# --------------------------
# Sorts FITS files by start date of observation
#  Returns: ordered filenames
# --------------------------
def sortByDate(fns, dir):
    dir_fns = [dir + i for i in fns]
    dates = []
    for f in dir_fns:
        mast, header = openFITS(f)
        dates.append(header['DATE-OBS'])
    dates, fns = np.sort(np.array([dates, fns]))
    return fns

# -------------------------- 
# Creates segmentation map of sources
#  Returns: Segmentation map
# -------------------------- 
def makeSegMap(mast):
    # Identifies image background
    bkg_est = MedianBackground()
    bkg     = Background2D(mast, (50,50), filter_size = (3,3), bkg_estimator = bkg_est)
    # Identifies sources >= 2 sigma above the backgroun
    threshold = bkg.background + (2. * bkg.background_rms)
    sigma     = 2. * gaussian_fwhm_to_sigma
    # Creates segmentation map for sources >= 2 sigma & >= 4 pixels
    segm = detect_sources(mast, threshold, npixels = 3)
    return segm
    

# -------------------------- 
# Identifies the nearest source
# Returns: distance to nearest source, index of nearest source
# --------------------------
def nearest(x, y, x_list, y_list):
    # Removes itself from the list
    x_list = np.delete(x_list, np.where(x_list == x))
    y_list = np.delete(y_list, np.where(y_list == y))
    # Finds index of nearest object
    closest = np.sqrt( (x-x_list)**2 + (y-y_list)**2 ).argmin()
    # Returns distance to closest point and index of closest point
    return np.sqrt( (x-x_list[closest])**2 + (y-y_list[closest])**2 ), closest

# --------------------------
# Identifies isolated sources
# Returns: indices of isolated sources, 
#          indices of isolated segmentation map sources
# --------------------------
def findLoners(x, y, segX, segY):
    isoSource= []
    for i in range(len(x)):
        cdist, cdist_arg = nearest(x[i], y[i], x, y)
        sdist, sdist_arg = nearest(x[i], y[i], segX, segY)
        if cdist > 5.0 and sdist <= 2.0:
            isoSource.append(cdist_arg)
    return isoSource

# --------------------------
# Writes mean & median correction factors to file
# --------------------------
def correctionFile(deltaX, deltaY, deltaT, file, cadence):
    meanX, meanY, meanT = np.mean(deltaX), np.mean(deltaY), np.mean(deltaT)
    medX , medY , medT  = np.median(deltaX), np.median(deltaY), np.median(deltaT)

    row  = [cadence, meanX, meanY, meanT, medX, medY, medT]
    with open(file, 'a') as tf:
        tf.write('{}\n'.format(' '.join(str(e) for e in row)))
    return

# --------------------------
# Creates TPF for isolated sources
# in order to calculate shift between cadences
# --------------------------
def calcShift(fns, x, y, camera, chip, corrFile):
    matrix = np.zeros((len(fns),len(x),2))

    for i in range(len(x)):
        tpf = ktpf.from_fits_images(images=fns, position=(x[i],y[i]), size=(5,5))
        for f in range(len(tpf.flux)):
            com = ni.measurements.center_of_mass(tpf.flux[f].T-np.median(tpf.flux[f]))
            if f == 0:
                matrix[f][i][0] = com[0] - 2.5
                matrix[f][i][1] = com[1] - 2.5
            else:
                matrix[f][i][0] = com[0] - 2.5#matrix[f-1][i][0]
                matrix[f][i][1] = com[1] - 2.5#matrix[f-1][i][1]
    for i in range(len(fns)):
        delX = np.mean(matrix[i][:,0])
        delY = np.mean(matrix[i][:,1])
        with open(corrFile, 'a') as tf:
            tf.write('{}\n'.format(str(i) + ' ' + str(delX) + ' ' + str(delY)))
        
    return

# --------------------------
# Corrects source position using centroids of segmentation map
# --------------------------
def correctionFactors():
    year, camera, chip, filenames, dir = passInfo()

    # Gets only FITS files from directory & sorts by date
    filenames = filenames[np.array([i for i,item in enumerate(filenames) if "fits" in item])]
    filenames = sortByDate(filenames, dir)

    x, y, g   = radec2pixel(dir+filenames[0])
    statMast, statHeader  = openFITS(dir+filenames[0])
    print("Obtained x, y, and Gmag successfully!")

    corrFile = '{}corrFactors_{}_{}-{}.txt'.format(dir, year, camera, chip)
    sourcesI = []
    if os.path.isfile(corrFile) == False:
        for i in range(2):
            cadence = filenames[i][11:17]
            mast, mheader = openFITS(dir+filenames[i])
            segm = makeSegMap(mast)
            print("Making sementation map catalog")
            cat  = source_properties(mast, segm).to_table()
            print("Getting segmentation map centroids")
            segCenX, segCenY = cat['xcentroid'].value, cat['ycentroid'].value  # Removies units
# only have to find centroids once to make sure it works then get rid of it
#            segCenX, segCenY = np.loadtxt('centroids.txt', unpack = True)
            print("Finding isolated sources")
            temp = findLoners(x, y, segCenX, segCenY)
            print temp
            [sourcesI.append(i) for i in temp]
        u = [item for item, count in collections.Counter(sourcesI).items() if count > 1]
        filePaths = [dir+i for i in filenames]
        calcShift(filePaths, x[u], y[u], camera, chip, corrFile)
    filePaths = [dir+i for i in filenames]
    corrections = np.loadtxt(corrFile)
    unique = [55]#, 133, 1031]
    return x[1031]+corrections[:,1], y[1031]+corrections[:,2], filePaths
"""    for i in unique:
        posX = x[i]+corrections[:,1]
        posY = y[i]+corrections[:,2]
        position = [(m,n) for m,n in zip(posX, posY)]
        tpf = ktpf.from_fits_images(images=filePaths, position=position, size=(5,5))
        tpf.to_fits(output_fn = '{}.fits'.format(str(i)))
        plt.close()

correctionFactors()
"""
