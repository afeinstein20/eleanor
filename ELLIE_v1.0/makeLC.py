import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from world2pix import getPixelCoords, openFITS
from astropy.io import fits
from lightkurve import KeplerTargetPixelFile as ktpf
from lightkurve import TessTargetPixelFile as ttpf
from photutils import CircularAperture, aperture_photometry, RectangularAperture, source_properties
from correctedSourceLoc import nearest, segMap, plotSegMap
from getFiles import passInfo
from numpy import linalg as LA
import matplotlib as mpl
import matplotlib.animation as animation
from astropy.wcs import WCS


# --------------------------
# Identify and plot isolated sources
# --------------------------
def isolatedFITS(corrX, corrY, mast):
    isolated = []
    for i in range(len(corrX)):
        dist, closest = nearest(corrX[i], corrX, corrY[i], corrY)
        if dist >= 6.0:
            isolated.append(i)

    plt.imshow(mast, origin = 'lower', interpolation = 'nearest', cmap = 'Greys', vmin = 50, vmax = 90)
    circle((corrX[isolated],corrY[isolated]), 2.5).plot(color = 'red', fill = 0.2, alpha = 0.3)
#    plt.show()
    plt.close()
    return isolated

# --------------------------  
# TPF & LC figures for circle & square apertures
# -------------------------- 
def testingApers(x, y, xMean, yMean, iso, fns, id):
    moreX, moreY = [], []
    for i in iso:
#        fig, ((ax1,ax2), (ax3,ax4), (ax5, ax6)) = plt.subplots(nrows = 3, ncols = 2, figsize = (16,8))
        center = (x[i],y[i])
        size = (5,5)

        tpf = ktpf.from_fits_images(images = fns, position = center, size = size, target_id = id[i])
        cenX, cenY = tpfCentroid(tpf)
        moreX.append(cenX)
        moreY.append(cenY)

#        plotTPF(ax1, ax2, 0., 0., tpf, 'k', 0.)

        plotApCen = (583.5, 92.5)
        plotSoCen = (cenX, cenY)
         
        lengths = [2.0, 1.5]
        colors  = ['red', 'yellow', 'green'] 

    return moreX, moreY


# --------------------------
#  Temporary Main Function
# --------------------------
def main():
    year, daynum, filenames, directory, camera, chip = passInfo()
#    fns = np.sort([directory + i for i in filenames])
    corrs = np.loadtxt('corrFactors_{}_{}-{}-{}.txt'.format(year, daynum, camera, chip), skiprows = 1)
    xMean, yMean = corrs[:,3], corrs[:,4]

    id, ra, dec, x, y, gmag, mast, mheader = getPixelCoords(filenames[0], directory)

    # Removes any repeating sources
    good = np.unique(id, return_index = True)[1]
    x, y, gmag, id = x[good], y[good], gmag[good], id[good]

    x, y = x-np.mean(xMean), y-np.mean(yMean)
    iso = isolatedFITS(x, y, mast)
    moreX, moreY = testingApers(x, y, xMean, yMean, iso, fns, id)


