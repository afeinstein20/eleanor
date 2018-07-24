import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from filesAndConvert import openFITS, radec2pixel, passInfo
from astropy.io import fits
from lightkurve import KeplerTargetPixelFile as ktpf
from photutils import CircularAperture, aperture_photometry, RectangularAperture, source_properties
from pixelCorrection import nearest, makeSegMap, correctionFactors
import matplotlib as mpl
from scipy import ndimage as ni
from astropy.wcs import WCS

mpl.rcParams['font.size'] = 7

# --------------------------
# Functions that create apertures
#   of different size & shape
# --------------------------
def square(pos, l, w, theta):
    return RectangularAperture(pos, l, w, theta)

def circle(pos, r):
    return CircularAperture(pos, r)

# --------------------------
# Plots FFI TPF & lightcurve
# --------------------------
def plotTPF(ax1, ax2, aperture, lc, tpf, color):
    tpf.plot(ax=ax1)
    aperture.plot(color=color, alpha=0.5, fill=0.2, ax=ax1)
    ax2.plot(tpf.to_lightcurve().time, lc, color=color)
    tpf.to_lightcurve.plot(color='k', ax=ax2)
    return

# --------------------------
# Extracts lightcurve from custom aperture
# --------------------------
def customLC(fns, aperture, tpf):
    lc = []
    for f in range(len(fns)):
        lc.append(aperture_photometry(tpf.flux[f], aperture)['aperture_sum'].data[0])
    return np.array(lc / np.nanmedian(lc))

# -------------------------- 
# Corrects (x,y) based on cadence
# -------------------------- 
def correctXY(x, y, xMean, yMean, tMean):
#    print tMean*180./np.pi
#    return (x+xMean)*tMean, (y+yMean)*tMean
    return np.abs(x-xMean), np.abs(y-yMean)

# -------------------------- 
#  Temporary Main Function
# -------------------------- 
def tempMain():
    corrFile, year, camera, chip, filenames, dir, x, y, g = correctionFactors()
    corrections = np.loadtxt(corrFile)
    
    for i in range(len(filenames[0:1])):
        mast, mheader = openFITS(dir+filenames[i])
        if filenames[i][11:17] == str(int(corrections[:,0][i])).zfill(6):
            meanX, meanY, meanT = corrections[:,1][i], corrections[:,2][i], corrections[:,3][i]
            corrX, corrY = correctXY(x, y, meanX, meanY, meanT)
            isolated = []
            for j in range(len(corrX)):
                dist, closest = nearest(corrX[j], corrY[j], corrX, corrY)
                if dist > 5.0:
                    isolated.append(j)

    filenames = [dir+i for i in filenames]
    for i in isolated[0:5]:
        center=(corrX[i],corrY[i])
        tpf = ktpf.from_fits_images(images=filenames, position=center, size=(5,5))

        com = ni.measurements.center_of_mass(tpf.flux[0].T-np.median(tpf.flux[0]))

#        print np.median(tpf.flux[0])
        plt.imshow(tpf.flux[0], origin = 'lower')
        plt.plot(com[0], com[1], 'rx', ms=5)
        print tpf.flux[0]

        plt.show()
        plt.close()

#    plt.imshow(mast, origin = 'lower', interpolation = 'nearest', cmap = 'Greys', vmin = 50, vmax = 90)
#    plt.plot(corrX[isolated], corrY[isolated], 'ro', alpha=0.4, ms=10)
#    plt.show()

tempMain()
