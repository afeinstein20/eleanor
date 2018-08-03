import sys, os
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import KeplerTargetPixelFile as ktpf
from scipy import ndimage
import collections
from astropy.wcs import WCS
from astropy.io import fits
from gaiaTIC import coneSearch, jsonTable
from gaiaTIC import ticSearchByContam as tsbc
from scipy.optimize import minimize

def openFITS(dir, fn):
    """ Opens FITS file to get mast, mheader """
    return fits.getdata(dir+fn, header=True)


def radec2pixel(header, r, contam):
    """ Completes a cone search around center of FITS file """
    pos = [header['CRVAL1'], header['CRVAL2']]
    print("Getting sources")
    data = tsbc(pos, r, contam)
    dataTable = jsonTable(data)
    ra, dec = dataTable['ra'], dataTable['dec']
    return WCS(header).all_world2pix(ra, dec, 1), dataTable['ID']

#################
##### ADDED #####
#################
def sortByDate(fns, dir):
    """
    Sorts FITS files by start date of observation
    Parameters
    ---------- 
        dir: directory the files are in
    Returns
    ---------- 
        fns: sorted filenames by date
    """
    dates = []
    for f in fns:
        print(f)
        mast, header = openFITS(dir, f)
        dates.append(header['DATE-OBS'])
    dates, fns = np.sort(np.array([dates, fns]))
    return fns

#################
##### ADDED #####
#################
def nearest(x, y, x_list, y_list):
    """
    Calculates the distance to the nearest source
    Parameters
    ---------- 
        x: x-coordinate of a source
        y: y-coordinate of a source
        x_list: x-coordinates for all sources in file
        y_list: y-coordinates for all sources in file
    Returns
    ---------- 
        dist: distance to closest source
        closest: index of closest source
    """
    x_list  = np.delete(x_list, np.where(x_list==x))
    y_list  = np.delete(y_list, np.where(y_list==y))
    closest = np.sqrt( (x-x_list)**2 + (y-y_list)**2 ).argmin()
    dist    = np.sqrt( (x-x_list[closest])**2 + (y-y_list[closest])**2 )
    return dist, closest

#################
##### ADDED #####
#################
def findIsolated(x, y):
    """
    Finds isolated sources in the image where an isolated source is >= 5 pixels away from
    any other source
    Parameters
    ---------- 
        x: a list of x-coordinates for all sources in file
        y: a list of y-coordinates for all sources in file
    Returns
    ---------- 
        isolated: a list of isolated source indices
    """
    isolated = []
    for i in range(len(x)):
        dist, dist_ind = nearest(x[i], y[i], x, y)
        if dist >= 5.0:
            isolated.append(i)
    return isolated



def calcShift(dir, fns, x, y, corrFile):
    """
    Calculates the deltaX, deltaY, and rotation for isolated sources in order to 
    put together a pointing model for the entire chip
    Parameters
    ---------- 
        fns: all FITS file names
        x: a list of x-coordinates for isolated sources
        y: a list of y-coordinates for isolated sources 
        corrFile: name of file to write corrections to for each cadence
    Returns
    ---------- 
    """

    def roll(theta, pos0, com0):
        rot = np.zeros((2,2))
        rot[1,0] = np.cos(theta)
        rot[1,1] = -1. * np.sin(theta)
        rot[0,0] = np.sin(theta)
        rot[0,1] = np.cos(theta)

        newPos = np.dot(pos0, rot)

        dist = np.sqrt( (newPos[0]-com0[0])**2 + (newPos[1]-com0[1])**2 ) 
        print(dist)
        return dist

    rand = np.random.randint(0, len(x), size=50)
    x, y = x[rand], y[rand]

    matrix = np.zeros((len(fns), len(x), 3))
    fns = [dir+i for i in fns]

    for f in range(len(fns)):
        print(f)
        for i in range(len(x)):
            tpf = ktpf.from_fits_images(images=[fns[f]], position=(x[i], y[i]), size=(5,5))
            com = ndimage.measurements.center_of_mass(tpf.flux.T-np.median(tpf.flux)) #subtracts background
            matrix[f][i][0] = com[0] - 2.5
            matrix[f][i][1] = com[1] - 2.5
            pos = [2.5, 2.5]
            theta = minimize(roll, 0.1, (np.array(pos), com))
            print(theta)
            matrix[f][i][2] = theta.x[0]
            

    for i in range(len(fns)):
        delX = np.mean(matrix[i][:,0])
        delY = np.mean(matrix[i][:,1])
        delT = np.mean(matrix[i][:,2])
        medX = np.median(matrix[i][:,0])
        medY = np.median(matrix[i][:,1])
        medT = np.median(matrix[i][:,2])
        row = [i, delX, delY, delT, medX, medY, medT]
        with open(corrFile, 'a') as tf:
            tf.write('{}\n'.format(' '.join(str(e) for e in row)))
    return


def correctionFactors(camera, chip, dir):
    """
    Creates a pointing model for a given camera and chip
    Parameters
    ---------- 
        camera: which camera to create a pointing model for
        chip: which chip on said camera to create a pointing model for
        dir: directory in which the FITS files can be found
    Returns 
    ---------- 
    """
    filenames = np.array(os.listdir(dir))
    fitsInds = np.array([i for i,item in enumerate(filenames) if "fits" in item])
    filenames = filenames[fitsInds]
    filenames = sortByDate(filenames, dir)

    mast, header = openFITS(dir, filenames[0])

    xy, id = radec2pixel(header, 6*np.sqrt(2), [0.0, 0.01])
    x, y = xy[0], xy[1]

    inds = np.where((x>=50.) & (x<len(mast)) & (y>=0.) & (y<len(mast[0])-100))[0]
    x, y = x[inds], y[inds]
    
    plt.imshow(mast, origin='lower', interpolation='nearest', vmin=40, vmax=100)
    plt.plot(x, y, 'ko', alpha=0.3, ms=3)
#    plt.show()
    plt.close()

    print(len(inds))
    calcShift(dir, filenames, x, y, 'pointingModel_{}-{}.txt'.format(camera, chip))

#correctionFactors(3, 1, './calFITS_2019_3-1/')
correctionFactors(1, 3, './2019/2019_1_1-3/')
