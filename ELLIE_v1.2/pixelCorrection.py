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
        mast, header = fits.getdata(dir+f, header=True)
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
        if dist >= 8.0:
            isolated.append(i)
    return isolated



def calcShift(dir, fns, x, y, corrFile, mast):
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

    def model(params):
        nonlocal xy, centroids
        theta, xT, yT = params
        theta = np.radians(theta)

        xRot = xy[:,0]*np.cos(theta) - xy[:,1]*np.sin(theta)
        yRot = xy[:,0]*np.sin(theta) + xy[:,1]*np.cos(theta)

        xRot += xT
        yRot += yT

        dist = np.sqrt((xRot-centroids[:,0])**2 + (yRot-centroids[:,1])**2)
        dist = np.square(dist)
        return np.sum(dist)

    rand = np.random.randint(0, len(x), size=200)
    x, y = x[rand], y[rand]

    fns = [dir+i for i in fns]

    x_cen, y_cen = len(mast)/2, len(mast[0])/2
    xy = np.zeros((len(x),2))


    for i in range(len(x)):
        xy[i][0] = x[i]-x_cen
        xy[i][1] = y[i]-y_cen
    with open(corrFile, 'w') as tf:
        tf.write('cadence medT medX medY\n')

    matrix = np.zeros((len(x), len(fns), 2))

    for i in range(len(x)):
        print(x[i], y[i])
        tpf = ktpf.from_fits_images(images=fns, position=(x[i], y[i]), size=(6,6))
        #tpf.to_fits(output_fn='test{}.fits'.format(i))
        for j in range(len(fns)):
            com = ndimage.measurements.center_of_mass(tpf.flux[j].T-np.median(tpf.flux[j])) # subtracts background
            matrix[i][j][0] = com[0]+xy[i][0]
            matrix[i][j][1] = com[1]+xy[i][1]


    for i in range(len(fns)):
        centroids = matrix[:,i]
        if i == 0:
            initGuess = [0.001, 0.1, -0.1]
        else:
            initGuess = solution.x

        bnds = ((-0.08, 0.08), (-5.0, 5.0), (-5.0, 5.0))
        solution = minimize(model, initGuess, method='L-BFGS-B', bounds=bnds, options={'ftol':5e-11, 
                                                                                       'gtol':5e-05})
        if i == 0:
            initSolution = solution.x
        else:
            sol = solution.x

        with open(corrFile, 'a') as tf:
            if i == 0:
                theta, delX, delY = initSolution[0], initSolution[1], initSolution[2]
            else:
                theta = initSolution[0] - sol[0]
                delX  = initSolution[1] - sol[1]
                delY  = initSolution[2] - sol[2]
            tf.write('{}\n'.format(str(i) + ' ' + str(theta) + ' ' + str(delX) + ' ' + str(delY)))
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

    xy, id = radec2pixel(header, 6*np.sqrt(2), [0.0, 5e-4])
    x, y = xy[0], xy[1]

    inds = np.where((y>=44.) & (y<len(mast)-45.) & (x>=0.) & (x<len(mast[0])-41.))[0]
    x, y = x[inds], y[inds]
 
    isolated = findIsolated(x, y)
    print(len(isolated))
    x, y = x[isolated], y[isolated]
   
    calcShift(dir, filenames, x, y, 'pointingModel_{}-{}.txt'.format(camera, chip), mast)


#correctionFactors(3, 1, './2019/2019_1_3-1/ffis/')
#correctionFactors(4, 4, './2019/2019_1_4-4/ffis/')
#correctionFactors(3, 3, './2019/2019_1_3-3/ffis/')
#correctionFactors(1, 3, './2019/2019_1_1-3/ffis/')
