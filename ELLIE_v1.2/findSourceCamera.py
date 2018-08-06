import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gaiaTIC import gaiaPositionByID as gaiaID
from gaiaTIC import ticPositionByID as ticID
from astropy.wcs import WCS

def findCameraChip(id, mission):
    """
    Uses center of each camera/chip and position of the source to find
         which files the source is located in
    Parameters
    ----------
        centers: list of centers for each camera/chip FITS file
        pos: [RA,Dec] position of the source 
    Returns
    ----------
        centers[i][0], centers[i][1]: the camera and chip numbers. Returns 0, 0 if source 
        is not found in any file
    """
    return_id, pos, tmag = ticID(int(id))
    for i in np.arange(1,5,1):
        for j in np.arange(1,5,1):
            dir  = './2019/2019_1_{}-{}/ffis/'.format(i, j)
            file = 'tess2019132000826-{}-{}-0016-s_ffic.fits'.format(i, j)
            if i == 3 and (j == 2 or j == 3):
                file = 'tess2019130000826-{}-{}-0016-s_ffic.fits'.format(i, j)
            mast, mheader = fits.getdata(dir+file, header=True)
            xy = WCS(mheader).all_world2pix(pos[0], pos[1], 1, quiet=True)
            if xy[0] >= 0. and xy[0] <= len(mast) and xy[1] >= 0. and xy[1] <= len(mast[0]):
                return i, j



def findSource():
    """
    Temporary main function
    """
    sources = np.loadtxt('sourcesToMovies.txt', dtype=str)
    for i in range(len(sources)):
        findCameraChip(sources[i][0], sources[i][1])


#findSource()
