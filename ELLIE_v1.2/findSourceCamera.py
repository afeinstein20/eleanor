import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from gaiaTIC import gaiaPositionByID as gaiaID
from gaiaTIC import ticPositionByID as ticID
from gaiaTIC import crossmatch
from astropy.wcs import WCS
from lightkurve import KeplerTargetPixelFile as ktpf
from pixelCorrection import sortByDate

def findCameraChip(id, pos):
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
    for i in np.arange(1,5,1):
        for j in np.arange(1,5,1):
            dir  = './2019/2019_1_{}-{}/ffis/'.format(i, j)
            file = 'tess2019132000826-{}-{}-0016-s_ffic.fits'.format(i, j)
            if i == 3 and (j == 2 or j == 3):
                file = 'tess2019130000826-{}-{}-0016-s_ffic.fits'.format(i, j)
            mast, mheader = fits.getdata(dir+file, header=True)
            xy = WCS(mheader).all_world2pix(pos[0], pos[1], 1, quiet=True)
            if xy[0] >= 0. and xy[0] <= len(mast) and xy[1] >= 0. and xy[1] <= len(mast[0]):
                return dir, xy


def findSource():
    """
    Temporary main function
    """
    sources = np.loadtxt('sourcesToMovies.txt', dtype=str)
    for i in range(len(sources)):
        findCameraChip(sources[i][0], sources[i][1])


def createTPF(id, mission):
    if mission == 'tic':
        return_id, pos, tmag = ticID(int(id))
    elif mission == 'gaia':
        return_id, pos, gmag, pmra, pmdec, plx = gaiaID(int(id))
    dir, xy = findCameraChip(id, pos)
    fns = np.array(os.listdir(dir))
    fns = fns[np.array([i for i,item in enumerate(fns) if "fits" in item])]
    fns = sortByDate(fns, dir)
    fns = [dir+i for i in fns]

    gaia = crossmatch(pos, 1, 'Mast.GaiaDR2.Crossmatch')
    gaia_id = gaia['MatchID']
    tpf = ktpf.from_fits_images(images=fns, position=xy, size=(9,9))
    output_fn = '{}_tpf.fits'.format(id)
    tpf.to_fits(output_fn=output_fn)
    
    names = ['TIC_ID', 'GAIA_ID', 'CEN_RA', 'CEN_DEC', 'CEN_X', 'CEN_Y']
    values = [return_id[0], gaia['MatchID'][0], pos[0], pos[1], float(xy[0]), float(xy[1])]

    for i in range(len(values)):
        print(names[i], values[i])
        fits.setval(output_fn, str(names[i]), value=values[i])

#createTPF(198593129, 'tic')

