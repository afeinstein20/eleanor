import os, sys
import numpy as np
import matplotlib.pyplot as plt
from lightkurve import KeplerTargetPixelFile as ktpf
from gaiaTIC import gaiaPositionByID as gaiaID
from gaiaTIC import ticPositionByID  as ticID
from astropy.wcs import WCS
from pixelCorrection import sortByDate, openFITS
import matplotlib.animation as animation
import imageio
import matplotlib
from astropy.io import fits
matplotlib.use("Agg")


def getSourcePos(id, mission, mheader):
    """ Finds (RA,Dec) for a single source, depending on mission """
    w = WCS(mheader)
    if mission == 'gaia':
        id, pos, gmag, pmra, pmdec, plx = gaiaID(int(id))
    elif mission == 'tic':
        id, pos, tmag = ticID(int(id))
    xy = w.all_world2pix(pos[0], pos[1], 1, quiet=True)
    return id, pos, xy


def correctXY(x, y, camera, chip):
    pointing = 'pointingModel_{}-{}.txt'.format(camera, chip)
    pointing = np.loadtxt(pointing, usecols=(1,2,3,4))
    new = []
    for i in range(len(pointing)):
        new.append((x+pointing[i][2], y+pointing[i][3]))
    return new


def main(camera, chip, id):
    """ Temporary Main Function """
    dir = './calFITS_2019_{}-{}/'.format(camera, chip)
    fns = np.array(os.listdir(dir))
    fns = fns[np.array([i for i,item in enumerate(fns) if "fits" in item])]
    fns = sortByDate(fns, dir)
    mast, mheader = openFITS(dir, fns[0])
    new_id, raDec, xy = getSourcePos(id, 'tic', mheader)
    xy_corr = correctXY(xy[0], xy[1], camera, chip)

    fns = [dir+i for i in fns]
    tpf = ktpf.from_fits_images(images=fns, position=xy_corr, size=(9,9))
    tpf.to_fits(output_fn = '{}.fits'.format(id))



def updateFig(fn):
    mast,mheader = fits.getdata(fn, header=True)
    return mast


def animate(i):
    global scats
    ax.imshow(updateFig(fns[i]), origin='lower', vmin=50, vmax=200)
    for scat in scats:
        scat.remove()
    scats = []
    scats.append(ax.scatter(x[i], y[i], s=16, c='k'))
    ax.text(x[0]+3, y[0]-4, 'Frame {}'.format(i), color='white', fontweight='bold')

scats = []
id = 229669377
main(3, 1, id)
id = 219870537
main(4,4, id)

