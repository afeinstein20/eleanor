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


def correctXY(x_s, y_s, camera, chip):
    pointing = 'pointingModel_{}-{}.txt'.format(camera, chip)
    delX, delY, delT = np.loadtxt(pointing, skiprows=1, usecols=(1,2,3), unpack=True)
    print(delX)
    delT = np.radians(delT)
    new = []
    for i in range(len(pointing)):
        x = (x_s*np.cos(delT[i]) - y_s*np.sin(delT[i])) + delX[i]
        y = (x_s*np.sin(delT[i]) + y_s*np.cos(delT[i])) + delY[i]
        new.append((x, y))
    return new


def main(camera, chip, id):
    """ Temporary Main Function """
    dir = './2019/2019_1_{}-{}/ffis/'.format(camera, chip)
    fns = np.array(os.listdir(dir))
    fns = fns[np.array([i for i,item in enumerate(fns) if "fits" in item])]
    fns = sortByDate(fns, dir)
    mast, mheader = openFITS(dir, fns[0])
    new_id, raDec, xy = getSourcePos(id, 'tic', mheader)
    print(xy)
    xy_corr = correctXY(xy[0], xy[1], camera, chip)
    print(xy_corr)
    fns = [dir+i for i in fns]
    tpf = ktpf.from_fits_images(images=fns, position=xy_corr, size=(9,9))
    tpf.to_fits(output_fn = '{}_rotation.fits'.format(id))


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
#id = 219870537
#main(4,4, id)

