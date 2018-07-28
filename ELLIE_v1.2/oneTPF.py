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
    pointing = np.loadtxt(pointing, usecols=(3,4))
    newX, newY = [], []
    for i in range(len(pointing)):
        newX.append(x+pointing[i][0])
        newY.append(y+pointing[i][1])
    return newX, newY


def main(camera, chip, id):
    """ Temporary Main Function """
    dir = './calFITS_2019_{}-{}/'.format(camera, chip)
    fns = np.array(os.listdir(dir))
    fns = fns[np.array([i for i,item in enumerate(fns) if "fits" in item])]
    fns = sortByDate(fns, dir)
    mast, mheader = openFITS(dir, fns[0])
    new_id, raDec, xy = getSourcePos(id, 'tic', mheader)
    x, y = correctXY(xy[0], xy[1], camera, chip)
    return x, y, [dir+i for i in fns]


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
x, y, fns = main(3, 1, id)

print("Making figure")
fig = plt.figure()
ax   = fig.add_subplot(111)

ax.set_xlim([x[0]-5, x[0]+5])
ax.set_ylim([y[1]-5, y[1]+5])
ax.set_title(id, color='black', fontweight='bold')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=8, metadata=dict(artist='Me'), bitrate=1800)

ani = animation.FuncAnimation(fig, animate, frames=len(fns))


ani.save('test1.mp4', writer=writer)
