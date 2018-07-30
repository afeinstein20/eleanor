import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from astropy.io import fits
import os, sys
import imageio
matplotlib.use("Agg")
from scipy import ndimage
from gaiaTIC import ticPositionByID  as ticID
import math
from lightkurve import KeplerTargetPixelFile as ktpf
import matplotlib.gridspec as gridspec


# --------------------------
# Functions that create apertures
#  of different size & shape
# --------------------------
def square(pos, l, w, theta):
    return RectangularAperture(pos, l, w, theta)

def circle(pos, r):
    return CircularAperture(pos, r)

# --------------------------
# Plots FFI TPF & lightcurve
# --------------------------
def plotAper(ax1, aperture, lc, tpf, color):
    aperture.plot(color=color, alpha=0.5, fill=0.2, ax=ax1)
    return

# --------------------------              
# Extracts lightcurve from custom aperture
# --------------------------              
def customLC(fns, aperture, tpf):
    lc = []
    for f in range(len(fns)):
        lc.append(aperture_photometry(tpf.flux[f], aperture)['aperture_sum'].data[0])
    return np.array(lc / np.nanmedian(lc))



def animate(i):
    global scats, text, lines, lc
    ax1.imshow(tpf.flux[i], origin='lower', vmin=40, vmax=150)#vmax=3000)
    for scat in scats:
        scat.remove()
    for line in lines:
        line.remove()
    scats = []
    scats.append(ax1.scatter(x[i], y[i], s=16, c='k'))
    time_text.set_text('Frame {}'.format(i))
    lines = []
    lcNorm = lc.flux / np.nanmedian(lc.flux)
    lines.append(ax.scatter(lc.time[i], lcNorm[i], s=20, c='r'))

id = str(sys.argv[1])
tpf = ktpf.from_fits('{}.fits'.format(id))
lc = tpf.to_lightcurve()

pointing = 'pointingModel_{}-{}.txt'.format(4, 4)
pointing = np.loadtxt(pointing, usecols=(1,2,3,4))


new_id, pos, tmag = ticID(int(id))
x, y, scats, lines = [], [], [], []

for i in range(len(tpf.flux)):
    x.append(5+pointing[i][2])
    y.append(5+pointing[i][3])
#for i in range(len(tpf.flux)):
#    com = ndimage.measurements.center_of_mass(tpf.flux[i].T-np.median(tpf.flux[i]))
#    x.append(com[0])
#    y.append(com[1])

fig = plt.figure(figsize=(18,5))
spec = gridspec.GridSpec(ncols=3, nrows=1)
ax  = fig.add_subplot(spec[0, 0:2])
ax1 = fig.add_subplot(spec[0, 2])


time_text = ax1.text(6.0, -0.25, '', color='white', fontweight='bold')
time_text.set_text('')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=6, metadata=dict(artist='Adina Feinstein'), bitrate=1800)

ani = animation.FuncAnimation(fig, animate, frames=len(tpf.flux))
plt.title('TIC {}'.format(id), color='black', fontweight='bold', loc='center')
lc.plot(ax=ax)

x_cen = math.ceil(pos[0])
y_cen = math.ceil(pos[1])

x_ticks = np.arange(x_cen-5, x_cen+4, 1)
y_ticks = np.arange(y_cen-5, y_cen+4, 1)

plt.xticks(np.arange(0,10,1), x_ticks)
plt.yticks(np.arange(0,10,1), y_ticks)
plt.colorbar(plt.imshow(tpf.flux[0], vmin=40, vmax=150), ax=ax1)
plt.tight_layout()
#plt.show()

ani.save('{}.mp4'.format(id), writer=writer)
