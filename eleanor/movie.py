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
from photutils import CircularAperture, RectangularAperture, aperture_photometry
from lightkurve import KeplerTargetPixelFile as ktpf
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection

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
def customLC(aperture, tpf):
    lc = []
    for f in range(len(tpf.flux)):
        mask = np.zeros_like(tpf.flux[f], dtype=bool)
        for i in range(len(tpf.flux[f])):
            for j in range(len(tpf.flux[f][0])):
                if tpf.flux[f][i][j] <= 0.35 * tpf.flux[f][3][4]:
                    mask[i,j] = True
                else:
                    mask[i,j] = False

        lc.append(aperture_photometry(tpf.flux[f], aperture, mask=mask)['aperture_sum'].data[0])

    return np.array(lc / np.nanmedian(lc))

# ----------------------
#   CREATES ANIMATION   
# ----------------------
def animate(i):
    global scats, text, lines, lc, custLCC, custLCR, ps

    ax1.imshow(tpf.flux[i], origin='lower')#, vmin=40, vmax=150)
    for scat in scats:
        scat.remove()
    for line in lines:
        line.remove()
    for c in ps:
        c.remove()
    scats, lines, ps= [], [], []
    scats.append(ax1.scatter(x[i], y[i], s=16, c='k'))
    time_text.set_text('Frame {}'.format(i))

    lcNorm = lc.flux / np.nanmedian(lc.flux)
#    lines.append(ax.scatter(lc.time[i], custLCC[i], s=20, c='r'))
#    lines.append(ax.scatter(lc.time[i], custLCR[i], s=20, c='k'))

#    circleShape = patches.Circle((x[i],y[i]), 1.5, fill=False, alpha=0.4)
#    rectanShape = patches.Rectangle((x[i]-1.5,y[i]-1.5), 3.0, 3.0, fill=False)
#    p = PatchCollection([rectanShape, circleShape], alpha=0.4)
#    colors = np.linspace(0,1,2)
#    p.set_array(np.array(colors))
#    p.set_edgecolor('face')
#    ps.append(ax1.add_collection(p))


id = str(sys.argv[1])
tpf = ktpf.from_fits('./figures/{}_tpf.fits'.format(id))
lc = tpf.to_lightcurve()

pointing = 'pointingModel_{}-{}.txt'.format(3,3)

theta, delX, delY = np.loadtxt(pointing, usecols=(1,2,3), skiprows=1, unpack=True)

new_id, pos, tmag = ticID(int(id))
x, y, scats, lines = [], [], [], []
ps = []

for i in range(len(tpf.flux)-1):
    if i == 0:
        x.append( 3.0*np.cos(theta[i+1]) - 4.0*np.sin(theta[i+1]) + delX[i+1] )
        y.append( 3.0*np.sin(theta[i+1]) + 4.0*np.cos(theta[i+1]) + delY[i+1] )
    else:
        x.append( x[i-1]*np.cos(theta[i+1]) - y[i-1]*np.sin(theta[i+1]) + delX[i+1] )
        y.append( x[i-1]*np.sin(theta[i+1]) + y[i-1]*np.cos(theta[i+1]) + delY[i+1] )


fig = plt.figure(figsize=(18,5))
spec = gridspec.GridSpec(ncols=3, nrows=1)
ax  = fig.add_subplot(spec[0, 0:2])
ax1 = fig.add_subplot(spec[0, 2])


time_text = ax1.text(6.0, -0.25, '', color='white', fontweight='bold')
time_text.set_text('')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=16, metadata=dict(artist='Adina Feinstein'), bitrate=1800)

ani = animation.FuncAnimation(fig, animate, frames=len(tpf.flux))
plt.title('TIC {}'.format(id), color='black', fontweight='bold', loc='center')

#apertureC = circle([x,y], 1.5)
#apertureR = square([x,y], 3.0, 3.0, 0.0)
#custLCC = customLC(apertureC, tpf)
#custLCR = customLC(apertureR, tpf)
#custLCC = custLCC / np.nanmedian(custLCC)
#custLCR = custLCR / np.nanmedian(custLCR)


# Writes light curve to FITS file
#lcFile  = '{}_lc.fits'.format(id)
#fits.writeto(lcFile, np.array([lc.time, custLCC]))



#ax.plot(lc.time, custLCC, 'r')
#ax.plot(lc.time, custLCR, 'k')
ax.plot(lc.time, lc.flux/np.nanmedian(lc.flux))

x_cen = math.ceil(pos[0])
y_cen = math.ceil(pos[1])

x_ticks = np.arange(x_cen-5, x_cen+4, 1)
y_ticks = np.arange(y_cen-5, y_cen+4, 1)

plt.xticks(np.arange(0,10,1), x_ticks)
plt.yticks(np.arange(0,10,1), y_ticks)
plt.colorbar(plt.imshow(tpf.flux[0]), ax=ax1)#, vmin=40, vmax=150), ax=ax1)
plt.tight_layout()
#plt.show()

ani.save('{}_customAp.mp4'.format(id), writer=writer)

