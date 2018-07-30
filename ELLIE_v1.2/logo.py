import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from lightkurve import KeplerTargetPixelFile as ktpf
import os, sys
import matplotlib.animation as animation
import imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
from scipy.misc import imread

def animate(i):
    global scats, lines, lc
    ax1.imshow(tpf.flux[i], origin='lower', vmax=2500)
    for line in lines:
        line.remove()
    lines = []
    lines.append(ax.scatter(lc.time[i], lcNorm[i], s=16, c='r'))


file = '219870537.fits'
tpf  = ktpf.from_fits(file)
lc   = tpf.to_lightcurve()
lcNorm   = lc.flux / np.nanmedian(lc.flux)

lines = []
img = imread('KeynoteEllie.png')
fig = plt.figure(figsize=(10,5))

plt.imshow(img)
plt.axis('off')
plt.tight_layout()

ax = fig.add_axes([0.685, 0.11, 0.17, 0.185])
ax.plot(lc.time, lcNorm, 'k')
ax.get_xaxis().set_ticks([])
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_ticks([])
ax.get_yaxis().set_visible(False)
ax.set_ylim([np.min(lcNorm)-0.05, np.max(lcNorm)+0.05])
ax.set_xlim([np.min(lc.time)-0.005, np.max(lc.time)+0.005])

ax1 = fig.add_axes([0.87, 0.11, 0.089, 0.185])
ax1.get_xaxis().set_ticks([])
ax1.get_yaxis().set_ticks([])

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Adina Feinstein'), bitrate=1000)

ani = animation.FuncAnimation(fig, animate, frames=len(tpf.flux))


#plt.show()

ani.save('logo.mp4', writer=writer)
