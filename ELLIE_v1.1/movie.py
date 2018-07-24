import numpy as np
import matplotlib.pyplot as plt
from pixelCorrection import correctionFactors
from filesAndConvert import openFITS
import matplotlib.animation as animation
import matplotlib
matplotlib.use("Agg")
import imageio
from lightkurve import KeplerTargetPixelFile as ktpf
from astropy.io import fits

def updatePoint(i):
    center.set_xdata(x[i])
    center.set_ydata(y[i])
    return center,

def animate(i):
    global scats
    print(i)
    ax.imshow(updateFig(fns[i]), origin = 'lower', vmin = 40, vmax = 100)
    
    for scat in scats:
        scat.remove()
    scats = []
    scats.append(ax.scatter(x[i], y[i], s=16, c='k'))

def init():
    center.set_ydata(np.ma.array(x, mask=True))
    return center,

def updateFig(fn):
    mast, mheader = fits.getdata(fn, header = True)
    return mast

x, y, fns = correctionFactors()

scats = []

print("Making figure")
fig = plt.figure()
ax   = fig.add_subplot(111)

ax.set_xlim([x[0]-5, x[0]+5])
ax.set_ylim([y[1]-5, y[1]+5])

print("Making animation")

Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani = animation.FuncAnimation(fig, animate, frames=len(fns))
print("Showing animation")
#plt.show()
ani.save('test1.mp4', writer=writer)
