import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.animation as animation
from astropy.io import fits
import os, sys
import imageio
matplotlib.use("Agg")
from scipy import ndimage

def animate(i):
    global scats, text
    ax.imshow(mast[i][4], origin='lower', vmax=3000)
 #   com = ndimage.measurements.center_of_mass(mast[i][4])
    for scat in scats:
        scat.remove()
    scats = []
 #   scats.append(ax.scatter(com[0], com[1], s=16, c='k'))
    scats.append(ax.scatter(x[i], y[i], s=16, c='k'))
    time_text.set_text('Frame {}'.format(i))

scats = []


id = str(sys.argv[1])
mast, mheader = fits.getdata('{}.fits'.format(id), header=True)

pointing = 'pointingModel_{}-{}.txt'.format(4, 4)
pointing = np.loadtxt(pointing, usecols=(1,2,3,4))
x, y = [], []
for i in range(len(pointing)):
    x.append(np.abs(pointing[i][2]))
    y.append(np.abs(pointing[i][3]))
    
fig, ax = plt.subplots()

time_text = ax.text(3.4, -0.25, '', color='white', fontweight='bold')
time_text.set_text('')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=16, metadata=dict(artist='Adina Feinstein'), bitrate=1800)

ani = animation.FuncAnimation(fig, animate, frames=len(mast))
ax.set_title('TIC {}'.format(id), color='black', fontweight='bold')

plt.colorbar(plt.imshow(mast[0][4], vmax=3000), ax=ax)

#plt.show()

ani.save('{}.mp4'.format(id), writer=writer)
