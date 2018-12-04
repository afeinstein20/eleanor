import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.wcs import WCS
from .eleanor import data_products
import matplotlib.animation as animation
from astropy.nddata import Cutout2D
from scipy import ndimage

pm_points, centroid = [], []

def animate(i):
    global pm_points, centroid
    for p in pm_points:
        p.remove()
    for c in centroid:
        c.remove()
    pm_points, centroid = [], []
    ax.imshow(tpf[i], origin='lower')#, **kwargs)
    pm_points.append(ax.scatter(pm_x[i], pm_y[i], s=20, c='k'))
    centroid.append(ax.scatter(cen_x[i], cen_y[i], s=20, c='r'))
    time_text.set_text('Frame {}'.format(i))



a = data_products()
fn = 'hlsp_eleanor_tess_ffi_198593129_v1_lc.fits'
hdu = fits.open(fn)
tpf = hdu[0].data
hdr = hdu[0].header
time, lc = hdu[1].data[0], hdu[1].data[1]

pm = a.get_pointing(header=hdr)
center = [hdr['CENTER_X'], hdr['CENTER_Y']]
theta, delx, dely = pm['medT'].data, pm['medX'].data, pm['medY'].data
# Corrects center to be at (4,4) of TPF following the pointing model
corr_x = center[0]*np.cos(theta) - center[1]*np.sin(theta) + delx
corr_y = center[0]*np.sin(theta) + center[1]*np.cos(theta) + dely
pm_x, pm_y = corr_x-np.median(corr_x)+4, corr_y-np.median(corr_y)+4

cen_x, cen_y =[], []
# Tracks centroid
for i in range(len(tpf)):
    tpf_com = Cutout2D(tpf[i], position=(4,4), size=(2,2))
    tpf_com = tpf_com.data
    com = ndimage.measurements.center_of_mass(tpf_com.T - np.median(tpf_com))
    cen_x.append(com[0])
    cen_y.append(com[1])
cen_x, cen_y = cen_x-np.median(cen_x)+4, cen_y-np.median(cen_y)+4

fig, ax = plt.subplots()
# Adds frame count
time_text = ax.text(5.5, -0.25, '', color='white', fontweight='bold')
time_text.set_text('')

Writer = animation.writers['ffmpeg']
writer = Writer(fps=20, metadata=dict(artist='Adina Feinstein'), bitrate=1800)

# Creates animation
ani = animation.FuncAnimation(fig, animate, frames=len(tpf))
# Adds colorbar
plt.colorbar(ax.imshow(tpf[0], origin='lower'), ax=ax)#, **kwargs), ax=ax)
plt.tight_layout()
#plt.show()
ani.save('track_both.mp4'.format(id), writer=writer)
