#from postcard import postcard
#a = postcard(sector=1, camera=2, chip=1)
#a = postcard(post_name='postcard_1_3-4_2-0.fits')
#a.grab()
#print(a.local_path)
#a.read()
#print(a.xy_center)


from ffi import ffi
import urllib.request
from astropy.table import Table
import numpy as np
from astropy.io import fits
from astropy.nddata import Cutout2D
from astropy.wcs import WCS
import matplotlib.pyplot as plt

a = ffi(sector=1, camera=3, chip=3)
a.download_ffis()
a.sort_by_date()

#a.pointing_model_per_cadence()


target_url = 'http://jet.uchicago.edu/tess_postcards/pointingModel_1_3-3.txt'

pm_link = urllib.request.urlopen(target_url)
pm = pm_link.read().decode('utf-8')
pm = Table.read(pm, format='ascii.basic')
row = pm[0]
matrix = []
for i in row.colnames:
    matrix.append(row[i])
matrix = np.reshape(np.array(matrix), (3,3))


pos = [265.89681, 48.054922]
pos = [265.788376, 48.133675]
pos = [265.945687, 48.239777]
pos = [265.707768, 48.005833]
hdu = fits.open(a.local_paths[0])
hdr = hdu[1].header
ffi = hdu[1].data

xy = WCS(hdr).all_world2pix(pos[0], pos[1], 1)
true_xy = a.use_pointing_model(xy, matrix)
true_xy = true_xy[0]

og_tpf  = Cutout2D(ffi, position=(xy[0], xy[1]), size=(9,9), mode='partial')
new_tpf = Cutout2D(ffi, position=(true_xy[0], true_xy[1]), size=(9,9), mode='partial')

fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
ax1.imshow(og_tpf.data,  origin='lower', vmin=60, vmax=300)
ax2.imshow(new_tpf.data, origin='lower', vmin=60, vmax=300)
plt.show()
