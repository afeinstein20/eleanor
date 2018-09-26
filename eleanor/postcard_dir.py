import os, sys
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table

t = Table()

dir = './postcards/'
files = os.listdir(dir)
dir_and_files = [dir+i for i in files]
hdu = fits.open(dir_and_files[0])
hdr = hdu[1].header

headerKeys = list(hdr.keys())
headerVals = list(hdr.values())

comments, dtype = [], []
for key in range(len(headerKeys)):
    dtype.append('U35')
    if headerKeys[key] == 'COMMENT':
        comments.append(key)
hdr = np.delete(headerKeys, comments)
val = np.delete(headerVals, comments)
dtype = np.delete(dtype, comments)

hdr = np.append(hdr, 'POST_NAME')
val = np.append(val, files[0])
dtype = np.append(dtype, 'U35')

t = Table(names=hdr, dtype=dtype)
t.add_row(val)

for i in range(1,len(files)):
    hdu = fits.open(dir_and_files[i])
    hdr = hdu[1].header
    vals = list(hdr.values())
    vals = np.delete(vals, comments)
    vals = np.append(vals, files[i])
    t.add_row(vals)

t.write('postcard.guide', format='ascii')

#plt.imshow(hdu[2].data[:,:,3], origin='lower', vmin=40, vmax=300)
#plt.show()
