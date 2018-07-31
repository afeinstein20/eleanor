from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys
from lightkurve import KeplerTargetPixelFile as ktpf
import matplotlib.patches as patches
from matplotlib.collections import PatchCollection


testFile = './calFits_2019_2-1/tess2019132000826-2-1-0016-s_ffic.fits'

mast, mheader = fits.getdata(testFile, header=True)

x = np.linspace(0, len(mast), 4, dtype=int)
y = np.linspace(0, len(mast[0]), 4, dtype=int)
x_cens, y_cens = [], []

for i in range(len(x)-1):
    for j in range(len(y)-1):
        fn = 'postcard_{}-{}.fits'.format(i, j)
        x_cen = (x[i]+x[i+1]) / 2.
        y_cen = (y[j]+y[j+1]) / 2.
        x_cens.append(x_cen)
        y_cens.append(y_cen)

        radec = WCS(mheader).all_pix2world(x_cen, y_cen, 1)
        tpf = ktpf.from_fits_images(images=[testFile], position=(x_cen,y_cen), size=(350,350))
        tpf.to_fits(output_fn=fn)

        data, hdr = fits.getdata(fn, header=True)

        hdr.set('LOC_C_X', np.round(x_cen,5))
        hdr.set('LOC_C_Y', np.round(y_cen,5))
        hdr.set('LOC_C_RA',  np.round(float(radec[0]),5))
        hdr.set('LOC_C_DE', np.round(float(radec[1]),5))

        fits.writeto(fn, data, hdr, clobber=True)


