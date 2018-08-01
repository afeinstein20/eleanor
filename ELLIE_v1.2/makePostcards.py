from astropy.io import fits
from astropy.wcs import WCS
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os, sys
from lightkurve import KeplerTargetPixelFile as ktpf


dir = './calFits_2019_3-3/'
fns = np.array(os.listdir(dir))
fns = fns[np.array([i for i,item in enumerate(fns) if "fits" in item])]
fns = [dir+i for i in fns]

mast, mheader = fits.getdata(fns[0], header=True)

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
        tpf = ktpf.from_fits_images(images=fns, position=(x_cen,y_cen), size=(350,350))
        tpf.to_fits(output_fn=fn)
        
        fits.setval(fn, 'LOC_C_X' , value=np.round(x_cen,5))
        fits.setval(fn, 'LOC_C_Y' , value=np.round(y_cen,5))
        fits.setval(fn, 'LOC_C_RA', value=float(radec[0]))
        fits.setval(fn, 'LOC_C_DE', value=float(radec[1]))




