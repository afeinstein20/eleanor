import numpy as np
import matplotlib.pyplot as plt
import os, sys
from astropy.io import fits
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

# --------------------------
# Opens & extracts FITS data
# --------------------------                
def openFITS(file, dir):
    return fits.getdata(dir + file, header = True)

# --------------------------  
# ADQL Code for Gaia Archive
# -------------------------- 
def adql(ra, dec, w, h):
    return "SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.phot_g_mean_mag FROM gaiadr2.gaia_source AS gaia WHERE 1=CONTAINS( POINT('ICRS', gaia.ra, gaia.dec), BOX('ICRS', {0},  {1}, {2}, {3})) AND gaia.phot_g_mean_mag<16.5".format(str(ra), str(dec), str(w), str(h))

# --------------------------
#    Queries Gaia Data
# --------------------------
def query(ra_cen, dec_cen, h, w):
    print ra_cen, dec_cen, h, w
    job = Gaia.launch_job(adql(ra_cen, dec_cen, h, w))
    print job
    table = job.get_results()
    return table['ra'].data, table['dec'].data, table['phot_g_mean_mag'].data, table['source_id'].data

# --------------------------
# Converts (RA, Dec) to (x,y) in pixel space
# --------------------------
def toPixels(ra, dec, mheader):
    w = WCS(mheader)                     
    x, y = w.all_world2pix(ra, dec, 1)
    return x, y

# --------------------------
# Creates a .txt file with (x,y)
# coordinates for all sources requested
# --------------------------
def getPixelCoords(fitsFile, dir):
    mast, mheader = openFITS(fitsFile, dir)
    id, ra, dec, x, y, gmag = np.loadtxt('pixelCoords.txt', skiprows = 1, unpack = True)
    return id, ra, dec, x, y, gmag, mast, mheader

    for i in np.arange(0.0, 1.0, 0.1):
    
        ra_cen  = mheader['CRVAL1']-i
        dec_cen = mheader['CRVAL2']-i

        h, w = 1., 1.

        ra, dec, gmag, id = query(ra_cen, dec_cen, float(h), float(w))
        x, y = toPixels(ra, dec, mheader)

        for i in range(len(id)):
            row = [id[i],ra[i],dec[i],x[i],y[i],gmag[i]]
            with open('pixelCoords.txt', 'a') as tf:
                tf.write('{}\n'.format(' '.join(str(e) for e in row)))
