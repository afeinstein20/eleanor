import numpy as np
from astropy.wcs import WCS
from starterFile import passInfo
from astropy.io import fits

# --------------------------              
#     Opens FITS file                       
#   Returns: data, header                   
# --------------------------                
def openFITS(dir, fn):
    return fits.getdata(dir+fn, header=True)

# --------------------------  
# Converts (RA,Dec) -> (x,y)
#  Returns: x, y
# --------------------------  
def world2pix(ra, dec, header):
    return WCS(header).all_world2pix(ra, dec, 1)

# --------------------------
# Finds sources in Gaia catalog
#  Returns: id, ra, dec, gmag
# --------------------------
def fromGaia(sector, ra_min, ra_max, dec_min, dec_max):
    gaia = np.loadtxt('gaia_sector{}.cat'.format(sector))
    inds = np.where((gaia[:,1] >= ra_min) & (gaia[:,1] <= ra_max) & (gaia[:,2] >= dec_min) & (gaia[:,2] <= dec_max))[0]
    return gaia[:,0][inds], gaia[:,1][inds], gaia[:,2][inds], gaia[:,3][inds]

# -------------------------- 
# Converts (RA,Dec) -> (x,y)
#  Returns: id, gmag, x, y
# -------------------------- 
def radec2pixel(dir, fn):
    mast, mheader = openFITS(dir, fn)
    ra_cen, dec_cen = mheader['CRVAL1'], mheader['CRVAL2']
    r = 5.
    id, ra, dec, gmag = fromGaia(ra_cen-r, ra_cen+r, dec_cen-r, dec_cen+r)
    print(id)
    w = WCS(mheader)
    return id, gmag, w.all_world2pix(ra, dec, 1)



radec2pixel('./cal_2019-3-3/', 'tess2019132000826-3-3-0016-s_ffic.fits')
