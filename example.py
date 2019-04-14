import numpy as np
import matplotlib.pyplot as plt
import eleanor
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
import sys, os

ra  = 11.096438 # In degrees
dec = -63.56094 # In degrees
coords = SkyCoord( Angle(ra, u.deg), Angle(dec, u.deg) )

star = eleanor.Source(coords=coords)
data0 = eleanor.TargetData(star, bkg_type='2d_bkg')
data1 = eleanor.TargetData(star, bkg_type='constant')

q = data0.quality == 0

plt.plot(data0.time[q], data0.corr_flux[q]/np.nanmedian(data0.corr_flux[q]), 'k.', label=str(data0.bkg_type))
plt.plot(data1.time[q], data1.corr_flux[q]/np.nanmedian(data1.corr_flux[q]), 'r.', label=str(data1.bkg_type))
plt.title('The labels in the legend should be 2D_BKG and CONSTANT.')
plt.legend()
plt.show()
plt.close()

print("Is FFIINDEX working?")
try:
    print(data0.post_obj.ffiindex)
except:
    pass
print("If an array of integers printed, the answer is yes!")
