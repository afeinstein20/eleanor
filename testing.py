import matplotlib.pyplot as plt
import eleanor
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from eleanor.source import Source
from eleanor.source import multi_sectors
from eleanor.targetdata import TargetData
from eleanor.visualize import Visualize
import numpy as np

#ra  = Angle('04:51:20.681', u.hourangle)
#dec = Angle(-68.069728, u.deg)
#star = Source(coords=SkyCoord(ra, dec), sector=1)
star = Source(tic=38825533)
data = TargetData(star)
#data.save()
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
q = data.quality==0
ax1.plot(data.time, data.x_com, 'k.')
ax2.plot(data.time, data.y_com, 'r.')
plt.show()
plt.close()
plt.plot(data.time[q], data.raw_flux[q], 'k.')
plt.plot(data.time[q], data.corr_flux[q], 'r.')
#plt.plot(data.time, data.quality, 'b*')
plt.show()
plt.imshow(data.tpf[0], origin='lower')
plt.show()
plt.close()

star = Source(tic=261136901)  # this is a binary of two bright stars separated by ~2 pixels
data = TargetData(star)
vis = Visualize(data)
vis.mark_gaia_sources()



star = Source(tic=261136246)  # this is a binary of two bright stars separated by 186 arcsec.
# The brighter star is not the target star
data = TargetData(star, height=15, width=15)
vis = Visualize(data)
vis.mark_gaia_sources()  # this shows the tpf, right?


# Multiple Sectors
star = multi_sectors(sectors=[1,2], tic=25155310)
data0 = TargetData(star[0])
data1 = TargetData(star[1])
plt.plot(data0.time, data0.corr_flux, 'k')
plt.plot(data1.time, data1.corr_flux, 'r')
plt.show()

# Only gets one sector
star = Source(tic=25155310, sector=1)
data = TargetData(star)
plt.plot(data0.time, data0.corr_flux, 'k')
plt.plot(data.time , data.corr_flux , 'g')
plt.show()

# Gets for all sectors available
star = multi_sectors(tic=25155310, sectors='all')
print(star[0].sector, star[1].sector)

# Gets for most recent sector observed
star = Source(tic=25155310, sector='recent')
data = TargetData(star)
plt.plot(data1.time, data1.corr_flux, 'k')
plt.plot(data.time , data.corr_flux , 'r')
plt.show()
