import numpy as np
import matplotlib.pyplot as plt
import findSourceCamera as fsc
from gaiaTIC import ticPositionByID as tpbi
from astropy.io import fits
from astropy.wcs import WCS
from lightkurve import KeplerTargetPixelFile as ktpf
import os, sys
from astroquery.mast import Catalogs
from customTPFs import visualize as vis
from pixelCorrection import sortByDate

jl = np.loadtxt('hotJupiters.txt', comments='#')

inds = np.where((jl[:,2] <= 5.) & (jl[:,4] >= 0.06) & (jl[:,9] >= 6500.))
objs = jl[:,0][inds]
print(jl[inds])
good = []
for o in objs:
    i, j = fsc.findCameraChip(o, 'tic')
    if i == 3 and j == 3:
        good.append(o)

dir = './2019/2019_1_3-3/ffis/'
fns = os.listdir(dir)
fns = sortByDate(fns, dir)
fns = [dir+i for i in fns]

mast, mheader = fits.getdata(fns[0], header=True)

#for tic_id in good:
#    tic_id = int(tic_id)
#    ticData = Catalogs.query_criteria(catalog='Tic', ID=tic_id)
#    id, pos, tmag = ticData['ID'].data, [ticData['ra'].data[0], ticData['dec'].data[0]], ticData['Tmag'].data

#    xy = WCS(mheader).all_world2pix(pos[0], pos[1], 1)
#    print(pos, xy)
#    tpf = ktpf.from_fits_images(images=fns, position=xy, size=(10,10))
#    tpf.to_fits('{}_tpf.fits'.format(int(tic_id)))


source = int(good[1])
tpfDir = './'


#call = vis(source, dir=tpfDir)
#ani = call.tpf_movie(plot_lc=True)
#plt.show()
