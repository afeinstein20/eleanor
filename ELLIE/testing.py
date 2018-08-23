from ellie import data_products
from ellie import find_sources
from astropy.io import fits
import numpy as np

camera, chip, sector = 3, 3, 1
dir   = './2019/2019_{}_{}-{}/ffis/'.format(sector, camera, chip)
files = data_products(dir=dir)
fns   = files.sort_by_date()
ffi, header = fits.getdata(dir+fns[0], header=True)
radec = [header['CRVAL1'], header['CRVAL2']]

#cone    = find_sources(pos=radec)
#r = 1
#sources = cone.cone_search(r=2, service='Mast.Catalogs.Tic.Cone')
#good = np.where(sources['Tmag'] < 16.0)[0]
#sources = sources[good]
#ids = sources['ID']

ex = data_products(dir=dir, camera=camera, chip=chip, sector=sector)
print("making pm")
ex.pointing_model()


#for i in ids[0:2]:
#    print(i)
#    example = data_products(dir=dir, camera=camera, chip=chip, sector=sector, id=i, mission='tic')  
#example.pointing_model()
#example.make_postcard()
#    example.individual_tpf()
