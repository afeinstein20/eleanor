from ellie import data_products
from ellie import find_sources
from ellie import visualize
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt



tic  = 198593129
tic  = 420888018
#pos  = [266.491526, 49.518268]
gaia = 1414861664385248640
#a = data_products(gaia=gaia)
#a.individual_tpf()
#a = data_products()
#a.download_ffis(sector=1, camera=[3], chips=[2])
#a.pointing_model(camera=2, chip=1, sector=1)



b = visualize(tic=tic)
b.mark_gaia()
#b.tpf_movie()
#lc = b.click_aperture()

