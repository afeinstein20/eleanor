from ellie import data_products
from ellie import find_sources
from ellie import visualize
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
from bokeh.plotting import show


vis = visualize(tic=193945542)
#ani = vis.tpf_movie(aperture=True, plot_lc=True, com=True)
#plt.show()
lc = vis.click_aperture()
plt.plot(np.arange(0,len(lc),1), lc, 'k')
plt.show()
