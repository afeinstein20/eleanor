from .source import Source
from .targetdata import TargetData
import matplotlib.pyplot as plt
from .visualize import Visualize



star = Source(tic=261136901)  # this is a binary of two bright stars separated by ~2 pixels
data = TargetData(star)
vis = Visualize(data)
vis.mark_gaia_sources()  # this shows the tpf, right?



star = Source(tic=261136246)  # this is a binary of two bright stars separated by 186 arcsec.
# The brighter star is not the target star
data = TargetData(star, height=15, width=15)
vis = Visualize(data)
vis.mark_gaia_sources()  # this shows the tpf, right?
 
              

#vis = Visualize(data)
#vis.mark_gaia_sources()
#vis.click_aperture()
