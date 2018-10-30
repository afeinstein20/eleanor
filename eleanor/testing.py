from source import Source
from targetdata import TargetData
import matplotlib.pyplot as plt
from visualize import Visualize

star = Source(tic=39825769)
data = TargetData(star)



vis = Visualize(data)
#vis.mark_gaia_sources()
vis.click_aperture()
