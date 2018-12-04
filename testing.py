import matplotlib.pyplot as plt
from eleanor.source import Source
from eleanor.targetdata import TargetData
from eleanor.visualize import Visualize


star = Source(tic=39825769)
data = TargetData(star)
star = Source(tic=39825769)
data = TargetData(star)

#vis = Visualize(data)
#vis.mark_gaia_sources()
#vis.click_aperture()
