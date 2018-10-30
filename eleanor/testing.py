from .source import Source
from .targetdata import TargetData
import matplotlib.pyplot as plt
from .visualize import Visualize

<<<<<<< HEAD
star = Source(tic=39825769)
data = TargetData(star)
=======
def test_sanity():
    star = Source(tic=229666555)#229669377)
    data = TargetData(star)
    print(data.x_com)
#data.custom_aperture(shape='circle', r=1.3, pos=(4,2))
#data.save()
#data.psf_lightcurve()
>>>>>>> eb6768aae8149fb5b2f15db7ada282c99efc95f6



vis = Visualize(data)
#vis.mark_gaia_sources()
vis.click_aperture()
