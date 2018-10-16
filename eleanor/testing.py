from source import Source
from targetdata import TargetData
import matplotlib.pyplot as plt
from visualize import Visualize

star = Source(tic=229666555)#229669377)
data = TargetData(star)
print(data.x_com)
#data.custom_aperture(shape='circle', r=1.3, pos=(4,2))
#data.save()
#data.psf_lightcurve()


#vis = Visualize(data)
#vis.movie(pointing_model=True)
#plt.show()

## Practicing Reading in pre-made files
#star = Source(fn='hlsp_ellie_tess_ffi_lc_TIC229669377.fits')
#data = TargetData(star)
