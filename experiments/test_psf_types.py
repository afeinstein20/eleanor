## Script to compare different PSF profiles, adapted from the quick-demo ipython notebook.

import numpy as np
import matplotlib.pyplot as plt
import eleanor
import lightkurve as lk

star = eleanor.Source(tic=120362128, sector=14, tc=True)

data = eleanor.TargetData(star, height=15, width=15, do_pca=True, do_psf=False)
sc = lk.search_targetpixelfile(target='tic120362128', sector=14).download()
sq = sc.quality < 5000
start = 2500
end = 3400

