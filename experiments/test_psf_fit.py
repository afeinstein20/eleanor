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

data.psf_lightcurve(data_arr = sc.flux[sq][start:end], err_arr = sc.flux_err[sq][start:end], 
                        bkg_arr=sc.flux_bkg[sq,0,0][start:end], verbose=True, nstars=3, xc=[4.9, 4.5, 4.7], 
                        yc=[3.0, 4.4, 7.0], ignore_pixels=1)

time = sc.time[sq][start:end]
photometry_flux = sc.to_lightcurve().flux[sq][start:end]

plt.plot(time.mjd, photometry_flux / np.max(photometry_flux), label='aperture photometry')
plt.plot(time.mjd, data.psf_flux[:end-start] / np.max(data.psf_flux[:end-start]), label='psf fit')
plt.legend()
plt.show()
