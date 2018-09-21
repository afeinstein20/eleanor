from muchbettermoments import quadratic_2d
from ellie import find_sources
from astropy.io import fits
import numpy as np
from astropy.wcs import WCS
import matplotlib.pyplot as plt
from astropy.nddata import Cutout2D


def pixel_cone(header, r, contam):
    """ Completes a cone search around center of FITS file """
    pos = [header['CRVAL1'], header['CRVAL2']]
    data = find_sources.tic_by_contamination(find_sources(), pos, r, contam)
    ra, dec = data['ra'], data['dec']
    id, mag = data['ID'], data['Tmag']
    inds = np.where(data['Tmag'].data < 12.5)
    xy   = WCS(header).all_world2pix(ra, dec, 1)
    return np.array([xy[0][inds], xy[1][inds]]), id[inds], mag[inds], ra[inds], dec[inds]


def find_isolated(x, y):
    """ Find the most isolated, least contaminated sources """
    def nearest(x_source, y_source):
        """ Calculates distance to the nearest source """
        nonlocal x, y
        x_list = np.delete(x, np.where(x==x_source))
        y_list = np.delete(y, np.where(y==y_source))
        closest = np.sqrt( (x_source-x_list)**2 + (y_source-y_list)**2 ).argmin()
        return np.sqrt( (x_source-x_list[closest])**2+ (y_source-y_list[closest])**2 )    
    isolated = []
    for i in range(len(x)):
        dist = nearest(x[i], y[i])
        if dist > 8.0:
            isolated.append(i)
    return isolated


def isolated_center(x, y, image):
    """ Finds the center of each isolated TPF with quadratic_2d """
    cenx, ceny, good = [], [], []
    for i in range(len(x)):
        if x[i] > 0. and y[i] > 0.:
            tpf = Cutout2D(image, position=(x[i], y[i]), size=(7,7), mode='partial')
            cen = quadratic_2d(tpf.data)
            cenx.append(cen[0]); ceny.append(cen[1])
            good.append(i)
    cenx, ceny = np.array(cenx), np.array(ceny)
    return cenx, ceny, good


def main():
    dir    = './.ellie/sector_1/ffis/'
    post   = 'tess2019129080826-3-3-0016-s_ffic.fits'
    r      = 6.0
    contam = [0.0, 5e-3]
    hdu    = fits.open(dir+post)
    xy, id, tmag, ra, dec = pixel_cone(hdu[1].header, r, contam)
    isolated = find_isolated(xy[0], xy[1])
    xy = np.array([xy[0][isolated], xy[1][isolated]])
    cenx, ceny, good = isolated_center(xy[0], xy[1], hdu[1].data)
    good = np.array(good)

    fig, (ax,ax1) = plt.subplots(nrows=1, ncols=2, figsize=(18,6))
    ax.scatter(xy[0][good], xy[1][good], c=cenx, vmin=0, vmax=6)
    p = ax1.scatter(xy[0][good], xy[1][good], c=ceny, vmin=0, vmax=6)
    plt.colorbar(p)
    plt.savefig('centroid_offset.png')
#    plt.show()

main()
