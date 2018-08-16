import numpy as np
import matplotlib.pyplot as plt
from lightkurve import KeplerTargetPixelFile as ktpf
from customTPFs import custom_tpf as ctpf
from astropy.io import fits
from astropy.wcs import WCS
from oneSource import files_in_dir
import mplcursors

def find_center(file):
    """ Finds the true center of the TPF """
    hdu = fits.open(file)
    header = hdu[0].header
    return (header['CEN_RA'], header['CEN_DEC']), (header['CEN_X'], header['CEN_Y'])


def cone_search_sources(pos, header):
    """ Completes a cone search for sources """
    locate  = ctpf(pos=pos)
    sources = locate.cone_search(r=0.05, service='Mast.Catalogs.GaiaDR2.Cone')
    xy_gaia = WCS(header).all_world2pix(sources['ra'].data, sources['dec'].data, 1)
    return xy_gaia, sources['source_id'], sources['phot_g_mean_mag']


def pointingCorr(xy, camera, chip):
    """ Corrects (x,y) coordinates based on pointing model """
    shift = np.loadtxt('pointingModel_{}-{}.txt'.format(camera, chip), skiprows=1, usecols=(1,2,3))[0]
    shift[0] = np.radians(shift[0])
    x = xy[0]*np.cos(shift[0]) - xy[1]*np.sin(shift[0]) - shift[1]
    y = xy[0]*np.sin(shift[0]) + xy[1]*np.cos(shift[0]) - shift[2]
    return np.array([x,y])
    

def in_tpf(xy, gaiaXY, gaiaID, gaiaMAG):
    gaiaX, gaiaY = xy[0]-gaiaXY[0]+4, xy[1]-gaiaXY[1]+4
    inds = np.where( (gaiaX > -0.5) & (gaiaX < 8.5) &
                     (gaiaY > -0.5) & (gaiaY < 8.5) )
    return [gaiaX[inds], gaiaY[inds]], gaiaID[inds], gaiaMAG[inds]



def plot_with_hover(tpf, gaiaXY, gaiaID, gaiaMAG, ticID, tmag):
    fig, ax = plt.subplots()

    ax.imshow(tpf.flux[0], origin='lower')
    sc = ax.scatter(gaiaXY[0], gaiaXY[1], c='k', s=10)

    plt.xlim([-0.5,8.5])
    plt.ylim([-0.5,8.5])

    mplcursors.cursor(sc).connect(
        "add", lambda sel: sel.annotation.set_text("TIC ID = {}\nTmag = {}\nGaia ID = {}\nGmag = {}".format(ticID[sel.target.index],
                                                                                                            tmag[sel.target.index],
                                                                                                            gaiaID[sel.target.index], 
                                                                                                            gaiaMAG[sel.target.index])))
    plt.show()
    


def main(id, camera, chip):
    """ Temporary main function """
    file = './figures/TIC{}_tpf.fits'.format(id)
    dir  = './2019/2019_1_{}-{}/ffis/'.format(camera, chip)
    fns  = files_in_dir(dir)

    flux, header = fits.getdata(fns[0], header=True)
    pos, xy = find_center(file)
    gaiaXY, gaiaID, gaiaMAG = cone_search_sources(pos, header)
    gaiaXY = pointingCorr(gaiaXY, camera, chip)

    gaiaXY, gaiaID, gaiaMAG = in_tpf(xy, gaiaXY, gaiaID, gaiaMAG)

    cross = ctpf(multiFile='crossmatch.txt')
    in_tic = cross.crossmatch_multi_to_tic()

    inds = np.where(in_tic['separation'].data <= 1.0)
    ticLabel, tmagLabel = np.zeros(len(gaiaID), dtype=str), np.zeros(len(gaiaID), dtype=str)

    for i in range(len(ticLabel)):
        if i in inds[0]:
            print(i)
            ticLabel[i]  = str(in_tic['TIC_ID'].data[i])
            tmagLabel[i] = str(in_tic['Tmag'].data[i])
    print(tmagLabel)

    tpf = ktpf.from_fits(file)
    plot_with_hover(tpf, gaiaXY, gaiaID, gaiaMAG, ticLabel, tmagLabel)

main(198593129, 3, 3)

