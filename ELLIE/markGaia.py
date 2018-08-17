import numpy as np
import matplotlib.pyplot as plt
from lightkurve import KeplerTargetPixelFile as ktpf
from customTPFs import custom_tpf as ctpf
from astropy.io import fits
from astropy.wcs import WCS
from oneSource import files_in_dir
import mplcursors
from astroquery.mast import Catalogs

def find_center(file):
    """ Finds the true center of the TPF """
    hdu = fits.open(file)
    header = hdu[0].header
    # Finds center (RA,Dec) and (x,y) of TPF
    return (header['CEN_RA'], header['CEN_DEC']), (header['CEN_X'], header['CEN_Y']), header


def cone_search_sources(cen_ra, cen_dec):#gaia_id):
    """ Completes a cone search for sources """
#    locate  = ctpf(gaia=gaia_id)
#    table = locate.gaia_pos_by_ID()
    # Gets (RA,Dec) position of the associated Gaia ID
#    pos = [table['ra'].data[0], table['dec'].data[0]]
    pos = [cen_ra, cen_dec]
    # Finds Gaia sources around associated Gaia ID
    newlocate = ctpf(pos=pos)
    sources = newlocate.cone_search(0.05, 'Mast.Catalogs.GaiaDR2.Cone')
    return sources['ra'], sources['dec'], sources['source_id'], sources['phot_g_mean_mag']


def pointingCorr(xy, camera, chip):
    """ Corrects (x,y) coordinates based on pointing model """
    shift = np.loadtxt('pointingModel_{}-{}.txt'.format(camera, chip), skiprows=1, usecols=(1,2,3))[0]
    shift[0] = np.radians(shift[0])
    x = xy[0]*np.cos(shift[0]) - xy[1]*np.sin(shift[0]) - shift[1]
    y = xy[0]*np.sin(shift[0]) + xy[1]*np.cos(shift[0]) - shift[2]
    return np.array([x,y])
    

def in_tpf(xy, gaiaXY, gaiaID, gaiaMAG):
    """ Pushes the Gaia sources to the appropriate place in the TPF """
    gaiaX, gaiaY = gaiaXY[0]-xy[0]+4, gaiaXY[1]-xy[1]+5
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
    pos, xy, hdu1 = find_center(file)

    gaiaRA, gaiaDEC, gaiaID, gaiaMAG = cone_search_sources(pos[0], pos[1])#hdu1['GAIA_ID'])

    gaiaXY = WCS(header).all_world2pix(gaiaRA, gaiaDEC, 1)
    gaiaXY = pointingCorr(gaiaXY, camera, chip)
    gaiaXY, gaiaID, gaiaMAG = in_tpf(xy, gaiaXY, gaiaID, gaiaMAG)

    crossmatch = ctpf()
    crossTable = crossmatch.crossmatch_multi_to_tic(list=gaiaID.data)

    ticLabel, tmagLabel = np.zeros(len(gaiaID.data)), np.zeros(len(gaiaID.data))
    for i in range(len(gaiaID.data)):
        row = crossTable[i]
#        print(row['separation'])
        if row['separation'] <= 1.0 and row['Gmag'] <= 16.5:
            ticLabel[i]  = row['TIC_ID']
            tmagLabel[i] = row['Tmag']

    tpf = ktpf.from_fits(file)
    plot_with_hover(tpf, gaiaXY, gaiaID, gaiaMAG, ticLabel, tmagLabel)


main(198593129, 3, 3)
#main(356149601, 3, 3)
#main(219870537, 4, 4)
