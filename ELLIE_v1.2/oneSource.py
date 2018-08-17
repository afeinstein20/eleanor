import os, sys
import numpy as np
import matplotlib.pyplot as plt
from scipy       import ndimage
from lightkurve  import KeplerTargetPixelFile as ktpf
from astropy.wcs import WCS
from astropy.io  import fits
import astropy.units as u
from customTPFs  import custom_tpf as ctpf


def files_in_dir(dir):
    """ Finds all FITS files in a given directory """
    """         Returns dir+filenames list        """
    fns = np.array(os.listdir(dir))
    fns = fns[np.array([i for i,item in enumerate(fns) if "fits" in item])]
    fns = [dir+i for i in fns]
    return sort_by_date(fns)


def sort_by_date(fns):
    """ Sorts FITS files by start date of observation """
    """           Returns: sorted filenames           """
    dates = []
    for f in fns:
        mast, header = fits.getdata(f, header=True)
        dates.append(header['DATE-OBS'])
    dates, fns = np.sort(np.array([dates, fns]))
    return fns


def find_camera_chip(id, pos):
    """
    Uses center of each camera/chip pair and position of source to find
         where the source is located in
    Parameters
    ----------
        id: ID of the source
        pos: [RA,Dec] position of the source
    Returns
    ----------
        dir: directory the files are in on the users computer
        xy : translated [RA,Dec] into [x,y] coordinates in file
        camera : the camera number
        chip : the chip number
    """
    for camera in np.arange(1,5,1):
        for chip in np.arange(1,5,1):
            dir   = './2019/2019_1_{}-{}/ffis/'.format(camera, chip)
            files = files_in_dir(dir)
            mast, mheader = fits.getdata(files[0], header=True)
            xy = WCS(mheader).all_world2pix(pos[0], pos[1], 1, quiet=True)
            if xy[0] >= 0. and xy[0] <= len(mast) and xy[1] >= 0. and xy[1] <= len(mast[0]):
                return files, xy, camera, chip
    return 'None', [], 0, 0


def edit_header(output_fn, tic_id, gaia_id, pos, xy):
    """ Adds extra comments to each TPF header """
    """                Returns                 """
    names = ['TIC_ID', 'GAIA_ID', 'CEN_RA', 'CEN_DEC', 'CEN_X', 'CEN_Y']
    values = [tic_id, gaia_id, pos[0], pos[1], float(xy[0]), float(xy[1])]
    for i in range(len(values)):
        fits.setval(output_fn, str(names[i]), value=values[i])
    return


def from_class(id, mission):
    """ Get crossmatching information  """
    """        Returns: table          """
    if mission == 'tic':
        
        locate = ctpf(tic=id)
        tic_id, pos, tmag = locate.tic_pos_by_ID()
        locate = ctpf(tic=id, pos=pos)
        table  = locate.find_by_position()

    elif mission == 'gaia':
        locate = ctpf(gaia=id)
        gaia_id, pos, gmag, pmra, pmdec, plx = locate.gaia_pos_by_ID()
        locate = ctpf(gaia=id, pos=pos)
        table = locate.find_by_position()

    return table


def add_shift(tpf):
    """ Creates an additional shift to put source at (4,4) of TPF file """
    """                          Returns: TPF                          """
    tpf_init = tpf.flux[0]
    tpf_com  = tpf_init[2:7, 2:7]
    com = ndimage.measurements.center_of_mass(tpf_com.T -  np.median(tpf_com))
    shift = [com[0]-2, com[1]-2]
    shift = [np.round(shift[0]+4.0,0), np.round(shift[1]+4.0,0)]
    return shift


def find_source_of_mag(files):
    mast, header = fits.getdata(files[0], header=True)
    pos = [header['CRVAL1'], header['CRVAL2']]
    print(pos)
    find = ctpf(pos=pos)
    sources = find.cone_search(3*np.sqrt(2), 'Mast.Catalogs.Tic.Cone')
    inds = np.where(sources['Tmag'].data <= 9.0)
    sources = sources[inds]

    xy = WCS(header).all_world2pix(sources['ra'].data, sources['dec'].data, 1)
    close = []
    for i in range(len(sources)):
        x, y = xy[0][i], xy[1][i]
        tempX = np.delete(xy[0], np.where(xy[0]==x))
        tempY = np.delete(xy[1], np.where(xy[1]==y))
        closest = np.sqrt( (x-tempX)**2 + (y-tempY)**2 ).argmin()
        if np.sqrt( (x-tempX[closest])**2 + (y-tempY[closest])**2 ) <= 3.0:
            close.append(i)
    close = np.array(close)
    return sources[close[0]], [xy[0][close[0]], xy[1][close[0]]]


def main(id, mission):
    """ Temporary main function """
    info = from_class(id, mission)

    pos = [info['RA'].data[0], info['Dec'].data[0]]
    files, xy, camera, chip = find_camera_chip(id, pos)

    # Shifts (x,y) coordinates based on pointing model
    initShift = np.loadtxt('pointingModel_{}-{}.txt'.format(camera, chip), skiprows=1,
                           usecols=(1,2,3))[0]
    initShift[0] = np.radians(initShift[0])
    x = xy[0]*np.cos(initShift[0]) - xy[1]*np.sin(initShift[0]) - initShift[1]
    y = xy[0]*np.sin(initShift[0]) + xy[1]*np.cos(initShift[0]) - initShift[2]

    # Creates first TPF 
    tpf = ktpf.from_fits_images(images=files, position=[x,y], size=(9,9))

    # Shifts (x,y) coordinates to center on centroid
    xy_new = add_shift(tpf)
    xy_new_start = [x+xy_new[0]-4.0, y+xy_new[1]-4.0]
    new_tpf = ktpf.from_fits_images(images=files, position=xy_new_start, size=(9,9))

    # Converts new (x,y) center to world coordinates
    flux, header = fits.getdata(files[0], header=True)
    pos = WCS(header).all_pix2world(xy_new_start[0], xy_new_start[1], 1)
    pos = [float(pos[0]), float(pos[1])]

    # Saves TPF to FITS file and edits header
    output_fn = 'TIC{}_tpf.fits'.format(id)
    new_tpf.to_fits(output_fn=output_fn)
    print(info)
    edit_header(output_fn, int(info['TIC_ID']), int(info['Gaia_ID']), pos, xy_new_start)



#main(219870537, 'tic')
#main(229669377, 'tic')
#main(420888018, 'tic')
#main(198593129, 'tic') 
