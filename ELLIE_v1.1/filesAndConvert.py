import os, sys
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from lightkurve import KeplerTargetPixelFile as ktpf
import requests
from bs4 import BeautifulSoup
from astropy.wcs import WCS
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import astropy.units as u

# --------------------------
# Finds all calibrated files from URL
# Returns: All files for each chip
# --------------------------
def getAllFiles(year, camera, chip, url, dayMin, dayMax):
    calFiles = []
    days = np.arange(dayMin, dayMax+1, 1)
    for d in days:
        path = str(url) + '/' + str(year) + '/' + str(d) + '/'  + str(camera) + '-' + str(chip) + '/'
        print path
        for fn in BeautifulSoup(requests.get(path).text, "lxml").find_all('a'):
            fn = fn.get('href')
            if fn[-7::] == 'ic.fits':
                calFiles.append(path + fn)
    return calFiles

# --------------------------
# Makes directory & scrapes files
# --------------------------
def scrapeFiles(directory, filenames):
    if os.path.isdir(directory) == False:
        os.mkdir(directory)
    for i in filenames:
        os.system('cd {} && curl -O -L {}'.format(directory, i))
    return

# -------------------------- 
# Opens & Extracts FITS data
#   Returns: data, header
# -------------------------- 
def openFITS(file):
    return fits.getdata(file, header = True)

# -------------------------- 
#    Queries Gaia Data
# Returns: ID, RA, Dec, Gmag
# -------------------------- 
def query(ra_cen, dec_cen, h, w):
    adql  = "SELECT gaia.source_id, gaia.ra, gaia.dec, gaia.phot_g_mean_mag FROM gaiadr2.gaia_source AS gaia WHERE 1=CONTAINS( POINT('ICRS', gaia.ra, gaia.dec), BOX('ICRS', {0}, {1}, {2}, {3})) AND gaia.phot_g_mean_mag<16.5".format(str(ra_cen), str(dec_cen), str(w), str(h))
    job   = Gaia.launch_job(adql)
    table = job.get_results()
    return table['source_id'].data, table['ra'].data, table['dec'].data, table['phot_g_mean_mag'].data

# --------------------------
# Converts (RA,Dec) to (x,y) in pixel space
#       Returns: x, y
# -------------------------- 
def toPixels(ra, dec, header):
    w = WCS(header)
    return w.all_world2pix(ra, dec, 1)

# -------------------------- 
# Stores Gaia query as text file
# -------------------------- 
def makeTXT(id, ra, dec, x, y, gmag, fn):
    for i in range(len(id)):
        row = [id[i],ra[i],dec[i],x[i],y[i],gmag[i]]
        with open(fn, 'a') as tf:
            tf.write('{}\n'.format(' '.join(str(e) for e in row)))
    return fn

# -------------------------
# Queries Gaia data & saves to text file
# -------------------------
def radec2pixel(fitsFile):
    year, camera, chip, filenames, dir = passInfo()
    fn = '{}pixelCoords-{}-{}.txt'.format(dir, camera, chip)
    
    # Only if the file does not exist will it be created
    if os.path.isfile(fn) == False:
        mast, mheader = openFITS(fitsFile)
    
        ra_cen, dec_cen = mheader['CRVAL1'],  mheader['CRVAL2']
        h, w = np.float(4), np.float(4)
    
        id, ra, dec, gmag = query(ra_cen, dec_cen, h, w)
        x, y = toPixels(ra, dec, mheader)
        print("printing pixel file")
        makeTXT(id, ra, dec, x, y, gmag, fn)

    return np.loadtxt(fn, usecols = (3,4,5), unpack = True)

# ------------------------- 
# Gets information from user
# ------------------------- 
def usrInput():
    year   = str(sys.argv[1])
    dayMin = int(sys.argv[2])
    dayMax = int(sys.argv[3])
    camera = str(sys.argv[4])
    chip   = str(sys.argv[5])
    url    = str(sys.argv[6])
    dir    = './calFits_{}_{}-{}/'.format(year, camera, chip)
    return year, dayMin, dayMax, camera, chip, url, dir

# ------------------------- 
# Used to call downloading routine
# ------------------------- 
def download(year, camera, chip, url, dayMin, dayMax, dir):
    filenames = getAllFiles(year, camera, chip, url, dayMin, dayMax)
    scrapeFiles(dir, filenames)
    return filenames
 
# ------------------------- 
# Gets files from directory AFTER download
# ------------------------- 
def passInfo():
    year, dayMin, dayMax, camera, chip, url, dir = usrInput()
    if os.path.isdir(dir) == False:
        filenames = download(year, camera, chip, url, dayMin, dayMax, dir)
    elif os.path.isdir(dir) == True:
        filenames = os.listdir(dir)
    return year, camera, chip, np.array(filenames), dir
