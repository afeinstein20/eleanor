import numpy as np
import sys, os
import urllib.parse as urlparse
import requests
from bs4 import BeautifulSoup

from tqdm import trange
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
from astroquery.mast import Tesscut


def listFD(url, ext=''):
"""
   Finds the co-trending basis vector files in the directory structure at STScI
   Input
   -----
      type(url) == str
"""
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

def download_cbvs(sectors=np.arange(1,14,1)):
"""
   Downloads the co-trending basis vectors for a given list of sectors
   Input
   -----
      type(sectors) == list
"""
    for sector in sectors:
        sector = int(sector)
        if sector <= 6:
            year = 2018
        else:
            year = 2019

        url = 'https://archive.stsci.edu/missions/tess/ffi/s{0:04d}/{1}/'.format(sector, year)
    
        directs = []
        for file in listFD(url):
            directs.append(file)
        directs = np.sort(directs)[1::]

        subdirects = []
        for file in listFD(directs[0]):
            subdirects.append(file)
        subdirects = np.sort(subdirects)[1:-4]

        fns = []
        for i in range(len(subdirects)):
            file = listFD(subdirects[i], ext='cbv.fits')[0]
            os.system('curl -O -L {}'.format(file))
            fns.append(file.split('/')[-1])
        return fns

        
def convolve_cbvs(sectors=np.arange(1,14,1)):
"""
   Bins the co-trending basis vectors into FFI times;
   Calls download_cbvs to get filenames
   Input
   -----
      type(sectors) == list
"""
    # Gets the cutout for a target in the CVZ
    coord = SkyCoord('04:35:50.330 -64:01:37.33', unit=(u.hourangle, u.deg))
    sector_table = Tesscut.get_sectors(coord)

    for sector in sectors:
        files = download_cbvs(sector)
        manifest = Tesscut.download_cutouts(coord, 31, sector = sector)
        cutout = fits.open(manifest['Local Path'][0])
        time = cutout[1].data['TIME'] - cutout[1].data['TIMECORR']

        for c in trange(len(files)):
            cbvs_short = np.zeros((len(time),16))
        
            cbvs   = fits.open(cbv_dir+files[c])
            
            camera = cbvs[1].header['CAMERA']
            ccd    = cbvs[1].header['CCD']
            
            new_fn = './s{0:04d}/cbv_components_s{0:04d}_{1:04d}_{2:04d}.txt'.format(sector, camera, ccd)

            for i in range(len(time)):
                g = np.where(np.abs(time[i] - cbvs[1].data['Time']) == np.min(np.abs(time[i] - cbvs[1].data['Time'])))[0][0]
                for j in range(16):
                    string = 'VECTOR_' + str(j+1)
                    cbvs_short[i,j] = np.mean(cbvs[1].data[string][g-7:g+8])

                np.savetxt(new_fn, cbvs_short)
