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

from datetime import datetime

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

def download_cbvs(sector):
    """
   Downloads the co-trending basis vectors for a given list of sectors
   Input
   -----
      type(sectors) == int
    """
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
        files = download_cbvs(int(sector))
        manifest = Tesscut.download_cutouts(coord, 31, sector = sector)
        cutout = fits.open(manifest['Local Path'][0])
        time = cutout[1].data['TIME'] - cutout[1].data['TIMECORR']

        for c in trange(len(files)):
            cbv      = fits.open(files[c])
            camera   = cbv[1].header['CAMERA']
            ccd      = cbv[1].header['CCD']
            cbv_time = cbv[1].data['Time']
            
            new_fn = './eleanor/metadata/s{0:04d}/cbv_components_s{0:04d}_{1:04d}_{2:04d}.txt'.format(sector, camera, ccd)
    
            convolved = np.zeros((len(time), 16))
            
            for i in range(len(time)):
                g = np.argmin( np.abs(time[i] - cbv_time) )
                for j in range(16):
                    index = 'VECTOR_{0}'.format(j+1)
                    cads  = np.arange(g-7, g+8, 1)
                    convolved[i,j] = np.mean(cbv[1].data[index][cads])

            np.savetxt(new_fn, convolved)


def set_quality_flags(sector=np.arange(1,14,1)):
    """ Uses the quality flags in a 2-minute target to create quality flags
        in the postcards.
    We create our own quality flag as well, using our pointing model.
    """
    
    coord = SkyCoord('04:35:50.330 -64:01:37.33', unit=(u.hourangle, u.deg))
    
    sector_table = Tesscut.get_sectors(coord)


    manifest = Tesscut.download_cutouts(coord, 31, sector = sector)
    
    cutout = fits.open(manifest['Local Path'][0])
    ffi_time = cutout[1].data['TIME'] - cutout[1].data['TIMECORR']
    
    
    shortCad_fn = 'eleanor/metadata/s{0:04d}/target_s{0:04d}.fits'.format(sector)
    
    # Binary string for values which apply to the FFIs
    ffi_apply = int('100010101111', 2)
    
    # Obtains information for 2-minute target
    twoMin     = fits.open(shortCad_fn)
    twoMinTime = twoMin[1].data['TIME']-twoMin[1].data['TIMECORR']
    finite     = np.isfinite(twoMinTime)
    twoMinQual = twoMin[1].data['QUALITY']

    twoMinTime = twoMinTime[finite]
    twoMinQual = twoMinQual[finite]

    convolve_ffi = []
    nodata = np.zeros_like(ffi_time)
    for i in range(len(ffi_time)):
        where = np.where(np.abs(ffi_time[i] - twoMinTime) == np.min(np.abs(ffi_time[i] - twoMinTime)))[0][0]
        
        sflux = np.sum(cutout[1].data['FLUX'][i])
        if sflux == 0:
            nodata[i] = 4096
            
        if (ffi_time[i] > 1420) and (ffi_time[i] < 1424):
            nodata[i] = 4096
        
        v = np.bitwise_or.reduce(twoMinQual[where-7:where+8])
        convolve_ffi.append(v)

        
    convolve_ffi = np.array(convolve_ffi)

    flags    = np.bitwise_and(convolve_ffi, ffi_apply)
    
    np.savetxt('eleanor/metadata/s{0:04d}/quality_s{0:04d}.txt'.format(sector), flags+nodata, fmt='%i')

    return flags


def create_ffiindex(sectors=np.arange(1,14,1)):
    """
   Creates FFIINDEX for a given sector
   Input
   -----
      type(sectors) == list
    """

    def hmsm_to_days(hour=0,min=0,sec=0,micro=0):
        """
        Convert hours, minutes, seconds, and microseconds to fractional days.
        """
        days = sec + (micro / 1.e6) 
        days = min + (days / 60.)
        days = hour + (days / 60.)
        return days / 24.

    def date_to_jd(year,month,day):
        """
        Convert a date to Julian Day.
        """
        if month == 1 or month == 2:
            yearp = year - 1
            monthp = month + 12
        else:
            yearp = year
            monthp = month
    
            # this checks where we are in relation to October 15, 1582, the beginning
            # of the Gregorian calendar.
        if ((year < 1582) or
            (year == 1582 and month < 10) or
            (year == 1582 and month == 10 and day < 15)):
            # before start of Gregorian calendar
            B = 0
        else:
            # after start of Gregorian calendar
            A = math.trunc(yearp / 100.)
            B = 2 - A + math.trunc(A / 4.)
            
        if yearp < 0:
            C = math.trunc((365.25 * yearp) - 0.75)
        else:
            C = math.trunc(365.25 * yearp)
            
        D = math.trunc(30.6001 * (monthp + 1))
        jd = B + C + D + day + 1720994.5 + 0.0008  # including leap second correction
        return jd 


    curlfile = 'tesscurl_sector_{0}_ffic.sh'

    later_sector_curl = curlfile.format(7)
    curr_sector_curl  = curlfile.format(sector)
    
    os.system('curl -O -L https://archive.stsci.edu/missions/tess/download_scripts/sector/{0}'.format(later_sector_curl))

    for sector in sectors:
        os.system('curl -O -L https://archive.stsci.edu/missions/tess/download_scripts/sector/{0}'.format(curr_sector_curl))
    
        index_sector = open('/Users/AdinaFeinstein/Documents/ELLIE/metadata/s0007/{0}'.format(later_sector_curl))
        download_file = []
        for line in index_sector:
            if len(line) > 30:
                download_file.append(line)
                break
        
        download_file = download_file[-1]
        os.system(download_file)
        fn = download_file.split(' ')[5]
        a  = fits.open(fn)
        
        outarr =np.array([])
        indexlist = open('/Users/AdinaFeinstein/Documents/ELLIE/metadata/s{0:04d}/{1}'.format(sector, curr_sector_curl))

        for line in indexlist:
            if len(line) > 30:
                outarr = np.append(outarr, (line.split('tess')[1][0:13]))

        times = np.sort(np.unique(outarr))

        outarr = np.zeros_like(times, dtype=int)
        for i in range(len(times)):
            date = datetime.strptime(str(times[i]), '%Y%j%H%M%S')
            days = date.day + hmsm_to_days(date.hour,date.minute,date.second,date.microsecond)
            tjd = date_to_jd(date.year,date.month,days) - 2457000
            cad = (tjd - a[0].header['tstart'])/(30./1440.)
            outarr[i] = (int(np.round(cad))+a[0].header['ffiindex'])
        
        np.savetxt('cadences_s{0:04d}.txt'.format(sector), outarr, fmt='%i')
        os.remove(fn)
        os.remove(curr_sector_curl)

    os.remove(later_sector_curl)


convolve_cbvs(sectors=[11])
