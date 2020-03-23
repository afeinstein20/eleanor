import os
from urllib.request import urlopen
from datetime import datetime
import math
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Tesscut
from astropy.io import fits
import numpy as np
from lightkurve import TessLightCurveFile, search_targetpixelfile
import sys
import urllib.parse as urlparse
import requests
from bs4 import BeautifulSoup


eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')
if not os.path.exists(eleanorpath):
    try:
        os.mkdir(eleanorpath)
    except OSError:
        eleanorpath = os.path.dirname(__file__)

def hmsm_to_days(hour=0,min=0,sec=0,micro=0):
    days = sec + (micro / 1.e6)
    days = min + (days / 60.)
    days = hour + (days / 60.)
    return days / 24.
def date_to_jd(year,month,day):
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

def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + node.get('href') for node in soup.find_all('a') if node.get('href').endswith(ext)]

__all__  = ['Update', 'update_all']

def update_all():
    sector = 1
    good = 1
    while good:
        try:
            Update(sector=sector)
        except AttributeError:
            good = 0
        sector += 1

class Update(object):

    def __init__(self, sector=None):

        if sector is None:
            print('Please pass a sector into eleanor.Update()!')
            return

        self.sector = sector

        try:
            os.mkdir(eleanorpath + '/metadata/s{:04d}'.format(sector))
            success = 1
        except FileExistsError:
            print('Sector {:d} metadata directory exists already!'.format(sector))
            success = 0

        if success == 1:

            tic_north_cvz = 198237770
            tic_south_cvz = 38846515

            if self.sector < 13.5:
                self.tic = tic_south_cvz
            elif self.sector < 26.5:
                self.tic = tic_north_cvz
            else:
                self.tic = tic_south_cvz

            if self.tic == 198237770:
                coord = SkyCoord('16:35:50.667 +63:54:39.87', unit=(u.hourangle, u.deg))
            elif self.tic == 38846515:
                coord = SkyCoord('04:35:50.330 -64:01:37.33', unit=(u.hourangle, u.deg))

            sector_table = Tesscut.get_sectors(coord)


            manifest = Tesscut.download_cutouts(coord, 31, sector = self.sector)

            self.cutout = fits.open(manifest['Local Path'][0])

            print('This is the first light curve you have made for this sector. Getting eleanor metadata products for Sector {0:2d}...'.format(self.sector))
            print('This will only take a minute, and only needs to be done once. Any other light curves you make in this sector will be faster.')
            self.get_target()
            print('Target Acquired')
            self.get_cadences()
            print('Cadences Calculated')
            self.get_quality()
            print('Quality Flags Assured')
            self.get_cbvs()
            print('CBVs Made')
            print('Success! Sector {:2d} now available.'.format(self.sector))
            os.remove(manifest['Local Path'][0])
            self.try_next_sector()

    def get_cbvs(self):
        if self.sector <= 6:
            year = 2018
        elif self.sector <= 20:
            year = 2019
        else:
            year = 2020


        url = 'https://archive.stsci.edu/missions/tess/ffi/s{0:04d}/{1}/'.format(self.sector, year)

        directs = []
        for file in listFD(url):
            directs.append(file)
        directs = np.sort(directs)[1::]

        subdirects = []
        for file in listFD(directs[0]):
            subdirects.append(file)
        subdirects = np.sort(subdirects)[1:-4]
        for i in range(len(subdirects)):
            file = listFD(subdirects[i], ext='cbv.fits')[0]
            os.system('curl -O -L {}'.format(file))

        time = self.cutout[1].data['TIME'] - self.cutout[1].data['TIMECORR']

        files = os.listdir('.')
        files = [i for i in files if i.endswith('cbv.fits') and 's{0:04d}'.format(self.sector) in i]

        for c in range(len(files)):
            cbv      = fits.open(files[c])
            camera   = cbv[1].header['CAMERA']
            ccd      = cbv[1].header['CCD']
            cbv_time = cbv[1].data['Time']

            new_fn = eleanorpath + '/metadata/s{0:04d}/cbv_components_s{0:04d}_{1:04d}_{2:04d}.txt'.format(self.sector, camera, ccd)

            convolved = np.zeros((len(time), 16))
            inds = np.array([], dtype=int)
            for i in range(len(time)):
                g = np.argmin( np.abs(time[i] -cbv_time) )
                for j in range(16):
                    index = 'VECTOR_{0}'.format(j+1)
                    cads  = np.arange(g-7, g+8,1)
                    convolved[i,j] = np.mean(cbv[1].data[index][cads])
            np.savetxt(new_fn, convolved)
            cbv.close()
        files = [i for i in files if i.endswith('.fits')]
        for c in range(len(files)):
            os.remove(files[c])


    def try_next_sector(self):
        codepath = os.path.dirname(__file__)
        f1 = open(codepath + '/maxsector.py', 'r')
        oldmax = float(f1.readline().split('=')[-1])
        if self.sector > oldmax:
            f = open(codepath + '/maxsector.py', 'w')
            f.write('maxsector = {:2d}'.format(self.sector))
            f.close()


    def get_target(self):
        filelist = urlopen('https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{:d}_lc.sh'.
                           format(self.sector))
        for line in filelist:
            if len(str(line)) > 30:
                import shutil
                os.system(str(line)[2:-3])
                fn = str(line)[2:-3].split()[5]
                shutil.move(fn, eleanorpath + '/metadata/s{0:04d}/target_s{0:04d}.fits'.format(self.sector, self.sector))
                break
        return

    def get_cadences(self):
        index_zeropoint = 12680
        index_t0 = 1491.625533688852

        times = np.array([], dtype=int)
        filelist = urlopen('https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{:d}_ffic.sh'.
                          format(self.sector))
        for line in filelist:
            if len(str(line)) > 30:
                times = np.append(times, int(str(line).split('tess')[1][0:13]))

        times = np.sort(np.unique(times))

        outarr = np.zeros_like(times)
        for i in range(len(times)):
            date = datetime.strptime(str(times[i]), '%Y%j%H%M%S')
            days = date.day + hmsm_to_days(date.hour,date.minute,date.second,date.microsecond)
            tjd = date_to_jd(date.year,date.month,days) - 2457000
            cad = (tjd - index_t0)/(30./1440.)
            outarr[i] = (int(np.round(cad))+index_zeropoint)

        np.savetxt(eleanorpath + '/metadata/s{0:04d}/cadences_s{0:04d}.txt'.format(self.sector, self.sector), outarr, fmt='%i')
        return


    def get_quality(self):
        """ Uses the quality flags in a 2-minute target to create quality flags
            in the postcards.
        """

        ffi_time = self.cutout[1].data['TIME'] - self.cutout[1].data['TIMECORR']


        shortCad_fn = eleanorpath + '/metadata/s{0:04d}/target_s{0:04d}.fits'.format(self.sector)

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

            sflux = np.sum(self.cutout[1].data['FLUX'][i])
            if sflux == 0:
                nodata[i] = 4096

            if (ffi_time[i] > 1420) and (ffi_time[i] < 1424):
                nodata[i] = 4096

            v = np.bitwise_or.reduce(twoMinQual[where-7:where+8])
            convolve_ffi.append(v)


        convolve_ffi = np.array(convolve_ffi)

        flags    = np.bitwise_and(convolve_ffi, ffi_apply)

        np.savetxt(eleanorpath + '/metadata/s{0:04d}/quality_s{0:04d}.txt'.format(self.sector), flags+nodata, fmt='%i')

        return
