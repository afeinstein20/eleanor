import os
from urllib.request import urlopen
from datetime import datetime
import math
from astropy.coordinates import SkyCoord
from astropy import units as u
from astroquery.mast import Tesscut
from astropy.io import fits
import numpy as np
import sys
import requests
from bs4 import BeautifulSoup
import shutil

eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')
if not os.path.exists(eleanorpath):
    try:
        os.mkdir(eleanorpath)
    except OSError:
        eleanorpath = os.path.dirname(__file__)


def hmsm_to_days(hour=0, min=0, sec=0, micro=0):
    days = sec + (micro / 1.e6)
    days = min + (days / 60.)
    days = hour + (days / 60.)
    return days / 24.


def date_to_jd(year, month, day):
    if month == 1 or month == 2:
        yearp = year - 1
        monthp = month + 12
    else:
        yearp = year
        monthp = month

    # this checks where we are in relation to October 15, 1582, the beginning
    # of the Gregorian calendar.
    if ((year < 1582) or (year == 1582 and month < 10) or
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

    # including leap second correction
    jd = B + C + D + day + 1720994.5 + 0.0008

    return jd


def listFD(url, ext=''):
    page = requests.get(url).text
    soup = BeautifulSoup(page, 'html.parser')
    return [url + node.get('href') for node in soup.find_all('a') if
            node.get('href').endswith(ext)]


__all__ = ['Update', 'update_all', 'update_max_sector']

def update_max_sector():
    baseurl = "https://archive.stsci.edu/missions/tess/ffi/"
    
    page = urlopen(baseurl)
    read = page.read()
    html = read.decode('utf-8').split('href="')
    current_sectors = [i for i in html if 's00' in i]
    sectors = [int(i[1:5]) for i in current_sectors]
    
    with open('maxsector.py', 'w') as tf:
        tf.write('maxsector = {0}'.format(int(np.nanmax(sectors))))

    print("Most recent sector available = ", int(np.nanmax(sectors)))


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
            print('Please pass a sector into eleanor.Update().')
            return

        if not os.path.exists(eleanorpath + '/metadata'):
            os.mkdir(eleanorpath + '/metadata')

        # Updates max sector file first
        update_max_sector()

        self.sector = sector
        self.metadata_path = os.path.join(eleanorpath, 'metadata/s{0:04d}'.format(self.sector))
        lastfile = 'cbv_components_s{0:04d}_0004_0004.txt'.format(self.sector)

        # Checks to see if directory contains all necessary files first
        if os.path.isdir(self.metadata_path):
            if lastfile in os.listdir(self.metadata_path):
                print('This directory already exists!')
                return

        self.north_coords = SkyCoord('16:35:50.667 +63:54:39.87',
                                     unit=(u.hourangle, u.deg))
        self.south_coords = SkyCoord('04:35:50.330 -64:01:37.33',
                                     unit=(u.hourangle, u.deg))

        self.ecliptic_coords_a = SkyCoord('04:00:00.000 +10:00:00.00',
                                          unit=(u.hourangle, u.deg))
        self.ecliptic_coords_b = SkyCoord('08:20:00.000 +12:00:00.00',
                                          unit=(u.hourangle, u.deg))


        if self.sector < 14 or (self.sector > 26 and self.sector < 40):
            use_coords = self.south_coords
        elif self.sector in [42, 43, 44]:
            use_coords = self.ecliptic_coords_a
        elif self.sector in [45, 46]:
            use_coords = self.ecliptic_coords_b
        else:
            use_coords = self.north_coords

        try:
            manifest = Tesscut.download_cutouts(coordinates=use_coords, size=31, sector=self.sector)
            success = 1
        except:
            print("This sector isn't available yet.")
            return

        if success == 1:
            if os.path.isdir(self.metadata_path) == True:
                pass
            else:
                os.mkdir(self.metadata_path)


        # memmap=False as wokaround for https://github.com/afeinstein20/eleanor/issues/204
        self.cutout = fits.open(manifest['Local Path'][0], memmap=False)


        print('This is the first light curve you have made for this sector. '
              'Getting eleanor metadata products for '
              'Sector {0:2d}...'.format(self.sector))
        print('This will only take a minute, and only needs to be done once. '
              'Any other light curves you make in this sector will be faster.')
        print('Acquiring target...')
        self.get_target()
        print('Calculating Cadences...')
        self.get_cadences()
        print('Assuring Quality Flags...')
        self.get_quality()
        print('Making CBVs...')
        self.get_cbvs()
        print('Success! Sector {:2d} now available.'.format(self.sector))
        self.cutout.close()
        os.remove(manifest['Local Path'][0])
        self.try_next_sector()

    def get_cbvs(self):
        if self.sector <= 6:
            year = 2018
        elif self.sector <= 20:
            year = 2019
        elif self.sector <= 33:
            year = 2020
        elif self.sector <= 47:
            year = 2021
        else:
            year = 2022

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
            file = listFD(subdirects[i], ext='_cbv.fits')[0]
            os.system('curl -O -L {}'.format(file))

        time = self.cutout[1].data['TIME'] - self.cutout[1].data['TIMECORR']

        files = os.listdir('.')
        files = [i for i in files if i.endswith('_cbv.fits') and
                 's{0:04d}'.format(self.sector) in i]

        for c in range(len(files)):
            # memmap=False as wokaround for https://github.com/afeinstein20/eleanor/issues/204
            cbv      = fits.open(files[c], memmap=False)
            camera   = cbv[1].header['CAMERA']
            ccd      = cbv[1].header['CCD']
            cbv_time = cbv[1].data['Time']

            new_fn = eleanorpath + '/metadata/s{0:04d}/cbv_components_s{0:04d}_{1:04d}_{2:04d}.txt'.format(self.sector, camera, ccd)

            convolved = np.zeros((len(time), 16))
            for i in range(len(time)):
                g = np.argmin(np.abs(time[i] - cbv_time))
                for j in range(16):
                    index = 'VECTOR_{0}'.format(j+1)
                    if self.sector < 27:
                        cads = np.arange(g-7, g+8, 1)
                    else:
                        # XXX: need to test when TESSCut becomes available
                        cads = np.arange(g-2, g+3, 1)
                    convolved[i, j] = np.mean(cbv[1].data[index][cads])
            np.savetxt(new_fn, convolved)
            cbv.close()
        files = [i for i in files if i.endswith('_cbv.fits') and
                 's{0:04d}'.format(self.sector) in i]
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
                os.system(str(line)[2:-3])
                fn = str(line)[2:-3].split()[5]
                shutil.move(fn, eleanorpath + '/metadata/s{0:04d}/target_s{0:04d}.fits'.format(self.sector))
                break
        return

    def get_cadences(self):
        if self.sector < 27:
            # these come from the first FFI cadence of S7, in particular
            # Camera 1, CCD 1 for the t0. The t0s vary by ~1 minute because of
            # barycentric corrections on different cameras
            index_zeropoint = 12680
            index_t0 = 1491.625533688852
        else:
            # first FFI cadence of S27 from Cam 1, CCD 1
            index_zeropoint = 116470
            index_t0 = 2036.283350837239

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
            days = date.day + hmsm_to_days(date.hour, date.minute,
                                           date.second, date.microsecond)
            tjd = date_to_jd(date.year, date.month, days) - 2457000
            if self.sector < 27:
                cad = (tjd - index_t0)/(30./1440.)
            else:
                cad = (tjd - index_t0)/(10./1440.)
            outarr[i] = (int(np.round(cad))+index_zeropoint)

        np.savetxt(eleanorpath + '/metadata/s{0:04d}/cadences_s{0:04d}.txt'.format(self.sector), outarr, fmt='%i')
        return

    def get_quality(self):
        """ Uses the quality flags in a 2-minute target to create quality flags
            in the postcards.
        """

        ffi_time = self.cutout[1].data['TIME'] - self.cutout[1].data['TIMECORR']

        shortCad_fn = eleanorpath + '/metadata/s{0:04d}/target_s{0:04d}.fits'.format(self.sector)

        # Binary string for values which apply to the FFIs
        if self.sector > 26:
            ffi_apply = int('100000000010101111', 2)
        else:
            ffi_apply = int('100010101111', 2)

        # Obtains information for 2-minute target
        twoMin     = fits.open(shortCad_fn)
        twoMinTime = twoMin[1].data['TIME']-twoMin[1].data['TIMECORR']
        finite     = np.isfinite(twoMinTime)
        twoMinQual = twoMin[1].data['QUALITY']

        twoMinTime = twoMinTime[finite]
        twoMinQual = twoMinQual[finite]

        convolve_ffi = []
        for i in range(len(ffi_time)):
            where = np.where(np.abs(ffi_time[i] - twoMinTime) == np.min(np.abs(ffi_time[i] - twoMinTime)))[0][0]

            sflux = np.sum(self.cutout[1].data['FLUX'][i])
            nodata = 0
            if sflux == 0:
                nodata = 131072

            if (ffi_time[i] > 1420) and (ffi_time[i] < 1424):
                nodata = 131072

            if self.sector < 27:
                v = np.bitwise_or.reduce(twoMinQual[where-7:where+8])
            else:
                # XXX: need to test when TESSCut is available in S27
                v = np.bitwise_or.reduce(twoMinQual[where - 2:where + 3])
            convolve_ffi.append(np.bitwise_or(v, nodata))

        convolve_ffi = np.array(convolve_ffi)

        flags = np.bitwise_and(convolve_ffi, ffi_apply)

        np.savetxt(eleanorpath + '/metadata/s{0:04d}/quality_s{0:04d}.txt'.format(self.sector), flags, fmt='%i')
