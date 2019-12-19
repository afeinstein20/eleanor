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

__all__  = ['Update']

class Update(object):

    def __init__(self, sector):
        
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

            if sector < 13.5:
                self.tic = tic_south_cvz
            elif sector < 26.5:
                self.tic = tic_north_cvz
            else:
                self.tic = tic_south_cvz

            print('Searching for Data...')
            self.get_target()
            print('Target Acquired')
            self.get_cadences()
            print('Cadences Calculated')
            self.get_quality()
            print('Quality Flags Assured')
            print('Success!')
            

    def get_target(self):
        tpf = search_targetpixelfile('tic %s'%str(self.tic), mission='TESS', sector=self.sector).download()
        tpf.to_fits(output_fn=eleanorpath + '/metadata/s{0:04d}/target_s{0:04d}.fits'.format(self.sector, self.sector))
        return

    def get_cadences(self):
        index_zeropoint = 12680
        index_t0 = 1491.625533688852

        times = np.array([], dtype=int)
        filelist = urlopen('https://archive.stsci.edu/missions/tess/download_scripts/sector/tesscurl_sector_{:2d}_ffic.sh'.
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

        np.savetxt(eleanorpath + '/metadata/s{0:04d}/cadences_s{0:04d}.fits'.format(self.sector, self.sector), outarr, fmt='%i')
        return


    def get_quality(self):
        """ Uses the quality flags in a 2-minute target to create quality flags
            in the postcards.
        """
        if self.tic == 198237770:
            coord = SkyCoord('16:35:50.667 +63:54:39.87', unit=(u.hourangle, u.deg))
        elif self.tic == 38846515:
            coord = SkyCoord('04:35:50.330 -64:01:37.33', unit=(u.hourangle, u.deg))

        sector_table = Tesscut.get_sectors(coord)


        manifest = Tesscut.download_cutouts(coord, 31, sector = self.sector)

        cutout = fits.open(manifest['Local Path'][0])
        ffi_time = cutout[1].data['TIME'] - cutout[1].data['TIMECORR']


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

            sflux = np.sum(cutout[1].data['FLUX'][i])
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