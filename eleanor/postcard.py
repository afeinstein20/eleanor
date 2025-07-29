import os, sys
from datetime import datetime

from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
import numpy as np
import warnings
import pandas as pd
import copy
from astropy.stats import SigmaClip
from photutils.background import MMMBackground
from .mast import crossmatch_by_position
from urllib.request import urlopen


__all__ = ['Postcard']

class Postcard(object):
    """TESS FFI data for one postcard across one sector.

    A postcard is an rectangular subsection cut out from the FFIs.
    It's like a TPF, but bigger.
    The Postcard object contains a stack of these cutouts from all available
    FFIs during a given sector of TESS observations.

    Parameters
    ----------
    filename : str
        Filename of the downloaded postcard.
    location : str, optional
        Filepath to `filename`.

    Attributes
    ----------
    dimensions : tuple
        (`x`, `y`, `time`) dimensions of postcard.
    flux, flux_err : numpy.ndarray
        Arrays of shape `postcard.dimensions` containing flux or error on flux
        for each pixel.
    time : float
        ?
    header : dict
        Stored header information for postcard file.
    center_radec : tuple
        RA & Dec coordinates of the postcard's central pixel.
    center_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of
        the postcard's central pixel on the FFI.
    origin_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of
        the postcard's (0,0) pixel on the FFI.
    background2d : np.ndarray
        The 2D modeled background array.
    """
    def __init__(self, filename, background, filepath):
        self.filename = os.path.join(filepath, filename)
        self.local_path = copy.copy(self.filename)
        self.hdu = fits.open(self.local_path)
        self.background2d = fits.open(os.path.join(filepath, background))[1].data

    def __repr__(self):
        return "eleanor postcard ({})".format(self.filename)

    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        """Plots a single frame of a postcard.

        Parameters
        ----------
        frame : int, optional
            Index of frame. Default 0.

        ax : matplotlib.axes.Axes, optional
            Axes on which to plot. Creates a new object by default.

        scale : str
            Scaling for colorbar; acceptable inputs are 'linear' or 'log'.
            Default 'linear'.

        **kwargs : passed to matplotlib.pyplot.imshow

        Returns
        -------
        ax : matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale == 'log':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dat = np.log10(self.flux[:, :, frame])
                dat[~np.isfinite(dat)] = np.nan
        else:
            dat = self.flux[:, :, frame]

        if ('vmin' not in kwargs) & ('vmax' not in kwargs):
            kwargs['vmin'] = np.nanpercentile(dat, 1)
            kwargs['vmax'] = np.nanpercentile(dat, 99)

        im = ax.imshow(dat, **kwargs)
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        cbar = plt.colorbar(im, ax=ax)
        if scale == 'log':
            cbar.set_label('log$_{10}$ Flux')
        else:
            cbar.set_label('Flux')

        # Reset the x/y ticks to the position in the ACTUAL FFI.
        xticks = ax.get_xticks() + self.center_xy[0]
        yticks = ax.get_yticks() + self.center_xy[1]
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        return ax

    def find_sources(self):
        """Finds the cataloged sources in the postcard and returns a table.

        Returns
        -------
        result : astropy.table.Table
            All the sources in a postcard with TIC IDs or Gaia IDs.
        """
        result = crossmatch_by_position(self.center_radec, 0.5, 'Mast.Tic.Crossmatch').to_pandas()
        result = result[['MatchID', 'MatchRA', 'MatchDEC', 'pmRA', 'pmDEC', 'Tmag']]
        result.columns = ['TessID', 'RA', 'Dec', 'pmRA', 'pmDEC', 'Tmag']
        return result


    @property
    def header(self):
        return self.hdu[1].header

    @property
    def center_radec(self):
        return(self.header['CEN_RA'], self.header['CEN_DEC'])

    @property
    def center_xy(self):
        return (self.header['CEN_X'],  self.header['CEN_Y'])

    @property
    def origin_xy(self):
        return (self.header['POSTPIX1'], self.header['POSTPIX2'])

    @property
    def flux(self):
        return self.hdu[2].data

    @property
    def dimensions(self):
        return self.flux.shape

    @property
    def flux_err(self):
        return self.hdu[3].data

    @property
    def time(self):
        return (self.hdu[1].data['TSTOP'] + self.hdu[1].data['TSTART'])/2

    @property
    def wcs(self):
        return WCS(self.header)

    @property
    def quality(self):
        return self.hdu[1].data['QUALITY']

    @property
    def bkg(self):
        return self.hdu[1].data['BKG']

    @property
    def barycorr(self):
        return self.hdu[1].data['BARYCORR']


    @property
    def ffiindex(self):
        return self.hdu[1].data['FFIINDEX']

class Postcard_tesscut(object):
    """TESS FFI data for one postcard across one sector.

    TESSCut is a service from MAST to produce TPF cutouts from the TESS FFIs. If
    `eleanor.Source()` is called with `tc=True`, TESSCut is used to produce a large
    postcard-like cutout region rather than downlading a standard eleanor postcard.

    Parameters
    ----------
    filename : str
        Filename of the downloaded postcard.
    location : str, optional
        Filepath to `filename`.

    Attributes
    ----------
    dimensions : tuple
        (`x`, `y`, `time`) dimensions of postcard.
    flux, flux_err : numpy.ndarray
        Arrays of shape `postcard.dimensions` containing flux or error on flux
        for each pixel.
    time : float
        ?
    header : dict
        Stored header information for postcard file.
    center_radec : tuple
        RA & Dec coordinates of the postcard's central pixel.
    center_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of
        the postcard's central pixel on the FFI.
    origin_xy : tuple
        (`x`, `y`) coordinates corresponding to the location of
        the postcard's (0,0) pixel on the FFI.
    """
    def __init__(self, cutout, location=None, eleanorpath=None):

        if eleanorpath is None:
            self.eleanor_path = os.path.join(os.path.expanduser('~'), '.eleanor')
        else:
            self.eleanor_path = eleanorpath

        if location is None:
            self.local_path = os.path.join(self.eleanor_path, 'tesscut')
        else:
            self.local_path = location


        self.hdu = cutout


    def plot(self, frame=0, ax=None, scale='linear', **kwargs):
        """Plots a single frame of a postcard.

        Parameters
        ----------
        frame : int, optional
            Index of frame. Default 0.

        ax : matplotlib.axes.Axes, optional
            Axes on which to plot. Creates a new object by default.

        scale : str
            Scaling for colorbar; acceptable inputs are 'linear' or 'log'.
            Default 'linear'.

        **kwargs : passed to matplotlib.pyplot.imshow

        Returns
        -------
        ax : matplotlib.axes.Axes
        """

        if ax is None:
            _, ax = plt.subplots(figsize=(8, 7))
        if scale == 'log':
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                dat = np.log10(self.flux[:, :, frame])
                dat[~np.isfinite(dat)] = np.nan
        else:
            dat = self.flux[:, :, frame]

        if ('vmin' not in kwargs) & ('vmax' not in kwargs):
            kwargs['vmin'] = np.nanpercentile(dat, 1)
            kwargs['vmax'] = np.nanpercentile(dat, 99)

        im = ax.imshow(dat, **kwargs)
        ax.set_xlabel('Row')
        ax.set_ylabel('Column')
        cbar = plt.colorbar(im, ax=ax)
        if scale == 'log':
            cbar.set_label('log$_{10}$ Flux')
        else:
            cbar.set_label('Flux')

        # Reset the x/y ticks to the position in the ACTUAL FFI.
        xticks = ax.get_xticks() + self.center_xy[0]
        yticks = ax.get_yticks() + self.center_xy[1]
        ax.set_xticklabels(xticks)
        ax.set_yticklabels(yticks)
        return ax

    def find_sources(self):
        """Finds the cataloged sources in the postcard and returns a table.

        Returns
        -------
        result : astropy.table.Table
            All the sources in a postcard with TIC IDs or Gaia IDs.
        """
        result = crossmatch_by_position(self.center_radec, 0.5, 'Mast.Tic.Crossmatch').to_pandas()
        result = result[['MatchID', 'MatchRA', 'MatchDEC', 'pmRA', 'pmDEC', 'Tmag']]
        result.columns = ['TessID', 'RA', 'Dec', 'pmRA', 'pmDEC', 'Tmag']
        return result


    @property
    def header(self):
        return self.hdu[1].header

    @property
    def center_radec(self):
        return(self.header['RA_OBJ'], self.header['DEC_OBJ'])

    @property
    def center_xy(self):
        return (self.header['1CRV4P']+16,  self.header['1CRV4P']+16)

    @property
    def origin_xy(self):
        return (self.header['1CRV4P'], self.header['1CRV4P'])

    @property
    def flux(self):
        return self.hdu[1].data['FLUX']

    @property
    def dimensions(self):
        return self.flux.shape

    @property
    def flux_err(self):
        return self.hdu[1].data['FLUX_ERR']

    @property
    def time(self):
        return self.hdu[1].data['TIME']

    @property
    def wcs(self):
        return WCS(self.header)

    @property
    def quality(self):
        sector = self.header['SECTOR']
        A = np.loadtxt(self.eleanor_path + '/metadata/s{0:04d}/quality_s{0:04d}.txt'.format(sector))
        if sector == 65:
            if self.header['CAMERA'] == 4:
                if self.header['CCD'] == 4:
                    A = np.append(A[:6448],A[7910:])
        if len(A) != len(self.flux):
            # Workaround for general problem of
            # https://github.com/afeinstein20/eleanor/issues/267
            warnings.warn(
                f"Num. of cadences mismatch between postcard TPF ({len(self.flux)})"
                f" and sector-wide quality flags ({len(A)})"
                f" for {source_id_str(self)}."
                " Regenerate them."
            )
            A = calc_quality(self.hdu, sector, eleanorpath=self.eleanor_path)
        return A

    @property
    def bkg(self):
        sigma_clip = SigmaClip(sigma=3.)
        bkg = MMMBackground(sigma_clip=sigma_clip)
        b = bkg.calc_background(self.flux, axis=(1,2))
        return b

    @property
    def barycorr(self):
        return self.hdu[1].data['TIMECORR']

    @property
    def ffiindex(self):
        sector = self.header['SECTOR']
        A = np.loadtxt(self.eleanor_path + '/metadata/s{0:04d}/cadences_s{0:04d}.txt'.format(sector))
        if sector == 65:
            if self.header['CAMERA'] == 4:
                if self.header['CCD'] == 4:
                    A = np.append(A[:6448],A[7910:])
        if len(A) != len(self.flux):
            # Workaround for general problem of
            # https://github.com/afeinstein20/eleanor/issues/267
            warnings.warn(
                f"Num. of cadences mismatch between postcard TPF ({len(self.flux)})"
                f" and sector-wide quality ffiindex ({len(A)})"
                f" for {source_id_str(self)}."
                " Regenerate them."
            )
            ffi_filelist = self.hdu[1].data['FFI_FILE']
            A = calc_ffiindex(ffi_filelist, sector)
        return A


def source_id_str(post_obj):
    """Helper to return a short string describing the target's source."""
    h = post_obj.header
    # the post card does not have the TIC ID (at least not from TessCut)
    return f"sector {h.get('SECTOR')}, camera {h.get('CAMERA')}, CCD {h.get('CCD')}"


def calc_quality(ffi_hdu, sector, eleanorpath=None):
    """ Uses the quality flags in a 2-minute target to create quality flags
        for the given postcard.
    """

    # Note: essentially the same code as Update.get_quality(),
    # adapted to be used standalone.

    ffi_time = ffi_hdu[1].data['TIME'] # - ffi_hdu[1].data['TIMECORR']

    if eleanorpath is None:
        eleanorpath = os.path.join(os.path.expanduser('~'), '.eleanor')
    shortCad_fn = eleanorpath + '/metadata/s{0:04d}/target_s{0:04d}.fits'.format(sector)

    # Binary string for values which apply to the FFIs
    if sector > 26:
        ffi_apply = int('100000000010101111', 2)
    else:
        ffi_apply = int('100010101111', 2)

    # Obtains information for 2-minute target
    twoMin     = fits.open(shortCad_fn)
    twoMinTime = twoMin[1].data['TIME'] # - twoMin[1].data['TIMECORR']
    # finite     = np.isfinite(twoMinTime)
    twoMinQual = twoMin[1].data['QUALITY']

    # if you only take the finite values, you're not guaranteeing
    # that the twoMinQual[where - 2:where + 3] is getting 5 contiguous
    # cadences that you expect. Some might have been removed, and you're
    # applying quality flags to the wrong cadence.

    # twoMinTime = twoMinTime[finite]
    # twoMinQual = twoMinQual[finite]

    convolve_ffi = []
    for i in range(len(ffi_time)):
        where = np.where(np.abs(ffi_time[i] - twoMinTime) == np.nanmin(np.abs(ffi_time[i] - twoMinTime)))[0][0]

        sflux = np.sum(ffi_hdu[1].data['FLUX'][i])
        nodata = 0
        if sflux == 0:
            nodata = 131072

        if (ffi_time[i] > 1420) and (ffi_time[i] < 1424):
            nodata = 131072

        if sector < 27:
            v = np.bitwise_or.reduce(twoMinQual[where-7:where+8])
        elif sector < 56:
            # XXX: need to test when TESSCut is available in S27
            v = np.bitwise_or.reduce(twoMinQual[where - 2:where + 3])
        else:
            v = np.bitwise_or.reduce(twoMinQual[where - 5:where + 5])
        convolve_ffi.append(np.bitwise_or(v, nodata))

    convolve_ffi = np.array(convolve_ffi)

    flags = np.bitwise_and(convolve_ffi, ffi_apply)
    return flags


def calc_ffiindex(ffi_filelist, sector):

    # Note: essentially the same code as Update.get_cadences(),
    # adapted to be used standalone.
    from .update import hmsm_to_days, date_to_jd

    if sector < 27:
        # these come from the first FFI cadence of S7, in particular
        # Camera 1, CCD 1 for the t0. The t0s vary by ~1 minute because of
        # barycentric corrections on different cameras
        index_zeropoint = 12680
        index_t0 = 1491.625533688852
    elif sector < 56:
        # first FFI cadence of S27 from Cam 1, CCD 1
        index_zeropoint = 116470
        index_t0 = 2036.283350837239
    else:
        # first FFI cadence of S56 from Cam 1, CCD 1
        index_zeropoint = 690247
        index_t0 = 2825.252246759366

    times = np.array([], dtype=int)
    for line in ffi_filelist:
        # to skip the line "#!/bin/sh", in case the list is from a download .sh file
        if len(str(line)) > 30:
            times = np.append(times, int(str(line).split('tess')[1][0:13]))

    times = np.sort(np.unique(times))

    outarr = np.zeros_like(times)
    for i in range(len(times)):
        date = datetime.strptime(str(times[i]), '%Y%j%H%M%S')
        days = date.day + hmsm_to_days(date.hour, date.minute,
                                        date.second, date.microsecond)
        tjd = date_to_jd(date.year, date.month, days) - 2457000
        if sector < 27:
            cad = (tjd - index_t0)/(30./1440.)
        elif sector < 56:
            cad = (tjd - index_t0)/(10./1440.)
        else:
            cad = (tjd - index_t0) / (200. / (1440.*60))
        outarr[i] = (int(np.round(cad))+index_zeropoint)

    return outarr
