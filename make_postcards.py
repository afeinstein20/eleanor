#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

__all__ = ["make_postcards"]

import os
import glob
import tqdm
import fitsio
import numpy as np
from time import strftime
from astropy.wcs import WCS
from astropy.table import Table
from astropy.stats import SigmaClip
from photutils import MMMBackground
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.interpolate import interp1d


from eleanor.ffi import ffi, set_quality_flags
#from eleanor.version import __version__


def bkg(flux, sigma=2.5):
    # Returns background for a single cadence. Default sigma=2.5
    sigma_clip = SigmaClip(sigma=sigma)
    bkg = MMMBackground(sigma_clip=sigma_clip)
    return bkg.calc_background(flux)

def do_pca(i, j, data, vv, q):
    xvals = xhat(vv[:], data[i,j,:][q]) # does the regression
    #cm = np.column_stack((pca.components_[0:modes,q].T, np.ones_like(pca.components_[0,q])))
    fmod = fhat(xvals, vv) # builds a predicted flux at each cadence from the regression (centered around zero)
    lc_pred = (fmod+1) # now centered around 1
    return xvals

def xhat(mat, lc):
    ATA = np.dot(mat.T, mat)
    ATAinv = np.linalg.inv(ATA)
    ATf = np.dot(mat.T, lc)
    xhat = np.dot(ATAinv, ATf)
    return xhat

def fhat(xhat, data):
    return np.dot(data, xhat)

def calc_2dbkg(flux, qual, time):
    q = qual == 0
    med = np.nanmedian(flux[:,:,:], axis=(2))
    g = np.ma.masked_where(med < np.percentile(med, 50.), med)
    
    pca = PCA(n_components=5)
    pca.fit(flux[g.mask])
    
    modes = 3
    pv = pca.components_[0:modes].T[q]

    vv = np.column_stack((pv.T))

    for i in range(-19, 19, 6):
        if i != 0:
            if i > 0:
                rolled = np.pad(pv.T, ((0,0),(i,0)), mode='constant')[:, :-i].T
            else:
                rolled = np.pad(pv.T, ((0,0),(0,-i)), mode='constant')[:, -i:].T
            vv = np.column_stack((vv, rolled))

    vv = np.column_stack((vv, np.ones_like(vv[:,0])))

    maskvals = np.zeros((104, 148, np.shape(vv)[1]))
    GD = np.zeros_like(maskvals)
    for i in range(len(g)):
        for j in range(len(g[0])):
            if g.mask[i][j] == True:     
                maskvals[i,j] = do_pca(i,j, flux, vv, q)
                
    noval = maskvals[:,:,:] == 0
    maskvals[noval] = np.nan
    
    outmeasure = np.zeros_like(maskvals[:,:,0])
    for i in range(len(maskvals[0,0])):
        outmeasure += (np.abs(maskvals[:,:,i]-np.nanmean(maskvals[:,:,i])))/np.nanstd(maskvals[:,:,i])

    metric = outmeasure/len(maskvals[0,0])
    maskvals[metric > 1.00,:] = np.nan
        

    x = np.arange(0, maskvals.shape[1])
    y = np.arange(0, maskvals.shape[0])    

    xx, yy = np.meshgrid(x, y)


    for i in range(np.shape(vv)[1]):
        array = np.ma.masked_invalid(maskvals[:,:,i])
        #get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]
        

        GD[:,:,i] = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='linear')

        array = np.ma.masked_invalid(GD[:,:,i])
        xx, yy = np.meshgrid(x, y)
        #get only the valid values
        x1 = xx[~array.mask]
        y1 = yy[~array.mask]
        newarr = array[~array.mask]

        GD[:,:,i] = interpolate.griddata((x1, y1), newarr.ravel(),
                                  (xx, yy),
                                     method='nearest')
        


    bkg_arr = np.zeros_like(flux)

    for i in range(104):
        for j in range(148):
            bkg_arr[i,j,q] = np.dot(GD[i,j,:], vv.T)

    f = interp1d(time[q], bkg_arr[:,:,q], kind='linear', axis=2, bounds_error=False, fill_value='extrapolate')
    fout = f(time)

    return fout

def make_postcards(fns, outdir, sc_fn, width=104, height=148, wstep=None, hstep=None):
    # Make sure that the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # We'll assume that the filenames can be sorted like this (it is true for
    # the ETE-6 test data
    fns = list(sorted(fns))
    total_ffis = len(fns)
    # Save the middle header as the primary header
    middle_fn = fns[total_ffis//2]
    data, primary_header = fitsio.read(middle_fn, 1, header=True)

    # Add the eleanor info to the header
    primary_header.add_record("COMMENT   ***********************")
    primary_header.add_record("COMMENT   *    eleanor INFO     *")
    primary_header.add_record("COMMENT   ***********************")
    primary_header.add_record(
        dict(name='AUTHOR', value='Adina D. Feinstein'))
    primary_header.add_record(
        dict(name='VERSION', value='0.0.4'))
    primary_header.add_record(
        dict(name='GITHUB',
             value='https://github.com/afeinstein20/eleanor'))
    primary_header.add_record(
        dict(name='CREATED', value=strftime('%Y-%m-%d'),
             comment='eleanor file creation date (YYY-MM-DD)'))

    # Build the WCS for this middle exposure
    primary_wcs = WCS(primary_header)

    # Is this a raw frame? If so, there won't be any error information
    is_raw = primary_header["IMAGTYPE"].strip() == "uncal"

    # Set the output filename format
    sector = os.path.split(middle_fn)[-1].split("-")[1] # Scrapes sector from the filename

    info   = (sector, primary_header["CAMERA"],
              primary_header["CCD"], primary_header["IMAGTYPE"].strip())
    info_str = '{0}-{1}-{2}-{3}'.format(info[0], info[1], info[2], info[3])

    outfn_fmt = "hlsp_eleanor_tess_ffi_postcard-{0}-{{0:04d}}-{{1:04d}}.fits".format(info_str)
    outfn_fmt = os.path.join(outdir, outfn_fmt).format

    # Build the pointing model
    f = ffi(sector=int(info[0][1:]), camera=info[1], chip=info[2])
    f.local_paths = fns
    f.sort_by_date()
    pm = f.pointing_model_per_cadence()

    # We want to shift the WCS for each postcard so let's store the default
    # reference pixel
    crpix_h = float(primary_header["CRPIX1"])
    crpix_w = float(primary_header["CRPIX2"])

    # Work out the dimensions of the problem
    dtype = data.dtype
    shape = data.shape
    total_width, total_height = shape
    width = int(width)
    height = int(height)
    if wstep is None:
        wstep = width - 50
    if hstep is None:
        hstep = height - 50
    wstep = int(wstep)
    hstep = int(hstep)

    # Make a grid of postcard origin coordinates
    ws = np.arange(0, 2049, wstep)#total_width - width + wstep + 1, wstep)
    hs = np.arange(44, 2093, hstep)#total_height - height + hstep + 1, hstep)

    # Compute the total numbers for progress bars
    num_times = len(fns)
    total_num_postcards = len(ws) * len(hs)

    # Allocate the memory for the stacked FFIs
    all_ffis = np.empty((total_width, total_height, len(fns)), dtype=dtype,
                        order="F")
    if not is_raw:
        all_errs = np.empty((total_width, total_height, len(fns)), dtype=dtype,
                            order="F")

    # We'll have the same primary HDU for each postcard - this will store the
    # time dependent header info
    primary_cols = ["TSTART", "TSTOP", "BARYCORR", "DATE-OBS", "DATE-END", "BKG", "QUALITY"]
    primary_dtype = [np.float32, np.float32, np.float32, "O", "O", np.float32, np.int64]
    primary_data = np.empty(len(fns), list(zip(primary_cols, primary_dtype)))

    # Make sure that the sector, camera, chip, and dimensions are the
    # same for all the files
    for i, name in tqdm.tqdm(enumerate(fns), total=num_times):
        data, hdr = fitsio.read(name, 1, header=True)

        # FIXME: when `sector` is added to the header, we should check
        # it too!  -- still not added (dfm)
        new_shape = (hdr["NAXIS2"], hdr["NAXIS1"])
        new_info = (sector, hdr["CAMERA"], hdr["CCD"], hdr["IMAGTYPE"].strip())
        if shape != new_shape or new_info != info:
            raise ValueError("the header info for '{0}' does not match"
                             .format(name))
        info = new_info

        # Save the info for the primary HDU
        for k, dtype in zip(primary_cols[0:len(primary_cols)-2], primary_dtype[0:len(primary_dtype)-2]):
            if dtype == "O":
                primary_data[k][i] = hdr[k].encode("ascii")
            else:
                primary_data[k][i] = hdr[k]


        # Save the data
        all_ffis[:, :, i] = data

        if not is_raw:
            all_errs[:, :, i] = fitsio.read(name, 2)

    wmax, hmax = 2048, 2092

    quality = np.empty(len(fns))

    # Loop over postcards
    with tqdm.tqdm(total=total_num_postcards) as bar:
        for i, h in enumerate(hs):
            for j, w in enumerate(ws):
                dw = width#min(width, total_width - w)
                dh = height#min(height, total_height - h)

                hdr = fitsio.FITSHDR(primary_header)

                if np.shape(all_ffis[w:w+dw, h:h+dh, :]) != (width,height,total_ffis):
                    if w+dw > wmax:
                        w = wmax-dw
                    if h+dh > hmax:
                        h = hmax-dh

                # Shift the reference pixel for the WCS to
                # account for the postcard location
                hdr.add_record(
                    dict(name="CRPIX1", value=crpix_h - h,
                         comment="X reference pixel"))
                hdr.add_record(
                    dict(name="CRPIX2", value=crpix_w - w,
                         comment="Y reference pixel"))

                # Save the postcard coordinates in the header
                hdr.add_record(
                    dict(name="POSTPIX1", value=h,
                         comment="origin of postcard axis 1"))
                hdr.add_record(
                    dict(name="POSTPIX2", value=w,
                         comment="origin of postcard axis 2"))

                xcen = h + 0.5*dh
                ycen = w + 0.5*dw

                outfn = outfn_fmt(int(xcen), int(ycen))

                rd = primary_wcs.all_pix2world(xcen, ycen, 1)
                hdr.add_record(
                    dict(name="CEN_X", value=xcen,
                         comment=("central pixel of postcard in FFI")))
                hdr.add_record(
                    dict(name="CEN_Y", value=ycen,
                         comment=("central pixel of postcard in FFI")))
                hdr.add_record(
                    dict(name="CEN_RA", value=float(rd[0]),
                         comment="RA of central pixel"))
                hdr.add_record(
                    dict(name="CEN_DEC", value=float(rd[1]),
                         comment="Dec of central pixel"))
                hdr.add_record(
                    dict(name="POST_H", value=float(height),
                         comment="Height of postcard in pixels"))
                hdr.add_record(
                    dict(name="POST_W", value=float(width),
                         comment="Width of postcard in pixels"))
                hdr.add_record(
                    dict(name="SECTOR", value=sector[1::],
                         comment="TESS sector"))

                pixel_data = all_ffis[w:w+dw, h:h+dh, :] + 0.0
                
                bkg_array = []

                # Adds in quality column for each cadence in primary_data
                for k in range(len(fns)):
                    b = bkg(pixel_data[:, :, k])
                    primary_data[k][len(primary_cols)-2] = b
                    pixel_data[:, :, k] -= b
                    if i==0 and j==0 and k==0:
                        print("Getting quality flags")
                        quality_array = set_quality_flags( primary_data['TSTART']-primary_data['BARYCORR'],
                                                           primary_data['TSTOP']-primary_data['BARYCORR'],
                                                           sc_fn, sector[1::], new_info[1], new_info[2],
                                                           pm=pm)
                    primary_data[k][len(primary_cols)-1] = quality_array[k]

                grid_bkg = calc_2dbkg(pixel_data, quality_array, primary_data['TSTART'])

                # Saves the primary hdu
                fitsio.write(outfn, primary_data, header=hdr, clobber=True)

                # Save the image data
                fitsio.write(outfn, pixel_data)

                # Save the background data
                fitsio.write(outfn, grid_bkg)

                if not is_raw:
                    fitsio.write(outfn, all_errs[w:w+dw, h:h+dh, :])

                bar.update()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Make postcards from a list of FFIs")
    parser.add_argument('file_pattern',
                        help='the pattern for the input FFI filenames')
    parser.add_argument('output_dir',
                        help='the output directory')
    parser.add_argument('sc_fn',
                        help='the short cadence filename for this sector, camera, chip')
    parser.add_argument('--width', type=int, default=104,
                        help='the width of the postcards')
    parser.add_argument('--height', type=int, default=148,
                        help='the height of the postcards')
    parser.add_argument('--wstep', type=int, default=None,
                        help='the step size in the width direction')
    parser.add_argument('--hstep', type=int, default=None,
                        help='the step size in the height direction')
    args = parser.parse_args()

    fns = sorted(glob.glob(args.file_pattern))
    outdir = args.output_dir
    sc_fn  = args.sc_fn
    make_postcards(fns, outdir, sc_fn,
                   width=args.width, height=args.height,
                   wstep=args.wstep, hstep=args.hstep)
