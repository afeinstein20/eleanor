#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function

import os
import glob
import tqdm
import fitsio
import numpy as np
from functools import partial
from sklearn.decomposition import PCA
from scipy import interpolate
from scipy.signal import medfilt2d as mf
from scipy.interpolate import interp1d
from multiprocessing import Pool

from eleanor import fill_grid

# Pre-compute the low pass filter
fc = 0.12
b = 0.08
N = int(np.ceil((4 / b)))
if not N % 2:
    N += 1
n = np.arange(N)
sinc_func = np.sinc(2 * fc * (n - (N - 1) / 2.))
window = 0.42 - 0.5 * np.cos(2 * np.pi * n / (N - 1))
window += 0.08 * np.cos(4 * np.pi * n / (N - 1))
sinc_func = sinc_func * window
sinc_func = sinc_func / np.sum(sinc_func)

def lowpass(vec):
    new_signal = np.convolve(vec, sinc_func)
    lns = len(new_signal)
    diff = int(np.abs(lns - len(vec))/2)  # low pass filter to remove high order modes
    return new_signal[diff:-diff]

def do_pca(i, j, data, factor, q):
    xvals = np.dot(factor, data[i,j,:][q])
    return xvals

def calc_2dbkg(flux, qual, time, fast=True):
    q = qual == 0

    med = np.percentile(flux[:,:,:], 1, axis=(2))   # build a single frame in shape of detector. This was once the median image, not sure why it was changed

    med = med-mf(med, 21) # subtract off median to remove stable background emission, which is especially apparent in corners
    g = np.ma.masked_where(med < np.percentile(med, 70.), med)  # mask which should separate pixels dominated by starlight from background

    modes = 21
    X = flux[g.mask][:, q]
    pca = PCA(n_components=modes)
    pca.fit(X)
    pv = pca.components_[0:modes]

    vv = np.column_stack((pv))
    for i in range(-15, 16, 3):
        if i != 0:
            if i > 0:
                rolled = np.pad(pv, ((0,0),(i,0)), mode='constant')[:, :-i].T
            else:
                rolled = np.pad(pv, ((0,0),(0,-i)), mode='constant')[:, -i:].T # assumption: background is a linear sum of what is seen on pixels on the
                                                                                 # detector. Because the earthshine varies with axis, this can be time-shifted
                                                                                 # so that individual pixels see the same response at different times
            vv = np.column_stack((vv, rolled))

    vv = np.column_stack((vv, np.ones_like(vv[:,0])))
    factor = np.linalg.solve(np.dot(vv.T, vv), vv.T)

    maskvals = np.ma.masked_array(data=np.zeros((104,148,np.shape(vv)[1])),
                                  mask=np.zeros((104,148,np.shape(vv)[1])))

    for i in range(len(g)):
        for j in range(len(g[0])):
            if g.mask[i][j] == True:
                maskvals.data[i, j] = do_pca(i, j, flux, factor, q)  # build eigenvectors to model the background at each pixel

    noval = maskvals.data[:,:,:] == 0
    maskvals.mask[noval] = True

    outmeasure = np.zeros_like(maskvals[:,:,0])
    for i in range(len(maskvals[0,0])):
        outmeasure += (np.abs(maskvals.data[:,:,i]-np.nanmean(maskvals.data[:,:,i])))/np.nanstd(maskvals.data[:,:,i])

    metric = outmeasure/len(maskvals.data[0,0]) # these eigenvectors give a bad fit when something astrophysical happens to these pixels, like a bright
                                                # asteroid crossing the FOV. Isolate these pixels and add them into the mask of pixels we don't use
    maskvals.mask[metric > 1.00,:] = True

    x = np.arange(0, maskvals.data.shape[1])
    y = np.arange(0, maskvals.data.shape[0])

    xx, yy = np.meshgrid(x, y)

    GD = np.copy(maskvals.data)
    if fast:
        count = np.zeros(GD.shape[:2], dtype=np.int32)
    for i in range(np.shape(vv)[1]):
        array = np.ma.masked_invalid(maskvals[:,:,i])

        if fast:
            fill_grid.fill_grid(GD[:, :, i], ~array.mask, count)

        else:
            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]


            GD[:,:,i] = interpolate.griddata((x1, y1), newarr.ravel(),  # take the masked pixels (where the stars are, in theory) and interpolate the
                                    (xx, yy),                         # inferred background onto these pixels
                                        method='linear')

            array = np.ma.masked_invalid(GD[:,:,i])
            xx, yy = np.meshgrid(x, y)

            x1 = xx[~array.mask]
            y1 = yy[~array.mask]
            newarr = array[~array.mask]

            GD[:,:,i] = interpolate.griddata((x1, y1), newarr.ravel(), # in the corners, just extend the background based on the nearest pixel. This shouldn't
                                    (xx, yy),                         # be too important since these pixels are in theory closer to the center of another postcard
                                    method='nearest')               # in which case inferring the background is going to be a challenge regardless

    bkg_arr = np.zeros_like(flux)
    bkg_arr[:, :, q] = np.dot(GD, vv.T)
    for i in range(104):
        for j in range(148):
            bkg_arr[i,j,q] = lowpass(bkg_arr[i,j,q])

    f = interp1d(time[q], bkg_arr[:,:,q], kind='linear', axis=2, bounds_error=False, fill_value='extrapolate') # then interpolate the background smoothly across
    fout = f(time)                                                                                             # all times, where we've only been learning with the
                                                                                                               # good quality cadences

    return fout

def do_background(fn, fast=True, suffix="_bkg"):
    base, ext = os.path.splitext(fn)
    new_filename = base + suffix + ext

    with fitsio.FITS(fn) as fits:
        hdr = fits[1].read_header()
        tbl = fits[1].read()
        bkg = calc_2dbkg(fits[2].read(), tbl['QUALITY'], tbl['TSTART'],
                         fast=fast)

    hdr["BACKGROUND"] = True
    fitsio.write(new_filename, bkg, header=dict(hdr), clobber=True)

def run_sector_camera_chip(base_dir, sector, camera, chip, threads=1,
                           fast=True, suffix="_bkg"):
    postdir = os.path.join(base_dir, "s{0:04d}".format(sector),
                           "{0:d}-{1:d}".format(camera, chip))
    pattern = os.path.join(
        postdir,
        "hlsp_eleanor_tess_ffi_postcard-s{0:04d}-{1:d}-{2:d}-cal-*-*[0-9].fits"
        .format(sector, camera, chip))

    # pattern = os.path.join(postdir, "hlsp_eleanor_*.fits")

    fns = list(sorted(glob.glob(pattern)))

    # Ensures no postcards have been repeated
    fns = np.unique(fns)

    # Writes in the background after making the postcards
    print("Computing backgrounds...")
    func = partial(do_background, fast=fast, suffix=suffix)
    if threads > 1:
        with Pool(threads) as pool:
            list(tqdm.tqdm(pool.map(func, fns), total=len(fns)))
    else:
        list(tqdm.tqdm(map(func, fns), total=len(fns)))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create backgrounds from a list of postcards")
    parser.add_argument('sector', type=int,
                        help='the sector number')
    parser.add_argument('base_dir', type=str,
                        help='the base data directory')
    parser.add_argument('--camera', type=int, default=None,
                        help='the camera number')
    parser.add_argument('--chip', type=int, default=None,
                        help='the chip number')
    parser.add_argument('--threads', type=int, default=1,
                        help='the number of threads to run in parallel')
    parser.add_argument('--slow', action="store_true",
                        help='use the old slower version of the interpolation')
    parser.add_argument('--suffix', type=str, default="_bkg",
                        help='use the old slower version of the interpolation')
    args = parser.parse_args()

    if args.camera is None:
        cameras = [1, 2, 3, 4]
    else:
        assert int(args.camera) in [1, 2, 3, 4]
        cameras = [int(args.camera)]

    if args.chip is None:
        chips = [1, 2, 3, 4]
    else:
        assert int(args.chip) in [1, 2, 3, 4]
        chips = [int(args.chip)]

    for camera in cameras:
        for chip in chips:
            print("Running {0:04d}-{1}-{2}".format(args.sector, camera, chip))
            run_sector_camera_chip(args.base_dir,
                                   args.sector, camera, chip,
                                   threads=args.threads, fast=not args.slow,
                                   suffix=args.suffix)
