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

from .version import __version__


def make_postcards(fns, outdir, width=104, height=148, wstep=None, hstep=None):
    # Make sure that the output directory exists
    os.makedirs(outdir, exist_ok=True)

    # We'll assume that the filenames can be sorted like this (it is true for
    # the ETE-6 test data
    fns = list(sorted(fns))
    total_ffis = len(fns)
    # Save the middle header as the primary header
    middle_fn = fns[total_ffis//2]
    data, primary_header = fitsio.read(middle_fn, 1, header=True)

    # Add the ELLIE info to the header
    primary_header.add_record("COMMENT   ***********************")
    primary_header.add_record("COMMENT   *     ELLIE INFO      *")
    primary_header.add_record("COMMENT   ***********************")
    primary_header.add_record(
        dict(name='AUTHOR', value='Adina D. Feinstein'))
    primary_header.add_record(
        dict(name='VERSION', value=__version__))
    primary_header.add_record(
        dict(name='GITHUB',
             value='https://github.com/afeinstein20/ELLIE'))
    primary_header.add_record(
        dict(name='CREATED', value=strftime('%Y-%m-%d'),
             comment='ELLIE file creation date (YYY-MM-DD)'))

    # Build the WCS for this middle exposure
    primary_wcs = WCS(primary_header)

    # Is this a raw frame? If so, there won't be any error information
    is_raw = primary_header["IMAGTYPE"].strip() == "uncal"

    # Set the output filename format
    info = (primary_header["CAMERA"], primary_header["CCD"],
            primary_header["IMAGTYPE"].strip())
    outfn_fmt = "hlsp_ellie_tess_ffi_postcard-{0}-{{0:04d}}-{{1:04d}}.fits".format(
        "-".join(map("{0}".format, info)))
    outfn_fmt = os.path.join(outdir, outfn_fmt).format

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
    primary_cols = ["TSTART", "TSTOP", "BARYCORR", "DATE-OBS", "DATE-END"]
    primary_dtype = [np.float32, np.float32, np.float32]
    primary_data = np.empty(len(fns), list(zip(primary_cols, primary_dtype)))

    # Make sure that the sector, camera, chip, and dimensions are the
    # same for all the files
    for i, name in tqdm.tqdm(enumerate(fns), total=num_times):
        data, hdr = fitsio.read(name, 1, header=True)
        #hdr = fitsio.read_header(name, 1)

        # FIXME: when `sector` is added to the header, we should check
        # it too!
        new_shape = (hdr["NAXIS2"], hdr["NAXIS1"])
        new_info = (hdr["CAMERA"], hdr["CCD"], hdr["IMAGTYPE"].strip())
        if shape != new_shape or new_info != info:
            raise ValueError("the header info for '{0}' does not match"
                             .format(name))
        info = new_info

        # Save the info for the primary HDU
        for k in primary_cols:
            primary_data[k][i] = hdr[k]

        # Save the data
        all_ffis[:, :, i] = data
        if not is_raw:
            all_errs[:, :, i] = fitsio.read(name, 2)


    wmax, hmax = 2048, 2092

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
                    dict(name="SECTOR", value=1,
                         comment="TESS sector (temporary)"))

                # Save the primary HDU
                fitsio.write(outfn, primary_data, header=hdr, clobber=True)

                # Save the image data
                fitsio.write(outfn, all_ffis[w:w+dw, h:h+dh, :])

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
    parser.add_argument('--width', '-w', type=int, default=104,
                        help='the width of the postcards')
    parser.add_argument('--height', '-h', type=int, default=148,
                        help='the height of the postcards')
    parser.add_argument('--wstep', type=int, default=None,
                        help='the step size in the width direction')
    parser.add_argument('--hstep', type=int, default=None,
                        help='the step size in the height direction')
    args = parser.parse_args()

    fns = sorted(glob.glob(args.file_pattern))
    outdir = args.output_dir
    make_postcards(fns, outdir,
                   width=args.width, height=args.height,
                   wstep=args.wstep, hstep=args.hstep)
