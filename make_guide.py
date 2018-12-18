import sys
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os
import glob
import tqdm


def postcard_names(loc):
    return list(glob.glob(os.path.join(loc, "*.fits")))


def get_headers(cards):
    for i in tqdm.tqdm(range(len(cards)), total=len(cards)):
        hdr = fitsio.read_header(cards[i], 1)

        # Initiate table using first postcard
        if i == 0:
            names, counts = [], []
            hdrKeys = list(hdr.keys())
            dtype = []
            for k in range(len(hdrKeys)):
                if hdrKeys[k] not in names:
                    names.append(hdrKeys[k])
                    counts.append(k)
            names.append('POSTNAME')

            counts = np.array(counts)

            row = [hdr[k] for k in hdrKeys]
            row.append(os.path.split(cards[i])[-1])
            row = np.array(row)
            for r in range(len(row[counts])+1):
                dtype.append('S60')
            t = Table(names=names, dtype=dtype)

        row = np.array([hdr[k] for k in hdrKeys])
        row = row[counts]
        row = np.append(row, os.path.split(cards[i]))
        t.add_row(row)
    return t


if len(sys.argv) > 1:
    dirname = sys.argv[1]
else:
    dirname = "."

postcards = postcard_names(dirname)
table = get_headers(postcards)

ascii.write(table, os.path.join(dirname, 'postcard.guide'), overwrite=True)
