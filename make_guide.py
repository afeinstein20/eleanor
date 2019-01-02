import sys
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os
import glob
import tqdm


def postcard_names(loc):
    return list(sorted(glob.glob(os.path.join(loc, "*/*.fits"))))


def get_headers(cards):
    for i in tqdm.tqdm(range(len(cards)), total=len(cards)):
        try:
            hdr = fitsio.read_header(cards[i], 1)
            key = 'TMOFST{0}{1}'.format(hdr['CAMERA'], hdr['CCD'])

            # Initiate table using first postcard
            if i == 0:
                names, counts = [], []
                hdrKeys = [k for k in hdr.keys() if not k.startswith('TMOFST')]
                dtype = []
                for k in range(len(hdrKeys)):
                    if hdrKeys[k] not in names:
                        names.append(hdrKeys[k])
                        counts.append(k)
                names.append('POSTNAME')
                names.append('TMOFST')

                counts = np.array(counts)

                dtype = ['S60' for _ in range(len(names))]
                t = Table(names=names, dtype=dtype)

            row = np.array([hdr[k] for k in hdrKeys])
            row = row[counts]
            row = np.append(row, os.path.split(cards[i])[-1])
            row = np.append(row, hdr[key])
            t.add_row(row)
        except:
            print(key, cards[i])
            raise
    return t


if len(sys.argv) > 1:
    dirname = sys.argv[1]
else:
    dirname = "."

postcards = postcard_names(dirname)
table = get_headers(postcards)

ascii.write(table, os.path.join(dirname, 'postcard.guide'), overwrite=True)
