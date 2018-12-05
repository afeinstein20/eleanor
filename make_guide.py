import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os


def postcard_names(loc):
    files = os.listdir(loc)
    files = [i for i in files if '.fits' in i]
    return np.array(files)


def get_headers(cards):
    for i in range(len(cards)):
        hdu = fits.open(cards[i])
        hdr = hdu[1].header

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
            counts.append(k+1)
            counts = np.array(counts)
            
            row = list(hdr.values())
            row.append(cards[i])
            row = np.array(row)
            for r in row[counts]:
                dtype.append('S60')
            t = Table(names=names, dtype=dtype)

        row = list(hdr.values())
        row.append(cards[i])
        row=np.array(row)
        row = row[counts]
        t.add_row(row)
    return t


postcards = postcard_names('.')
table = get_headers(postcards)

ascii.write(table, 'postcard.guide')
