import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import fitsio
import os


def postcard_names(loc):
    files = os.listdir(loc)
    files = [i for i in files if 'postcard' in i]
    return np.array(files)


def get_headers(cards):
    colnames = np.array(['SECTOR', 'CAMERA', 'CCD', 'POSTPIX1', 'POSTPIX2',
                         'POST_H', 'POST_W', 'CEN_X', 'CEN_Y', 'POSTNAME'])

    t = Table(names=colnames, dtype=('f4', 'f4', 'f4', 'f4', 
                                     'f4', 'f4', 'f4', 'f4', 
                                     'f4', 'S60'))
    for c in cards:
        data, header = fitsio.read(c, 1, header=True)
        row = []
        for n in colnames[0:len(colnames)-1]:
            row.append(header[n])
        row.append(c)
        t.add_row(row)
    return t


postcards = postcard_names('.')
table = get_headers(postcards)

ascii.write(table, 'postcard.guide')
