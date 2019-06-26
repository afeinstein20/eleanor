import sys
import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table
import os, tqdm
from glob import glob


def get_headers(cards, exists=False, t=None):
    for i in tqdm.tqdm(range(len(cards)), total=len(cards)):
        try:
            hdr = fits.open(cards[i])[1].header
            key = 'TMOFST{0}{1}'.format(hdr['CAMERA'], hdr['CCD'])

            sector = os.path.split(cards[i])[-1].split("-")[1]

            # Initiate table using first postcard
            if i == 0 and not exists:
                names, counts = [], []
                hdrKeys = [k for k in hdr.keys() if not k.startswith('TMOFST')]
                for k in range(len(hdrKeys)):
                    if hdrKeys[k] not in names:
                        names.append(hdrKeys[k])
                        counts.append(k)
                # names.append('SECTOR')
                names.append('POSTNAME')
                names.append('TMOFST')

                counts = np.array(counts)

                dtype = ['S60' for _ in range(len(names))]
                t = Table(names=names, dtype=dtype)

            elif i == 0 and exists:
                names = t.colnames
                counts = np.arange(len(names))
                hdrKeys = [k for k in hdr.keys() if not k.startswith('TMOFST')]

            row = np.array([hdr.get(k, 0) for k in hdrKeys])
            row[hdrKeys.index('COMMENT')] = row[hdrKeys.index('COMMENT')][0]
            row = row[counts]
            # row = np.append(row, sector)
            row = np.append(row, os.path.split(cards[i])[1])
            row = np.append(row, hdr[key])
            print(row)
            if exists:
                sec_ind = [i for i in np.arange(0,len(names),1) if 'SECTOR' in names[i]][0]
                cam_ind = [i for i in np.arange(0,len(names),1) if 'CAMERA' in names[i]][0]
                ccd_ind = [i for i in np.arange(0,len(names),1) if 'CCD'    in names[i]][0]
                file_check = [row[sec_ind], int(row[cam_ind]), int(row[ccd_ind])]

                checker = 0
                for j in range(len(t)):
                    check = [t['SECTOR'][j], t['CAMERA'][j], t['CCD'][j]]
                    if check==file_check:
                        checker += 1
                if checker == 0:
                    t.add_row(row)

            else:
                t.add_row(row)

        except:
            print(key, cards[i])
            raise
    return t


if len(sys.argv) > 1:
    # directory to place the output file
    dirname = sys.argv[1]
    # location of the postcards for this sector.
    ffidir = sys.argv[2]
else:
    raise Exception('must be called from command line')

fn_output = 'postcard.guide'
fn_path = os.path.join(dirname, fn_output)
exists = os.path.exists(fn_path)
if exists:
    os.remove(fn_path)
    exists = False

ffis = glob(os.path.join(ffidir, '*', '*eleanor*postcard*fits'))

if exists:
    t = Table.read(fn_path, format='ascii')
    table = get_headers(ffis, exists=exists, t=t)
else:
    table = get_headers(ffis)

ascii.write(table, fn_path, overwrite=True)

t = Table.read(fn_path, format='ascii')
print(len(t['SECTOR']))
