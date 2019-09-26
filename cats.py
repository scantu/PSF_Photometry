## Sarah Cantu 2018/2019 
## made for Grus I and IndII photometry project
##
## this script reads in fits image and phot files
## created by apcorrection.py with_suffix('.ins') 
## .ins for instrumental mags
## pixel coords are converted to ra/dec w/ header WCS
## final catalog files saved as '.conv'

import numpy as np
from astropy.table import Table, join, vstack
import astropy.wcs as wcs
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from pathlib import Path


direc = Path(input("what directory? "))
file = sorted(direc.glob(input("what are the photometry files? ")))
pic = sorted(direc.glob(input("what are the pic files? ")))

maxid = []

for ids in file:
    df = ascii.read(ids)
    maxid.append(max(df['starid']))
    
count=max(maxid)+10000

for f, p in zip(file, pic):
    
    tbdata = ascii.read(f)
    tbdata['starid'] = tbdata['starid'] + count
    header = fits.open(p)[0].header
    w = wcs.WCS(header)
    count += 10000
    xy = np.column_stack((tbdata['x'], tbdata['y']))
    coord = w.wcs_pix2world(xy, 1)

    tbdata['ra'] = coord[:,0].tolist()
    tbdata['dec'] = coord[:,1].tolist()
    
    print(f.stem, p.stem)
    ascii.write(tbdata, f.with_suffix('.conv'), overwrite=True)

