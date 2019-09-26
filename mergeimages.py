import numpy as np
from astropy.table import Table, join, vstack, hstack
import astropy.wcs as wcs
from astropy.io import fits, ascii
import matplotlib.pyplot as plt
from pathlib import Path
import re
import matplotlib.pyplot as plt
from astropy import units as u
from astropy.coordinates import SkyCoord
import pickle

direc = Path(input("Where are the files? "))
file = sorted(direc.glob(input("what are the photometry files? ")))

maxid = []

for ids in file:
    df = ascii.read(ids)
    maxid.append(len(df))
    
mtbd = ascii.read(file[maxid.index(max(maxid))])
coo_mega = SkyCoord(mtbd['ra']*u.deg, mtbd['dec']*u.deg)
tbd = {}
for f in file:
    
    dtbd = ascii.read(f)
    coo_des = SkyCoord(dtbd['ra']*u.deg, dtbd['dec']*u.deg)
    idx_mega, d2d_mega, d3d_mega = coo_des.match_to_catalog_sky(coo_mega)
    dtbd['uid'] = mtbd[idx_mega]['starid']
    dtbd['d2d'] = d2d_mega
    tbd[f.stem] = dtbd

    
    
pickle.dump(tbd, open(file[0].stem[:-4]+'.all', 'wb'))

