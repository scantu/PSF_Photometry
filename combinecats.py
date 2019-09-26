import numpy as np
from astropy.io import fits, ascii
from photutils import find_peaks
from astropy.wcs import WCS
from regions import CircleSkyRegion, write_ds9
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from pathlib import Path
from astropy.table import Table

filepath = Path(input('where are the .dat files? '))
cats = sorted(filepath.glob('indii*dat'))

df = []
for als in cats:
    df.append(ascii.read(als, data_start=1))
    
df.write('./test.dat')
    