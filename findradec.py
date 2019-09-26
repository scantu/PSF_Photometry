## Sarah A. Cantu 04/29/2019
## made for GrusI and IndII photometry project
##
## This find ra and dec after the keywords in the image are fixed

import astropy.wcs as wcs
import numpy as np
from astropy.io import fits, ascii
from pathlib import Path

test = 0
while test == 0:
    
    filepath = Path(input('what directory are we working in? '))
    filename = input('what are the image names? ' )
    catalogs = input('what are the .alf names? ' )

    for files, cats in zip(sorted(filepath.glob(filename)), sorted(filepath.glob(catalogs))):
        df = ascii.read(cats, data_start=2, names=('starid', 'x', 'y', 'mag', 'err', 'sky', 'iter',
                                                   'chi', 'sharp'))
        print(files)
        header = fits.getheader(files)
        w = wcs.WCS(header)

        xy = np.column_stack((df['x'], df['y']))
        dec = w.wcs_pix2world(xy, 1)
        df['ra'] = dec[:,0]
        df['dec'] = dec[:,1]
        df.write(cats.with_suffix('.coords'), format='ascii')
    test = input('if there are any more files/images, enter 0? ')
