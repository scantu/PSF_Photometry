import numpy as np
from astropy.io import fits
from photutils import find_peaks
from astropy.wcs import WCS
from regions import CircleSkyRegion, write_ds9
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from pyregion.mpl_helper import properties_func_default 
import pyregion
from pathlib import Path

test = 0
ang = 2.*u.arcsec
radius = Angle(ang.to('deg'))

while test == 0:
    try:
        filename = Path(input("Enter path to file: "))
    except (KeyError, TypeError, FileNotFoundError):
        filename = Path(input("perhaps a typo, try again"))
    names = input('image name? ')
    images = sorted(filename.glob(names))
    for names in images:
        image = fits.open(names)
        data = image[0].data
        wcs = WCS(image[0].header)
        
        # try:
        #     threshold = image[0].header['SATURATE']
        # except (KeyError, TypeError):
        #     threshold = 40000
        threshold = 55000    
        tbl = find_peaks(data, threshold, box_size=10, wcs=wcs)
        regions = []
    
    
        center = tbl['skycoord_peak']   
        for i in range(len(tbl)):
            region = CircleSkyRegion(center[i],radius)
            region.visual['color'] = 'cyan'
            region.visual['width'] = '3'
            regions.append(region)
        write_ds9(regions, names.parent.joinpath('saturated'+names.stem+'.reg'))
        print(names)
        image.close()
    test = input("If there is another image, enter 0. If done, enter any key. ")
    
