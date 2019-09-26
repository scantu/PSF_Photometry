import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyregion
import matplotlib.cm as cm
from photutils import find_peaks
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy import units as u
from pyregion.mpl_helper import properties_func_default 
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import WCSAxes
from pathlib import Path
# open fits image
test = 0
while test == 0:
    try:
        filename = Path(input("Enter path to file: "))
        rfile = Path(input("enter region file path: "))
    except (KeyError, TypeError, FileNotFoundError):
        filename = str(input("perhaps a typo, try again"))

    var = input('everything or small set? ')
    names = str(input('which file group? '))
    
    if var == 'todos':
        images = sorted(filename.glob(names))
    elif var == 'small':
        images = ['indiir8032_10.fits', 'indiir8032_15.fits', 'indiir8032_16.fits', 'indiir8032_22.fits', 'indiir8032_24.fits', 'indiir8032_28.fits', 'indiir8032_31.fits', 'indiir8032_34.fits', 'indiir8032_5.fits', 'indiir8032_8.fits']
    
    region_name = sorted(rfile.glob(input("What's the region file name(s)? ")))

    print(region_name)    
#     print(region_name)
    
    for namess in images:
#         names = filename.parent.joinpath(namess)
        names = namess
        image = fits.open(names)
        data = image[0].data
        wcs = WCS(image[0].header)

# this is the bit that actually creates the final fits image
# so this file could be streamlined without all the plotting stuff
# for example keeping something like lines: 13, 14, 32, 60-67
        for regs in region_name:
            r = pyregion.open(regs).as_imagecoord(header=image[0].header)
            mymask = r.get_mask(hdu=image[0])
            new_mask = mymask.astype(int)
            new_mask[new_mask==1] = -100000
        #new_mask[new_mask==0] = 1
            print(names)
            final_data = new_mask + image[0].data
            final_data = final_data.astype(np.float32)
            image[0].header['BITPIX'] = -32
            primary_hdu = fits.PrimaryHDU(data=final_data, header=image[0].header)
            primary_hdu.writeto(names.parent.joinpath(names.stem+names.suffix), overwrite=True)
            
            print(regs)
    test = int(input("If there is another image, enter 0. If done, enter any key. "))

