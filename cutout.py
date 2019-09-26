import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyregion
import matplotlib.cm as cm
from photutils import find_peaks
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy import units as u

# cutout2d(data, position, size) where position=(x,y) and size=(ny,nx)=(h,w)

filename = './Wed0529/gswarp.fits'
image = fits.open(filename)[0]
wcs = wcs.WCS(image.header)

region_name = '/home/scantu/092018_ufdg_followup/psfphot/grui/gband/ds9.reg'
r = pyregion.open(region_name)
x, y, w, h, ang = r[-1].coord_list
print(x,y,w,h, ang)
position = (x,y)
size = (h,w)
cutout = Cutout2D(image.data, position=position, size=size, wcs=wcs)
image.data = cutout.data
image.header.update(cutout.wcs.to_header())
cutout_filename = './Wed0529/gwarpcutout.fits'
image.writeto(cutout_filename,overwrite=True)

# issues issues issues issues == so the ra/dec region file still loads properly... but not the xy-coords (which is how i made the cutout) maybe cuz the wcs changed

# new_image = fits.PrimaryHDU()
# mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# tbl = find_peaks(data, threshold, box_size=50)

