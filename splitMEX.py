import numpy as np
from astropy.io import fits
from photutils import find_peaks
from astropy.wcs import WCS
from regions import CircleSkyRegion, write_ds9
from astropy.coordinates import Angle, SkyCoord
import astropy.units as u
from pathlib import Path


filename = Path(input("Enter path to file and filename: "))

images = sorted(filename.glob(input("What's the image name form? ")))
print(images)
for names in images:
        
    image = fits.open(names)

    a = np.arange(int(image[1].header['MEXTNO']), int(image[-1].header['MEXTNO'])+1)
    print(a)
    for ext in a:
            
#         image[ext].header['RADESYSa'] = image[ext].header['RADECSYS']
#         del image[ext].header['RADECSYS']

        cutout_filename = names.parent.joinpath(names.stem+'_'+str(ext)+names.suffix)
        primary_hdu = fits.PrimaryHDU(data=image[ext].data, header=image[ext].header)#, overwrite=True)
        primary_hdu.writeto(cutout_filename, overwrite=True)
        
    #test = input("If there is another image, enter 0. If done, enter any key. ")
# issues issues issues issues == so the ra/dec region file still loads properly... but not the xy-coords (which is how i made the cutout) maybe cuz the wcs changed

# new_image = fits.PrimaryHDU()
# mean, median, std = sigma_clipped_stats(data, sigma=3.0)

# tbl = find_peaks(data, threshold, box_size=50)

