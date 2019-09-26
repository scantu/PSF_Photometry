import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
import pyregion
import matplotlib.cm as cm
from photutils import find_peaks
from astropy import wcs
from astropy.nddata import Cutout2D
from astropy import units as u

filename = './rcutout.fits'
image = fits.open(filename)

try:
    from astropy.wcs import WCS
    from astropy.visualization.wcsaxes import WCSAxes

    wcs = WCS(image[0].header)
    fig = plt.figure()
    ax = WCSAxes(fig, [.1,.1,.8,.8], wcs=wcs)
    fig.add_axes(ax)
except ImportError:
    ax = plt.subplot(111)

ax.imshow(image[0].data, cmap=cm.gray, vmin=0, vmax=0.00038, origin='lower')

region_name = './satpeaks4.reg'
r = pyregion.open(region_name).as_imagecoord(header=image[0].header)

from pyregion.mpl_helper import properties_func_default

def fixed_color(shape, saved_attrs):
    attr_list, attr_dict = saved_attrs
    attr_dict["color"] = "red"
    kwargs = properties_func_default(shape, (attr_list,attr_dict))

    return kwargs

r1 = pyregion.ShapeList([rr for rr in r if rr.attr[1].get("tag") == "Group 1"])
patch_list1, artist_list1 = r1.get_mpl_patches_texts(fixed_color)
r2 = pyregion.ShapeList([rr for rr in r if rr.attr[1].get("tag") != "Group 1"])
patch_list2, artist_list2 = r2.get_mpl_patches_texts(fixed_color)

for p in patch_list1 + patch_list2:
    ax.add_patch(p)
for t in artist_list1 + artist_list2:
 0 0 0 0a0x0.0a0d00d_0a00

.00000000000000000000000rtist(t)

plt.show()

mymask = r.get_mask(hdu=image[0])
# myfilter = r.get_filter()

inverted_mask = np.invert(mymask)
new_mask = inverted_mask.astype(int)
final_data = new_mask * image[0].data

primary_hdu = fits.PrimaryHDU(data=final_data, header=image[0].header)
primary_hdu.writeto('rmasked3.fits')


