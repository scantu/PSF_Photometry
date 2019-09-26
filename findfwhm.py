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
from imexam.imexamine import Imexamine
from imexam.math_helper import mfwhm, gfwhm
from astropy.io import ascii
test = 0
while test == 0:
    try:
        filename = Path(input("Enter path to file: "))
        rfile = Path(input("enter path to region file: "))
    except (KeyError, TypeError, FileNotFoundError):
        filename = str(input("perhaps a typo, try again"))
        
    images = sorted(filename.glob('indiir1_20*.fits'))
    region_name = sorted(rfile.glob('indiir1_*.coo'))
    plots=Imexamine()
    
    for names,stars in zip(images, region_name):
        data=fits.getdata(names)
        plots.set_data(data)
        starlist = ascii.read(stars, data_start=3)
        plots.line_fit_pars['func'] = ['Moffat1D']
        plots.column_fit_pars['func'] = ['Moffat1D']    
        results = dict()
        yresults = dict()
        for star in starlist:


            sid,x,y,a4,a5,a6,a7=star
            moff = plots.line_fit(x,y,genplot=False)
            ymoff = plots.column_fit(x,y,genplot=False)
#            print(moff)
            results[x] = mfwhm(alpha=moff.alpha_0,gamma=moff.gamma_0)
            yresults[y] = mfwhm(alpha=ymoff.alpha_0,gamma=ymoff.gamma_0)
#            gresults[x] = gfwhm(moff.stddev_0)[0]
#            gyresults[y] = gfwhm(ymoff.stddev_0)[1]
            print(names,np.median(list(results.values())), np.median(list(yresults.values())))
#        plt.figure()
#        plt.subplot(121)
#        plt.hist(yresults.values(), bins='auto', alpha=.7, range=(3.,7.5))
#        plt.hist(results.values(), bins='auto', alpha=.7, range=(3.,7.5))
        
#        plt.subplot(122)
 #       plt.hist(gyresults.values(), bins='auto', alpha=.7, range=(3.,7.5))
 #       plt.hist(gresults.values(), bins='auto', alpha=.7, range=(3.,7.5))
#        plt.show()
        
        