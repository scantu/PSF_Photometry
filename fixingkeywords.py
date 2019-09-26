## Sarah Cantu 04/18/2019
## made for Grus I and IndII photometry project
##
## Takes a multiextension fits file and replaces RADECSYS
## with RADESYS. Should work on a single fits image, but 
## only if it has a primary hdu as well
## Can do multiple images at the same time
##
## 04/29/2019
## Fixed to work with either single fits images or MEX
## Needs to be fixed to except upper case MEX input

from astropy.io import fits
from pathlib import Path

test = 0
while test == 0:
    MEX = None
    coords = str(input('what coordinate system? '))
    MEX = str(input('Is MEX "true" or "false"? '))

    if MEX == 'false':
        filepath = Path(input('what directory are the images in? '))
        filename = input('what are the image filenames? ')
        hdus = sorted(filepath.glob(filename))

        for hdu in hdus:
            print(hdu)
            fits.setval(hdu, 'RADESYS', value=coords,comment=
                            'Coordinate System (keyword changed by SCantu)',
                            before='RADECSYS')
            fits.delval(hdu, 'RADECSYS')
            
    elif MEX == 'true':
        
        filepath = Path(input('where are the MEX files? '))
        filename = input('what is the image filename? ')
        hdus = sorted(filepath.glob(filename))
        for hdu in hdus:
            print(hdu)
            for i in range(len(fits.info(hdu, output=False))-1):
                fits.setval(hdu, 'RADESYS', value=coords,comment=
                            'Coordinate System (keyword changed by SCantu)',
                            before='RADECSYS', ext=i+1)
        
                fits.delval(hdu, 'RADECSYS', ext=i+1)
                print('ext '+str(i+1))
#         print(!pwd)

            
    test = int(input("If there are more images enter 0, otherwise enter any number. "))
