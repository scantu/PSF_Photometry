## Sarah A. Cantu
## 04/24/2019
## I should be writing a letter to a middle schooler
## I want to remove the stars that aren't in .nei from the .lso files (cuz they're usually bad right?)

import numpy as np
from astropy.table import Table
from astropy.io import ascii
from opensave import sigd
from pathlib import Path

filepath = Path(input('what directory are the files in? '))
filename = input('what is the nei filename(s)? ')
filenamel = input('what is the lso filename(s)? ')


nei = sorted(filepath.glob(filename))
lsos = sorted(filepath.glob(filenamel))

for neifiles,files in zip(nei,lsos):
    print(files)
    t, rm, cut = None, None, None
    
    t = ascii.read(neifiles, data_start=2, names=('starid', 'x', 'y', 'mag', 'sky'))
    rm = ascii.read(files, data_start=2, names=('starid', 'x', 'y', 'mag', 'chi'))
    cut = rm[np.where(np.isin(rm['starid'], t['starid']))]
    print(len(rm)-len(cut))
    
    fp = open(files, 'r')
    lines = fp.readlines()
    fp.close()


    fp = open(files.parent.joinpath('new/'+str(files.name)), 'w')

    for l in lines:
        if l.strip().startswith(tuple(str(ids) for ids in cut['starid'])):
            fp.write(l)
        else:
            pass
    fp.close()

    
