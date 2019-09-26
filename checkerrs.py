import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from opensave import sigd
from pathlib import Path

filepath = Path(input('where are the .als files? '))
filename = input('what is the file group? ')
print(filepath)

alss = sorted(filepath.glob(filename))

for als in alss:
    print(als)
    tb = ascii.read(als, data_start=2, names=('starid', 'x', 'y','mag','err',
                                              'sky', 'iter', 'chi', 'sharp'))
    plt.figure(figsize=(16,14))
    plt.subplot(211)
    plt.hist(tb['err'], bins='auto', label=als.stem)
    plt.xlabel('err')
    plt.legend(loc='best')
    plt.tight_layout()
    
    plt.subplot(212)
    plt.plot(tb['mag'], tb['err'], 'ko',label=als.stem)
    plt.xlabel('mag')
    plt.ylabel('err')
    plt.legend(loc='best')
    plt.tight_layout()
    plt.show()
