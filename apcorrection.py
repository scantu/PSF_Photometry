import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from opensave import sigd
from pathlib import Path
import re

filepath = Path(input('where are the files? '))
filename = input('what is the apc filename? ')
filenamel = input('what is the als filename? ')
print(filepath)

apcs = sorted(filepath.glob(filename))
als = sorted(filepath.glob(filenamel))
 
for afiles,files in zip(apcs,als):
    print(files)
    fp = open(afiles, 'r')
    lines = fp.read().splitlines()
    fp.close()
    first = lines[4::3]
    second = lines[5::3]
    rows1 = []
    rows2 = []

    for i in range(len(first)):
        rows1.append(first[i].split())
        rows2.append(second[i].split())

    datas = [np.asarray(rows1), np.asarray(rows2)]
    t = None
    cut = None
    data_rows = np.hstack(datas)
    t = Table(rows=data_rows, names=('starid', 'x', 'y', 'ap1', 'ap2', 'ap3',
                   'ap4', 'ap5', 'sky', 'std', 'skew', 'err1',
                  'err2', 'err3', 'err4', 'err5'), 
            dtype=('i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'
                  , 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    diff = t['ap5']-t['ap1']
    lg = open(files.with_suffix('.aplog'), 'w')
    print(np.mean(diff), np.median(diff))
    lg.write(('Before sigd '+str(np.mean(diff))+str(np.median(diff))+'\n'))
    low, high = sigd(diff, 3.,3.)
    cut = t[(diff>low) & (diff<high)]
    
    diff = cut['ap5']-cut['ap1']
    print(np.mean(diff), np.median(diff))
    lg.write(('After sigd '+str(np.mean(diff))+str(np.median(diff))))
       
    lg.close()
    
    fp = open(files, 'r')
    lines = fp.read().splitlines()
    fp.close()

    #fp = open(files.with_suffix('.ins'), 'w')
    newnames = re.findall('[0-9]',files.stem)
    filt = re.findall('[^0-9_]', files.stem)[-1]
    datarows = []
    lines = lines[3:]
    for i in range(len(lines)):
        datarows.append(lines[i].split())
    
    tbl = Table(rows=np.asarray(datarows), 
                names=('starid', 'x', 'y', 'mag', 'merr', 'sky', 'iter',
                                                 'chi', 'sharp'), 
                dtype=('i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
        
    tbl['image'] = newnames[0]+newnames[1]+newnames[2]+newnames[3]
    tbl['chip'] = newnames[4]
    tbl['filter'] = filt
    tbl['apcor'] = np.median(diff)
    tbl['instr_mag'] = tbl['mag'] + np.median(diff)
    
    ascii.write(tbl, files.with_suffix('.ins'))