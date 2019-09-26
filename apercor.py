import matplotlib.pyplot as plt
import numpy as np
from astropy.table import Table
from astropy.io import ascii
from opensave import sigd
from pathlib import Path

filepath = Path(input('where are the .lso files? '))
filename = input('what is the apc filename? ')
filenamel = input('what is the lso filename? ')
print(filepath)
mins = []
apcs = sorted(filepath.glob(filename))
lsos = sorted(filepath.glob(filenamel))
aps = [] 
for afiles,files in zip(apcs,lsos):
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
    rm = None
    cut = None
    lim = None
    conv = 1000000
    data_rows = np.hstack(datas)
    t = Table(rows=data_rows, names=('starid', 'x', 'y', 'ap1', 'ap2', 'ap3',
                   'ap4', 'ap5', 'sky', 'std', 'skew', 'err1',
                  'err2', 'err3', 'err4', 'err5'), 
            dtype=('i4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'
                  , 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4', 'f4'))
    ls = ascii.read(files,data_start=2, names=('starid','x','y','mag','chi'))
    t = t[np.where(np.isin(t['starid'],ls['starid']))]
    diff = t['ap5']-t['ap1']

    lim = len(diff)
    print(np.median(diff), lim, conv)
    low, high = sigd(diff, 3.,3.)
    cut = t[(diff>low) & (diff<high)]

    while lim != conv:
        
        conv = len(diff)        
        diff = cut['ap5']-cut['ap1']
        low, high = sigd(diff, 3.,3.)
        cut = cut[(diff>low) & (diff<high)]
        lim = len(diff)
        print(np.median(diff), lim, conv)

    rm = t[np.where(np.isin(t['starid'], cut['starid'], invert=True))]
    print(len(cut),len(rm))
    mins.append(len(cut))
    aps.append(np.median(diff))
    fp = open(files, 'r')
    lines = fp.readlines()
    fp.close()

    fp = open('./new/'+str(files), 'w')
    fp.write(lines[0])
    fp.write(lines[1])
#    fp.write(lines[2])

    for l in lines[2:]:
        if l.strip().startswith(tuple(str(ids) for ids in rm['starid'])):
            pass
        else:
            fp.write(l)
    fp.close()
print(min(mins), max(mins))
plt.figure()
plt.subplot(121)
plt.hist(mins, bins='auto')
plt.xlabel('# of stars')

plt.subplot(122)
plt.hist(aps, bins='auto')
plt.xlabel('ap correction')

plt.tight_layout()
plt.show()
