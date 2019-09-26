import numpy as np
from pathlib import Path
import re
import os
from glob import glob

path = input("Enter path for working directory: ")
name = input("Enter name: ")

file_list = glob(str(path)+str(name)+'*.txt')
print(file_list)

for fname in file_list:
    f1 = np.genfromtxt(fname,dtype=None, names=['wave', 'I'])
    res = re.findall("[-+]?[0-9]*\.?[0-9]+", fname)
    if not res: continue
    t = np.full_like(f1['wave'], res[0])
    f1 = np.stack((f1['wave'], f1['I'], t), axis=1)
    np.savetxt(fname, f1)