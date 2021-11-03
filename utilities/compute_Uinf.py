import numpy as np
from read_fortran_data import read_fortran_data

datadir = '/Users/ryanhass/Documents/MATLAB/CS_230/data/'
tid_vec = np.arange(146000, 179300, 100, dtype = int)
Uinf = 0.

for tid in tid_vec:
    fname = 'Run01_' + str(tid).zfill(7) + '.h5'
    u = read_fortran_data(datadir + fname, 'uVel')
    U = np.mean(u,axis = (0,1))
    Uinf += U[-1]

Uinf /= tid_vec.size
print(Uinf)
