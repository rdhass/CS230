import numpy as np
import numpy.matlib as mat
from scipy.fft import rfft irfft fftshift

def ddx(f,dx):
    nx, ny, nz = np.shape(f)
    nxf, nyf, nzf = np.float64((nx,ny,nz))
    k1 = fftshift( np.multiply( np.arange(-nxf/2., nxf/2.-1., 1., \
            dtype = np.float64), 2.*np.pi/(nxf*dx) ) )
    kx = mat.repmat(k1.T,1,ny)
    fhat = rfft(f, axis = 0)
    for kk in range(nz):
        fhat[:,:,kk] = np.multiply(1j*kx,fhat[:,:,kk])
    dfdx = irfft(fhat,axis = 0)
    return dfdx
