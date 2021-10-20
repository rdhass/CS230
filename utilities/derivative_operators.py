import numpy as np
import numpy.matlib as mat
from scipy.fft import fft, ifft, fftshift

def __Fourier_coll__(f,dim,nx,ny,nz,h):
    if dim == 0:
        N = nx
        Nrep = ny
    elif dim == 1:
        N = ny
        Nrep = nx

    Nf = np.float64(N)
    kvec = fftshift( np.multiply( np.arange(-Nf/2., Nf/2., 1., \
            dtype = np.float64), 2.*np.pi/(Nf*h) ) )
    if dim == 0:
        kvec = kvec.reshape(kvec.size,1)
        kmat = mat.repmat(kvec,1,Nrep)
    elif dim == 1:
        kvec = kvec.reshape(1,kvec.size)
        kmat = mat.repmat(kvec,Nrep,1)

    fhat = fft(f, axis = dim)
    for kk in range(nz):
        fhat[:,:,kk] = np.multiply(1j*kmat,fhat[:,:,kk])
    fprime = np.real(ifft(fhat,axis = dim))
    return fprime

def ddx(f,dx):
    nx, ny, nz = np.shape(f)
    dfdx = __Fourier_coll__(f,0,nx,ny,nz,dx)
    return dfdx

def ddy(f,dy):
    nx, ny, nz = np.shape(f)
    dfdy = __Fourier_coll__(f,1,nx,ny,nz,dy)
    return dfdy
