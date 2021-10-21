import numpy as np
from scipy.fft import fft, ifft, fftshift

def __Fourier_coll__(f,dim,h):
    nx, ny, nz = np.shape(f)
    N = [nx,ny]

    Nf = np.float64(N[dim])
    kvec = fftshift( np.multiply( np.arange(-Nf/2., Nf/2., 1., \
            dtype = np.float64), 2.*np.pi/(Nf*h) ) )
    if dim == 0:
        kvec = kvec.reshape(kvec.size,1)
        kmat = mat.repmat(kvec,1,N[1])
    elif dim == 1:
        kvec = kvec.reshape(1,kvec.size)
        kmat = mat.repmat(kvec,N[0],1)

    fhat = fft(f, axis = dim)
    for kk in range(nz):
        fhat[:,:,kk] = np.multiply(1j*kmat,fhat[:,:,kk])
    fprime = np.real(ifft(fhat,axis = dim))
    return fprime

def ddx(f,dx):
    #dfdx = __Fourier_coll__(f,0,dx)
    nx, ny, nz = np.shape(f)

    Nf = np.float64(nx)
    k1 = fftshift( np.multiply( np.arange(-Nf/2., Nf/2., 1., \
            dtype = np.float64), 2.*np.pi/(Nf*dx) ) )
    k1 = k1.reshape(k1.size,1)
    kx = np.matlib.repmat(k1,1,ny)

    fhat = fft(f, axis = 0)
    for kk in range(nz):
        fhat[:,:,kk] = np.multiply(1j*kx,fhat[:,:,kk])
    dfdy = np.real(ifft(fhat,axis = 0))
    return dfdx

def ddy(f,dy):
    #dfdy = __Fourier_coll__(f,1,dy)
    nx, ny, nz = np.shape(f)

    Nf = np.float64(ny)
    k2 = fftshift( np.multiply( np.arange(-Nf/2., Nf/2., 1., \
            dtype = np.float64), 2.*np.pi/(Nf*dy) ) )
    k2 = k2.reshape(1,k2.size)
    ky = np.matlib.repmat(k2,nx,1)

    fhat = fft(f, axis = 1)
    for kk in range(nz):
        fhat[:,:,kk] = np.multiply(1j*ky,fhat[:,:,kk])
    dfdy = np.real(ifft(fhat,axis = 1))
    return dfdy
