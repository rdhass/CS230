# ME 451B Homework 5 (Autumn 2020)
# This is the startup code for the linear stability analysis of the Rayleigh-Benard-Poiseuille flow.
# In this file the class of differential operator is defined and implemented. The default parameters
# for the class construction is based on the problem configuration. The first derivative in x and y
# are based on the spactral method, and the first derivative in z uses the 6th order compact finite
# difference scheme.
#
# HOW TO USE:
# from diff import DiffOps
#
# Programmer: Hang Song (songhang@stanford.edu)

import numpy as np
from numpy import pi
from scipy.fft import fft, ifft, fftshift
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import splu

import matplotlib.pyplot as plt


class DiffOps:
    def __init__(self, nx=256, ny=128, nz=64, Lx = 4.0*pi, Ly = 2.0*pi, Lz = 1.0):
        self.NX, self.NY, self.NZ = nx, ny, nz
        self.buf_cmplx = np.empty((nx,ny,nz), dtype=np.dtype(np.complex128, align=True), order='F')
        self.buf_z = np.empty((nz, nx*ny), dtype=np.float64, order='F')
        self.rhs = np.empty((nz, nx*ny), dtype=np.float64, order='F')
        #Lx, Ly = 4.0*pi, 2.0*pi
        kx = fftshift(np.arange(-nx//2, nx//2)) * (2.0*pi/Lx)
        ky = fftshift(np.arange(-ny//2, ny//2)) * (2.0*pi/Ly)
        self.KX, self.KY = np.meshgrid(kx, ky, indexing='ij')

        
        A = lil_matrix((nz, nz), dtype=np.float64)
        B = lil_matrix((nz, nz), dtype=np.float64)
        idz = np.float64(nz/Lz)

        A[0, 0], A[-1, -1] = 1.00, 1.00
        A[1, 1], A[-2, -2] = 1.00, 1.00
        A[2, 2], A[-3, -3] = 1.00, 1.00
        A[0, 1], A[-1, -2] = 3.00, 3.00
        A[1, 2], A[-2, -1] = 0.25, 0.25
        A[1, 0], A[-2, -3] = 0.25, 0.25
        A[2, 3], A[-3, -2] = 163./508., 163./508.
        A[2, 1], A[-3, -4] = 163./508., 163./508.

        B[ 0, [ 0, 1, 2, 3]] = [-17./6.*idz, 3./2.*idz, 3./2.*idz, -1./6*idz]
        B[-1, [-1,-2,-3,-4]] = [ 17./6.*idz,-3./2.*idz,-3./2.*idz,  1./6*idz]
        B[ 1, [ 0, 2]]       = [-3./4.*idz, 3./4.*idz]
        B[-2, [-3,-1]]       = [-3./4.*idz, 3./4.*idz]
        B[ 2, [ 0, 1, 3, 4]] = [-3./127.*idz, -393./508.*idz, 393./508.*idz, 3./127.*idz]
        B[-3, [-5,-4,-2,-1]] = [-3./127.*idz, -393./508.*idz, 393./508.*idz, 3./127.*idz]

        for i in range(2, nz-2):
            A[i, [i-1, i, i+1]] = [1./3., 1.0, 1./3.]
            B[i, [i-2, i-1, i+1, i+2]] = [-1./36.*idz, -7./9.*idz, 7./9.*idz, 1./36.*idz]
        
        self.LU = splu(A.tocsc())
        self.B = B.tocsc()


    def ddx(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdx
        '''
        self.buf_cmplx[:] = fft(f, axis=0, workers=-1)
        self.buf_cmplx[:] *= 1j * self.KX[:,:,None].astype(dtype=np.float64)
        self.buf_cmplx[self.NX//2,:,:] = 0.0+0.0j
        return np.real(ifft(self.buf_cmplx, axis=0, workers=-1))


    def ddy(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdy
        '''
        self.buf_cmplx[:] = fft(f, axis=1, workers=-1)
        self.buf_cmplx[:] *= 1j * self.KY[:,:,None].astype(dtype=np.float64)
        self.buf_cmplx[:,self.NY//2,:] = 0.0+0.0j
        return np.real(ifft(self.buf_cmplx, axis=1, workers=-1))


    def ddz(self, f):
        '''
        Input: f -- 3D array in Fortran ordering
        Return dfdz
        '''
        fz = np.empty((self.NX, self.NY, self.NZ), dtype=np.dtype(np.float64, align=True), order='F')
        self.buf_z[:] = np.transpose(f.reshape((self.NX*self.NY, self.NZ), order='F'))
        self.rhs[:] = self.B.dot(self.buf_z)
        fz[:] = np.transpose(self.LU.solve(self.rhs)).reshape((self.NX, self.NY, self.NZ), order='F')
        return fz



def test():
    NX, NY, NZ = 256, 128, 64
    dz = 1.0 / NZ
    x = np.linspace(0, 4*pi, num=NX, endpoint=False)
    y = np.linspace(0, 2*pi, num=NY, endpoint=False)
    z = np.linspace(-0.5*(1.0-dz), 0.5*(1.0-dz), num=NZ)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    f   = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fxe = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fye = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fze = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fxn = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fyn = np.empty((NX, NY, NZ), dtype=np.float64, order='F')
    fzn = np.empty((NX, NY, NZ), dtype=np.float64, order='F')

    f  [:] = np.cos(3.0 * X) * np.cos(2.0 * Y) * np.cos(4*pi*Z) 
    fxe[:] =-np.sin(3.0 * X) * np.cos(2.0 * Y) * np.cos(4*pi*Z) * 3.0
    fye[:] =-np.cos(3.0 * X) * np.sin(2.0 * Y) * np.cos(4*pi*Z) * 2.0
    fze[:] =-np.cos(3.0 * X) * np.cos(2.0 * Y) * np.sin(4*pi*Z) * 4.0 * pi

    diff = DiffOps(NX, NY, NZ)

    fxn[:] = diff.ddx(f)
    print("Error dfdx: {:.5E}".format(np.max(np.abs(fxe-fxn))))

    fyn[:] = diff.ddy(f)
    print("Error dfdy: {:.5E}".format(np.max(np.abs(fye-fyn))))

    fzn[:] = diff.ddz(f)
    print("Error dfdz: {:.5E}".format(np.max(np.abs(fze-fzn))))

if __name__ == '__main__':
    test()
