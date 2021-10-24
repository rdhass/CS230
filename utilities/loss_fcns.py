import h5py
import numpy as np
from numpy import pi, cos, sin
from diff import DiffOps
from domain_setup import setup_domain

def MSE(f,g):
    # Computes the mean square error
    nx, ny, nz = f.shape
    L = np.linalg.norm(f - g)**2./(nx*ny*nz)
    return L 

def xy_avg(f):
    nx, ny, nz = f.shape
    f_avg = np.mean(f, axis = (1,2), keepdims = False)
    return f_avg

def extract_field_variables_from_input_layer(X, nx, ny, nz):
    X = X.reshape(nx, ny, nz, 4, order = 'F')
    u = X[:,:,:,0]
    v = X[:,:,:,1]
    w = X[:,:,:,2]
    p = X[:,:,:,3]
    return u, v, w, p

def extract_3D_outputs_from_output_layer(Y, nx, ny, nz):
    Y = Y.reshape(nx, ny, nz, 10, order = 'F')
    tauij = Y[:,:,:,:6]
    A = Y[:,:,:,6:9]
    B = Y[:,:,:,9]
    return tauij, A, B

def L_mom(X,Y,dop,nx,ny,nz):
    # Compute the residual of the pressure Poisson equations
    u, v, w, p = extract_field_variables_from_input_layer(X)
    
    # tauij[:,:,:,0] --> tau_11
    # tauij[:,:,:,1] --> tau_12
    # tauij[:,:,:,2] --> tau_13
    # tauij[:,:,:,3] --> tau_22
    # tauij[:,:,:,4] --> tau_23
    # tauij[:,:,:,5] --> tau_33
    tauij, A, B = extract_3D_outputs_from_output_layer(Y)
    A1u = np.multiply(A[:,:,:,1],u)
    A2v = np.multiply(A[:,:,:,2],v)
    A3w = np.multiply(A[:,:,:,3],w)
    Bp  = np.multiply(B,p)

    Intertial_term = dop.ddx(dop.ddx( np.multiply(A1u,A1u) ) ) +\
                  2.*dop.ddx(dop.ddy( np.multiply(A1u,A2v) ) ) +\
                  2.*dop.ddx(dop.ddz( np.multiply(A1u,A3w) ) ) +\
                     dop.ddy(dop.ddy( np.multiply(A2v,A2v) ) ) +\
                  2.*dop.ddy(dop.ddz( np.multiply(A2v,A3w) ) ) +\
                     dop.ddz(dop.ddz( np.multiply(A3w,A3w) ) )
    Pressure_term = dop.ddx(dop.ddx(Bp)) + dop.ddy(dop.ddy(Bp)) + dop.ddz(dop.ddz(Bp))
    Stress_term   = dop.ddx(dop.ddx(tauij[:,:,:,0])) + \
                 2.*dop.ddx(dop.ddy(tauij[:,:,:,1])) + \
                 2.*dop.ddx(dop.ddz(tauij[:,:,:,2])) + \
                    dop.ddy(dop.ddy(tauij[:,:,:,3])) + \
                 2.*dop.ddy(dop.ddz(tauij[:,:,:,4])) + \
                    dop.ddz(dop.ddz(tauij[:,:,:,5]))     

    L_mom = np.sum(Intertial_term + Pressure_term + Stress_term)
    return L_mom

if __name__ == '__main__':
    # Test xy_avg
    nx, ny, nz = 32, 32, 32
    coefs = [2.0, 0.5]
    for i in range(2):
      Lx, Ly, Lz = coefs[i]*pi, coefs[i]*pi, coefs[i]*pi
      _, _, _, X, Y, Z = setup_domain(Lx, Ly, Lz, nx, ny, nz)

      xcos = cos(X)
      ycos = cos(Y)
      zcos = cos(Z)

      f = np.multiply(xcos,np.multiply(ycos,zcos))
      favg = xy_avg(f)
      if i == 0:
        assert np.amax(favg) < 1.e-12, "np.amax(favg) < 1.e-12 | np.amax(favg): {}".\
                format(np.amax(favg))
      else:
        assert np.amax(favg - zcos) < 1.e-12, "np.amax(favg - zcos) < 1.e-12" +\
                " | np.amax(favg - zcos): {}".format(np.amax(favg))
      print("xy_avg test {} PASSED!".format(i+1))

    # Test MSE
    xsin = sin(X)
    L2_err_true = 0.
    for i in range(nx):
       for j in range(ny):
           for k in range(nz):
               L2_err_true += (xcos[i,j,k] - xsin[i,j,k])**2.
    L2_err_true /= nx*ny*nz
    L2_err = MSE(xcos,xsin)
    assert np.abs(L2_err - L2_err_true) < 1.e-12, "np.abs(L2_err - "\
            + "L2_err_true) = {}".format(np.abs(L2_err - L2_err_true))
    L2_err = MSE(xsin,xcos)
    assert np.abs(L2_err - L2_err_true) < 1.e-12, "np.abs(L2_err - "\
            + "L2_err_true) = {}".format(np.abs(L2_err - L2_err_true))
    print("MSE test PASSED!")

    # Test extract_field_variables_from_input_layer
    u = np.random.randn(nx,ny,nz)
    v = np.random.randn(nx,ny,nz)
    w = np.random.randn(nx,ny,nz)
    p = np.random.randn(nx,ny,nz)
    X = np.empty((nx,ny,nz,4), dtype = np.float64)
    X[:,:,:,0], X[:,:,:,1], X[:,:,:,2], X[:,:,:,3] = u, v, w, p
    X = X.flatten('F')

    uc, vc, wc, pc = extract_field_variables_from_input_layer(X,nx,ny,nz)
    assert np.amax(uc - u) < 1.e-12, 'np.amax(uc - u) = {}'.format(np.amax(uc - u))
    assert np.amax(vc - v) < 1.e-12, 'np.amax(vc - v) = {}'.format(np.amax(vc - v))
    assert np.amax(wc - w) < 1.e-12, 'np.amax(wc - w) = {}'.format(np.amax(wc - w))
    assert np.amax(pc - p) < 1.e-12, 'np.amax(pc - p) = {}'.format(np.amax(pc - p))
    print("extract_field_variables_from_input_layer test PASSED!")

    # Test extract_3D_outputs_from_output_layer
    tauij = np.random.randn(nx,ny,nz,6)
    A     = np.random.randn(nx,ny,nz,3)
    B     = np.random.randn(nx,ny,nz)
    Y     = np.empty((nx,ny,nz,10), dtype = np.float64)
    Y[:,:,:,:6] = tauij
    Y[:,:,:,6:9] = A
    Y[:,:,:,9] = B
    Y = Y.flatten('F')

    tau_c, Ac, Bc = extract_3D_outputs_from_output_layer(Y,nx,ny,nz)
    assert np.amax(A - Ac) < 1.e-12, 'np.amax(A - Ac) = {}'.format(np.amax(A - Ac))
    assert np.amax(B - Bc) < 1.e-12, 'np.amax(B - Bc) = {}'.format(np.amax(B - Bc))
    assert np.amax(tauij - tau_c) < 1.e-12, 'np.amax(tauij - tau_c) = {}'.\
            format(np.amax(tau - tauc))
    print("extract_3D_outputs_from_output_layer test PASSED!")

    # Test L_mom
    def read_test_data(fname):
        ds = h5py.File(fname,'r')
        u = np.array(ds["uVel"][:,:,:])
        v = np.array(ds["vVel"][:,:,:])
        w = np.array(ds["wVel"][:,:,:])
        p = np.array(ds["prss"][:,:,:])
        nx, ny, nz = u.shape
        tauij = np.empty([nx,ny,nz,6])
        tauij[:,:,:,0] = np.array(ds["tau11"][:,:,:])
        tauij[:,:,:,1] = np.array(ds["tau12"][:,:,:])
        tauij[:,:,:,2] = np.array(ds["tau13"][:,:,:])
        tauij[:,:,:,3] = np.array(ds["tau22"][:,:,:])
        tauij[:,:,:,4] = np.array(ds["tau23"][:,:,:])
        tauij[:,:,:,5] = np.array(ds["tau33"][:,:,:])
        return u, v, w, p, tauij
    
    fname = '/Users/ryanhass/Documents/MATLAB/CS_230/DNS_test_data.h5'
    u, v, w, p, tauij = read_test_data(fname)
    print(u.shape)
    print(v.shape)
    print(w.shape)
    print(p.shape)
    print(tauij.shape)
