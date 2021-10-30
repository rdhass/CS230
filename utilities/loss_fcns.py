from diff import DiffOps
from domain_setup import setup_domain, setup_domain_1D
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin
from read_fortran_data import read_fortran_data, get_domain_size
from scipy import interpolate
import sys
from time import time
from io_mod import load_dataset_V2

class Loss:
    def __init__(self,nx,ny,nz,Lx,Ly,Lz):
        # Inputs:
        #   nx, ny, nz --> the number of grid points in domain "Course" grid
        #   Lx, Ly, Lz    --> Domain size in the "i"-dimension
        self.z = setup_domain_1D(Lz/nz*0.5,Lz-Lz/nz*0.5,Lz/nz)
        self.nx, self.ny, self.nz = nx, ny, nz

        # Initialize the ground truth dictionary
        self.ground_truth = {}
        self.ground_truth_keys = ("meanU","u1u1","u1u2","u1u3","u2u2","u2u3","u3u3",\
                "tau11_mod","tau12_mod","tau13_mod","tau22_mod",\
                "tau23_mod","tau33_mod","meanP")
        for i, key in enumerate(self.ground_truth_keys):
            self.ground_truth[key] = np.empty((nz,1), dtype = np.float64, order = 'F')

        # initialize derivative operator
        self.dop = DiffOps(nx = nx, ny = ny, nz = nz, Lx = Lx, Ly = Ly, Lz = Lz)
        
        # Allocate memory for the flow variables
        self.u     = np.empty((nx,ny,nz),   dtype = np.float64, order = 'F')
        self.v     = np.empty((nx,ny,nz),   dtype = np.float64, order = 'F')
        self.w     = np.empty((nx,ny,nz),   dtype = np.float64, order = 'F')
        self.p     = np.empty((nx,ny,nz),   dtype = np.float64, order = 'F')
        self.tauij = np.empty((nx,ny,nz,6), dtype = np.float64, order = 'F')
        self.A     = np.empty((nx,ny,nz,3), dtype = np.float64, order = 'F')
        self.B     = np.empty((nx,ny,nz),   dtype = np.float64, order = 'F')

    def mean_square(self,f):
        return np.mean(f*f)
    
    def MSE(self,f,g):
        assert len(f.shape) == len(g.shape)
        for i in range(len(f.shape)):
            assert f.shape[i] == g.shape[i], \
                    "f.shape[{}] = {}; g.shape[{}] = {}".format(i,f.shape[i],\
                    i,g.shape[i])
        L = self.mean_square(f - g)
        return L 
    
    def confirm_dimensions(self,nx,ny,nz):
        assert self.nx == nx
        assert self.ny == ny
        assert self.nz == nz
        return None

    def xy_avg(self,f):
        nx, ny, nz = f.shape
        self.confirm_dimensions(nx, ny, nz)
        return np.mean(f, axis = (0,1), keepdims = False)
    
    def fluct(self,u):
        nx, ny, nz = u.shape
        self.confirm_dimensions(nx, ny, nz)
        uAvg = self.xy_avg(u)
        uprime = np.zeros((nx,ny,nz))
        for i in range(nx):
            for j in range(ny):
                uprime[i,j,:] = u[i,j,:] - uAvg
        return uprime
    
    def extract_field_variables_from_input_layer(self,X, inc_prss = True):
        nx, ny, nz = self.nx, self.ny, self.nz
        if inc_prss:
            X = X.reshape(nx, ny, nz, 4, order = 'F')
            self.p = X[:,:,:,3]
        else:
            X = X.reshape(nx, ny, nz, 3, order = 'F')
        self.u = X[:,:,:,0]
        self.v = X[:,:,:,1]
        self.w = X[:,:,:,2]
        return None
    
    def extract_3Dfields_from_output_layer(self,Yhat, Y, inc_tauij = True):
        nx, ny, nz = self.nx, self.ny, self.nz
        if inc_tauij:
          Yhat = Yhat.reshape(nx, ny, nz, 10, order = 'F')
          self.tauij = Yhat[:,:,:,:6]
          self.A = Yhat[:,:,:,6:9]
          self.B = Yhat[:,:,:,9]
          nprofs = 14
        else:
          Yhat = Yhat.reshape(nx, ny, nz, 3, order = 'F')
          self.A = Yhat
          nprofs = 7

        for i in range(nprofs):
            self.ground_truth[self.ground_truth_keys[i]] = \
                    Y.reshape((nz,nprofs), order = 'F')[:,i]
        return None
    
    def L_mass(self):
        return self.mean_square(self.dop.ddx(self.u) + self.dop.ddy(self.v) + self.dop.ddz(self.w))
    
    def L_mom(self):
        # Compute the residual of the pressure Poisson equations
        
        # tauij[:,:,:,0] --> tau_11
        # tauij[:,:,:,1] --> tau_12
        # tauij[:,:,:,2] --> tau_13
        # tauij[:,:,:,3] --> tau_22
        # tauij[:,:,:,4] --> tau_23
        # tauij[:,:,:,5] --> tau_33
    
        Inertial_term = self.dop.ddx(self.dop.ddx( self.u*self.u ) ) +\
                 2.*self.dop.ddx(self.dop.ddy( self.u*self.v ) ) +\
                 2.*self.dop.ddx(self.dop.ddz( self.u*self.w ) ) +\
                    self.dop.ddy(self.dop.ddy( self.v*self.v ) ) +\
                 2.*self.dop.ddy(self.dop.ddz( self.v*self.w ) ) +\
                    self.dop.ddz(self.dop.ddz( self.w*self.w ) )
        Pressure_term = self.dop.ddx(self.dop.ddx(self.p)) + \
           self.dop.ddy(self.dop.ddy(self.p)) + self.dop.ddz(self.dop.ddz(self.p))
        Stress_term   = self.dop.ddx(self.dop.ddx(self.tauij[:,:,:,0])) + \
                     2.*self.dop.ddx(self.dop.ddy(self.tauij[:,:,:,1])) + \
                     2.*self.dop.ddx(self.dop.ddz(self.tauij[:,:,:,2])) + \
                        self.dop.ddy(self.dop.ddy(self.tauij[:,:,:,3])) + \
                     2.*self.dop.ddy(self.dop.ddz(self.tauij[:,:,:,4])) + \
                        self.dop.ddz(self.dop.ddz(self.tauij[:,:,:,5]))     
    
        return self.mean_square(Inertial_term + Pressure_term + Stress_term)
    
    def __L_mean_profile__(self,f,dict_var):
        F_GT = self.ground_truth[dict_var]
        F_ML = self.xy_avg(f)
        return self.MSE(F_GT,F_ML)

    def L_U(self,u):
        return self.__L_mean_profile__(u,"meanU")
    
    def L_P(self,p):
        return self.__L_mean_profile__(p,"meanP")
    
    def L_uiuj(self):
        L_uiuj = 0.
        inputs = {}
        inputs["u1"] = self.fluct(self.u);
        inputs["u2"] = self.fluct(self.v);
        inputs["u3"] = self.fluct(self.w);
        for i in range(3):
            for j in range(3):
                if i <= j:
                    uiujGT = self.ground_truth["u"+str(i+1)+"u"+str(j+1)]
                    uiujML = self.xy_avg(inputs["u"+str(i+1)]*inputs["u"+str(j+1)])
                    L_uiuj += self.MSE(uiujGT,uiujML)
        return L_uiuj
    
    def modify_fields(self):
        self.u = self.A[:,:,:,0]*self.u
        self.v = self.A[:,:,:,1]*self.v
        self.w = self.A[:,:,:,2]*self.w
        self.p = self.B*self.p
        return None

    def compute_loss(self, X, Yhat, Y, lambda_p = 0.5, lambda_tau = 0.5, inc_mom = True):
        nx,ny,nz = self.nx, self.ny, self.nz
        if inc_mom:
            assert nx*ny*nz*4 == X.size
            assert nx*ny*nz*10 == Y.size
            assert nx*ny*nz*10 == Yhat.size
        else:
            assert nx*ny*nz*3 == X.size
            assert nx*ny*nz*3 == Y.size
            assert nx*ny*nz*3 == Yhat.size
        
        # Step 1: Extract data and apply scaling, e.g. u -> Au
        self.extract_field_variables_from_input_layer(X,inc_prss = inc_mom)
        self.extract_3Dfields_from_output_layer(Yhat,Y,inc_tauij = inc_mom)
        self.modify_fields()

        # Compute loss functions
        Lphys = self.L_mass()
        Lcontent = self.L_uiuj() + self.L_U()
        if inc_mom:
            Lphys += self.L_mom()
            Lcontent = (1. - lambda_tau)*(Lcontent + self.L_P())
            #Lcontent += lambda_tau*self.L_tauij()
        self.total_loss = lambda_p*Lphys + (1. - lambda_p*Lcontent)
        return None

def read_test_data(fname):
    ds_name = ['uVel','vVel','wVel','prss']
    nx, ny, nz = get_domain_size(fname,ds_name[0])
    ncube = nx*ny*nz
    X = np.empty((nx,ny,nz,4), dtype = np.float64, order = 'F')
    for i, dsname in enumerate(ds_name):
        X[:,:,:,i] = read_fortran_data(fname,dsname)
    X = X.flatten(order = 'F').reshape(4*ncube,1,order = 'F')
    
    Y = np.empty((nx,ny,nz,10), dtype = np.float64, order = 'F')
    dsname = ['tau11','tau12','tau13','tau22','tau23','tau33']
    for i, ds_name in enumerate(dsname):
        Y[:,:,:,i]   = read_fortran_data(fname,ds_name)

    Y[:,:,:,6:] = np.ones((nx,ny,nz,4))
    Y = Y.flatten(order = 'F').reshape(10*ncube,1,order = 'F')
    return X, Y, nx, ny, nz
   
def test_xy_avg(nx,ny,nz,coefs):
    for i in range(2):
        Lx, Ly, Lz = coefs[i]*pi, coefs[i]*pi, coefs[i]*pi
        L = Loss(nx,ny,nz,6.*pi,3.*pi,1.)
        _, _, _, _, _, _, X, Y, Z = setup_domain(Lx, Ly, Lz, nx, ny, nz, zPeriodic = True)

        xcos = cos(X)
        ycos = cos(Y)
        zcos = cos(Z)

        f = xcos*ycos*zcos
        favg = L.xy_avg(f)
        if i == 0:
            assert np.amax(favg) < 1.e-12, "np.amax(favg) < 1.e-12 | np.amax(favg): {}".\
                  format(np.amax(favg))
        else:
            zcos = np.squeeze(zcos[0,0,:])
            assert np.amax(favg - zcos) < 1.e-12, "np.amax(favg - zcos) < 1.e-12" +\
                  " | np.amax(favg - zcos): {}".format(np.amax(favg))
        print("xy_avg test {} PASSED!".format(i+1))
    return X, Y, Z

def test_MSE(nx,ny,nz,Lx,Ly,Lz):
    L = Loss(nx,ny,nz,Lx,Ly,Lz)
    _, _, _, _, _, _, X, _, _ = setup_domain(Lx,Ly,Lz,nx,ny,nz,zPeriodic = True)
    xsin = sin(X)
    xcos = cos(X)
    L2_err_true = 0.
    for i in range(nx):
       for j in range(ny):
           for k in range(nz):
               L2_err_true += (xcos[i,j,k] - xsin[i,j,k])**2.
    L2_err_true /= nx*ny*nz
    L2_err = L.MSE(xcos,xsin)
    assert np.abs(L2_err - L2_err_true) < 1.e-12, "np.abs(L2_err - "\
            + "L2_err_true) = {}".format(np.abs(L2_err - L2_err_true))
    L2_err = L.MSE(xsin,xcos)
    assert np.abs(L2_err - L2_err_true) < 1.e-12, "np.abs(L2_err - "\
            + "L2_err_true) = {}".format(np.abs(L2_err - L2_err_true))
    print("MSE test PASSED!")
    return None

def test_extract_field_variables_from_input_layer(nx,ny,nz):
    u = np.random.randn(nx,ny,nz)
    v = np.random.randn(nx,ny,nz)
    w = np.random.randn(nx,ny,nz)
    p = np.random.randn(nx,ny,nz)
    X = np.empty((nx,ny,nz,4), dtype = np.float64)
    X[:,:,:,0], X[:,:,:,1], X[:,:,:,2], X[:,:,:,3] = u, v, w, p
    X = X.flatten('F')
    
    L = Loss(nx,ny,nz,1.,1.,1.)
    L.extract_field_variables_from_input_layer(X)
    assert np.amax(L.u - u) < 1.e-12, 'np.amax(uc - u) = {}'.format(np.amax(L.u - u))
    assert np.amax(L.v - v) < 1.e-12, 'np.amax(vc - v) = {}'.format(np.amax(L.v - v))
    assert np.amax(L.w - w) < 1.e-12, 'np.amax(wc - w) = {}'.format(np.amax(L.w - w))
    assert np.amax(L.p - p) < 1.e-12, 'np.amax(pc - p) = {}'.format(np.amax(L.p - p))
    print("extract_field_variables_from_input_layer test PASSED!")
    return None

def test_extract_3Dfields_from_output_layer(nx,ny,nz):
    tauij = np.random.randn(nx,ny,nz,6)
    A     = np.random.randn(nx,ny,nz,3)
    B     = np.random.randn(nx,ny,nz)
    Yhat     = np.empty((nx,ny,nz,10), dtype = np.float64)
    Yhat[:,:,:,:6] = tauij
    Yhat[:,:,:,6:9] = A
    Yhat[:,:,:,9] = B
    Yhat = Yhat.flatten('F')
    Y = np.random.randn(nz*14,1)

    L = Loss(nx,ny,nz,1.,1.,1.)
    L.extract_3Dfields_from_output_layer(Yhat,Y)
    assert np.amax(A - L.A) < 1.e-12, 'np.amax(A - Ac) = {}'.format(np.amax(A - L.A))
    assert np.amax(B - L.B) < 1.e-12, 'np.amax(B - Bc) = {}'.format(np.amax(B - L.B))
    assert np.amax(tauij - L.tauij) < 1.e-12, 'np.amax(tauij - L.tauij) = {}'.\
            format(np.amax(tau - L.tauij))
    print("extract_3Dfields_from_output_layer test PASSED!")
    return None

def test_L_mass(L,fname):
    X, Y, nx, ny, nz = read_test_data(fname)
    L.extract_field_variables_from_input_layer(X)
    L.dop = DiffOps(nx = nx, ny = ny, nz = nz, Lx = 2.*pi, Ly = 2.*pi, Lz = 2.*pi)
   
    Lmass = L.L_mass()

    #assert Lmass < 1.e-4, 'Lmass = {}'.format(Lmass)
    print("Lmass = {}".format(Lmass))
    return None

def test_fluct(fname):
    X, _, nx, ny, nz = read_test_data(fname)
    L = Loss(nx,ny,nz,2.*pi,2.*pi,2.*pi)
    L.extract_field_variables_from_input_layer(X,inc_prss = True)
    
    ufluct = L.fluct(L.u)
    vfluct = L.fluct(L.v)
    wfluct = L.fluct(L.w)
    pfluct = L.fluct(L.p)
    assert np.mean(ufluct) < 1.e-12, 'np.mean(ufluct) = {}'.format(np.mean(ufluct))
    assert np.mean(vfluct) < 1.e-12, 'np.mean(vfluct) = {}'.format(np.mean(vfluct))
    assert np.mean(wfluct) < 1.e-12, 'np.mean(wfluct) = {}'.format(np.mean(wfluct))
    assert np.mean(pfluct) < 1.e-12, 'np.mean(pfluct) = {}'.format(np.mean(pfluct))
    print("fluct test PASSED!")
    return None

def test_L_U(L):
    L_U = L.L_U(L.u)
    L_U_Mat = 0.028057428411232
    assert L_U - L_U_Mat < 1.e-12, "L_U computed: {}. Expected result: {}".format(L.L_U(u),L_U_Mat)
    print('L_U test PASSED!') 
    return None

def test_L_uiuj(L):
    st = time()
    L_uiuj = L.L_uiuj()
    en = time() - st
    print("L_uiuj took {}s to compute".format(en))
    L_uiuj_Mat = sum((0.148239052336250,6.535248054341532e-04,0.001795615654557,\
            0.006632703981999,1.396591062801845e-04,0.009991396638865))
    assert L_uiuj - L_uiuj_Mat < 1.e-14, "L_uiuj = {}, L_uiuj_Mat = {}".forme(\
            L_uiuj, L_uiuj_Mat)
    print("L_uiuj test PASSED!")
    return None

def load_data_for_loss_tests(datadir,nzC,nzF,Lz = 1.):
    nx, ny, nz = 192, 192, nzC
    zC = setup_domain_1D(Lz/nzC*0.5, Lz - Lz/nzC*0.5, Lz/nzC)
    zF = setup_domain_1D(Lz/nzF*0.5, Lz - Lz/nzF*0.5, Lz/nzF)
    x_tid_vec = np.array([179300])
    y_tid_vec = np.array([73200])
    X, Y, _, _ = load_dataset_V2(datadir,192,192,64,zF,zC,x_tid_vec,x_tid_vec,\
            y_tid_vec,y_tid_vec,navg = 3020)
    Yhat = np.ones((nx*ny*nz*10,1), dtype = np.float64, order = 'F')
    return X, Y, Yhat

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print("Usage:")
        print("  python3 loss_fcns.py <fname of HIT data> "\
                +"<fname of HR PBL averages> <fname of LR PBL data> <datadir>")
        sys.exit()

#### For milestone ######
    # Test xy_avg
    nx, ny, nz = 32, 32, 32
    coefs = [2.0, 0.5]
    fname_averages = sys.argv[2]
    L_test = Loss(192,192,64,6.*pi,3.*pi,1.)
    X,Y,Z = test_xy_avg(nx,ny,nz,coefs)

    # Test fluct
    fname_HIT = sys.argv[1]
    test_fluct(fname_HIT)

    # Test MSE
    test_MSE(nx,ny,nz,coefs[1]*pi,coefs[1]*pi,coefs[1]*pi)

    # Test extract_field_variables_from_input_layer
    test_extract_field_variables_from_input_layer(nx,ny,nz)

    # Test extract_3Dfields_from_output_layer
    test_extract_3Dfields_from_output_layer(nx,ny,nz)

    # Load actual channel data for loss tests
    fname_LR = sys.argv[3]
    datadir = sys.argv[4] + '/'
    X, Y, Yhat = load_data_for_loss_tests(datadir,64,128)

    # Load data into Loss class
    L_test.extract_field_variables_from_input_layer(X)
    L_test.extract_3Dfields_from_output_layer(Yhat, Y)
    
    # Plot some ground-truth profiles to confirm proper initialization of the class
    #print(L_test.ground_truth["u1u1"][:10])
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.plot(L_test.ground_truth["u1u1"],L_test.z)
    ax2.plot(L_test.ground_truth["u2u2"],L_test.z)
    ax3.plot(L_test.ground_truth["u3u3"],L_test.z)
    ax4.plot(L_test.ground_truth["u1u3"],L_test.z)
    plt.show()

    # Test L_U
    test_L_U(L_test)
    
    # Test L_uiuj
    test_L_uiuj(L_test)
    
    # Test L_mass
    L_test = Loss(32,32,32,2.*pi,2.*pi,2.*pi)
    test_L_mass(L_test,fname_HIT)

###### For final project ######
    # Test P_U
    
    # Test L_tauij

    # Test L_mom
    #Lmom = L_mom(u,v,w,p,tauij,A,B,dop,nx,ny,nz)
    #assert Lmom < 1.e-4, 'Lmom = {}'.format(Lmom)
    #print("L_mom test PASSED!")
