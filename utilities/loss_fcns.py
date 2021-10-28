from diff import DiffOps
from domain_setup import setup_domain
import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy import pi, cos, sin
from read_fortran_data import read_fortran_data, get_domain_size
from scipy import interpolate
import sys

class Loss:
    def __init__(self,nzF,nxC,nyC,nzC,Lx,Ly,Lz,fname):
        # Inputs:
        #   nzF   --> the number of grid points in z for the "Fine" grid
        #   nzC   --> the number of grid points in z for the "Course" grid
        #   Lz    --> Domain size in the z-dimension
        #   fname --> file path and name of the mean profiles 

        # Defined the z-axis
        _, _, _, _, _, zmesh = setup_domain(Lz, Lz, Lz, nzF, nzF, nzF)
        zF = zmesh[0,0,:].reshape([nzF,1])
        _, _, _, _, _, zmesh = setup_domain(Lz, Lz, Lz, nzC, nzC, nzC)
        self.z = zmesh[0,0,:].reshape([nzC,1])

        # Read in average profiles from disk
        avgF = np.genfromtxt(fname,dtype = np.float64)

        # Interpolate the average profiles to the self.z locations
        avgC = np.zeros((nzC,avgF.shape[1]))
        for i in range(avgF.shape[1]):
            tck = interpolate.splrep(zF, avgF[:,i], s=0)
            avgC[:,i] = interpolate.splev(np.squeeze(self.z), tck, der=0)

        # Define the ground-truth from the interpolated values
        self.ground_truth = {"meanU":avgC[:,0],\
                "u1u1":avgC[:,3], "u1u2":avgC[:,4], "u1u3":avgC[:,5],\
                "u2u2":avgC[:,6], "u2u3":avgC[:,7], "u3u3":avgC[:,8],\
                "tau11_mod":avgC[:,17], "tau12_mod":avgC[:,18],\
                "tau13_mod":avgC[:,13], "tau22_mod":avgC[:,19],\
                "tau23_mod":avgC[:,14], "tau33_mod":avgC[:,20]}

        # initialize derivative operator
        self.dop = DiffOps(nx = nxC, ny = nyC, nz = nzC, Lx = Lx, Ly = Ly, Lz = Lz)

    def mean_square(f):
        mean_sq = np.mean(np.power(f,2.))
        return mean_sq
    
    def MSE(f,g):
        L = mean_square(f - g)
        return L 
    
    def xy_avg(f):
        nx, ny, nz = f.shape
        f_avg = np.mean(f, axis = (1,2), keepdims = False)
        return f_avg
    
    def fluct(u):
        nx, ny, nz = u.shape
        uAvg = xy_avg(u)
        uprime = np.zeros((nx,ny,nz))
        for i in range(nx):
            for j in range(ny):
                uprime[i,j,:] = u[i,j,:] - uAvg
        return uprime
    
    def extract_field_variables_from_input_layer(self,X, nx, ny, nz, inc_prss = True):
        if inc_prss:
            X = X.reshape(nx, ny, nz, 4, order = 'F')
            p = X[:,:,:,3]
        else:
            X = X.reshape(nx, ny, nz, 3, order = 'F')
            p = None
        u = X[:,:,:,0]
        v = X[:,:,:,1]
        w = X[:,:,:,2]
        return u, v, w, p
    
    def extract_3Dfields_from_output_layer(self,Y, nx, ny, nz, inc_tauij = True):
        if inc_tauij:
          Y = Y.reshape(nx, ny, nz, 10, order = 'F')
          tauij = Y[:,:,:,:6]
          A = Y[:,:,:,6:9]
          B = Y[:,:,:,9]
        else:
          Y = Y.reshape(nx, ny, nz, 4, order = 'F')
          tauij = None
          A = Y[:,:,:,:3]
          B = Y[:,:,:,3]
        return tauij, A, B
    
    def L_mass(self,u,v,w):
        return mean_square(self.dop.ddx(u) + self.dop.ddy(v) + self.dop.ddz(w))
    
    def L_mom(self,u,v,w,p,tauij):
        # Compute the residual of the pressure Poisson equations
        
        # tauij[:,:,:,0] --> tau_11
        # tauij[:,:,:,1] --> tau_12
        # tauij[:,:,:,2] --> tau_13
        # tauij[:,:,:,3] --> tau_22
        # tauij[:,:,:,4] --> tau_23
        # tauij[:,:,:,5] --> tau_33
    
        Intertial_term = self.dop.ddx(self.dop.ddx( np.multiply(u,u) ) ) +\
                      2.*self.dop.ddx(self.dop.ddy( np.multiply(u,v) ) ) +\
                      2.*self.dop.ddx(self.dop.ddz( np.multiply(u,w) ) ) +\
                         self.dop.ddy(self.dop.ddy( np.multiply(v,v) ) ) +\
                      2.*self.dop.ddy(self.dop.ddz( np.multiply(v,w) ) ) +\
                         self.dop.ddz(self.dop.ddz( np.multiply(w,w) ) )
        Pressure_term = self.dop.ddx(self.dop.ddx(Bp)) + \
                self.dop.ddy(self.dop.ddy(Bp)) + self.dop.ddz(self.dop.ddz(Bp))
        Stress_term   = self.dop.ddx(self.dop.ddx(tauij[:,:,:,0])) + \
                     2.*self.dop.ddx(self.dop.ddy(tauij[:,:,:,1])) + \
                     2.*self.dop.ddx(self.dop.ddz(tauij[:,:,:,2])) + \
                        self.dop.ddy(self.dop.ddy(tauij[:,:,:,3])) + \
                     2.*self.dop.ddy(self.dop.ddz(tauij[:,:,:,4])) + \
                        self.dop.ddz(self.dop.ddz(tauij[:,:,:,5]))     
    
        return mean_square(Intertial_term + Pressure_term + Stress_term)
    
    def L_U(self,u):
        U_GT = self.ground_truth["meanU"]
        U_ML = xy_avg(u)
        return MSE(U_GT,U_ML)
    
    def L_uiuj(self,u,v,w):
        inputs = {"u1":u,"u2":v,"u3":w}
        for i in range(3):
            for j in range(3):
                if i <= j:
                    uiujGT = ground_truth["u"+str(i)+"u"+str(j)]
                    uiujML = xy_avg(np.multiply(fluct(inputs["u"+str(i)]),\
                            fluct(inputs["u"+str(j)]) ) )
                    L_uiuj += MSE(uiujGT,uiujML)
        return L_uiuj
    
    def modify_fields(self, u, v, w, p, A, B):
        u = np.multiply(A[:,:,:,0],u)
        v = np.multiply(A[:,:,:,1],v)
        w = np.multiply(A[:,:,:,2],w)
        p = np.multiply(B,p)
        return u, v, w, p

    def compute_loss(self,X,Y,nx,ny,nz,lambda_p = 0.5, inc_mom = True):
        # Step 1: Extract data and apply scaling, e.g. u -> Au
        u, v, w, p  = self.extract_field_variables_from_input_layer(X,nx,ny,nz)
        tauij, A, B = self.extract_3Dfields_from_output_layer(Y,nx,ny,nz)
        A1u, A2v, A3w, Bp = self.modify_fields(u, v, w, p, A, B)

        # Compute loss functions
        Lphys = self.L_mass(A1u,A2v,A3w)
        Lcontent = self.L_uiuj(A1u,A2v,A3w) + self.L_U(A1u) # + self.L_P(Bp)
        if inc_mom:
            Lphys += self.L_mom(A1u,A2v,A3w,Bp,tauij)
            #Lcontent += self.L_tauij(...)

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
        _, _, _, X, Y, Z = setup_domain(Lx, Ly, Lz, nx, ny, nz, zPeriodic = True)

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
    return X, Y, Z

def test_MSE(nx,ny,nz,Lx,Ly,Lz):
    _, _, _, X, _, _ = setup_domain(Lx,Ly,Lz,nx,ny,nz,zPeriodic = True)
    xsin = sin(X)
    xcos = cos(X)
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
    return None

def test_extract_field_variables_from_input_layer(nx,ny,nz):
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
    return None

def test_extract_3Dfields_from_output_layer(nx,ny,nz):
    tauij = np.random.randn(nx,ny,nz,6)
    A     = np.random.randn(nx,ny,nz,3)
    B     = np.random.randn(nx,ny,nz)
    Y     = np.empty((nx,ny,nz,10), dtype = np.float64)
    Y[:,:,:,:6] = tauij
    Y[:,:,:,6:9] = A
    Y[:,:,:,9] = B
    Y = Y.flatten('F')

    tau_c, Ac, Bc = extract_3Dfields_from_output_layer(Y,nx,ny,nz)
    assert np.amax(A - Ac) < 1.e-12, 'np.amax(A - Ac) = {}'.format(np.amax(A - Ac))
    assert np.amax(B - Bc) < 1.e-12, 'np.amax(B - Bc) = {}'.format(np.amax(B - Bc))
    assert np.amax(tauij - tau_c) < 1.e-12, 'np.amax(tauij - tau_c) = {}'.\
            format(np.amax(tau - tauc))
    print("extract_3Dfields_from_output_layer test PASSED!")
    return None

def test_L_mass(fname):
    X, Y, nx, ny, nz = read_test_data(fname)
    u, v, w, p  = extract_field_variables_from_input_layer(X,nx,ny,nz)
    tauij, A, B = extract_3Dfields_from_output_layer(Y,nx,ny,nz)
    dop = DiffOps(nx = nx, ny = ny, nz = nz, Lx = 2.*pi, Ly = 2.*pi, Lz = 2.*pi)
    
    Lmass = L_mass(u,v,w,A,dop,nx,ny,nz)
    assert Lmass < 1.e-4, 'Lmass = {}'.format(Lmass)
    print("Lmass test PASSED!")
    return None

def test_fluct(fname):
    X, _, nx, ny, nz = read_test_data(fname)
    u, v, w, p  = extract_field_variables_from_input_layer(X,nx,ny,nz)
    
    ufluct = fluct(u)
    vfluct = fluct(v)
    wfluct = fluct(w)
    pfluct = fluct(p)
    assert np.mean(ufluct) < 1.e-12, 'np.mean(ufluct) = {}'.format(np.mean(ufluct))
    assert np.mean(vfluct) < 1.e-12, 'np.mean(vfluct) = {}'.format(np.mean(vfluct))
    assert np.mean(wfluct) < 1.e-12, 'np.mean(wfluct) = {}'.format(np.mean(wfluct))
    assert np.mean(pfluct) < 1.e-12, 'np.mean(pfluct) = {}'.format(np.mean(pfluct))
    print("fluct test PASSED!")
    return None

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage:")
        print("  python3 loss_fcns.py <fname of test data> <fname of test data averages>")
        sys.exit()

#### For milestone ######
    # Test xy_avg
    nx, ny, nz = 32, 32, 32
    coefs = [2.0, 0.5]
    fname_averages = sys.argv[2]
    L_test = Loss(128,64,1.,fname_averages)
    print(L_test.ground_truth["u1u1"][:10])
    fig, ((ax1,ax2),(ax3,ax4)) = plt.subplots(2,2)
    ax1.plot(L_test.ground_truth["u1u1"],L_test.z)
    ax2.plot(L_test.ground_truth["u2u2"],L_test.z)
    ax3.plot(L_test.ground_truth["u3u3"],L_test.z)
    ax4.plot(L_test.ground_truth["u1u3"],L_test.z)
    plt.show()
    exit(0)
    X,Y,Z = test_xy_avg(nx,ny,nz,coefs)

    # Test fluct
    fname_full_fields = sys.argv[1]
    test_fluct(fname_full_fields)

    # Test MSE
    test_MSE(nx,ny,nz,coefs[1]*pi,coefs[1]*pi,coefs[1]*pi)

    # Test extract_field_variables_from_input_layer
    test_extract_field_variables_from_input_layer(nx,ny,nz)

    # Test extract_3Dfields_from_output_layer
    test_extract_3Dfields_from_output_layer(nx,ny,nz)

    # Test L_U and L_P

    # Test L_uiuj

    # Test L_mass
    test_L_mass(fname_full_fields)

###### For final project ######
    # Test L_tauij

    # Test L_mom
    #Lmom = L_mom(u,v,w,p,tauij,A,B,dop,nx,ny,nz)
    #assert Lmom < 1.e-4, 'Lmom = {}'.format(Lmom)
    #print("L_mom test PASSED!")
