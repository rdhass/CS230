import code
from domain_setup import setup_domain_1D
import numpy as np
from numpy import pi
from read_fortran_data import read_fortran_data, get_domain_size
import sys
from time import time
from io_mod import load_dataset_V2
from diff_tf import DiffOps
import tensorflow as tf
from mse_custom import mse_custom

# Array shapes: 
# X, Yhat: [m, nvars, nx, ny, nz]
# Y:       [m, nprofs, nz]

class Loss:
    def __init__(self,nx,ny,nz,Lx,Ly,Lz,n_examples,inc_mom = False):
        # Inputs:
        #   nx, ny, nz --> the number of grid points in domain "Course" grid
        #   inc_mom    --> logical whether or not to include "momentum" terms
        #                  in model (e.g. tauij, L_mom, pressure)
        self.inc_mom = inc_mom
        if self.inc_mom:
            self.nprofs = 14
            self.nvars = 4
        else:
            self.nprofs = 7
            self.nvars = 3


        self.m = n_examples
        self.nx, self.ny, self.nz = nx, ny, nz

        # Initialize derivative operator
        self.dop = DiffOps(nx, ny, nz, Lx, Ly, Lz)

        # Allocate memory for the flow variables
        self.fields = {}
        self.fields["u1"] = tf.zeros((n_examples,nx,ny,nz), dtype = tf.float32)  
        self.fields["u2"] = tf.zeros((n_examples,nx,ny,nz), dtype = tf.float32)  
        self.fields["u3"] = tf.zeros((n_examples,nx,ny,nz), dtype = tf.float32)  
        if self.inc_mom:
            self.fields["p"] = tf.zeros((n_examples,nx,ny,nz), dtype = tf.float32)  

        # Allocate memory for the current state
        current_state = tf.zeros((n_examples,self.nprofs,nz), dtype = tf.float32)

    def mean_square(self,f):
        return tf.math.reduce_mean(tf.multiply(f,f), name = 'mean_square')
    
    def confirm_dimensions(self,nx,ny,nz):
        assert self.nx == nx
        assert self.ny == ny
        assert self.nz == nz
        return None

    def xy_avg(self,f):
        _, nx, ny, nz = tuple(f.shape.as_list())
        self.confirm_dimensions(nx, ny, nz)
        return tf.math.reduce_mean(f, axis = (1,2), keepdims = False, name = 'xy_avg')
    
    def L_mass(self):
        return self.mean_square(self.dop.ddx_pointed(self.fields['u1']) + \
                self.dop.ddy_pointed(self.fields['u2']) + self.dop.ddz_pointed(self.fields['u3']))
    
    def L_mom(self):
        # Compute the residual of the pressure Poisson equations
        
        # tauij[:,:,:,0] --> tau_11
        # tauij[:,:,:,1] --> tau_12
        # tauij[:,:,:,2] --> tau_13
        # tauij[:,:,:,3] --> tau_22
        # tauij[:,:,:,4] --> tau_23
        # tauij[:,:,:,5] --> tau_33
    
#        Inertial_term = self.dop.ddx(self.dop.ddx( self.u*self.u ) ) +\
#                 2.*self.dop.ddx(self.dop.ddy( self.u*self.v ) ) +\
#                 2.*self.dop.ddx(self.dop.ddz( self.u*self.w ) ) +\
#                    self.dop.ddy(self.dop.ddy( self.v*self.v ) ) +\
#                 2.*self.dop.ddy(self.dop.ddz( self.v*self.w ) ) +\
#                    self.dop.ddz(self.dop.ddz( self.w*self.w ) )
#        Pressure_term = self.dop.ddx(self.dop.ddx(self.p)) + \
#           self.dop.ddy(self.dop.ddy(self.p)) + self.dop.ddz(self.dop.ddz(self.p))
#        Stress_term   = self.dop.ddx(self.dop.ddx(self.tauij[:,:,:,0])) + \
#                     2.*self.dop.ddx(self.dop.ddy(self.tauij[:,:,:,1])) + \
#                     2.*self.dop.ddx(self.dop.ddz(self.tauij[:,:,:,2])) + \
#                        self.dop.ddy(self.dop.ddy(self.tauij[:,:,:,3])) + \
#                     2.*self.dop.ddy(self.dop.ddz(self.tauij[:,:,:,4])) + \
#                        self.dop.ddz(self.dop.ddz(self.tauij[:,:,:,5]))     
    
#        return self.mean_square(Inertial_term + Pressure_term + Stress_term)
        return None
    
    def set_fields(self,Yhat):
        self.fields['u1'] = Yhat[:,0,:,:,:]
        self.fields['u2'] = Yhat[:,1,:,:,:]
        self.fields['u3'] = Yhat[:,2,:,:,:]
        if self.inc_mom:
            self.fields['p'] = Yhat[:,3,:,:,:]
    
        return None
    
    def compute_averages(self):
        # Stack in the following order: mean(U), <u1u1> ,<u1u2>, <u1u3>, <u2u2>, <u2u3>, <u3u3>
        self.current_state = tf.transpose( tf.stack([\
                self.xy_avg(self.fields['u1']),\
                self.xy_avg(tf.math.multiply(self.fields['u1'],self.fields['u1'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u1']),self.xy_avg(self.fields['u1'])), \
                self.xy_avg(tf.math.multiply(self.fields['u1'],self.fields['u2'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u1']),self.xy_avg(self.fields['u2'])), \
                self.xy_avg(tf.math.multiply(self.fields['u1'],self.fields['u3'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u1']),self.xy_avg(self.fields['u3'])), \
                self.xy_avg(tf.math.multiply(self.fields['u2'],self.fields['u2'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u2']),self.xy_avg(self.fields['u2'])), \
                self.xy_avg(tf.math.multiply(self.fields['u2'],self.fields['u3'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u2']),self.xy_avg(self.fields['u3'])), \
                self.xy_avg(tf.math.multiply(self.fields['u3'],self.fields['u3'])) - \
                        tf.math.multiply(self.xy_avg(self.fields['u3']),self.xy_avg(self.fields['u3']))], \
                ), perm = [1,0,2] )

        if self.inc_mom:
            # TODO: Needs to be implemented
            None
        return None

    def MSE(self,Y):
        return tf.math.reduce_mean(tf.multiply(Y-self.current_state, \
                Y-self.current_state), axis=(0,2))

    def Lcontent(self,Y):
        mse = self.MSE(Y)
        Lcont = tf.math.reduce_sum(mse) #self.L_uiuj() + self.L_U()
        return Lcont

    def compute_loss(self, Yhat, Y, lambda_p = 0.5, lambda_tau = 0.5):
        # Inputs:
        #   Yhat       --> NN output layer 
        #                  Type: TensorFlow tensor
        #                  Dimension: [m,nvars,nx,ny,nz]
        #   Y          --> "labels" (Type: TensorFlow tensor)
        #                  Type: TensorFlow tensor
        #                  Dimension: [m,nprfs,nz]
        #   lambda_p   --> hyperparameter of model that determines the ...
        #                  ... relative importance of the physics loss term.
        #                  Type: np.float32
        #   lambda_tau --> hyperparameter of model that determines the ...
        #                  ... relative importance of the residual stress ...
        #                  ... in the content loss.
        #                  Type: np.float32

        nx,ny,nz = self.nx, self.ny, self.nz
        m = tf.shape(Yhat)[0] 
        
        # Verify dimensions of input arrays

        assert self.nvars == tf.shape(Yhat)[1]
        assert nx == tf.shape(Yhat)[2]
        assert ny == tf.shape(Yhat)[3]
        assert nz == tf.shape(Yhat)[4]

        assert m == tf.shape(Y)[0]
        assert self.nprofs == tf.shape(Y)[1]
        assert nz == tf.shape(Y)[2]
       
        #self.extract_avg_profiles_from_labels_array(Y)
        self.set_fields(Yhat)
        self.compute_averages()

        # Compute loss functions
        Lphys = self.L_mass()
        Lcont = self.Lcontent(Y)

        if self.inc_mom:
            # TODO: compute momentume relavant loss terms
            #Lphys += self.L_mom()
            #Lcontent = (1. - lambda_tau)*Lcontent
            #Lcontent += lambda_tau*self.L_tauij()
            None
        total_loss = lambda_p*Lphys + (1. - lambda_p)*Lcont
        return total_loss

def load_data_for_loss_tests(datadir,nx,ny,nzC,nzF,Lx,Ly,Lz,tidx,tidy,navg,inc_prss = False):
    tidx_vec = np.array([tidx])
    tidy_vec = np.array([tidy])
    
    zF =  setup_domain_1D(0.5*Lz/nzF, Lz - 0.5*Lz/nzF, Lz/nzF)
    zC =  setup_domain_1D(0.5*Lz/nzC, Lz - 0.5*Lz/nzC, Lz/nzC)
    Yhat, Y, _, _ = load_dataset_V2(datadir, nx, ny, nzC, zF, zC, tidx_vec, \
            tidx_vec, tidy_vec, tidy_vec, inc_prss = inc_prss, navg = navg)
   
    Yhat = Yhat.reshape((1,3,nx,ny,nzC), order = 'F')
    Y = tf.cast(Y, tf.float32)
    Yhat = tf.cast(Yhat, tf.float32)
    return Y, Yhat

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 loss_fcns.py <datadir>")
        sys.exit()

#### For milestone ######
    datadir = sys.argv[1] + '/'
    
    # Spatial domain
    nx   = 192
    ny   = 192
    nzC  = 64
    nzF  = 256

    Lx = 6.*pi
    Ly = 3.*pi
    Lz = 1.

    # fname parameters
    navg = 840
    tidx = 179300
    tidy = 25400

    # Load channel data for loss tests
    Y, Yhat = load_data_for_loss_tests(datadir,nx,ny,nzC,nzF,Lx,Ly,Lz,tidx,tidy,navg,\
            inc_prss = False)
    
    # Create placeholders for tf operations
    # Load data into Loss class
    L_test = Loss(nx,ny,nzC,Lx,Ly,Lz,1,inc_mom = False)
    L_test.set_fields(Yhat)
    L_test.compute_averages()
    Lphys = L_test.L_mass()
    Lcont = L_test.Lcontent(Y)
    code.interact(local=locals())
    
    # TODO: Plot some ground-truth profiles to confirm proper initialization of the class

###### For final project ######
    # Test L_P
    
    # Test L_tauij

    # Test L_mom
    #Lmom = L_mom(u,v,w,p,tauij,A,B,dop,nx,ny,nz)
    #assert Lmom < 1.e-4, 'Lmom = {}'.format(Lmom)
    #print("L_mom test PASSED!")
