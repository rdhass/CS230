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
from problem_parameters import nx, ny, nzC, nzF, Lx, Ly, Lz

# Array shapes: 
# X, Yhat: [m, nvars, nx, ny, nzC]
# Y:       [m, nprofs, nzC]

class Loss:
    def __init__(self,n_examples,inc_mom = False):
        # Inputs:
        #   n_examples --> Numer of training examples in current mini-batch
        #   inc_mom    --> logical whether or not to include "momentum" terms
        #                  in model (e.g. tauij, L_mom, pressure)
        self.inc_mom = inc_mom
        if self.inc_mom:
            self.nprofs = 14
            self.nvars = 10 # u, v, w, p, tau11, tau12, tau13, tau22, tau23, tau33
        else:
            self.nprofs = 7
            self.nvars = 3


        self.m = n_examples
        self.nx, self.ny, self.nz = nx, ny, nzC

        # Initialize derivative operator
        self.dop = DiffOps(nx, ny, nzC, Lx, Ly, Lz)

        # Allocate memory for the flow variables
        self.fields = {}
        self.fields["u1"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
        self.fields["u2"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
        self.fields["u3"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
        if self.inc_mom:
            self.fields["p"]     = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
            self.fields["tau11"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
            self.fields["tau12"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
            self.fields["tau13"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
            self.fields["tau22"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
            self.fields["tau23"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  
            self.fields["tau33"] = tf.zeros((n_examples,nx,ny,nzC), dtype = tf.float32)  

        # Allocate memory for the current state
        current_state = tf.zeros((n_examples,self.nprofs,nzC), dtype = tf.float32)

    def mean_square(self,f):
        return tf.math.reduce_mean(tf.multiply(f,f), name = 'mean_square')
    
    def confirm_dimensions(self,nx1,nx2,nx3):
        assert self.nx == nx1
        assert self.ny == nx2
        assert self.nz == nx3
        return None

    def xy_avg(self,f):
        _, nx1, nx2, nx3 = tuple(f.shape.as_list())
        self.confirm_dimensions(nx1, nx2, nx3)
        return tf.math.reduce_mean(f, axis = (1,2), keepdims = False, name = 'xy_avg')
    
    def L_mass(self):
        return self.mean_square(self.dop.ddx_pointed(self.fields['u1']) + \
                self.dop.ddy_pointed(self.fields['u2']) + self.dop.ddz_pointed(self.fields['u3']))
    
    def L_mom(self):
        # Compute the residual of the pressure Poisson equations
        
        Inertial_term = self.dop.ddx_pointed(self.dop.ddx_pointed( self.fields['u1']*self.fields['u1'] ) ) +\
                     2.*self.dop.ddx_pointed(self.dop.ddy_pointed( self.fields['u1']*self.fields['u2'] ) ) +\
                     2.*self.dop.ddx_pointed(self.dop.ddz_pointed( self.fields['u1']*self.fields['u3'] ) ) +\
                        self.dop.ddy_pointed(self.dop.ddy_pointed( self.fields['u2']*self.fields['u2'] ) ) +\
                     2.*self.dop.ddy_pointed(self.dop.ddz_pointed( self.fields['u2']*self.fields['u3'] ) ) +\
                        self.dop.ddz_pointed(self.dop.ddz_pointed( self.fields['u3']*self.fields['u3'] ) )

        Pressure_term = self.dop.ddx_pointed(self.dop.ddx_pointed(self.fields['p'])) + \
                        self.dop.ddy_pointed(self.dop.ddy_pointed(self.fields['p'])) + \
                        self.dop.ddz_pointed(self.dop.ddz_pointed(self.fields['p']))

        Stress_term   = self.dop.ddx_pointed(self.dop.ddx_pointed(self.fields['tau11'])) + \
                     2.*self.dop.ddx_pointed(self.dop.ddy_pointed(self.fields['tau12'])) + \
                     2.*self.dop.ddx_pointed(self.dop.ddz_pointed(self.fields['tau13'])) + \
                        self.dop.ddy_pointed(self.dop.ddy_pointed(self.fields['tau22'])) + \
                     2.*self.dop.ddy_pointed(self.dop.ddz_pointed(self.fields['tau23'])) + \
                        self.dop.ddz_pointed(self.dop.ddz_pointed(self.fields['tau33']))     
   
        return self.mean_square(Inertial_term + Pressure_term + Stress_term)
    
    def set_fields(self,Yhat):
        self.fields['u1'] = Yhat[:,0,:,:,:]
        self.fields['u2'] = Yhat[:,1,:,:,:]
        self.fields['u3'] = Yhat[:,2,:,:,:]
        if self.inc_mom:
            self.fields['p']     = Yhat[:,3,:,:,:]
            self.fields['tau11'] = Yhat[:,4,:,:]
            self.fields['tau12'] = Yhat[:,5,:,:]
            self.fields['tau13'] = Yhat[:,6,:,:]
            self.fields['tau22'] = Yhat[:,7,:,:]
            self.fields['tau23'] = Yhat[:,8,:,:]
            self.fields['tau33'] = Yhat[:,9,:,:]
    
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
                            tf.math.multiply(self.xy_avg(self.fields['u3']),self.xy_avg(self.fields['u3'])), \
                    self.xy_avg(self.fields['tau11']),\
                    self.xy_avg(self.fields['tau12']),\
                    self.xy_avg(self.fields['tau13']),\
                    self.xy_avg(self.fields['tau22']),\
                    self.xy_avg(self.fields['tau23']),\
                    self.xy_avg(self.fields['tau33']),\
                    self.xy_avg(self.fields['p'])],\
                    ), perm = [1,0,2] )
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
        #                  Dimension: [m,nvars,nx,ny,nzC]
        #   Y          --> "labels" (Type: TensorFlow tensor)
        #                  Type: TensorFlow tensor
        #                  Dimension: [m,nprfs,nzC]
        #   lambda_p   --> hyperparameter of model that determines the ...
        #                  ... relative importance of the physics loss term.
        #                  Type: np.float32
        #   lambda_tau --> hyperparameter of model that determines the ...
        #                  ... relative importance of the residual stress ...
        #                  ... in the content loss.
        #                  Type: np.float32
        # TODO: Incorporate lambda_tau. Right now the SGS stress loss is not ...
        #       weighted due to the challenges of scaling only a portion of a ...
        #       tf tensor (i.e. Y[:,7:-1,:] = sqrt(lambda_p)*Y[:,7:-1,:])

        nx,ny,nz = self.nx, self.ny, self.nz
        m = tf.shape(Yhat)[0] 
        
        # Verify dimensions of input arrays

        assert self.nvars == tf.shape(Yhat)[1]
        assert nx == tf.shape(Yhat)[2]
        assert ny == tf.shape(Yhat)[3]
        assert nzC == tf.shape(Yhat)[4]

        assert self.nprofs == tf.shape(Y)[1]
        assert nzC == tf.shape(Y)[2]
       
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

def load_data_for_loss_tests(datadir,tidx,tidy,navg,inc_prss = False):
    tidx_vec = np.array([tidx])
    tidy_vec = np.array([tidy])
    
    if inc_prss:
        _, Y, _, _ = load_dataset_V2(datadir, nx, ny, nzC, nzF, tidx_vec, \
                tidx_vec, tidy_vec, tidy_vec, inc_prss = inc_prss, navg = navg)
        Yhat = np.empty((1,10,nx,ny,nzC), dtype = np.float32)
        tidx = 17000
        fname = datadir + 'Run01_filt_t' + str(tidx).zfill(6) + '.h5'

        Yhat[:,0,:,:,:] = read_fortran_data(fname,'uVel')
        Yhat[:,1,:,:,:] = read_fortran_data(fname,'vVel')
        Yhat[:,2,:,:,:] = read_fortran_data(fname,'wVel')
        Yhat[:,3,:,:,:] = read_fortran_data(fname,'prss')
        Yhat[:,4,:,:,:] = read_fortran_data(fname,'tau11')
        Yhat[:,5,:,:,:] = read_fortran_data(fname,'tau12')
        Yhat[:,6,:,:,:] = read_fortran_data(fname,'tau13')
        Yhat[:,7,:,:,:] = read_fortran_data(fname,'tau22')
        Yhat[:,8,:,:,:] = read_fortran_data(fname,'tau23')
        Yhat[:,9,:,:,:] = read_fortran_data(fname,'tau33')
    else:
        Yhat, Y, _, _ = load_dataset_V2(datadir, nx, ny, nzC, nzF, tidx_vec, \
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
    
    # fname parameters
    navg = 840
    tidx = 179300
    tidy = 25400

    # Load channel data for loss tests
    Y, Yhat = load_data_for_loss_tests(datadir,tidx,tidy,navg,\
            inc_prss = False)
    
    # Load data into Loss class
    L_test = Loss(1,inc_mom = False)
    L_test.set_fields(Yhat)
    L_test.compute_averages()
    Lphys = L_test.L_mass()
    Lcont = L_test.Lcontent(Y)
    
    # TODO: Plot some ground-truth profiles to confirm proper initialization of the class

###### For final project ######
    Y, Yhat = load_data_for_loss_tests(datadir,tidx,tidy,navg,inc_prss = True)
    L_test = Loss(1,inc_mom = True)
    L_test.set_fields(Yhat)
    #code.interact(local=locals())
    L_test.compute_averages()
    Lmass = L_test.L_mass()
    Lmom  = L_test.L_mom()
    print('Lmom = {} | Expected output: {}'.format(Lmom,3.027846330297548e+04))
    Lcont = L_test.Lcontent(Y)
    # Test L_P
    
    # Test L_tauij

    # Test L_mom
    #Lmom = L_mom(u,v,w,p,tauij,A,B,dop,nx,ny,nz)
    #assert Lmom < 1.e-4, 'Lmom = {}'.format(Lmom)
    #print("L_mom test PASSED!")
