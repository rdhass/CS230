from domain_setup import setup_domain_1D
import numpy as np
from numpy import pi
from read_fortran_data import read_fortran_data
from scipy import interpolate

def get_x_vec(datadir,X,buff,tid_vec,dsname):
    for n, tid in enumerate(tid_vec):
        tidstr = str(tid).zfill(7)
        if n > 0:
            X[n-1,:] = buff.flatten(order = 'F')
        for i, ds in enumerate(dsname):
            buff[i,:,:,:] = read_fortran_data(datadir + 'Run01_' + \
                    tidstr + '.h5', ds)
    X[-1,:] = buff.flatten(order = 'F')
    return X

def get_y_vec(datadir, Y, buff, zF, zC, tid_vec, prof_ids, navg = 1):
    # Output: 
    #   Y --> labels vector of dimension [n_examples, nprofs, nzC]
    
    nzC = np.size(zC)
    nzF = np.size(zF)
    for n, tid in enumerate(tid_vec):
        fname = datadir + 'Run01_budget0_t' + str(tid).zfill(6) + '_n' + \
                str(navg).zfill(6) + '_nzF' + str(nzF) + '.stt'
        
        # Read in average profiles from disk. These correspond to the "Fine" grid
        avgF = np.genfromtxt(fname,dtype = np.float32).T

        # Interpolate the average profiles to the zC locations (i.e. "Course" grid)
        avgC = np.zeros((avgF.shape[0],nzC))
        for i in range(avgF.shape[0]):
            tck = interpolate.splrep(zF.T, avgF[i,:], s=0)
            avgC[i,:] = interpolate.splev(np.squeeze(zC.T), tck, der=0)

        # Copy the profiles we need to the buffer array
        for i, prof in enumerate(prof_ids):
            buff[i,:] = avgC[prof,:]

        # Copy the flattened buffer to the Y-vector
        Y[n,:,:] = buff
    return Y

def load_dataset_V2(data_directory,nx,ny,nz,zF,zC,x_tid_vec_train,x_tid_vec_test,\
        y_tid_vec_train, y_tid_vec_test, inc_prss = True, navg = 1):
    # Inputs:
    #   data_directory  --> directory path where raw data resides
    #   nx, ny, nz      --> number of grid points in computational domain
    #   x_tid_vec_test  --> vector of time ID's from the simulation for the training features 
    #   x_tid_vec_train --> "                                             " test features
    #   y_tid_vec_test  --> "                                             " training labels 
    #   y_tid_vec_train --> "                                             " test labels
    #   inc_prss        --> are we including pressure in the input layer?
    #   navg            --> how many snapshots the profiles are averaged over
 
    tsteps_train = x_tid_vec_train.size
    tsteps_test  = x_tid_vec_test.size
    assert tsteps_train == y_tid_vec_train.size
    assert tsteps_test  == y_tid_vec_test.size

    ncube = nx*ny*nz
    dsname = ['uVel','vVel','wVel'] # Dataset names in the hdf5 files
    if inc_prss:
        nfields = 4
        nprofs  = 14 # meanU, <uu>, <uv>, <uw>, <vv>, <vw>, <ww>, <tau11>, <tau12>
                     # <tau13>, <tau22>, <tau23>, <tau33>, meanP
        prof_id = (0,3,4,5,6,7,8,17,18,13,19,14,20,16)
        dsname.append('prss')
    else:
        nfields = 3
        nprofs  = 7 # meanU, <uu>, <uv>, <uw>, <vv>, <vw>, <ww>
        prof_id = (0,3,4,5,6,7,8)
    assert len(prof_id) == nprofs

    buff_x = np.empty((nfields,nx,ny,nz), dtype = np.float32, order = 'F')
    buff_y = np.empty((nprofs,nz),        dtype = np.float32, order = 'F')
    
    # Initialize training and test features and labels
    train_set_x = np.empty((tsteps_train, nfields*ncube), dtype = np.float32, order = 'F') 
    test_set_x  = np.empty((tsteps_test,  nfields*ncube), dtype = np.float32, order = 'F')
    train_set_y = np.empty((tsteps_train, nprofs, nz), dtype = np.float32, order = 'F') 
    test_set_y  = np.empty((tsteps_test,  nprofs, nz), dtype = np.float32, order = 'F')
    
    train_set_x = get_x_vec(data_directory, train_set_x, buff_x, x_tid_vec_train, dsname)
    test_set_x  = get_x_vec(data_directory, test_set_x,  buff_x, x_tid_vec_test,  dsname)

    train_set_y = get_y_vec(data_directory, train_set_y, buff_y, zF, zC, y_tid_vec_train, prof_id, navg = navg)
    test_set_y  = get_y_vec(data_directory, test_set_y,  buff_y, zF, zC, y_tid_vec_test,  prof_id, navg = navg)

    #train_set_y = train_set_y.reshape((1, train_set_y.shape[0]))
    #test_set_y = test_set_y.reshape((1, test_set_y.shape[0]))
    
    return train_set_x, train_set_y, test_set_x, test_set_y

################## TESTS ###############################
def test_load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = True, navg = 1):
    X_train, Y_train, X_test, Y_test = \
            load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = inc_prss, navg = navg)
    print("Shape of X_train: {}".format(X_train.shape))
    print("Shape of X_test: {}".format(X_test.shape))
    print("Shape of Y_train: {}".format(Y_train.shape))
    print("Shape of Y_test: {}".format(Y_test.shape))
    print(X_train[:10])
    print(X_test[:10])
    print(Y_train[0,0,:10])
    print(Y_test[0,0,:10])
    return None

if __name__ == '__main__':
    data_directory = '/Users/ryanhass/Documents/MATLAB/CS_230/data/'
    nx = 192
    ny = 192
    nz = 64
    nzF = 256
    Lx = 6.*pi
    Ly = 3.*pi
    Lz = 1.

    zC = setup_domain_1D(0.5*Lz/nz , Lz - 0.5*Lz/nz , Lz/nz)
    zF = setup_domain_1D(0.5*Lz/nzF, Lz - 0.5*Lz/nzF, Lz/nzF)
    print(zF[:3])

    x_tid_vec_test = np.array([179300])
    x_tid_vec_train = np.array([179300])
    y_tid_vec_test = np.array([25400])
    y_tid_vec_train = np.array([25400])
    test_load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = False, navg = 840)
