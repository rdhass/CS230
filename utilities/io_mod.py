from domain_setup import setup_domain
import numpy as np
from numpy import pi
from read_fortran_data import read_fortran_data
from scipy import interpolate

def get_x_vec(datadir,X,buff,tid_vec,dsname):
    for n, tid in enumerate(tid_vec):
        tidstr = str(tid).zfill(7)
        if n > 0:
            X[:,n-1] = buff.flatten(order = 'F')
        for i, ds in enumerate(dsname):
            buff[:,:,:,i] = read_fortran_data(datadir + 'Run01_' + \
                    tidstr + '.h5', ds)
    X[:,-1] = buff.flatten(order = 'F')
    return X

def get_y_vec(datadir, Y, buff, zF, zC, tid_vec, prof_ids, navg = 1):
  
    nzC = np.size(zC)
    nzF = np.size(zF)
    for n, tid in enumerate(tid_vec):
        fname = datadir + 'Run01_budget0_t' + str(tid).zfill(6) + '_n' + \
                str(navg).zfill(6) + '_nzF' + str(nzF) + '.stt'
        
        # Read in average profiles from disk. These correspond to the "Fine" grid
        avgF = np.genfromtxt(fname,dtype = np.float64)

        # Interpolate the average profiles to the zC locations (i.e. "Course" grid)
        avgC = np.zeros((nzC,avgF.shape[1]))
        for i in range(avgF.shape[1]):
            tck = interpolate.splrep(zF, avgF[:,i], s=0)
            avgC[:,i] = interpolate.splev(np.squeeze(zC), tck, der=0)

        # Copy the profiles we need to the buffer array
        for i, prof in enumerate(prof_ids):
            buff[:,i] = avgC[:,prof]

        # Copy the flattened buffer to the Y-vector
        Y[:,n] = buff.flatten(order = 'F')
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

    buff_x = np.empty((nx,ny,nz,nfields), dtype = np.float64, order = 'F')
    buff_y = np.empty((nz,nprofs),        dtype = np.float64, order = 'F')
    
    # Initialize training and test features and labels
    train_set_x_orig = np.empty((nfields*ncube,tsteps_train), dtype = np.float64, order = 'F') 
    test_set_x_orig  = np.empty((nfields*ncube,tsteps_test),  dtype = np.float64, order = 'F')
    train_set_y_orig = np.empty((nprofs*nz,tsteps_train), dtype = np.float64, order = 'F') 
    test_set_y_orig  = np.empty((nprofs*nz,tsteps_test),  dtype = np.float64, order = 'F')
    
    train_set_x_orig = get_x_vec(data_directory, train_set_x_orig, buff_x, x_tid_vec_train, dsname)
    test_set_x_orig  = get_x_vec(data_directory, test_set_x_orig,  buff_x, x_tid_vec_test,  dsname)

    train_set_y_orig = get_y_vec(data_directory, train_set_y_orig, buff_y, zF, zC, y_tid_vec_train, prof_id, navg = navg)
    test_set_y_orig  = get_y_vec(data_directory, test_set_y_orig,  buff_y, zF, zC, y_tid_vec_test,  prof_id, navg = navg)

    #train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    #test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig

################## TESTS ###############################
def test_load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = True, navg = 1):
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = \
            load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = inc_prss, navg = navg)
    print("Shape of X_train_orig: {}".format(X_train_orig.shape))
    print("Shape of X_test_orig: {}".format(X_test_orig.shape))
    print("Shape of Y_train_orig: {}".format(Y_train_orig.shape))
    print("Shape of Y_test_orig: {}".format(Y_test_orig.shape))
    print(X_train_orig[:10])
    print(X_test_orig[:10])
    print(Y_train_orig[:10])
    print(Y_test_orig[:10])
    return None

if __name__ == '__main__':
    data_directory = '/Users/ryanhass/Documents/MATLAB/CS_230/data/'
    nx = 192
    ny = 192
    nz = 64
    nzF = 128
    Lx = 6.*pi
    Ly = 3.*pi
    Lz = 1.

    _, _, _, _, _, zC, _, _, _ = setup_domain(nz = nz, Lz = Lz)
    _, _, _, _, _, zF, _, _, _ = setup_domain(nz = nzF, Lz = Lz)

    x_tid_vec_test = np.array([179300])
    x_tid_vec_train = np.array([179300])
    y_tid_vec_test = np.array([73200])
    y_tid_vec_train = np.array([73200])
    test_load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = False, navg = 3020)