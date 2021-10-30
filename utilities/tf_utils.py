from domain_setup import setup_domain
import h5py
import numpy as np
from numpy import pi
from read_fortran_data import read_fortran_data
import tensorflow as tf
import math
from scipy import interpolate

def load_dataset(data_directory):
    train_dataset = h5py.File(data_directory + 'train_signs.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:]) # your train set features
    train_set_y_orig = np.array(train_dataset["train_set_y"][:]) # your train set labels

    test_dataset = h5py.File(data_directory + 'test_signs.h5', "r")
    test_set_x_orig = np.array(test_dataset["test_set_x"][:]) # your test set features
    test_set_y_orig = np.array(test_dataset["test_set_y"][:]) # your test set labels

    classes = np.array(test_dataset["list_classes"][:]) # the list of classes
    
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes

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
    for n, tid in enumerate(tid_vec):
        fname = datadir + 'Run01_budget0_t' + str(tid).zfill(6) + '_n' + \
                str(navg).zfill(6) + '.stt'
        
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

def random_mini_batches(X, Y, mini_batch_size, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    
    Arguments:
    X -- input data, of shape (input size, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    mini_batch_size - size of the mini-batches, integer
    seed -- this is only for the purpose of grading, so that you're "random minibatches are the same as ours.
    
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    
    m = X.shape[1]                  # number of training examples
    mini_batches = []
    np.random.seed(seed)
    
    # Step 1: Shuffle (X, Y)
    permutation = list(np.random.permutation(m))
    shuffled_X = X[:, permutation]
    shuffled_Y = Y[:, permutation].reshape((Y.shape[0],m))

    # Step 2: Partition (shuffled_X, shuffled_Y). Minus the end case.
    num_complete_minibatches = math.floor(m/mini_batch_size) # number of mini batches of size mini_batch_size in your partitionning
    for k in range(0, num_complete_minibatches):
        mini_batch_X = shuffled_X[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch_Y = shuffled_Y[:, k * mini_batch_size : k * mini_batch_size + mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    # Handling the end case (last mini-batch < mini_batch_size)
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch_Y = shuffled_Y[:, num_complete_minibatches * mini_batch_size : m]
        mini_batch = (mini_batch_X, mini_batch_Y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def convert_to_one_hot(Y, C):
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y


def predict(X, parameters):
    
    W1 = tf.convert_to_tensor(parameters["W1"])
    b1 = tf.convert_to_tensor(parameters["b1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    b2 = tf.convert_to_tensor(parameters["b2"])
    W3 = tf.convert_to_tensor(parameters["W3"])
    b3 = tf.convert_to_tensor(parameters["b3"])
    
    params = {"W1": W1,
              "b1": b1,
              "W2": W2,
              "b2": b2,
              "W3": W3,
              "b3": b3}
    
    x = tf.placeholder("float", [12288, 1])
    
    z3 = forward_propagation_for_predict(x, params)
    p = tf.argmax(z3)
    
    sess = tf.Session()
    prediction = sess.run(p, feed_dict = {x: X})
        
    return prediction

def forward_propagation_for_predict(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
    
    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    # Retrieve the parameters from the dictionary "parameters" 
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3'] 
                                                           # Numpy Equivalents:
    Z1 = tf.add(tf.matmul(W1, X), b1)                      # Z1 = np.dot(W1, X) + b1
    A1 = tf.nn.relu(Z1)                                    # A1 = relu(Z1)
    Z2 = tf.add(tf.matmul(W2, A1), b2)                     # Z2 = np.dot(W2, a1) + b2
    A2 = tf.nn.relu(Z2)                                    # A2 = relu(Z2)
    Z3 = tf.add(tf.matmul(W3, A2), b3)                     # Z3 = np.dot(W3,Z2) + b3
    
    return Z3
   
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
