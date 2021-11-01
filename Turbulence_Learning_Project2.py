import sys
sys.path.insert(0,'/Users/ryanhass/Documents/MATLAB/CS_230/Final_project/utilities')
import numpy as np
from numpy import pi
import code
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict
from io_mod import load_dataset_V2
import time
import h5py
import os
import matplotlib.pyplot as plt


'''

Example inputs:
    
    N_neurons_in_layers = [1000,  50,  50,  1000]
                            ^     ^     ^      ^
                  input layer  hidden layers  output layer

NOTE: N_neurons_in_layers includes the input and output layers! The numbers in these slots must agree with the input and output shapes!
    
    FP_sequence = [["LINEAR", "RELU"], ["LINEAR", "RELU"], ["LINEAR", "SOFTMAX"]]

NOTE: - FP_sequence defines the sequence of transforms to be applied to the input (during forward propagation) to get to the output (i.e., the prediction;
        what you will plug into the loss function for training). The increments of applied transforms (i.e., ["LINEAR", "RELU"]) allow for easier
        understanding of where the neurons exist in the sequence (i.e., to get from the input to the output of the first hidden layer, we apply
        FP_sequence[0], which is a LINEAR transform, followed by a RELU transform).
      - The example input for FP_sequence corresponds to LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX. The LINEAR and RELU transforms are applied
        in the forward propagation function, and the SOFTMAX (i.e., the final transform in any sequence) is applied independently to allow for different
        transforms to be applied to the output for cost evaluation (this is just for generalization purposes).

    X_train_All = numpy array of shape (inputs, number of samples)

    Y_train_All = numpy array of shape (inputs, number of samples)
    
    X_test = numpy array of shape (inputs, number of samples)
    
    Y_test = numpy array of shape (inputs, number of samples)

'''

class NN_model:
    def __init__(self, N_neurons_in_layers, FP_sequence, X_train_All, Y_train_All, X_test, Y_test,\
            nx, ny, nz, Lx, Ly, Lz, lambda_p, lambda_tau, inc_mom):

        self.N_neurons_in_layers = N_neurons_in_layers
        self.FP_sequence = FP_sequence

        self.X_train_All = X_train_All
        self.Y_train_All = Y_train_All

        self.total_samples = self.X_train_All.shape[1]
        #for i in range(len(self.X_train_All)):
        #    self.total_samples += self.X_train_All[i].shape[1]

        self.X_test = X_test
        self.Y_test = Y_test


        ############################################################
        ############################################################
        ##################### Graph Creation #######################
        ############################################################
        ############################################################
        
        # "Graphs" essentially hold all the weights, variables, etc. Multiple can be created, but
        # we really just need 1.

        #################################################################################################
        ########################################### Graph 1 #############################################
        #################################################################################################

        # Initiate the graph
        self.Graph1 = tf.Graph()

        # Initiate the loss function object
        self.Loss = Loss(nx,ny,nz,Lx,Ly,Lz,inc_mom = inc_mom)

        # Define our network in this graph
        with self.Graph1.as_default():

            # Initialize the weights and biases for the network (in dictionary "self.parameters").
            self.standard_initialize_parameters(initialization_method = "Xavier")

            # Load labeled data from the data vector
            self.X_Graph1 = self.X_train_All
            self.Y_Graph1 = self.Y_train_All

            # Create placeholders for the input and labeled output vectors based on the number of neurons per layer provided in N_neurons_in_layers.
            self.X_Graph1_PH, self.Y_Graph1_PH = self.create_placeholders(input_size = N_neurons_in_layers[0], output_size = N_neurons_in_layers[-1], input_name = "Graph1_Input", output_name = "Graph1_Output")

            # Define the sequence of transforms to be applied to the input (during forward propagation) to get to the output (output = Y_NNpredicted, i.e., what you will plug into the loss function for the network prediction)
            # - Example input for FP_sequence_Graph1 for LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX would be FP_sequence_Graph1 = [["LINEAR", "RELU"], ["LINEAR", "RELU"], ["LINEAR", "SOFTMAX"]]
            # - FP_sequence_Graph1 is a list of lists. The inner lists are commands to be applied to the output of the previous layer to get to the next layer, i.e., ["LINEAR", "RELU"] applies the LINEAR tranformation, and then the RELU transformation
            # - By default (i.e., FP_sequence_Graph1 = []), the code will define this function as a list of ["LINEAR", "RELU"] for all layers, except for the output layer, which is ["LINEAR", "SOFTMAX"].
            FP_sequence_Graph1 = self.FP_sequence

            # Define use forward propagation sequence to define the output tensor (prediction of the network)
            self.Y_Graph1_NNpredict = self.standard_forward_propagation(X = self.X_Graph1_PH, FP_sequence = FP_sequence_Graph1)

            # Calculate the loss, which involves the forward propagation outputs, the label data, and a loss function (which should be in last entry of the last list of FP_sequence_Graph1)
            self.Loss.comput_loss(self.X_Graph1_PH, self.Y_Graph1_NNpredict, self.Y_Graph1, \
                    lambda_p = lambda_p, lambda_tau = lambda_tau)
            #self.loss_Graph1 = self.compute_loss(Y_NNpredict = self.Y_Graph1_NNpredict, \
            #        Y = self.Y_Graph1_PH, FP_sequence = FP_sequence_Graph1)
            #self.loss_Graph1_PHYSICS = self.compute_loss(Y_NNpredict = self.Y_Graph1_NNpredict, Y = self.Y_Graph1_PH, FP_sequence = FP_sequence_Graph1)

            #################################################################################################
            ############################ Prepare Total Loss, Optimizer, etc. ################################
            #################################################################################################

            # Sum the losses to acquire the total loss function
            self.total_loss = self.loss_Graph1

            # Set the optimizer
            self.set_optimizer(Opt_type = "Adam")

        






    def create_placeholders(self, input_size, output_size, input_name = None, output_name = None):
        # Creates placeholders for the input and output vectors.
        
        if input_name == None:
            X = tf.placeholder(dtype = tf.float32, shape = (input_size, None))
        else:
            X = tf.placeholder(dtype = tf.float32, shape = (input_size, None), name = input_name)

        if output_name == None:
            Y = tf.placeholder(dtype = tf.float32, shape = (output_size, None))
        else:
            Y = tf.placeholder(dtype = tf.float32, shape = (output_size, None), name = output_name)
        
        return X, Y


    def standard_initialize_parameters(self, initialization_method = "Xavier"):
        # Initializes weights and biases.
        
        if initialization_method == "Xavier":
            # To initiate a random number seed  ->  tf.set_random_seed(1)
            self.parameters = {}
            for i in range(1, len(self.N_neurons_in_layers)):
                ############## OLD VERSION ##############
                # Example with seed  ->  tf.contrib.layers.xavier_initializer(seed = 1)
                #W_dummy = tf.Variable("W" + str(i), [self.N_neurons_in_layers[i], self.N_neurons_in_layers[i - 1]], initializer = tf.contrib.layers.xavier_initializer())
                #self.parameters.update({"W" + str(i) : W_dummy})
                #b_dummy = tf.Variable("b" + str(i), [self.N_neurons_in_layers[i], 1], initializer = tf.contrib.layers.xavier_initializer())
                #self.parameters.update({"b" + str(i) : b_dummy})

                ############## NEW VERSION ############## (works for tensorflow version 1)
                W_dummy = self.xavier_init(size = [self.N_neurons_in_layers[i], self.N_neurons_in_layers[i - 1]])
                self.parameters.update({"W" + str(i) : W_dummy})
                b_dummy = self.xavier_init(size = [self.N_neurons_in_layers[i], 1])
                #b_dummy = tf.Variable(tf.zeros([self.N_neurons_in_layers[i], 1], dtype=tf.float32), dtype=tf.float32)
                self.parameters.update({"b" + str(i) : b_dummy})

        else:
            print("ERROR: Inappropriate initialization method")


    def xavier_init(self, size):
        out_dim = size[0]
        in_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([out_dim, in_dim], stddev = xavier_stddev), dtype = tf.float32)
    

    def standard_forward_propagation(self, X, FP_sequence = []):
        """
        - Implements the forward propagation for the model.
        - Example input for FP_sequence for LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX would be FP_sequence = [["LINEAR", "RELU"], ["LINEAR", "RELU"], ["LINEAR", "SOFTMAX"]
        - FP_sequence is a list of lists. The inner lists are commands to be applied to the output of the previous layer to get to the next layer, i.e., ["LINEAR", "RELU"] applies the LINEAR tranformation, and then the RELU transformation
        - By default (i.e., FP_sequence = []), this function will create an FP_sequence list of ["LINEAR", "RELU"] for all layers, including the output layer.

        Arguments:
        X -- input dataset placeholder, of shape (input size, number of examples)

        Returns:
        Zi -- the output of the last unit prior to the output of the network (i.e., the output to the last LINEAR block in the above example)
        """

        # As a default, create the typical fully-connect neural network using LINEAR -> RELU at all layers, except the output layer, which uses ["LINEAR", "SOFTMAX"]
        if FP_sequence == []:
            for i in range(len(N_neurons_in_layers) - 1):
                if i == len(N_neurons_in_layers) - 2:
                    FP_sequence.append(["LINEAR", "SOFTMAX"])
                else:
                    FP_sequence.append(["LINEAR", "RELU"])


        # Make sure the sequence of commands corresponds to the number of layers of the current model
        assert len(FP_sequence) == len(N_neurons_in_layers) - 1


        # Apply transforms to the input according to the sequence provided in FP_sequence
        Zi_dummy = None
        for i in range(len(FP_sequence)):
            for j in range(len(FP_sequence[i])):
                
                if FP_sequence[i][j] == "LINEAR":
                    if i == 0 and j == 0:
                        Zi_dummy = self.linear_transform(W = self.parameters["W" + str(i + 1)], X = X, b = self.parameters["b" + str(i + 1)])
                    else:
                        Zi_dummy = self.linear_transform(W = self.parameters["W" + str(i + 1)], X = Zi_dummy, b = self.parameters["b" + str(i + 1)])

                elif FP_sequence[i][j] == "RELU":
                    if i == 0 and j == 0:
                        Zi_dummy = self.relu_transform(X)
                    else:
                        Zi_dummy = self.relu_transform(Zi_dummy)

                elif FP_sequence[i][j] == "TANH":
                    if i == 0 and j == 0:
                        Zi_dummy = self.tanh_transform(X)
                    else:
                        Zi_dummy = self.tanh_transform(Zi_dummy)

                elif FP_sequence[i][j] == "SIGMOID":
                    if i == 0 and j == 0:
                        Zi_dummy = self.sigmoid_transform(X)
                    else:
                        Zi_dummy = self.sigmoid_transform(Zi_dummy)

                # Do not apply the last transformation in FP_sequence. This transform considered as the transform to compute the loss/cost. 
                if i == len(FP_sequence) - 1 and j == len(FP_sequence[i]) - 1:
                    break

        # The output just before the last transformation in FP_sequence
        Zi = Zi_dummy

        return Zi

    def linear_transform(self, W, X, b):
        return tf.add(tf.matmul(W, X), b)

    def relu_transform(self, Z):
        return tf.nn.relu(Z)

    def tanh_transform(self, Z):
        return tf.tanh(Z)

    def sigmoid_transform(self, Z):
        return tf.sigmoid(Z)


    def compute_loss(self, Y_NNpredict, Y, FP_sequence):
        """
        - Computes different types of loss using the labeled data (Y) and the predicted output (Y_NNpredict).
        - Y_NNpredict is the output of forward propagation (output of the transform before the last transform) (shape is (# of outputs, # of samples) )
        - Y is a placeholder for the outputs (same shape as Y_NNpredict)
        
        Returns:
        - loss is a function of the forward propagation, the loss function, and the input and output placeholders 
        """
        
        if FP_sequence[-1][-1] == "SOFTMAX":
            # To fit the TensorFlow requirement for tf.nn.softmax_cross_entropy_with_logits(...,...)
            logits = tf.transpose(Y_NNpredict)
            labels = tf.transpose(Y)
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = labels, logits = logits))
        
        elif FP_sequence[-1][-1] == "SQUARED_ABSOLUTE_ERROR":
            loss = tf.reduce_mean(tf.square(Y_NNpredict - Y))

        elif FP_sequence[-1][-1] == "PHYSICS_CONTINUITY_EQ":
            print("ERROR: compute loss option not coded yet")
            # tf.gradients(output, input)
        else:
            print("ERROR: Nothing was selected to compute the loss")
        
        return loss


    def set_optimizer(self, Opt_type = "Adam", learning_rate = 0.0001):
        # Select the optimizer type.

        self.learning_rate = learning_rate

        if Opt_type == "Adam":
            self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.total_loss)
        else:
            print("ERROR: Invalid optimizer selection.")


    def train(self, N_epochs = 1500, minibatch_size = 32, print_cost = True):
        # Train the network.

        # To be able to rerun the model without overwriting tf variables
        ops.reset_default_graph()
        
        costs = []

        # Start the TensorFlow session (using the formulated Graph, here assumed to be Graph 1!)
        with tf.Session(graph = self.Graph1) as sess:

            # Initialize all the variables
            sess.run(tf.global_variables_initializer())
            
            # Do the training loop
            for epoch in range(N_epochs):

                # Defines a cost related to an epoch
                epoch_cost = 0.

                # number of minibatches of size minibatch_size in the train set
                num_minibatches = int(self.total_samples / minibatch_size)
                
                # use a seed for randomizing minibatches
                #seed = seed + 1
                #minibatches = random_mini_batches(X_train, Y_train, minibatch_size, seed)
                minibatches = random_mini_batches(X = self.X_train_All, Y = self.Y_train_All, mini_batch_size = minibatch_size)

                for minibatch in minibatches:

                    # Select a minibatch
                    (minibatch_X, minibatch_Y) = minibatch
                    
                    # Run the optimizer on a minibatch
                    _ , minibatch_cost = sess.run([self.optimizer, self.total_loss], feed_dict = {self.X_Graph1_PH : minibatch_X, self.Y_Graph1_PH : minibatch_Y})
                    
                    epoch_cost += minibatch_cost / minibatch_size

                # Print the cost every epoch
                if print_cost and epoch % 100 == 0:
                    print ("Cost after epoch %i: %f" % (epoch, epoch_cost))
                if print_cost and epoch % 5 == 0:
                    costs.append(epoch_cost)
                
            # plot the cost
            plt.plot(np.squeeze(costs))
            plt.ylabel('cost')
            plt.xlabel('iterations (per fives)')
            plt.title("Learning rate =" + str(self.learning_rate))
            plt.savefig('Cost_vs_Iteration.png', dpi=300)
            plt.close()
            

            # Lets save the parameters in a variable
            self.parameters = sess.run(self.parameters)
            print ("Parameters have been trained!")

            # Calculate the correct predictions
            correct_prediction = tf.equal(tf.argmax(self.Y_Graph1_NNpredict), tf.argmax(self.Y_Graph1_PH))

            # Calculate accuracy on the test set
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            print ("Train Accuracy:", accuracy.eval({self.X_Graph1_PH : self.X_train_All, self.Y_Graph1_PH: self.Y_train_All}))
            print ("Test Accuracy:", accuracy.eval({self.X_Graph1_PH : self.X_test, self.Y_Graph1_PH : self.Y_test}))
        
        #return self.parameters


    def predict(self, X_for_prediction):

        # Start the TensorFlow session
        with tf.Session(graph = self.Graph1) as sess:
            Y_prediction = sess.run(self.Y_Graph1_NNpredict, feed_dict = {self.X_Graph1_PH : X_for_prediction})
        
        return Y_prediction



if __name__ == "__main__":
    
    if len(sys.argv) < 4:
        print("Usage: ")
        print("  python3 Turbulence_Learning_Project2.py <nzF> <lambda_p> <lambda_tau>")
        exit(0)

    # Reverting to TensorFlow version 1
    tf.disable_v2_behavior()
    
    # File path from code to data directory
    data_directory = "Finger_Number_Data/"

    # Spatial domain
    nxC = 192
    nyC = 192
    nzC = 64
    nzF = sys.argv[1]

    Lx = 6.*pi
    Ly = 3.*pi
    Lz = 1.
    
    # Define the low and high resolution computational domains (we actually only need the z-vectors)
    zC = setup_domain_1D(Lz/nzC*0.5, Lz - Lz/nzC*0.5, Lz/nzC)
    zF = setup_domain_1D(Lz/nzF*0.5, Lz - Lz/nzF*0.5, Lz/nzF)

    # Loss function parameters
    lambda_p = sys.argv[2]
    lambda_tau = sys.argv[3]
    inc_mom = False

    ### Data IO parameters ###
    # Specify which time steps are included in the training/test sets for both...
    # ... input features and labels
    x_tid_vec_train = np.array([])
    x_tid_vec_test = np.array([])
    y_tid_vec_train = np.array([])
    y_tid_vec_test = np.array([])

    # How many timesteps were used to generate the averages (if tavg is used)
    navg = 840

    # Load the data (this uses a function already written from CS230)
    #X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset(data_directory)
    X_train_orig, Y_train_orig, X_test_orig, Y_test_orig = \
            load_dataset_V2(data_directory, nx, ny, nz, zF, zC, x_tid_vec_train, \
            x_tid_vec_test, y_tid_vec_train, y_tid_vec_test, \
            inc_prss = False, nsteps_avg = navg)

    #*** NOTE: load_dataset_V2 returns flattened data ***
    
    # Flatten the training and test images
    #X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
    #X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
    
    # Normalize image vectors
    # TODO: Properly normalize the data
    #X_train = X_train_flatten/255.
    #X_test = X_test_flatten/255.
    
    # Convert training and test labels to one hot matrices
    #Y_train = convert_to_one_hot(Y_train_orig, 6)
    #Y_test = convert_to_one_hot(Y_test_orig, 6)



    #### These will eventually be helpful in using the new data.
    '''
    # Go into data directory
    os.chdir(data_directory)

    # Get the file names
    file_names = os.listdir()

    H5_file_test = h5py.File(file_names[0], 'r')
    H5_file_train = h5py.File(file_names[1], 'r')

    
    for i in range(1080):
        if i == 0:
            X_training = H5_file_train['train_set_x'][i].reshape( (1, 3*64*64) )
        else:
            X_training = np.append(X_training, H5_file_train['train_set_x'][i].reshape( (1, 3*64*64) ), 0)
    X_training = np.transpose(X_training)/255
    Y_training = H5_file_train['train_set_y'][:].reshape( (1, 1080) )
    
    for i in range(120):
        if i == 0:
            X_test = H5_file_test['test_set_x'][i].reshape( (1, 3*64*64) )
        else:
            X_test = np.append(X_test, H5_file_test['test_set_x'][i].reshape( (1, 3*64*64) ), 0)
    X_test = np.transpose(X_test)/255
    Y_test = H5_file_test['test_set_y'][:].reshape( (1, 120) )
    '''



    
    FP_sequence = [["LINEAR", "RELU"], ["LINEAR", "RELU"], ["LINEAR", "SOFTMAX"]]
    N_neurons_in_layers = [X_train.shape[0], 25, 25, Y_train.shape[0]]


    #### These will eventually be helpful in using the new data.
    '''
    H5_file = h5py.File(file_names[0], 'r')

    prss = H5_file['prss'].astype(np.float32())[:]
    prss_input_data = prss.reshape( (prss.shape[0]*prss.shape[1]*prss.shape[2], 1) )
    uVel = H5_file['uVel'].astype(np.float32())[:]
    uVel_input_data = uVel.reshape( (uVel.shape[0]*uVel.shape[1]*uVel.shape[2], 1) )
    vVel = H5_file['vVel'].astype(np.float32())[:]
    vVel_input_data = vVel.reshape( (vVel.shape[0]*vVel.shape[1]*vVel.shape[2], 1) )
    wVel = H5_file['wVel'].astype(np.float32())[:]
    wVel_input_data = wVel.reshape( (wVel.shape[0]*wVel.shape[1]*wVel.shape[2], 1) )

    X_train_All = uVel_input_data
    Y_train_All = vVel_input_data
    X_test = uVel_input_data
    Y_test = vVel_input_data
    FP_sequence = [["LINEAR", "RELU"], ["LINEAR", "RELU"], ["LINEAR", "SOFTMAX"]]

    N_neurons_in_layers = [uVel_input_data.shape[0], 50, 50, uVel_input_data.shape[0]]
    '''




    # Define the model
    model = NN_model(N_neurons_in_layers, FP_sequence, X_train, Y_train, X_test, Y_test, \
            nx, ny, nz, Lx, Ly, Lz, lambda_p, lambda_tau, inc_mom)

    # Train the model
    model.train()

    code.interact(local = locals())
