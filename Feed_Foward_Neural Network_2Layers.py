# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 01:22:27 2021

@author: SAYED
"""

from Neural_Network import *

import time
import numpy as np
import h5py


def two_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):
    """
    Implements a two-layer neural network: LINEAR->RELU->LINEAR->SIGMOID.
    
    Arguments:
    X -- input data, of shape (n_x, number of examples)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- dimensions of the layers (n_x, n_h, n_y)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- If set to True, this will print the cost every 100 iterations 
    
    Returns:
    parameters -- a dictionary containing W1, W2, b1, and b2
    """
    
    np.random.seed(1)
    grads = {}
    costs = []                              # to keep track of the cost
    m = X.shape[1]                           # number of examples
    (n_x, n_h, n_y) = layers_dims
    
    # Initialize parameters dictionary, by calling one of the functions you'd previously implemented
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        
        
        # Compute cost
        
        cost = compute_cost(A2, Y)
        
        if i % 500 == 0:
            print('Iteration Cost', i, cost)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        
        dA1, dW2, db2 =  linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        
        parameters = update_parameters(parameters, grads, learning_rate)

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        
    return parameters



def main():
    
    train_data, train_labels, test_data, test_labels, classes = load_data()
    
    
    train_x_flatten = train_data.reshape(train_data.shape[0], -1).T   
    test_x_flatten = test_data.reshape(test_data.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_data = train_x_flatten / 255.0
    test_data = test_x_flatten / 255.0
    
    print('Dataset Name: Cat vs Not Cat')
    
    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test labels: ", test_labels.shape)
    
    
    n_x = 12288     
    n_h = 7
    n_y = 1
    layers_dims = (n_x, n_h, n_y)
    
    parameters = two_layer_model(train_data, train_labels, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    
    
    predictions_train = predict(train_data, train_labels, parameters)
    
    predictions_test = predict(test_data, test_labels, parameters)
    
    
    
    
if __name__ == "__main__":
    main()