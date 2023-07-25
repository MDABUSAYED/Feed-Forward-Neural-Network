# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 01:22:27 2021

@author: SAYED
"""

from Neural_Network import *

import numpy as np
import h5py
import time

def L_layer_model(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 3000, print_cost=False):#lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []                         # keep track of cost
    
    # Parameters initialization. (≈ 1 line of code)
    parameters = initialize_parameters_deep(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = L_model_forward(X, parameters)
        
        # Compute cost.
        cost =  compute_cost(AL, Y)
        
        if i % 500 == 0:
            print('Iteration Cost',i, cost)
    
        # Backward propagation.
        grads = L_model_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
    
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
    
    
    layers_dims = [12288,  7, 1] 
    
    print("Layer dims: ", layers_dims)
    
    parameters = L_layer_model(train_data, train_labels, layers_dims, num_iterations = 2500, print_cost = True)
    
    predictions_train = predict(train_data, train_labels, parameters)
    
    predictions_test = predict(test_data, test_labels, parameters)
    
    
    
    
if __name__ == "__main__":
    main()