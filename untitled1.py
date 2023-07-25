# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:26:05 2021

@author: SAYED
"""



import numpy as np
import matplotlib.pyplot as plt
import pickle
from Neural_Network import *
"""
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 
training images and 10000 test images.
The dataset is divided into five training batches and one test batch, each with 10000 images. The test batch contains 
exactly 1000 randomly-selected images from each class. The training batches contain the remaining images in random 
order, but some training batches may contain more images from one class than another. Between them, the training 
batches contain exactly 5000 images from each class.
"""
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
    ### START CODE HERE ### (≈ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Get W1, b1, W2 and b2 from the dictionary parameters.
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    # Loop (gradient descent)

    for i in range(0, num_iterations):

        # Forward propagation: LINEAR -> RELU -> LINEAR -> SIGMOID. Inputs: "X, W1, b1, W2, b2". Output: "A1, cache1, A2, cache2".
        ### START CODE HERE ### (≈ 2 lines of code)
        A1, cache1 = linear_activation_forward(X, W1, b1, activation="relu")
        A2, cache2 = linear_activation_forward(A1, W2, b2, activation="sigmoid")
        ### END CODE HERE ###
        
        #print('A2 Y', A2, Y)
        cost = compute_cost_cifar_10(A2, Y)
        ### END CODE HERE ###
        #print('Iteration Cost',i, cost)
        #print(cost.shape)
        #print('Iteration Cost',i, cost)
        ### END CODE HERE ###
        if i % 500 == 0:
            print('Iteration Cost',i, cost)
        
        # Initializing backward propagation
        dA2 = - (np.divide(Y, A2) - np.divide(1 - Y, 1 - A2))
        
        # Backward propagation. Inputs: "dA2, cache2, cache1". Outputs: "dA1, dW2, db2; also dA0 (not used), dW1, db1".
        ### START CODE HERE ### (≈ 2 lines of code)
        dA1, dW2, db2 =  linear_activation_backward(dA2, cache2, activation = "sigmoid")
        dA0, dW1, db1 = linear_activation_backward(dA1, cache1, activation = "relu")
        ### END CODE HERE ###
        
        # Set grads['dWl'] to dW1, grads['db1'] to db1, grads['dW2'] to dW2, grads['db2'] to db2
        grads['dW1'] = dW1
        grads['db1'] = db1
        grads['dW2'] = dW2
        grads['db2'] = db2
        
        # Update parameters.
        ### START CODE HERE ### (approx. 1 line of code)
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###

        # Retrieve W1, b1, W2, b2 from parameters
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]
        

    return parameters




if __name__ == "__main__":
    """show it works"""


    train_data, train_labels, test_data, test_labels, = load_cifar_10_data('cifar-10-batches-py')

    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test labels: ", test_labels.shape)
    
    train_data = train_data[0:1000]
    train_labels = train_labels[0][0:1000]
    

    b = np.unique(train_labels)
    #print('b ', b, train_labels.shape)
    
    train_labels = np.array([[1 if i == j else 0 for i in train_labels] for j in b])
    
    #train_labels = np.expand_dims(train_labels, axis=0)
    
    test_data = test_data[0:100]
    test_labels = test_labels[0][0:100]
    #test_labels = np.expand_dims(test_labels, axis=0)
    
    
    b = np.unique(test_labels)
    
    test_labels = np.array([[1 if i == j else 0 for i in test_labels] for j in b])
    
    train_x_flatten = train_data.reshape(train_data.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_data.reshape(test_data.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_data = train_x_flatten / 255.0
    test_data = test_x_flatten / 255.0
    
    
    
    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test labels: ", test_labels.shape)
    
    n_x = 3072     # num_px * num_px * 3
    n_h = 7
    n_y = 10
    layers_dims = (n_x, n_h, n_y)
    
    parameters = two_layer_model(train_data, train_labels, layers_dims = (n_x, n_h, n_y), num_iterations = 2500, print_cost=True)
    
    
    predictions_train = predict_cifar_10(train_data, train_labels, parameters)
    
    predictions_test = predict_cifar_10(test_data, test_labels, parameters)
    

    