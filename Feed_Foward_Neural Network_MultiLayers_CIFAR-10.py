# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 18:26:05 2021

@author: SAYED
"""


import numpy as np
import pickle


def unpickle(file):
    """load the cifar-10 data"""

    with open(file, 'rb') as fo:
        data = pickle.load(fo, encoding='bytes')
    return data


def load_cifar_10_data(data_dir, negatives=False):
    """
    Load cifar-10 dataset in python without using any Machine Learning Library, only use numpy.
    
    Arguments:
    data_dir -- Directroty of cifar-10 dataset 
    
    Returns:
    train_data -- 50000 image with a dimension of (32, 32, 3)
    train_labels -- Output labels(value form 0-9) for 50000 train images
    test_data -- 10000 image with a dimension of (32, 32, 3)
    test_labels --Output labels(value form 0-9) for 10000 test images
    """


    meta_data_dict = unpickle(data_dir + "/batches.meta")
    label_names = meta_data_dict[b'label_names']
    label_names = np.array(label_names)

    # training data
    train_data = None
    train_labels = []


    for i in range(1, 6):
        train_data_dict = unpickle(data_dir + "/data_batch_{}".format(i))
        if i == 1:
            train_data = train_data_dict[b'data']
        else:
            train_data = np.vstack((train_data, train_data_dict[b'data']))
        train_labels += train_data_dict[b'labels']

    train_data = train_data.reshape((len(train_data), 3, 32, 32))
    if negatives:
        train_data = train_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        train_data = np.rollaxis(train_data, 1, 4)
        

    train_labels = np.array(train_labels)
    
    train_labels = np.expand_dims(train_labels, axis=0)


    test_data_dict = unpickle(data_dir + "/test_batch")
    test_data = test_data_dict[b'data']
    

    test_labels = test_data_dict[b'labels']

    test_data = test_data.reshape((len(test_data), 3, 32, 32))
    if negatives:
        test_data = test_data.transpose(0, 2, 3, 1).astype(np.float32)
    else:
        test_data = np.rollaxis(test_data, 1, 4)
        
        
    test_labels = np.array(test_labels)
    
    test_labels = np.expand_dims(test_labels, axis=0)

    return train_data, train_labels, test_data, test_labels


def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1 / (1 + np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0, Z)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy = True)
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    
    return dZ



def initialize_parameters_multilayer(layer_dims):
    """
    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network
    
    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL":
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """
    
    np.random.seed(1)
    
    parameters = {}
    L = len(layer_dims)           

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) / np.sqrt(layer_dims[l-1]) 
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        
    return parameters

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
    Z = W.dot(A) + b
    
    cache = (A, W, b)
    
    return Z, cache

def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value 
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """
    
    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)
    
    elif activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)
    
    cache = (linear_cache, activation_cache)

    return A, cache

def linear_multilayer_forward(X, parameters):
    """
    Implement forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation
    
    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters_deep()
    
    Returns:
    AL -- last post-activation value
    caches -- list of caches containing:
                every cache of linear_relu_forward() (there are L-1 of them, indexed from 0 to L-2)
                the cache of linear_sigmoid_forward() (there is one, indexed L-1)
    """

    caches = []
    A = X
    L = len(parameters) // 2
    
    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A 
        A, cache = linear_activation_forward(A_prev, parameters['W' + str(l)], parameters['b' + str(l)], activation = "relu")
        caches.append(cache)
    
    # Implement LINEAR -> SIGMOID. Add "cache" to the "caches" list.
    AL, cache = linear_activation_forward(A, parameters['W' + str(L)], parameters['b' + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    return AL, caches

def compute_cost_cifar_10(AL, Y):
    """
    Implement the cost function for L-layer multi-class neural network.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (10, number of examples)
    Y -- true "label" vector (for example: (containing 1 if belongs to corresponding class, otherwise 0)), shape (10, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from AL and Y.
    cost = 0.0
    for i in range(m):
        cost += (-np.dot(Y[:, i].T, np.log(AL[:, i])) - np.dot((1 - Y[:, i]).T, np.log(1 - AL[:, i])))
    cost /= m
    
    return cost

def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = 1./m * np.dot(dZ,A_prev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T,dZ)
    
    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
    Arguments:
    dA -- post-activation gradient for current layer l 
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"
    
    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    
    linear_cache, activation_cache = cache
    
    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
        
    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def linear_multilayer_backward(AL, Y, caches):
    """
    Implement the backward propagation for the [LINEAR->RELU] * (L-1) -> LINEAR -> SIGMOID group
    
    Arguments:
    AL -- probability vector, output of the forward propagation (L_model_forward())
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing:
                every cache of linear_activation_forward() with "relu" (there are (L-1) or them, indexes from 0 to L-2)
                the cache of linear_activation_forward() with "sigmoid" (there is one, index L-1)
    
    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
    """
    
    grads = {}
    L = len(caches) # the number of layers
   
    Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
    # Initializing the backpropagation
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
    # Lth layer (SIGMOID -> LINEAR) gradients. Inputs: "AL, Y, caches". Outputs: "grads["dAL"], grads["dWL"], grads["dbL"]
    current_cache = caches[L-1]
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, activation = "sigmoid")
    
    for l in reversed(range(L-1)):
        # lth layer: (RELU -> LINEAR) gradients.
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation = "relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    grads -- python dictionary containing your gradients, output of L_model_backward
    
    Returns:
    parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
    """
    
    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        
    return parameters

def predict_cifar_10(X, y, parameters, dataset):
    """
    This function is used to predict the results of CIFAR-10 Dataset using  any-layer multi-class neural network.
    
    
    Arguments:
    X -- data set of examples you would like to label
    y -- true label
    parameters -- parameters of the trained model
    dataset -- dataset description such as training or test
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1,m))
    true_label = np.zeros((1,m))
    
    # Forward propagation
    probas, caches = linear_multilayer_forward(X, parameters)

    # convert probas to [0-9] predictions
    for i in range(0, probas.shape[1]):
        p[0,i] = np.argmax(probas[:,i])
        true_label[0,i] = np.argmax(y[:,i])
    
    print("Accuracy on " + dataset + ' : ' + str((np.sum(p == true_label) * 100) / m) + '%')
    
    return p


def Feed_Foward_Neural_Network_MultiLayers(X, Y, layers_dims, learning_rate = 0.0075, num_iterations = 5000):
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.
    
    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 1 if belongs to corresponding class, otherwise 0), of shape (10, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    learning_rate -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    
    # Parameters initialization.
    parameters = initialize_parameters_multilayer(layers_dims)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations + 1):

        # Forward propagation: [LINEAR -> RELU]*(L-1) -> LINEAR -> SIGMOID.
        AL, caches = linear_multilayer_forward(X, parameters)
        
        # Compute cost.
        cost =  compute_cost_cifar_10(AL, Y)
        
        if i % 500 == 0:
            print('Iteration Cost',i, cost)
    
        # Backward propagation: LINEAR -> SIGMOID -> [LINEAR -> RELU]*(L-1).
        grads = linear_multilayer_backward(AL, Y, caches)
 
        # Update parameters.
        parameters = update_parameters(parameters, grads, learning_rate)
                
        
    return parameters

def main():
    
    train_data, train_labels, test_data, test_labels, = load_cifar_10_data('cifar-10-batches-py')
    
    print('Dataset Name: CIFAR-10')
    print('Original Data Shape : ')
    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test labels: ", test_labels.shape)
    
    #Take a portion of training data as running whole dataset takes too much time
    train_data = train_data[0:2500]
    train_labels = train_labels[0][0:2500]
    
    
    # Split train_labels for multiclass into [0, 1]
    b = np.unique(train_labels)
    train_labels = np.array([[1 if i == j else 0 for i in train_labels] for j in b])
    
    #train_labels = np.expand_dims(train_labels, axis=0)
    
    test_data = test_data[0:250]
    test_labels = test_labels[0][0:250]
    #test_labels = np.expand_dims(test_labels, axis=0)
    
    
    
    # Split test_labels for multiclass into [0, 1]
    b = np.unique(test_labels)
    test_labels = np.array([[1 if i == j else 0 for i in test_labels] for j in b])
    
    train_x_flatten = train_data.reshape(train_data.shape[0], -1).T   # The "-1" makes reshape flatten the remaining dimensions
    test_x_flatten = test_data.reshape(test_data.shape[0], -1).T

    # Standardize data to have feature values between 0 and 1.
    train_data = train_x_flatten / 255.0
    test_data = test_x_flatten / 255.0
    
    
    print('Selected Date Shape : ')
    print("Train data: ", train_data.shape)
    print("Train labels: ", train_labels.shape)
    print("Test data: ", test_data.shape)
    print("Test labels: ", test_labels.shape)
    
    layers_dims = [3072, 40, 20, 17, 12, 10] 
    
    print("Layer dims: ", layers_dims)
    
    parameters = Feed_Foward_Neural_Network_MultiLayers(train_data, train_labels, layers_dims, num_iterations = 2500)
    
    prediction_train = predict_cifar_10(train_data, train_labels, parameters, dataset = 'Training Data')
    
    prediction_test = predict_cifar_10(test_data, test_labels, parameters, dataset = 'Test Data')

if __name__ == "__main__":
    main()


    