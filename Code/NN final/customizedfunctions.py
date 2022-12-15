import numpy as np
import matplotlib.pyplot as plt

# Sigmoid Activation function
def sigmoidActivation(linearTransform):
    # To calculate: S(x)= 1/(1+e^(-x))
    activationFunction = 1/(1+np.exp(-linearTransform))
    return activationFunction, linearTransform

# Rectified Linear Unit (ReLU) Activation function
def reluActivation(linearTransform):
    activationFunction = np.maximum(0,linearTransform)
    return activationFunction, linearTransform

# To calculate and return ReLU backward derivatve
def reluDerivative(derActivation, stash):
    linearTransform = stash
    derivativeLinearTranformation = np.array(derActivation, copy=True)
    derivativeLinearTranformation[linearTransform <= 0] = 0
    return derivativeLinearTranformation

# To calculate and return Sigmoid backward derivatve
def sigmoidDerivative(derActivation, stash):
    linearTransform = stash
    activatn = 1/(1+np.exp(-linearTransform))
    derivativeLinearTranformation = derActivation * activatn * (1 - activatn)
    return derivativeLinearTranformation

# Forward Propogation methods
def linearForward(A, W, b):
    Z = W.dot(A) + b
    return Z, (A, W, b)

def linearActivationForward(aPrev, W, b, activation):
    if activation == "sigmoid":
        Z, linearStash = linearForward(aPrev, W, b)
        A, activationStash = sigmoidActivation(Z)
    elif activation == "relu":
        Z, linearStash = linearForward(aPrev, W, b)
        A, activationStash = reluActivation(Z)
    return A, (linearStash, activationStash)

def linearModelForward(X, parameters):
    stashs = []
    A = X
    paramLength = len(parameters) // 2
    i = 1
    while i < paramLength:
        aPrev = A
        A, stash = linearActivationForward(aPrev, parameters['W' + str(i)], parameters['b' + str(i)], activation = "relu")
        stashs.append(stash)
        i += 1
    AL, stash = linearActivationForward(A, parameters['W' + str(paramLength)], parameters['b' + str(paramLength)], activation = "sigmoid")
    stashs.append(stash)
    return AL, stashs

def computedCost(AL, Y):
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y,np.log(AL).T) - np.dot(1-Y, np.log(1-AL).T))
    cost = np.squeeze(cost)
    return cost


# Backward Propogation methods
def linearBackward(dZ, stash):
    aPrev, W, b = stash
    m = aPrev.shape[1]
    dW = 1./m * np.dot(dZ,aPrev.T)
    db = 1./m * np.sum(dZ, axis = 1, keepdims = True)
    aDerivativePrev = np.dot(W.T,dZ)
    return aDerivativePrev, dW, db

def linearActivationBackward(dA, stash, activation):
    linearStash, activationStash = stash
    if activation == "relu":
        dZ = reluDerivative(dA, activationStash)
        aDerivativePrev, dW, db = linearBackward(dZ, linearStash)
    elif activation == "sigmoid":
        dZ = sigmoidDerivative(dA, activationStash)
        aDerivativePrev, dW, db = linearBackward(dZ, linearStash)
    return aDerivativePrev, dW, db

def linearModelBackward(AL, Y, stashs):
    grads = {}
    paramStashes = len(stashs)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_stash = stashs[paramStashes-1]
    grads["dA" + str(paramStashes-1)], grads["dW" + str(paramStashes)], grads["db" + str(paramStashes)] = linearActivationBackward(dAL, current_stash, activation = "sigmoid")
    l = 0
    for l in reversed(range(paramStashes - 1)):
        current_stash = stashs[l]
        aDerivativePrev_temp, dW_temp, db_temp = linearActivationBackward(grads["dA" + str(l + 1)], current_stash, activation = "relu")
        grads["dA" + str(l)] = aDerivativePrev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp
    return grads

def updateParameters(parameters, grads, learning_rate):
    paramLength = len(parameters) // 2
    l = 0
    while l < paramLength:
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate * grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate * grads["db" + str(l+1)]
        l += 1
    return parameters
