import numpy as np 

##### Activations #####

def sigmoid(x):             #Sigmoid Activation
    return 1/(1 + np.exp(-x))

def tanh(x):                #Tanh Activation 
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def softsign(x):            #SoftSign Activation
    return x/(1 + abs(x))

def ReLU(x):                #Relu Activation
    return np.maximum(0, x)

def softmax(x):             #Softmax Activation
    e_x = np.exp(x)
    return e_x/e_x.sum()

def linear(x):
    return x 


##### Derivatives #####

def d_sigmoid(x, dA):
    s = sigmoid(x)
    return dA*s*(1-s) 

def d_tanh(x, dA):
    return dA*(1 - tanh(x)**2)  

def d_softsign(x, dA):
    return dA/((1 + abs(x))**2)

def d_ReLU(x, dA):
    dz = dA.copy()
    dz[x <= 0] = 0 
    return dz

def d_linear(x, dA):
    return dA 