from scipy.special import expit
import numpy as np

# Activation functions for random basis
def SiLU(x):
    return x * expit(x)      

def ReLU(x):
    return np.maximum(x, 0.0)     

def tanh(x):
    return np.tanh(x)

def LeakyReLU(x, alpha=0.03):
    return np.where(x > 0, x, alpha * x)        

def Softplus(x, beta=1.0):
    return np.logaddexp(0.0, beta * x) / beta