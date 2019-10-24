import numpy as np
import matplotlib.pylab as plt

def step(x):
    y = x>0
    return y.astype(np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

def identity(x):
    return x

def softmax(x):
    c = np.max(x)
    exp_a = np.exp(x - c) # Overflow 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a/sum_exp_a
    return y
