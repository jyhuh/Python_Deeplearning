import numpy as np
import matplotlib.pylab as plt
import NN_func
#sigmoid
#relu
#identity
#step
#softmax

#book1 91p
import sys, os

from os import chdir
chdir("/Users/jaeyeonic/Desktop/Code/Python_Deeplearning/Python_Deeplearning/Python_Study")
print(os.getcwd())
print(os.listdir('.'))
sys.path.append(os.pardir)
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=True)

print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)
zz
