import numpy as np
import matplotlib.pylab as plt

# book1 99p

import sys, os
import NN_func
from os import chdir
chdir("/Users/jaeyeonic/Desktop/Code/Python_Deeplearning/Python_Deeplearning/Python_Study")

sys.path.append(os.pardir)

from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]

print(label) #5

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)
img_show(img)
