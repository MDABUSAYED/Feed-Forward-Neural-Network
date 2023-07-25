# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 16:03:41 2021

@author: SAYED
"""

from mnist import MNIST
import numpy as np

mndata = MNIST('mnist/')
mndata.gz = True
images, labels = mndata.load_training()


data = [images[i] for i in range(10000)]
data = np.array(data)
print(data.shape)
#data = np.expand_dims(data, axis=0)
    
l = [labels[i] for i in range(10000)]
l = np.array(l)
print(data.shape)
print(l.shape)