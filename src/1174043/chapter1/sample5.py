# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 20:04:46 2020

@author: lenovo
"""

import numpy as np #import dari library sklearn
from sklearn import random_projection #import dari library sklearn

rng = np.random.RandomState(0) #Menggunakan fungsi random
X = rng.rand(10, 2000) #mengambil nilai random
X = np.array(X, dtype='float32') #membuat array bertipe float32
X.dtype