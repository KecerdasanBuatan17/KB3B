# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 23:24:21 2020

@author: User
"""

import numpy as npy #mengimport numpy sebagai npy
from sklearn import random_projection #mengimport random_projection dari library sklearn

rng = npy.random.RandomState(0) #Menggunakan fungsi random dari numpy
X = rng.rand(10, 2000) #membuat range random diantara 10 sampai 2000
X = npy.array(X, dtype='float32') #yang dijadikan array dengan tipe data float32
X.dtype