# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 19:32:36 2020

@author: lenovo
"""

from sklearn import datasets #import class/fungsi dataset dari scikit-learn library

import matplotlib.pyplot as plt #import matplotlib

#Load the digits dataset
digits = datasets.load_digits() #memuat dan memasukkan dataset digits ke variabel digits

#menampilkan digit pertama
plt.figure(1, figsize=(3, 3))
plt.imshow(digits.images[-1], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()