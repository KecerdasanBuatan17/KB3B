# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:21:09 2020

@author: Liyana
"""

from sklearn import datasets # digunakan untunk mengimport dataset mulai dari library learn-scikit
iris = datasets.load_iris() # Dapat membuat sebuah variable iris yang mempunyai isi yaitu dataset iris
digits = datasets.load_digits() # Digunakan untuk membuat sebuah variable digits yang mempunyai isi yaitu dataset digits

print(digits.data) # Dapat memberikan akses ke fitur yang dapat digunakan untuk mengklasifikasikan sampel digit dan menampilkannya di console

digits.target # Dapat memberikan informasi tentang data yang berhubungan atau juga dapat dijadikan sebagai label

digits.images[0] #Data dapat berupa array 2D, shape (n_samples, n_features), meskipun data aslinya mungkin memiliki bentuk yang berbeda.
