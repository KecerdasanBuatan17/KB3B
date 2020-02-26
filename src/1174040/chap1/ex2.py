# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 18:19:36 2020

@author: User
"""


from sklearn import datasets #untunk mengimport dataset dari library learn-scikit
iris = datasets.load_iris() #membuat sebuah variable iris yang mempunyai isi yaitu dataset iris
digits = datasets.load_digits() #membuat sebuah variable digits yang mempunyai isi yaitu dataset digits

print(digits.data) #memberikan akses ke fitur yang dapat digunakan untuk mengklasifikasikan sampel digit dan menampilkannya di console

digits.target #memberikan informasi tentang data yang berhubungan atau juga dapat dijadikan sebagai label

digits.images[0] #Data selalu berupa array 2D, shape (n_samples, n_features), meskipun data aslinya mungkin memiliki bentuk yang berbeda.
