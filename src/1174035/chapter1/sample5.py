# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:56:18 2020

@author: Luthfi Muhammad Nabil
"""

import numpy as np #Memuat library numpy yang akan diinisialisasikan menjadi np
from sklearn import random_projection #Memuat class random_projection dari library sklearn

rng = np.random.RandomState(0) #Memuat angka random dan mengekspos angka untuk menghasilkan dari berbagai probabilitas distribusi, angka tersebut dimasukkan ke variabel rng
X = rng.rand(10, 2000) #Membatasi jarak angka random
X = np.array(X, dtype='float32') #Memuat angka menjadi ke dalam array
X.dtype #Mengambil tipe data dari variabel X
#%%
transformer = random_projection.GaussianRandomProjection() #Mengimplementasikan modul yang simpel dan efisien secara komputasi untuk mengurangi dimensi dari data
X_new = transformer.fit_transform(X) #Untuk membuat penengahan data
X_new.dtype #Mengambil tipe data dari variabel X

#%%
from sklearn import datasets #Mengimport datasets dari library sklearn
from sklearn.svm import SVC 
iris = datasets.load_iris() #Untuk memuat dan memasukkan dataset iris ke variabel bernama iris
clf = SVC() #Memasukkan implementasi dari "Support Vector Classification" ke variabel clf
clf.fit(iris.data, iris.target) #Untuk melakukan pengiriman data training set ke method fit
#%%
list(clf.predict(iris.data[:3])) #Untuk memuat dan men de-serialize hirarki dari object, data tersebut akan dimasukkan ke fungsi list
#%%
clf.fit(iris.data, iris.target_names[iris.target]) #Untuk melakukan pengiriman data training set ke method fit
#%%
list(clf.predict(iris.data[:3])) #Untuk memuat dan men de-serialize hirarki dari object, data tersebut alan dimasukkan ke fumgsi list

