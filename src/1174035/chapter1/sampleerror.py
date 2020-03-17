# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:32:23 2020

@author: Luthfi Muhammad Nabil
"""

from sklearn import svm #Untuk mengimport class sv dari library sklearn
from sklearn import datasets #Untuk import class/fungsi dataset dari scikit-learn library
clf = svm.SVC(gamma=0.001, C=100) #Memasukkan implementasi dari "Support Vector Classification" ke variabel clf

#%%

clf.fit(digits.data[:-1], digits.target[:-1]) #Untuk melakukan pengiriman data training set ke method fit
#%%
clf.predict(digits.data[-1:]) #Untuk melakukan prediksi nilai yang baru berdasarkan gambar terakhir dari digits.data

