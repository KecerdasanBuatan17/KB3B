# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 15:55:58 2020

@author: Luthfi Muhammad Nabil
"""

from sklearn import datasets #Untuk import class/fungsi dataset dari scikit-learn library
iris = datasets.load_iris() #Untuk memuat dan memasukkan dataset iris ke variabel bernama iris
digits = datasets.load_digits() #Untuk memuat dan memasukkan dataset digits ke variabel digits
#%%
print(digits.data) #Menampilkan object berformat Dictionary-like yang nanti akan ditampilkan pada console
#%%
digits.target #Menunjukkan data angka yang berhubungan dengan setiap digit gambar yang sedang dipelajari
#%%
digits.images[0] #Akan mengambil data dengan berformat array 2 Dimensi, dengan bentuk parameter (n_samples, n_features)