# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 10:37:49 2020

@author:Luthfi Muhammad Nabil
"""

from sklearn import svm #Mengimport class SVM dari library sklearn
from sklearn import datasets #Mengimport datasets dari library sklearn
clf = svm.SVC() #Memasukkan implementasi dari "Support Vector Classification" ke variabel clf
X, y = datasets.load_iris(return_X_y=True) #Untuk memuat dan memasukkan dataset iris ke variabel bernama iris
clf.fit(X, y) #Untuk melakukan pengiriman data training set ke method fit
#%%
import pickle #Mengimport pickle untuk implementasi protokol serializing dan de-serializing
s = pickle.dumps(clf) #Untuk serialize hirarki dari object
clf2 = pickle.loads(s) #Untuk memuat dan men de-serialize hirarki dari object
clf2.predict(X[0:1]) #Untuk melakukan prediksi nilai yang baru berdasarkan gambar terakhir dari digits.data
#%%
y[0] #Untuk menampilkan data iris koordinat y
#%%
from joblib import dump, load #Mengambil class dump dan load dari library joblib
dump(clf, 'filename.joblib') #Untuk memasukkan data ke dalam sebuah file
#%%
clf = load('filename.joblib') #Untuk memuat data dari file