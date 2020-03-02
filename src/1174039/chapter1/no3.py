# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:29:41 2020

@author: Liyana
"""

from sklearn.linear_model import LogisticRegression # digunakan untuk mengimport linear_model dari library sklearn
from sklearn.datasets import make_blobs #digunakan untuk mengimport library datasets dari sklearn

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1) # dapat untuk generate dataset dengan klasifikasi 2D
model = LogisticRegression() # dapat menggunakan metode loginstic regression
model.fit(X, y)

Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1) # dapat menentukan 1 buah contoh baru dimana jawabannya tidak dapat diketahui

ynew = model.predict_proba(Xnew) # membuat sebuah prediksi dan memasukkan nya kedalam variable ynew
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i])) #menampilkan hasil prediksi