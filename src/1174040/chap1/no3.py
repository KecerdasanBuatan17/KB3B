# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:20:55 2020

@author: User
"""

from sklearn.linear_model import LogisticRegression #untuk mengimport linear_model dari library sklearn
from sklearn.datasets import make_blobs #untuk mengimport library datasets dari sklearn

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1) # untuk generate dataset dengan klasifikasi 2D
model = LogisticRegression() #menggunakan metode loginstic regression
model.fit(X, y)

Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1) # menentukan 1 buah contoh baru dimana jawabannya tidak diketahui

ynew = model.predict_proba(Xnew) # membuat sebuah prediksi dan memasukkan nya kedalam variable ynew
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i])) #menampilkan hasil prediksi