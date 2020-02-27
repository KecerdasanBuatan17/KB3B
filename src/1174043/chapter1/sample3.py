# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:07:08 2020

@author: lenovo
"""

from sklearn.linear_model import LogisticRegression #import dari library sklearn
from sklearn.datasets import make_blobs #import dari library sklearn

X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1) # generate dataset klasifikasi 2d
# fit final model
model = LogisticRegression()
model.fit(X, y)

Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1) # menentukan 1 buah contoh baru dimana jawabannya tidak diketahui

ynew = model.predict_proba(Xnew) # membuat prediksi
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i])) #menampilkan hasil prediksi