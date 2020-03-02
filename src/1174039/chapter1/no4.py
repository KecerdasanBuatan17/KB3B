# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:30:06 2020

@author: Liyana
"""

from sklearn import svm #dapat mengimport svm dari library sklearn
from sklearn import datasets #digunakan untuk mengimport datasets dari library sklearn
clf = svm.SVC() # digunakan  dengan menggunakan method SVC
iris = datasets.load_iris() # digunakan dengan menggunakan dataset iris
X, y = iris.data, iris.target #memasukkan x sebagai iris data , dan y sebagai iris target
clf.fit(X, y) #laalu menggunakan metod fit .