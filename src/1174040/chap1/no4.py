# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 22:27:13 2020

@author: User
"""

from sklearn import svm #menigmport svm dari library sklearn
from sklearn import datasets #menigmport datasets dari library sklearn
clf = svm.SVC() #menggunakan method SVC
iris = datasets.load_iris() #menggunakan dataset iris
X, y = iris.data, iris.target #memasukkan x sebagai iris data , dan y sebagai iris target
clf.fit(X, y) #laalu menggunakan metod fit .