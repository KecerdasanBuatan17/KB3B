# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 19:36:22 2020

@author: lenovo
"""

from sklearn import svm #import dari library sklearn
from sklearn import datasets #import dari library sklearn
clf = svm.SVC() #menggunakan Support Vector Classifier
iris = datasets.load_iris() #menggunakan iris dataset
X, y = iris.data, iris.target
clf.fit(X, y)