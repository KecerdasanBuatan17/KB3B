# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:08:34 2020

@author: User
"""

#Multiclass vs. Multilabel Fitting
from sklearn.svm import SVC
#pada baris ini merupakan sebuah perintah untuk mengimport class SVC dari packaged sklearn.svm
from sklearn.multiclass import OneVsRestClassifier
#pada baris ini merupakan sebuah perintah untuk mengimport class OneVsRestClassifier dari packaged sklearn.multiclass
from sklearn.preprocessing import LabelBinarizer
#pada baris ini merupakan sebuah perintah untuk mengimport class LabelBinarizer dari packaged sklearn.preprocessing
X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
y = [0, 0, 1, 1, 2]
classif = OneVsRestClassifier(estimator=SVC(gamma='scale',random_state=0))
classif.fit(X, y).predict(X)
y = LabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)


from sklearn.preprocessing import MultiLabelBinarizer
y = [[0, 1], [0, 2], [1, 3], [0, 2, 3], [2, 4]]
y = MultiLabelBinarizer().fit_transform(y)
classif.fit(X, y).predict(X)