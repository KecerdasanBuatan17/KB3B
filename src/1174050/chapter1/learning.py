# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:17:01 2020

@author: User
"""
from sklearn import svm
#merupakan sebuah perintah untuk mengimport class svm dari packaged sklearn
from sklearn import datasets
# merupakan sebuah perintah untuk mengimport class datasets dari packaged sklearn
clf = svm.SVC(gamma='scale')
#clf sebagai estimator/parameter, svm.SVC sebagai class, gamma sebagai parameter untuk menetapkan nilai secara manual dengan nilai scale
iris = datasets.load_iris()
#iris sebagai estimator/parameter, datasets.load iris() sebagai item dari suatu nilai
X, y = iris.data, iris.target
#X, y sebagai estimator/parameter, iris.data, iris.target sebagai item dari 2 nilai yang ada
clf.fit(X, y)
#clf sebagai estimator/parameter dengan menggunakan metode fit untuk memanggil estimator X, y dengan outputannya



