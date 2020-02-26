# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 22:45:29 2020

@author: User
"""

#Type Casting
from sklearn import svm
#pada baris ini merupakan sebuah perintah untuk mengimport class svm dari packaged sklearn
from sklearn import random_projection
#pada baris ini merupakan sebuah perintah untuk mengimport class random projection dari packaged sklearn
rng = np.random.RandomState(0)
#rng sebagai estimator/parameter dengan nilai suatu itemnya yaitu np.random.RandomState(0)
X = rng.rand(10, 2000)
#X sebagai estimator/parameter dengan nilai item rng.rand
X = np.array(X, dtype='float32')
#X sebagai estimator/parameter dengan nilai item np.array
X.dtype
#X.dtype sebagai item pemanggil
transformer = random_projection.GaussianRandomProjection()
#transformer sebagai estimator/parameter dengan memanggil class random projection
X_new = transformer.fit_transform(X)
#X new di sini sebagai estomator/parameter dan menggunakan metode fit
X_new.dtype
#X new.dtype sebagai item


from sklearn import datasets
#pada baris ini merupakan sebuah perintah untuk mengimport class datasets dari packaged sklearn
from sklearn.svm import SVC
#pada baris ini merupakan sebuah perintah untuk mengimport class SVC dari packaged sklearn.svm
iris = datasets.load_iris()
#iris sebagai estimator/parameter dengan item datasets.load iris()
clf = SVC(gamma='scale')
#clf sebagai estimator/parameter dengan nilai class SVC pada parameter gamma sebagai set penilaian
clf.fit(iris.data, iris.target)
#estimator/parameter clf menggunakan metode fit dengan itemnya
list(clf.predict(iris.data[:3]))
#menambahkan item list dengan metode predict
clf.fit(iris.data, iris.target_names[iris.target])
#estimator/parameter clf menggunakan metode fit dengan itemnya
list(clf.predict(iris.data[:3]))
#menambahkan item list dengan metode predict




