# -*- coding: utf-8 -*-
"""
Created on Wed Feb 26 23:10:03 2020

@author: User
"""

#Refitting and Updating Parameters
import numpy as np
#pada baris ini merupakan sebuah perintah untuk mengimport class svm dari np
from sklearn.svm import SVC
#pada baris ini merupakan sebuah perintah untuk mengimport class SVC dari packaged sklearn.svm
rng = np.random.RandomState(0)
#rng sebagai estimator/parameter dengan nilai suatu itemnya yaitu np.random.RandomState(0)
X = rng.rand(100, 10)
#X sebagai estimator/parameter dengan nilai item rng.rand
y = rng.binomial(1, 0.5, 100)
#y sebagai estimator/parameter dengan nilai item rng.binomial
X_test = rng.rand(5, 10)
#X test sebagai estimator/parameter dengan nilai item rng.rand
clf = SVC()
#clf sebagai estimator/parameter dan class SVC
clf.set_params(kernel='linear').fit(X, y)
#set params sebagai item
clf.predict(X_test)
#menggunakan metode predict
clf.set_params(kernel='rbf', gamma='scale').fit(X, y)
clf.predict(X_test)