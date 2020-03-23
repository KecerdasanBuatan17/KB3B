# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:00:11 2020

@author: User
"""

#Coding untuk melakukan klasifikasi SVM
from sklearn import svm
clfsvm = svm.SVC()
clfsvm.fit(dk_train_att, dk_train_label)
clfsvm.score(dk_test_att, dk_test_label)