# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:11:57 2020

@author: User
"""

#Melakukan klasifikasi Decision Tree
from sklearn import tree
clftree = tree.DecisionTreeClassifier()
clftree.fit(dk_train_att, dk_train_label)
clftree.score(dk_test_att, dk_test_label)