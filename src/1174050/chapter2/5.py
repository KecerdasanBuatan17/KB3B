# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 21:56:41 2020

@author: User
"""

# fit a decision tree
from sklearn import tree
lontong = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
lontong = lontong.fit(pecel,_train_att, pecel_train_pass)