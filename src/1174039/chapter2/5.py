# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:01:08 2020

@author: Liyana
"""

# fit a decision tree
from sklearn import tree
lobener = tree.DecisionTreeClassifier(criterion="entropy", max_depth=5)
lobener = lobener.fit(jatibarang_train_att, jatibarang_train_pass)