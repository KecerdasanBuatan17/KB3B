# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:03:38 2020

@author: User
"""

for max_depth in range(1, 100):
    lontong = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(lontong, pecel_att, pecel_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))