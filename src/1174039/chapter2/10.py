# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:10:49 2020

@author: Liyana
"""

for max_depth in range(1, 100):
    lobener= tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
    scores = cross_val_score(lobener, jatibarang_att, jatibarang_pass, cv=5)
    print("Max depth: %d, Accuracy: %0.2f (+/- %0.2f)" % (max_depth, scores.mean(), scores.std() * 2))