# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:03:32 2020

@author: User
"""

# save tree
tree.export_graphviz(lontong, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(pecel_train_att), class_names=["fail", "pass"], filled=True, rounded=True)