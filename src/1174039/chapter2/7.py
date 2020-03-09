# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:08:17 2020

@author: Liyana
"""

tree.export_graphviz(lobener, out_file="student-performance.dot", label="all", impurity=False, proportion=True,
                     feature_names=list(jatibarang_train_att), class_names=["fail", "pass"], filled=True, rounded=True)