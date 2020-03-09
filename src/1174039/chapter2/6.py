# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:05:14 2020

@author: Liyana
"""

# visualize tree
import graphviz
dot_data = tree.export_graphviz(lobener, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(jatibarang_train_att), class_names=["fail", "pass"],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph