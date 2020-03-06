# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 22:00:57 2020

@author: User
"""

# visualize tree
import graphviz
dot_data = tree.export_graphviz(lontong, out_file=None, label="all", impurity=False, proportion=True,
                                feature_names=list(pecel_train_att), class_names=["fail", "pass"],
                                filled=True, rounded=True)
graph = graphviz.Source(dot_data)
graph