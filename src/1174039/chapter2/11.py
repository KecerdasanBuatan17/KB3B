# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:11:44 2020

@author: Liyana
"""

depth_acc = np.empty((19,3), float)
i = 0
for max_depth in range(1, 20):
   lobener = tree.DecisionTreeClassifier(criterion="entropy", max_depth=max_depth)
   scores = cross_val_score(lobener , jatibarang_att, jatibarang_pass, cv=5)
   depth_acc[i,0] = max_depth
   depth_acc[i,1] = scores.mean()
   depth_acc[i,2] = scores.std() * 2
   i += 1

print(depth_acc)