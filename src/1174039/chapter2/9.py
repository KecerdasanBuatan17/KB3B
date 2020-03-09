# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 23:09:58 2020

@author: Liyana
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(lobener, jatibarang_att, jatibarang_pass, cv=5)
# show average score and +/- two standard deviations away
#(covering 95% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))