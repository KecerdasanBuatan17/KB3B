# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:14:06 2020

@author: User
"""

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree,dk_train_att,dk_train_label,cv=5)
scorerata2=scores.mean()
scorersd=scores.std()

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
# show average score and +/- two standard deviations away (covering 95
#% of scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

scorestree = cross_val_score(clftree, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scorestree.mean(), scorestree.std() * 2))

scoressvm = cross_val_score(clfsvm, dk_train_att, dk_train_label, cv=5)
print("Accuracy: %0.2f (+/- %0.2f)" % (scoressvm.mean(), scoressvm.std() * 2))