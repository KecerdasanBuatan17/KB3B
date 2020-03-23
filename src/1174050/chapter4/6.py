# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 22:13:25 2020

@author: User
"""

#Melakukan confusion matrix
from sklearn.metrics import confusion_matrix
pred_labels = clftree.predict(dk_test_att)
cm = confusion_matrix(dk_test_label, pred_labels)
cm