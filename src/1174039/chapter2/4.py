# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:57:59 2020

@author: Liyana
"""

# shuffle rows
jatibarang = jatibarang.sample(frac=1)
# split training and testing data
jatibarang_train = jatibarang[:500]
jatibarang_test = jatibarang[500:]
jatibarang_train_att = jatibarang_train.drop(['pass'], axis=1)
jatibarang_train_pass = jatibarang_train['pass']
jatibarang_test_att = jatibarang_test.drop(['pass'], axis=1)
jatibarang_test_pass = jatibarang_test['pass']
jatibarang_att = jatibarang.drop(['pass'], axis=1)
jatibarang_pass = jatibarang['pass']
# number of passing students in whole dataset:
import numpy as np
print("Passing: %d out of %d (%.2f%%)" % (np.sum(jatibarang_pass), len(jatibarang_pass),100*float(np.sum(jatibarang_pass)) / len(jatibarang_pass)))